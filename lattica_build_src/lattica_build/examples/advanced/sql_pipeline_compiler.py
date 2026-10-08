"""Compile a small, explicit SQL subset into homomorphic selection pipelines.

The SQL query and table schema are public compilation inputs. Database values
and named query parameters are prepared as separate plaintext tensors and are
encrypted by the normal client runtime. Results retain their physical row
positions while encrypted.

Version intentionally supports one numeric table, simple projections, a
required ``WHERE`` clause, named parameters, ``>``/``<``, and ``AND``/``OR``.
Unsupported SQL is rejected during compilation instead of being approximated
with surprising semantics.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from numbers import Real
from typing import ClassVar, Literal, TypeAlias

import sqlglot
import torch
from sqlglot import exp
from sqlglot.errors import ParseError

from lattica_build.base_classes.hom_op import HomOp
from lattica_build.base_classes.hom_pipeline import HomomorphicPipeline
from lattica_build.base_classes.hom_value import HomValue
from lattica_build.operators.arithmetic.h_const_mul import HomConstMul
from lattica_build.operators.comparison.h_compare import HomCompare
from lattica_build.operators.composite.sequential import SequentialHomOp
from lattica_build.operators.fhe.h_bootstrap import Bootstrap
from lattica_build.operators.shape.h_reshape import HomReshape
from lattica_build.operators.shape.h_slice import HomSlice
from lattica_build.operators.shape.h_squeeze import HomSqueeze
from lattica_build.operators.shape.h_unsqueeze import HomUnsqueeze
from lattica_build.operators.slots.h_rotate_sum import HomRotateSum
from lattica_build.params.level_and_scale_tracing import (
    ModulusChain,
    init_active_rows_cols,
)
from lattica_build.params.params import HomParams

SqlColumnKind: TypeAlias = Literal["real", "integer"]
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_VALIDITY_CHANNEL = 0


class SqlCompileError(ValueError):
    """Raised when SQL is invalid or outside the supported encrypted subset."""


def _validate_identifier(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER_PATTERN.fullmatch(value):
        raise ValueError(
            f"{label} must be an unquoted SQL identifier containing only letters, "
            f"digits, and underscores; got {value!r}."
        )
    return value


def _validate_real(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a real number; got {value!r}.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite; got {value!r}.")
    return result


@dataclass(frozen=True, kw_only=True)
class SqlColumn:
    """Public numeric-column metadata used for packing and normalization."""

    name: str
    min_value: float
    max_value: float
    kind: SqlColumnKind = "real"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "name", _validate_identifier(self.name, label="column name")
        )
        minimum = _validate_real(self.min_value, label=f"{self.name}.min_value")
        maximum = _validate_real(self.max_value, label=f"{self.name}.max_value")
        if minimum >= maximum:
            raise ValueError(
                f"{self.name}.min_value must be smaller than max_value; "
                f"got [{minimum}, {maximum}]."
            )
        if self.kind not in ("real", "integer"):
            raise ValueError(
                f"{self.name}.kind must be 'real' or 'integer'; got {self.kind!r}."
            )
        object.__setattr__(self, "min_value", minimum)
        object.__setattr__(self, "max_value", maximum)


@dataclass(frozen=True, kw_only=True)
class SqlTableSchema:
    """Public metadata for one numeric SQL table."""

    name: str
    columns: tuple[SqlColumn, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "name", _validate_identifier(self.name, label="table name")
        )
        columns = tuple(self.columns)
        if not columns:
            raise ValueError("SqlTableSchema.columns cannot be empty.")
        if not all(isinstance(column, SqlColumn) for column in columns):
            raise TypeError(
                "SqlTableSchema.columns must contain only SqlColumn values."
            )

        names: set[str] = set()
        for column in columns:
            key = column.name.casefold()
            if key in names:
                raise ValueError(f"Duplicate SQL column name: {column.name!r}.")
            names.add(key)

        object.__setattr__(self, "columns", columns)


@dataclass(frozen=True, kw_only=True)
class SqlSelectOptions:
    """Compilation and decoding settings for encrypted SELECT."""

    dialect: str | None = None
    x_accuracy: int = 9
    y_accuracy: int = 10
    selection_threshold: float = 0.5
    bootstrap: Literal["auto", "never", "always"] = "auto"

    def __post_init__(self) -> None:
        if self.bootstrap not in ("auto", "never", "always"):
            raise ValueError("bootstrap must be auto, never, or always.")
        if self.dialect is not None and (
            not isinstance(self.dialect, str) or not self.dialect.strip()
        ):
            raise ValueError(
                "SqlSelectOptions.dialect must be None or a non-empty string."
            )
        if self.dialect is not None:
            object.__setattr__(self, "dialect", self.dialect.strip())
        for name in ("x_accuracy", "y_accuracy"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"SqlSelectOptions.{name} must be a positive integer.")

        threshold = _validate_real(
            self.selection_threshold,
            label="SqlSelectOptions.selection_threshold",
        )
        if not 0 < threshold < 1:
            raise ValueError(
                "SqlSelectOptions.selection_threshold must be between 0 and 1."
            )
        object.__setattr__(self, "selection_threshold", threshold)


@dataclass(frozen=True)
class _ParameterBinding:
    name: str
    column_index: int
    column_is_greater: bool


@dataclass(frozen=True)
class _Comparison:
    binding_index: int


@dataclass(frozen=True)
class _BooleanPredicate:
    operator: Literal["and", "or"]
    left: "_Predicate"
    right: "_Predicate"


_Predicate: TypeAlias = _Comparison | _BooleanPredicate


def _flatten_boolean_operands(
    predicate: _Predicate,
    *,
    operator: Literal["and", "or"],
) -> list[_Predicate]:
    if isinstance(predicate, _BooleanPredicate) and predicate.operator == operator:
        return [
            *_flatten_boolean_operands(predicate.left, operator=operator),
            *_flatten_boolean_operands(predicate.right, operator=operator),
        ]
    return [predicate]


def _build_balanced_boolean(
    operator: Literal["and", "or"],
    operands: list[_Predicate],
) -> _Predicate:
    while len(operands) > 1:
        next_level: list[_Predicate] = []
        for index in range(0, len(operands), 2):
            if index + 1 == len(operands):
                next_level.append(operands[index])
            else:
                next_level.append(
                    _BooleanPredicate(
                        operator=operator,
                        left=operands[index],
                        right=operands[index + 1],
                    )
                )
        operands = next_level
    return operands[0]


def _balance_predicate(predicate: _Predicate) -> _Predicate:
    """Balance associative Boolean chains to keep multiplicative depth logarithmic."""

    if isinstance(predicate, _Comparison):
        return predicate
    balanced = _BooleanPredicate(
        operator=predicate.operator,
        left=_balance_predicate(predicate.left),
        right=_balance_predicate(predicate.right),
    )
    return _build_balanced_boolean(
        balanced.operator,
        list(
            dict.fromkeys(
                _flatten_boolean_operands(balanced, operator=balanced.operator)
            )
        ),
    )


@dataclass(frozen=True)
class _SelectPlan:
    predicate: _Predicate
    parameter_bindings: tuple[_ParameterBinding, ...]
    projected_column_indices: tuple[int, ...]


@dataclass(frozen=True, kw_only=True)
class SqlPackingPlan:
    """Public shapes and periods; independent of the private table values."""

    row_count: int
    ring_slots: int
    row_slots: int
    binding_count: int
    pack_width: int
    pack_count: int
    working_slots: int
    output_channels: int

    @property
    def parameter_shape(self) -> tuple[int, int]:
        return (self.pack_count, self.working_slots)

    @property
    def output_shape(self) -> tuple[int, int]:
        return (self.output_channels, self.working_slots)


def _plan_packing(
    row_count: int, binding_count: int, output_columns: int, hom_params: HomParams
) -> SqlPackingPlan:
    if isinstance(row_count, bool) or not isinstance(row_count, int) or row_count <= 0:
        raise ValueError("row_count must be a positive integer.")
    ring_slots = hom_params.internal_n
    if row_count > ring_slots:
        raise ValueError(
            f"{row_count} rows exceed the ring capacity of {ring_slots} slots."
        )
    row_slots = 1 << (row_count - 1).bit_length()
    pack_width = min(ring_slots // row_slots, 1 << (binding_count - 1).bit_length())
    return SqlPackingPlan(
        row_count=row_count,
        ring_slots=ring_slots,
        row_slots=row_slots,
        binding_count=binding_count,
        pack_width=pack_width,
        pack_count=(binding_count + pack_width - 1) // pack_width,
        working_slots=pack_width * row_slots,
        output_channels=output_columns + 1,
    )


@dataclass(frozen=True, kw_only=True)
class SqlSelectResult:
    """Decoded, compact rows returned in SQL projection order.

    ``row_indices`` contains zero-based physical positions from the source
    table. Selected column tensors have the same compact length.
    Column tensors remain floating point because CKKS results are approximate,
    including columns declared with ``kind="integer"``.
    """

    row_indices: torch.Tensor
    columns: dict[str, torch.Tensor]

    def __getitem__(self, column_name: str) -> torch.Tensor:
        return self.columns[column_name]

    def __len__(self) -> int:
        return int(self.row_indices.numel())


def _normalize(values: torch.Tensor, column: SqlColumn) -> torch.Tensor:
    return (
        2.0 * (values - column.min_value) / (column.max_value - column.min_value) - 1.0
    )


def _denormalize(values: torch.Tensor, column: SqlColumn) -> torch.Tensor:
    return (values + 1.0) * (
        column.max_value - column.min_value
    ) / 2.0 + column.min_value


def _as_real_tensor(value: object, *, label: str) -> torch.Tensor:
    try:
        tensor = torch.as_tensor(value)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise TypeError(f"{label} must contain real numeric values.") from exc
    if tensor.dtype == torch.bool or tensor.is_complex():
        raise TypeError(f"{label} must contain real numeric values, excluding bool.")
    try:
        tensor = tensor.detach().to(dtype=torch.float64, device="cpu")
    except (TypeError, ValueError, RuntimeError) as exc:
        raise TypeError(f"{label} must contain real numeric values.") from exc
    if not bool(torch.all(torch.isfinite(tensor)).item()):
        raise ValueError(f"{label} must contain only finite values.")
    return tensor


def _validate_in_bounds(values: torch.Tensor, column: SqlColumn, *, label: str) -> None:
    outside = (values < column.min_value) | (values > column.max_value)
    if bool(torch.any(outside).item()):
        observed_min = float(torch.min(values).item())
        observed_max = float(torch.max(values).item())
        raise ValueError(
            f"{label} must be within [{column.min_value}, {column.max_value}]; "
            f"observed [{observed_min}, {observed_max}]."
        )


def _prepare_database_tensor(
    data: Mapping[str, object],
    *,
    schema: SqlTableSchema,
    expected_row_count: int | None = None,
) -> torch.Tensor:
    """Validate and normalize a private database with one exact row count."""

    if not isinstance(data, Mapping):
        raise TypeError("database must be a mapping of column names to values.")

    expected_names = tuple(column.name for column in schema.columns)
    missing = [name for name in expected_names if name not in data]
    unknown = [name for name in data if name not in expected_names]
    if missing or unknown:
        raise ValueError(
            "Database columns must match the schema exactly. "
            f"Missing: {missing or 'none'}; unknown: {unknown or 'none'}."
        )

    prepared: list[torch.Tensor] = []
    row_count: int | None = None
    for column in schema.columns:
        values = _as_real_tensor(data[column.name], label=f"column {column.name!r}")
        if values.ndim != 1:
            raise ValueError(
                f"column {column.name!r} must be one-dimensional; "
                f"got shape {tuple(values.shape)}."
            )
        if row_count is None:
            row_count = len(values)
            if row_count == 0:
                raise ValueError("Database must contain at least one row.")
            if expected_row_count is not None and row_count != expected_row_count:
                raise ValueError(
                    f"Database has {row_count} rows but the compiled pipeline expects "
                    f"{expected_row_count}. Recompile to change the row count."
                )
        elif len(values) != row_count:
            raise ValueError(
                "All database columns must have the same row count; "
                f"expected {row_count}, got {len(values)} for {column.name!r}."
            )
        _validate_in_bounds(values, column, label=f"column {column.name!r}")
        if column.kind == "integer" and bool(
            torch.any(values != torch.round(values)).item()
        ):
            raise ValueError(f"column {column.name!r} must contain integer values.")
        prepared.append(_normalize(values, column))

    assert row_count is not None
    packed = torch.empty(
        (len(schema.columns) + 1, row_count),
        dtype=torch.float64,
    )
    packed[_VALIDITY_CHANNEL].fill_(1.0)
    for column_index, values in enumerate(prepared):
        packed[column_index + 1] = values
    return packed


def _pad_last_dimension(
    values: torch.Tensor,
    *,
    size: int,
) -> torch.Tensor:
    """Zero-pad a two-dimensional tensor along its final dimension."""

    if values.ndim != 2 or values.shape[1] > size:
        raise ValueError(
            f"Cannot pad tensor with shape {tuple(values.shape)} to final size {size}."
        )
    padded = torch.zeros(
        (int(values.shape[0]), size),
        dtype=values.dtype,
        device=values.device,
    )
    padded[:, : values.shape[1]] = values
    return padded


def _normalize_comparison(values: torch.Tensor, column: SqlColumn) -> torch.Tensor:
    # Integer comparisons use half-integer boundaries, including just outside
    # the public bounds. Leave room for those boundaries in the fit domain.
    padding = 0.5 if column.kind == "integer" else 0.0
    return (values - (column.min_value - padding)) / (
        column.max_value - column.min_value + 2 * padding
    ) - 0.5


@dataclass(frozen=True, kw_only=True)
class CompiledSqlSelect:
    """An immutable plan, its graph, and codecs for runtime private inputs."""

    PARAMETERS_INPUT_NAME: ClassVar[str] = "parameters"
    COMPARISONS_INPUT_NAME: ClassVar[str] = "comparison_database"
    DATABASE_INPUT_NAME: ClassVar[str] = "database"

    sql: str
    schema: SqlTableSchema
    hom_params: HomParams
    options: SqlSelectOptions
    pipeline: HomomorphicPipeline
    packing: SqlPackingPlan
    parameter_names: tuple[str, ...]
    projected_columns: tuple[SqlColumn, ...]
    _plan: _SelectPlan = field(repr=False)
    prepared_database: dict[str, torch.Tensor] | None = field(
        default=None, repr=False, compare=False
    )

    @property
    def row_count(self) -> int:
        return self.packing.row_count

    @property
    def subring_slots(self) -> int:
        return self.packing.row_slots

    @property
    def parameter_binding_count(self) -> int:
        return self.packing.binding_count

    @property
    def parameter_pack_width(self) -> int:
        return self.packing.pack_width

    @property
    def parameter_ciphertext_count(self) -> int:
        return self.packing.pack_count

    @property
    def parameter_shape(self) -> tuple[int, int]:
        return self.packing.parameter_shape

    @property
    def database_shape(self) -> tuple[int, int]:
        """Shape of the projected database view (including validity)."""
        return self.packing.output_shape

    @property
    def output_shape(self) -> tuple[int, int]:
        return self.packing.output_shape

    @property
    def bootstrap_count(self) -> int:
        return len(self.pipeline.hom.bootstrap_predicates)

    def prepare_database(self, data: Mapping[str, object]) -> dict[str, torch.Tensor]:
        """Prepare both encrypted views once per database upload/update.

        The returned mapping is passed directly to encrypt_and_upload_custom_data.
        Query thresholds are absent from these reusable views.
        """
        logical = _prepare_database_tensor(
            data, schema=self.schema, expected_row_count=self.row_count
        )
        output_indices = (0, *(i + 1 for i in self._plan.projected_column_indices))
        output = _pad_last_dimension(
            logical[list(output_indices)], size=self.subring_slots
        )
        output = output.repeat(1, self.parameter_pack_width)
        comparisons = torch.zeros(self.parameter_shape, dtype=torch.float64)
        for index, binding in enumerate(self._plan.parameter_bindings):
            column = self.schema.columns[binding.column_index]
            values = _as_real_tensor(data[column.name], label=f"column {column.name!r}")
            direction = 1 if binding.column_is_greater else -1
            pack, block = divmod(index, self.parameter_pack_width)
            start = block * self.subring_slots
            comparisons[pack, start : start + self.row_count] = (
                direction * _normalize_comparison(values, column)
            )
        return {
            self.COMPARISONS_INPUT_NAME: comparisons,
            self.DATABASE_INPUT_NAME: output,
        }

    def prepare_parameters(self, values: Mapping[str, object]) -> torch.Tensor:
        """Pack runtime thresholds using the same normalization as the database."""
        if not isinstance(values, Mapping):
            raise TypeError(
                "prepare_parameters expects a mapping of parameter names to values."
            )
        missing = [name for name in self.parameter_names if name not in values]
        unknown = [name for name in values if name not in self.parameter_names]
        if missing or unknown:
            raise ValueError(
                f"Query parameters must match SQL placeholders. Missing: {missing}; unknown: {unknown}."
            )
        prepared = torch.zeros(self.parameter_shape, dtype=torch.float64)
        for index, binding in enumerate(self._plan.parameter_bindings):
            column = self.schema.columns[binding.column_index]
            value = _as_real_tensor(
                values[binding.name], label=f"parameter :{binding.name}"
            )
            if value.numel() != 1:
                raise ValueError(f"parameter :{binding.name} must be a scalar.")
            value = value.reshape(())
            _validate_in_bounds(value, column, label=f"parameter :{binding.name}")
            if column.kind == "integer":
                # x > t iff x > floor(t)+1/2, and x < t iff x < ceil(t)-1/2.
                # Equality is therefore separated from the sign polynomial's transition.
                value = (
                    torch.floor(value) + 0.5
                    if binding.column_is_greater
                    else torch.ceil(value) - 0.5
                )
            direction = 1 if binding.column_is_greater else -1
            pack, block = divmod(index, self.parameter_pack_width)
            start = block * self.subring_slots
            prepared[pack, start : start + self.subring_slots] = (
                direction * _normalize_comparison(value, column)
            )
        return prepared

    def apply_clear(
        self,
        prepared_parameters: torch.Tensor,
        *,
        prepared_database: Mapping[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Exact predicate reference, independent of the approximate operator graph."""
        parameters = _validate_prepared_tensor(
            prepared_parameters,
            expected_shape=self.parameter_shape,
            label="prepared SQL parameters",
        )
        data = (
            self.prepared_database if prepared_database is None else prepared_database
        )
        if data is None:
            raise ValueError(
                "Supply prepared_database when compiling with row_count only."
            )
        expected_names = {self.COMPARISONS_INPUT_NAME, self.DATABASE_INPUT_NAME}
        if set(data) != expected_names:
            raise ValueError(
                f"Prepared database must contain exactly {sorted(expected_names)}."
            )
        comparisons = _validate_prepared_tensor(
            data[self.COMPARISONS_INPUT_NAME],
            expected_shape=self.parameter_shape,
            label="prepared comparison database",
        )
        output = _validate_prepared_tensor(
            data[self.DATABASE_INPUT_NAME],
            expected_shape=self.output_shape,
            label="prepared output database",
        )
        values = []
        for index in range(self.parameter_binding_count):
            pack, block = divmod(index, self.parameter_pack_width)
            start = block * self.subring_slots
            block_slice = slice(start, start + self.subring_slots)
            values.append(
                comparisons[pack, block_slice] > parameters[pack, block_slice]
            )

        def evaluate(predicate):
            if isinstance(predicate, _Comparison):
                return values[predicate.binding_index]
            left, right = evaluate(predicate.left), evaluate(predicate.right)
            return left & right if predicate.operator == "and" else left | right

        selected = evaluate(self._plan.predicate).repeat(self.parameter_pack_width)
        return output * selected.to(output.dtype).unsqueeze(0)

    def decode_result(self, result: torch.Tensor) -> SqlSelectResult:
        """Compact physical rows, preserving projection order and original row indices."""
        decoded = _validate_prepared_tensor(
            result, expected_shape=self.output_shape, label="decrypted SQL result"
        )
        decoded = decoded[:, : self.row_count]
        selected = decoded[0] > self.options.selection_threshold
        columns = {
            column.name: _denormalize(decoded[i + 1, selected], column)
            for i, column in enumerate(self.projected_columns)
        }
        return SqlSelectResult(
            row_indices=torch.nonzero(selected, as_tuple=False).flatten(),
            columns=columns,
        )


def _validate_prepared_tensor(
    value: object, *, expected_shape: tuple[int, int], label: str
) -> torch.Tensor:
    tensor = _as_real_tensor(value, label=label)
    if tuple(tensor.shape) != expected_shape:
        raise ValueError(
            f"{label} must have shape {expected_shape}; got {tuple(tensor.shape)}."
        )
    return tensor


class _RotateAndAdd(HomOp):
    """Explicit x + rotate(x), with identical clear and encrypted semantics."""

    def __init__(self, offset: int):
        super().__init__()
        self.rotate = HomRotateSum(rotations=(offset,), perform_sum=False)
        self.squeeze = HomSqueeze(dim=0)

    def forward(self, x):
        return x + self.squeeze(self.rotate(x))


def _build_comparison_unpacking_pipeline(
    packing: SqlPackingPlan,
) -> SequentialHomOp | None:
    if packing.pack_width == 1:
        return None
    mask = torch.zeros(
        (1, packing.pack_width, packing.working_slots), dtype=torch.float64
    )
    for block in range(packing.pack_width):
        mask[0, block, block * packing.row_slots : (block + 1) * packing.row_slots] = 1
    mask_op = HomConstMul(dims=tuple(mask.shape))
    mask_op.set_data(mask)
    ops = [
        HomUnsqueeze(dim=1),
        mask_op,
        HomReshape((packing.pack_count * packing.pack_width, packing.working_slots)),
    ]
    # Remove dummy lanes before rotating; their encrypted work is unnecessary.
    if packing.pack_count * packing.pack_width != packing.binding_count:
        ops.append(HomSlice(dim=0, key=slice(0, packing.binding_count)))
    ops.extend(
        _RotateAndAdd(packing.row_slots * (1 << stage))
        for stage in range(packing.pack_width.bit_length() - 1)
    )
    return SequentialHomOp(*ops)


def _plan_bootstraps(
    plan: _SelectPlan,
    compare: HomCompare,
    packing: SqlPackingPlan,
    params: HomParams,
    policy: str,
) -> frozenset[_Predicate]:
    """Budget the comparator with the real tracer; schedule Boolean refreshes.

    Reserve two modulus columns at the output. The exact serialized modulus
    and scale are checked separately before returning a compiled pipeline.
    """
    probe = replace(params)
    probe.mod_chain = ModulusChain(probe)
    rows, cols = init_active_rows_cols(probe)
    available = sum(map(sum, cols))
    value = HomValue(
        id="a",
        tensor_shape=packing.parameter_shape,
        n_axis=1,
        n_slots=packing.working_slots,
        active_rows=rows,
        active_cols=cols,
        pt_scale=probe.pt_scale,
    )
    try:
        _, result = compare.serialize(
            {}, value, value.make_copy(id="b"), hom_params=probe
        )
    except (ValueError, RuntimeError, IndexError) as exc:
        raise SqlCompileError(
            "The comparison exceeds the modulus budget; increase it or reduce comparison accuracy."
        ) from exc
    comparison_depth = available - sum(map(sum, result.active_cols))
    limit = available - 2
    trunk_depth = comparison_depth + (packing.pack_width > 1)
    if trunk_depth > limit:
        raise SqlCompileError(
            "The comparison/unpacking leaves insufficient modulus headroom; increase the modulus budget."
        )
    refresh = set()
    depths = {}

    def refresh_at(predicate):
        if policy == "never":
            raise SqlCompileError(
                "The query needs bootstrapping or a larger modulus budget (bootstrap='never')."
            )
        refresh.add(predicate)
        depths[predicate] = 0

    def depth(predicate):
        if predicate in depths:
            return depths[predicate]
        if isinstance(predicate, _Comparison):
            result = trunk_depth
        else:
            depth(predicate.left)
            depth(predicate.right)
            for child in (predicate.left, predicate.right):
                if depths[child] + 1 > limit:
                    refresh_at(child)
            result = max(depths[predicate.left], depths[predicate.right]) + 1
        depths[predicate] = result
        return result

    if depth(plan.predicate) + 1 > limit or policy == "always":
        refresh_at(plan.predicate)
    if refresh and params.num_init_rows is not None:
        # A reduced initial chain needs a separate schedule from the full chain
        # after bootstrap. Refuse to silently build with an incorrect budget.
        raise SqlCompileError(
            "SQL automatic bootstrap planning requires num_init_rows=None."
        )
    return frozenset(refresh)


class HomSqlPipeline(HomOp):
    def __init__(
        self,
        plan: _SelectPlan,
        packing: SqlPackingPlan,
        options: SqlSelectOptions,
        hom_params: HomParams,
    ):
        super().__init__()
        self._predicate = plan.predicate
        self.parameters_shape = HomReshape(packing.parameter_shape)
        self.compare = HomCompare(
            x_accuracy=options.x_accuracy,
            y_accuracy=options.y_accuracy,
            left=-0.5,
            right=0.5,
        )
        self.unpack_comparisons = _build_comparison_unpacking_pipeline(packing)
        self.bootstrap_predicates = _plan_bootstraps(
            plan, self.compare, packing, hom_params, options.bootstrap
        )
        # Do not register an unused Bootstrap: its presence adds bootstrap primes
        # and keys even if forward never calls it.
        self.bootstrap = (
            Bootstrap(target_output_scale=hom_params.pt_scale)
            if self.bootstrap_predicates
            else None
        )

    def _evaluate_predicate(self, predicate, values, cache):
        if predicate in cache:
            return cache[predicate]
        if isinstance(predicate, _Comparison):
            result = HomSlice(dim=0, key=predicate.binding_index)(values)
        else:
            left = self._evaluate_predicate(predicate.left, values, cache)
            right = self._evaluate_predicate(predicate.right, values, cache)
            both = left * right
            result = both if predicate.operator == "and" else left + right - both
        if predicate in self.bootstrap_predicates:
            result = self.bootstrap(result)
        cache[predicate] = result
        return result

    def forward(
        self, parameters: HomValue, comparison_database: HomValue, database: HomValue
    ) -> HomValue:
        # Establish the primary ciphertext state before the backend binds a
        # custom input to it. This reshape only changes metadata.
        parameters = self.parameters_shape(parameters)
        values = self.compare(comparison_database, parameters)
        if self.unpack_comparisons is not None:
            values = self.unpack_comparisons(values)
        predicate = self._evaluate_predicate(self._predicate, values, {})
        return database * predicate


def _matches_identifier(identifier: exp.Identifier, expected: str) -> bool:
    actual = identifier.name
    if identifier.args.get("quoted"):
        return actual == expected
    return actual.casefold() == expected.casefold()


def _resolve_column(expression: exp.Column, schema: SqlTableSchema) -> int:
    if expression.db or expression.catalog:
        raise SqlCompileError(
            f"Qualified database/catalog names are not supported: {expression.sql()}."
        )
    if expression.table:
        table_identifier = expression.args.get("table")
        if not isinstance(table_identifier, exp.Identifier) or not _matches_identifier(
            table_identifier, schema.name
        ):
            raise SqlCompileError(
                f"Column {expression.sql()!r} does not belong to table {schema.name!r}."
            )

    identifier = expression.args.get("this")
    if not isinstance(identifier, exp.Identifier):
        raise SqlCompileError(f"Expected a simple column, got {expression.sql()!r}.")
    for index, column in enumerate(schema.columns):
        if _matches_identifier(identifier, column.name):
            return index
    available = ", ".join(column.name for column in schema.columns)
    raise SqlCompileError(
        f"Unknown column {expression.name!r}. Available columns: {available}."
    )


def _placeholder_name(expression: exp.Placeholder) -> str:
    name = expression.this
    if not isinstance(name, str) or not _IDENTIFIER_PATTERN.fullmatch(name):
        raise SqlCompileError(
            "Only named parameters such as :threshold are supported; "
            f"got {expression.sql()!r}."
        )
    return name


def _compile_predicate(
    expression: exp.Expression,
    *,
    schema: SqlTableSchema,
    bindings: list[_ParameterBinding],
) -> _Predicate:
    if isinstance(expression, exp.Paren):
        return _compile_predicate(expression.this, schema=schema, bindings=bindings)
    if isinstance(expression, (exp.And, exp.Or)):
        operator: Literal["and", "or"] = (
            "and" if isinstance(expression, exp.And) else "or"
        )
        return _BooleanPredicate(
            operator=operator,
            left=_compile_predicate(expression.this, schema=schema, bindings=bindings),
            right=_compile_predicate(
                expression.expression, schema=schema, bindings=bindings
            ),
        )
    if not isinstance(expression, (exp.GT, exp.LT)):
        raise SqlCompileError(
            "WHERE supports only >, <, AND, OR, and parentheses; "
            f"got {expression.sql()!r}."
        )

    left = expression.this
    right = expression.expression
    if isinstance(left, exp.Column) and isinstance(right, exp.Placeholder):
        column_index = _resolve_column(left, schema)
        parameter_name = _placeholder_name(right)
        column_is_greater = isinstance(expression, exp.GT)
    elif isinstance(left, exp.Placeholder) and isinstance(right, exp.Column):
        column_index = _resolve_column(right, schema)
        parameter_name = _placeholder_name(left)
        column_is_greater = isinstance(expression, exp.LT)
    else:
        raise SqlCompileError(
            "Each comparison must contain one column and one named parameter; "
            f"got {expression.sql()!r}."
        )

    binding = _ParameterBinding(
        name=parameter_name,
        column_index=column_index,
        column_is_greater=column_is_greater,
    )
    if binding not in bindings:
        bindings.append(binding)
    binding_index = bindings.index(binding)
    return _Comparison(binding_index=binding_index)


def _parse_select(
    sql: str,
    *,
    schema: SqlTableSchema,
    options: SqlSelectOptions,
) -> _SelectPlan:
    if not isinstance(sql, str) or not sql.strip():
        raise SqlCompileError("SQL must be a non-empty string.")
    try:
        statements = sqlglot.parse(sql, read=options.dialect)
    except (ParseError, ValueError) as exc:
        raise SqlCompileError(f"Invalid SQL: {exc}") from exc
    statements = [statement for statement in statements if statement is not None]
    if len(statements) != 1:
        raise SqlCompileError("Exactly one SQL statement is required.")

    statement = statements[0]
    if not isinstance(statement, exp.Select):
        raise SqlCompileError("Only SELECT statements are supported.")
    if statement.args.get("distinct"):
        raise SqlCompileError("SELECT DISTINCT is not supported.")
    for clause in (
        "group",
        "having",
        "order",
        "limit",
        "offset",
        "qualify",
        "with_",
        "hint",
        "exclude",
        "operation_modifiers",
    ):
        if statement.args.get(clause) is not None:
            raise SqlCompileError(f"{clause.rstrip('_').upper()} is not supported.")
    if statement.args.get("joins"):
        raise SqlCompileError("JOIN is not supported.")

    from_expression = statement.args.get("from_")
    if not isinstance(from_expression, exp.From) or not isinstance(
        from_expression.this, exp.Table
    ):
        raise SqlCompileError("SELECT must read from exactly one named table.")
    tables = list(statement.find_all(exp.Table))
    if len(tables) != 1:
        raise SqlCompileError("SELECT must read from exactly one named table.")
    table = tables[0]
    if table.alias or table.db or table.catalog:
        raise SqlCompileError(
            "Table aliases and database/catalog qualifiers are not supported."
        )
    table_identifier = table.args.get("this")
    if not isinstance(table_identifier, exp.Identifier) or not _matches_identifier(
        table_identifier, schema.name
    ):
        raise SqlCompileError(
            f"SQL table {table.name!r} does not match schema table {schema.name!r}."
        )

    projections = tuple(statement.expressions)
    if not projections:
        raise SqlCompileError("SELECT must project at least one column or *.")
    if len(projections) == 1 and isinstance(projections[0], exp.Star):
        if any(value for value in projections[0].args.values()):
            raise SqlCompileError("SELECT * modifiers are not supported.")
        projected_indices = tuple(range(len(schema.columns)))
    else:
        if any(isinstance(projection, exp.Star) for projection in projections):
            raise SqlCompileError("SELECT * cannot be mixed with explicit columns.")
        projected_indices_list: list[int] = []
        for projection in projections:
            if not isinstance(projection, exp.Column):
                raise SqlCompileError(
                    "SELECT projections must be simple columns without aliases or expressions; "
                    f"got {projection.sql()!r}."
                )
            projected_indices_list.append(_resolve_column(projection, schema))
        if len(set(projected_indices_list)) != len(projected_indices_list):
            raise SqlCompileError("Duplicate projected columns are not supported.")
        projected_indices = tuple(projected_indices_list)

    where = statement.args.get("where")
    if not isinstance(where, exp.Where):
        raise SqlCompileError("SELECT requires a WHERE clause.")
    bindings: list[_ParameterBinding] = []
    predicate = _balance_predicate(
        _compile_predicate(where.this, schema=schema, bindings=bindings)
    )
    return _SelectPlan(
        predicate=predicate,
        parameter_bindings=tuple(bindings),
        projected_column_indices=projected_indices,
    )


def compile_sql_select(
    sql: str,
    *,
    schema: SqlTableSchema,
    hom_params: HomParams,
    database: Mapping[str, object] | None = None,
    row_count: int | None = None,
    options: SqlSelectOptions | None = None,
) -> CompiledSqlSelect:
    """Compile a public query and shape; private inputs can be prepared later.

    Supply either row_count or database. The latter is a convenience which also
    prepares the two database views; neither view is embedded in the graph.
    Real comparisons retain a transition band around equality. Integer columns
    use half-integer thresholds so strict > and < exclude equal values.
    """
    if not isinstance(schema, SqlTableSchema):
        raise TypeError("schema must be a SqlTableSchema.")
    if not isinstance(hom_params, HomParams):
        raise TypeError("hom_params must be a HomParams.")
    if options is None:
        options = SqlSelectOptions()
    elif not isinstance(options, SqlSelectOptions):
        raise TypeError("options must be a SqlSelectOptions or None.")
    if (database is None) == (row_count is None):
        raise ValueError("Supply exactly one of database or row_count.")
    plan = _parse_select(sql, schema=schema, options=options)
    if database is not None:
        row_count = int(_prepare_database_tensor(database, schema=schema).shape[1])
    packing = _plan_packing(
        row_count,
        len(plan.parameter_bindings),
        len(plan.projected_column_indices),
        hom_params,
    )
    # Keep the caller's reusable HE configuration unchanged.
    params = replace(hom_params, n_slots=packing.working_slots)
    hom = HomSqlPipeline(plan, packing, options, params)
    pipeline = HomomorphicPipeline(
        hom=hom,
        input_shape={
            CompiledSqlSelect.PARAMETERS_INPUT_NAME: packing.parameter_shape,
            CompiledSqlSelect.COMPARISONS_INPUT_NAME: packing.parameter_shape,
            CompiledSqlSelect.DATABASE_INPUT_NAME: packing.output_shape,
        },
        custom_n_slots={
            CompiledSqlSelect.COMPARISONS_INPUT_NAME: packing.working_slots,
            CompiledSqlSelect.DATABASE_INPUT_NAME: packing.working_slots,
        },
        n_axis=-1,
    )
    # Reject empty/exhausted output chains, which the general tracer can serialize.
    # Check actual primes and scale, rather than assuming every level has 30 bits.
    try:
        graph, _ = pipeline.serialize(params)
        output = json.loads(graph)["pipeline_sections"]["hom"]["body_output"]
        modulus_bits = sum(
            math.log2(params.mod_chain.factors_per_row[row][col])
            for row, cols in zip(output["active_rows"], output["active_cols"])
            for col, active in enumerate(cols)
            if active
        )
        scale_bits = math.log2(float(output["pt_scale"]))
        if modulus_bits - scale_bits < 20:
            raise SqlCompileError(
                "The output has insufficient modulus headroom; increase the modulus budget or enable bootstrapping."
            )
    except (ValueError, RuntimeError, IndexError) as exc:
        if isinstance(exc, SqlCompileError):
            raise
        raise SqlCompileError(
            "Cannot fit SQL into the supplied HE parameters; increase the modulus budget."
        ) from exc
    compiled = CompiledSqlSelect(
        sql=sql,
        schema=schema,
        hom_params=params,
        options=options,
        pipeline=pipeline,
        packing=packing,
        parameter_names=tuple(dict.fromkeys(b.name for b in plan.parameter_bindings)),
        projected_columns=tuple(
            schema.columns[i] for i in plan.projected_column_indices
        ),
        _plan=plan,
    )
    if database is not None:
        compiled = replace(
            compiled, prepared_database=compiled.prepare_database(database)
        )
    return compiled

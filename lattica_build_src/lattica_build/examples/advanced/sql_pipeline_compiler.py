"""Compile a small, explicit SQL subset into homomorphic selection pipelines.

The SQL query and table schema are public compilation inputs. Database values
and named query parameters are prepared as separate plaintext tensors and are
encrypted by the normal client runtime. Results retain their physical row
positions while encrypted;

Version intentionally supports one numeric table, simple projections, a
required ``WHERE`` clause, named parameters, ``>``/``<``, and ``AND``/``OR``.
Unsupported SQL is rejected during compilation instead of being approximated
with surprising semantics.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Real
from typing import ClassVar, Literal, TypeAlias

import sqlglot
import torch
from sqlglot import exp
from sqlglot.errors import ParseError

from lattica_build.base_classes.hom_pipeline import HomomorphicPipeline
from lattica_build.base_classes.hom_value import HomValue
from lattica_build.base_classes.hom_op import HomOp
from lattica_build.operators.arithmetic.h_axis_sum import HomAxisSum
from lattica_build.operators.arithmetic.h_const_mul import HomConstMul
from lattica_build.operators.client_ops import Repeat
from lattica_build.operators.comparison.h_compare import HomCompare
from lattica_build.operators.composite.sequential import SequentialHomOp
from lattica_build.operators.fhe.h_bootstrap import Bootstrap
from lattica_build.operators.shape.h_reshape import HomReshape
from lattica_build.operators.shape.h_slice import HomSlice
from lattica_build.operators.shape.h_unsqueeze import HomUnsqueeze
from lattica_build.operators.slots.h_rotate_sum import HomRotateSum
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
        object.__setattr__(self, "name", _validate_identifier(self.name, label="column name"))
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
        object.__setattr__(self, "name", _validate_identifier(self.name, label="table name"))
        columns = tuple(self.columns)
        if not columns:
            raise ValueError("SqlTableSchema.columns cannot be empty.")
        if not all(isinstance(column, SqlColumn) for column in columns):
            raise TypeError("SqlTableSchema.columns must contain only SqlColumn values.")

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

    def __post_init__(self) -> None:
        if self.dialect is not None and (
            not isinstance(self.dialect, str) or not self.dialect.strip()
        ):
            raise ValueError("SqlSelectOptions.dialect must be None or a non-empty string.")
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
            raise ValueError("SqlSelectOptions.selection_threshold must be between 0 and 1.")
        object.__setattr__(self, "selection_threshold", threshold)


@dataclass(frozen=True)
class _ParameterBinding:
    name: str
    column_index: int


@dataclass(frozen=True)
class _Comparison:
    column_index: int
    binding_index: int
    column_is_greater: bool


@dataclass(frozen=True)
class _BooleanPredicate:
    operator: Literal["and", "or"]
    left: "_Predicate"
    right: "_Predicate"


_Predicate: TypeAlias = _Comparison | _BooleanPredicate


def _comparison_directions(predicate: _Predicate) -> dict[int, bool]:
    """Return whether the column is the greater operand for each binding."""

    if isinstance(predicate, _Comparison):
        return {predicate.binding_index: predicate.column_is_greater}
    return {
        **_comparison_directions(predicate.left),
        **_comparison_directions(predicate.right),
    }


def _flatten_boolean_operands(
    predicate: _Predicate,
    *,
    operator: Literal["and", "or"],
) -> list[_Predicate]:
    if (
        isinstance(predicate, _BooleanPredicate)
        and predicate.operator == operator
    ):
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
        _flatten_boolean_operands(balanced, operator=balanced.operator),
    )


@dataclass(frozen=True)
class _SelectPlan:
    predicate: _Predicate
    parameter_bindings: tuple[_ParameterBinding, ...]
    projected_column_indices: tuple[int, ...]
    projects_all_columns: bool


@dataclass(frozen=True)
class _SubringLayout:
    active_slots: int
    main_slots: int
    subring_slots: int
    log_n_subring: int
    repetitions: int


@dataclass(frozen=True)
class _ParameterLayout:
    binding_count: int
    pack_width: int
    pack_count: int
    packed_slots: int

    @property
    def input_shape(self) -> tuple[int, int]:
        return (self.pack_count, self.packed_slots)


def _subring_layout_for_active_slots(
    active_slots: int,
    *,
    ring_dimension: int,
) -> _SubringLayout:
    """Derive the smallest power-of-two subring that holds all logical rows."""

    if (
        isinstance(active_slots, bool)
        or not isinstance(active_slots, int)
        or active_slots <= 0
    ):
        raise ValueError(
            f"active_slots must be a positive integer; got {active_slots!r}."
        )
    if (
        isinstance(ring_dimension, bool)
        or not isinstance(ring_dimension, int)
        or ring_dimension < 2
        or ring_dimension & (ring_dimension - 1)
    ):
        raise ValueError(
            "ring_dimension must be a power of two greater than or equal to 2; "
            f"got {ring_dimension!r}."
        )

    main_slots = ring_dimension // 2
    if active_slots > main_slots:
        raise ValueError(
            f"{active_slots} active slots exceed the main ring capacity of "
            f"{main_slots} slots."
        )

    subring_slots = 1 << (active_slots - 1).bit_length()
    return _SubringLayout(
        active_slots=active_slots,
        main_slots=main_slots,
        subring_slots=subring_slots,
        log_n_subring=subring_slots.bit_length(),
        repetitions=main_slots // subring_slots,
    )


def _parameter_layout_for_bindings(
    binding_count: int,
    *,
    subring_layout: _SubringLayout,
) -> _ParameterLayout:
    """Pack comparison bindings into the unused row-sized SIMD blocks."""

    if (
        isinstance(binding_count, bool)
        or not isinstance(binding_count, int)
        or binding_count <= 0
    ):
        raise ValueError(
            f"binding_count must be a positive integer; got {binding_count!r}."
        )

    desired_width = 1 << (binding_count - 1).bit_length()
    pack_width = min(subring_layout.repetitions, desired_width)
    pack_count = (binding_count + pack_width - 1) // pack_width
    return _ParameterLayout(
        binding_count=binding_count,
        pack_width=pack_width,
        pack_count=pack_count,
        packed_slots=pack_width * subring_layout.subring_slots,
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
        2.0 * (values - column.min_value) / (column.max_value - column.min_value)
        - 1.0
    )


def _denormalize(values: torch.Tensor, column: SqlColumn) -> torch.Tensor:
    return (
        (values + 1.0) * (column.max_value - column.min_value) / 2.0
        + column.min_value
    )


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


def _repeat_subring_block(
    values: torch.Tensor,
    *,
    repetitions: int,
) -> torch.Tensor:
    """Repeat each row's complete subring block across the main ring."""

    if values.ndim != 2:
        raise ValueError(
            f"Expected a two-dimensional tensor; got {values.ndim} dimensions."
        )
    return values.repeat(1, repetitions)


@dataclass(frozen=True, kw_only=True)
class CompiledSqlSelect:
    """A built pipeline plus its private input and output codecs."""

    PARAMETERS_INPUT_NAME: ClassVar[str] = "parameters"
    DATABASE_INPUT_NAME: ClassVar[str] = "database"

    sql: str
    schema: SqlTableSchema
    hom_params: HomParams
    options: SqlSelectOptions
    pipeline: HomomorphicPipeline
    prepared_database: torch.Tensor = field(repr=False, compare=False)
    parameter_names: tuple[str, ...]
    projected_columns: tuple[SqlColumn, ...]
    _parameter_bindings: tuple[_ParameterBinding, ...] = field(repr=False)
    _predicate: _Predicate = field(repr=False)
    _projected_column_indices: tuple[int, ...] = field(repr=False)
    _subring_layout: _SubringLayout = field(repr=False)
    _parameter_layout: _ParameterLayout = field(repr=False)

    @property
    def database_shape(self) -> tuple[int, int]:
        return (
            int(self.prepared_database.shape[0]),
            int(self.prepared_database.shape[1]),
        )

    @property
    def row_count(self) -> int:
        return self._subring_layout.active_slots

    @property
    def subring_slots(self) -> int:
        return self._subring_layout.subring_slots

    @property
    def log_n_subring(self) -> int:
        return self._subring_layout.log_n_subring

    @property
    def repetitions(self) -> int:
        return self._subring_layout.repetitions

    @property
    def parameter_binding_count(self) -> int:
        return self._parameter_layout.binding_count

    @property
    def parameter_pack_width(self) -> int:
        return self._parameter_layout.pack_width

    @property
    def parameter_ciphertext_count(self) -> int:
        return self._parameter_layout.pack_count

    @property
    def parameter_shape(self) -> tuple[int, int]:
        """Packed client input shape before repetition to the main ring."""

        return self._parameter_layout.input_shape

    @property
    def output_shape(self) -> tuple[int, int]:
        return self.database_shape

    def prepare_database(self, data: Mapping[str, object]) -> torch.Tensor:
        """Prepare replacement values with the compiled database's row count."""

        logical_database = _prepare_database_tensor(
            data,
            schema=self.schema,
            expected_row_count=self.row_count,
        )
        subring_database = _pad_last_dimension(
            logical_database,
            size=self.subring_slots,
        )
        return _repeat_subring_block(
            subring_database,
            repetitions=self.repetitions,
        )

    def prepare_parameters(self, values: Mapping[str, object]) -> torch.Tensor:
        """Normalize and direction-sign private values into packed row blocks."""

        if not isinstance(values, Mapping):
            raise TypeError("prepare_parameters expects a mapping of parameter names to values.")
        missing = [name for name in self.parameter_names if name not in values]
        unknown = [name for name in values if name not in self.parameter_names]
        if missing or unknown:
            raise ValueError(
                "Query parameters must match the SQL placeholders exactly. "
                f"Missing: {missing or 'none'}; unknown: {unknown or 'none'}."
            )

        prepared = torch.zeros(self.parameter_shape, dtype=torch.float64)
        column_is_greater = _comparison_directions(self._predicate)
        for binding_index, binding in enumerate(self._parameter_bindings):
            column = self.schema.columns[binding.column_index]
            raw_value = _as_real_tensor(
                values[binding.name],
                label=f"parameter :{binding.name}",
            )
            if raw_value.numel() != 1:
                raise ValueError(f"parameter :{binding.name} must be a scalar.")
            raw_value = raw_value.reshape(())
            _validate_in_bounds(raw_value, column, label=f"parameter :{binding.name}")
            pack_index, block_index = divmod(
                binding_index,
                self.parameter_pack_width,
            )
            block_start = block_index * self.subring_slots
            block_end = block_start + self.subring_slots
            direction = 1.0 if column_is_greater[binding_index] else -1.0
            prepared[pack_index, block_start:block_end].fill_(
                direction * float(_normalize(raw_value, column).item())
            )
        return prepared

    def _unpack_parameters_for_clear(
        self,
        prepared_parameters: torch.Tensor,
    ) -> torch.Tensor:
        """Restore one repeated row block per comparison binding for clear evaluation."""

        unpacked = torch.empty(
            (self.parameter_binding_count, self._subring_layout.main_slots),
            dtype=prepared_parameters.dtype,
            device=prepared_parameters.device,
        )
        column_is_greater = _comparison_directions(self._predicate)
        for binding_index in range(self.parameter_binding_count):
            pack_index, block_index = divmod(
                binding_index,
                self.parameter_pack_width,
            )
            block_start = block_index * self.subring_slots
            block_end = block_start + self.subring_slots
            direction = 1.0 if column_is_greater[binding_index] else -1.0
            unpacked[binding_index] = direction * prepared_parameters[
                pack_index,
                block_start:block_end,
            ].repeat(self.repetitions)
        return unpacked

    def apply_clear(
        self,
        prepared_parameters: torch.Tensor,
        *,
        prepared_database: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Evaluate this SQL selection locally on prepared plaintext inputs.

        This is a client-side verification helper for the current multi-input
        SQL example. It returns the same normalized, masked tensor layout as
        the encrypted pipeline so callers can decode both paths identically.
        """

        parameters = _validate_prepared_tensor(
            prepared_parameters,
            expected_shape=self.parameter_shape,
            label="prepared SQL parameters",
        )
        parameters = self._unpack_parameters_for_clear(parameters)
        database = _validate_prepared_tensor(
            self.prepared_database if prepared_database is None else prepared_database,
            expected_shape=self.database_shape,
            label="prepared SQL database",
        )

        selected = _evaluate_clear_predicate(
            self._predicate,
            parameters=parameters,
            database=database,
        )
        projection = torch.zeros_like(database)
        projection[_VALIDITY_CHANNEL].fill_(1.0)
        for column_index in self._projected_column_indices:
            projection[column_index + 1].fill_(1.0)
        return database * projection * selected.to(dtype=torch.float64).unsqueeze(0)

    def decode_result(self, result: torch.Tensor) -> SqlSelectResult:
        """Compact and denormalize a decrypted masked result tensor."""

        decoded = _as_real_tensor(result, label="decrypted SQL result")
        if tuple(decoded.shape) != self.output_shape:
            raise ValueError(
                f"Expected decrypted SQL result shape {self.output_shape}, "
                f"got {tuple(decoded.shape)}."
            )

        decoded = decoded[:, : self.row_count]
        selected = decoded[_VALIDITY_CHANNEL] > self.options.selection_threshold
        row_indices = torch.nonzero(selected, as_tuple=False).flatten()
        columns: dict[str, torch.Tensor] = {}
        for column in self.projected_columns:
            column_index = self.schema.columns.index(column)
            values = _denormalize(decoded[column_index + 1, selected], column)
            columns[column.name] = values
        return SqlSelectResult(row_indices=row_indices, columns=columns)


def _validate_prepared_tensor(
    value: object,
    *,
    expected_shape: tuple[int, int],
    label: str,
) -> torch.Tensor:
    tensor = _as_real_tensor(value, label=label)
    if tuple(tensor.shape) != expected_shape:
        raise ValueError(
            f"{label} must have shape {expected_shape}; got {tuple(tensor.shape)}."
        )
    return tensor


def _evaluate_clear_predicate(
    predicate: _Predicate,
    *,
    parameters: torch.Tensor,
    database: torch.Tensor,
) -> torch.Tensor:
    if isinstance(predicate, _Comparison):
        column = database[predicate.column_index + 1]
        parameter = parameters[predicate.binding_index]
        return column > parameter if predicate.column_is_greater else parameter > column

    left = _evaluate_clear_predicate(
        predicate.left,
        parameters=parameters,
        database=database,
    )
    right = _evaluate_clear_predicate(
        predicate.right,
        parameters=parameters,
        database=database,
    )
    if predicate.operator == "and":
        return torch.logical_and(left, right)
    return torch.logical_or(left, right)


def _build_database_packing_pipeline(
    *,
    plan: _SelectPlan,
    database_channels: int,
    subring_layout: _SubringLayout,
    parameter_layout: _ParameterLayout,
) -> SequentialHomOp:
    """Select and sign each comparison column in its parameter block."""

    mask = torch.zeros(
        (
            parameter_layout.pack_count,
            database_channels,
            subring_layout.main_slots,
        ),
        dtype=torch.float64,
    )
    column_is_greater = _comparison_directions(plan.predicate)
    for binding_index, binding in enumerate(plan.parameter_bindings):
        pack_index, block_index = divmod(
            binding_index,
            parameter_layout.pack_width,
        )
        direction = 1.0 if column_is_greater[binding_index] else -1.0
        for repeated_block_index in range(
            block_index,
            subring_layout.repetitions,
            parameter_layout.pack_width,
        ):
            block_start = repeated_block_index * subring_layout.subring_slots
            mask[
                pack_index,
                binding.column_index + 1,
                block_start : block_start + subring_layout.subring_slots,
            ] = direction

    mask_database = HomConstMul(dims=tuple(mask.shape))
    mask_database.set_data(mask)
    return SequentialHomOp(
        HomUnsqueeze(dim=0),
        mask_database,
        HomAxisSum(dim=1),
    )


def _build_comparison_unpacking_pipeline(
    *,
    subring_layout: _SubringLayout,
    parameter_layout: _ParameterLayout,
) -> SequentialHomOp | None:
    """Broadcast packed comparison blocks into one ciphertext lane per binding."""

    if parameter_layout.pack_width == 1:
        return None

    mask = torch.zeros(
        (1, parameter_layout.pack_width, subring_layout.main_slots),
        dtype=torch.float64,
    )
    for block_index in range(parameter_layout.pack_width):
        for repeated_block_index in range(
            block_index,
            subring_layout.repetitions,
            parameter_layout.pack_width,
        ):
            block_start = repeated_block_index * subring_layout.subring_slots
            mask[
                0,
                block_index,
                block_start : block_start + subring_layout.subring_slots,
            ] = 1.0

    flattened_binding_count = (
        parameter_layout.pack_count * parameter_layout.pack_width
    )
    mask_comparisons = HomConstMul(dims=tuple(mask.shape))
    mask_comparisons.set_data(mask)
    ops: list[HomOp] = [
        HomUnsqueeze(dim=1),
        mask_comparisons,
        *(
            HomRotateSum(
                rotations=(subring_layout.subring_slots * (1 << stage),),
            )
            for stage in range(parameter_layout.pack_width.bit_length() - 1)
        ),
        HomReshape((flattened_binding_count, subring_layout.main_slots)),
    ]
    if flattened_binding_count != parameter_layout.binding_count:
        ops.append(
            HomSlice(
                dim=0,
                key=slice(0, parameter_layout.binding_count),
            )
        )
    return SequentialHomOp(*ops)


def _build_projection_selector(
    plan: _SelectPlan,
    *,
    database_channels: int,
) -> torch.Tensor | None:
    if plan.projects_all_columns:
        return None

    projection = torch.zeros((database_channels, 1), dtype=torch.float64)
    projection[_VALIDITY_CHANNEL, 0] = 1.0
    for column_index in plan.projected_column_indices:
        projection[column_index + 1, 0] = 1.0
    return projection


class HomSqlPipeline(HomOp):
    def __init__(
        self,
        *,
        plan: _SelectPlan,
        schema: SqlTableSchema,
        options: SqlSelectOptions,
        subring_layout: _SubringLayout,
        parameter_layout: _ParameterLayout,
        log_n_subring: int,
        bootstrap_target_output_scale: int,
    ) -> None:
        super().__init__()
        self._predicate = plan.predicate
        database_channels = len(schema.columns) + 1

        self._binding_selectors = tuple(
            self._selector(len(plan.parameter_bindings), binding_index)
            for binding_index in range(len(plan.parameter_bindings))
        )
        self.pack_database = _build_database_packing_pipeline(
            plan=plan,
            database_channels=database_channels,
            subring_layout=subring_layout,
            parameter_layout=parameter_layout,
        )
        self.unpack_comparisons = _build_comparison_unpacking_pipeline(
            subring_layout=subring_layout,
            parameter_layout=parameter_layout,
        )
        self._projection_selector = _build_projection_selector(
            plan,
            database_channels=database_channels,
        )

        self.encrypted_zero = HomConstMul(
            dims=(),
            with_modswitch=False,
            pt_scale=1,
        )
        self.encrypted_zero.set_data(torch.tensor(0.0, dtype=torch.float64))
        self.sum_channel = HomAxisSum(dim=0)
        self.compare = HomCompare(
            x_accuracy=options.x_accuracy,
            y_accuracy=options.y_accuracy,
            left=-1.0,
            right=1.0,
        )
        self.bootstrap_predicate = Bootstrap(
            log_n_subring=log_n_subring,
            target_output_scale=bootstrap_target_output_scale,
        )

    @staticmethod
    def _selector(channel_count: int, channel_index: int) -> torch.Tensor:
        selector = torch.zeros((channel_count, 1), dtype=torch.float64)
        selector[channel_index, 0] = 1.0
        return selector

    def _extract_channel(
        self,
        values: HomValue,
        selector: torch.Tensor,
    ) -> HomValue:
        return self.sum_channel(values * selector)

    def _evaluate_predicate(
        self,
        predicate: _Predicate,
        *,
        comparison_values: HomValue,
        binding_values: dict[int, HomValue],
    ) -> HomValue:
        if isinstance(predicate, _Comparison):
            if predicate.binding_index not in binding_values:
                binding_values[predicate.binding_index] = self._extract_channel(
                    comparison_values,
                    self._binding_selectors[predicate.binding_index],
                )
            return binding_values[predicate.binding_index]

        left = self._evaluate_predicate(
            predicate.left,
            comparison_values=comparison_values,
            binding_values=binding_values,
        )
        right = self._evaluate_predicate(
            predicate.right,
            comparison_values=comparison_values,
            binding_values=binding_values,
        )
        both = left * right
        if predicate.operator == "and":
            return both
        return left + right - both

    def _unpack_comparisons(self, packed_comparisons: HomValue) -> HomValue:
        if self.unpack_comparisons is None:
            return packed_comparisons
        return self.unpack_comparisons(packed_comparisons)

    def forward(self, parameters: HomValue, database: HomValue) -> HomValue:
        packed_database = self.pack_database(database)
        signed_difference = packed_database - parameters
        packed_comparisons = self.compare(
            signed_difference,
            self.encrypted_zero(signed_difference),
        )
        comparison_values = self._unpack_comparisons(packed_comparisons)
        predicate = self._evaluate_predicate(
            self._predicate,
            comparison_values=comparison_values,
            binding_values={},
        )
        predicate = self.bootstrap_predicate(predicate)
        projected_database = (
            database
            if self._projection_selector is None
            else database * self._projection_selector
        )
        return projected_database * predicate


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
        operator: Literal["and", "or"] = "and" if isinstance(expression, exp.And) else "or"
        return _BooleanPredicate(
            operator=operator,
            left=_compile_predicate(expression.this, schema=schema, bindings=bindings),
            right=_compile_predicate(expression.expression, schema=schema, bindings=bindings),
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

    binding_index = len(bindings)
    bindings.append(_ParameterBinding(name=parameter_name, column_index=column_index))
    return _Comparison(
        column_index=column_index,
        binding_index=binding_index,
        column_is_greater=column_is_greater,
    )


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
        raise SqlCompileError("Table aliases and database/catalog qualifiers are not supported.")
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
        projects_all = True
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
        projects_all = len(projected_indices) == len(schema.columns) and set(
            projected_indices
        ) == set(range(len(schema.columns)))

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
        projects_all_columns=projects_all,
    )


def compile_sql_select(
    sql: str,
    *,
    schema: SqlTableSchema,
    database: Mapping[str, object],
    hom_params: HomParams,
    options: SqlSelectOptions | None = None,
) -> CompiledSqlSelect:
    """Compile SQL for one exact database shape and retain its packed input."""

    if not isinstance(schema, SqlTableSchema):
        raise TypeError("schema must be a SqlTableSchema.")
    if not isinstance(hom_params, HomParams):
        raise TypeError("hom_params must be a HomParams.")
    if options is None:
        options = SqlSelectOptions()
    elif not isinstance(options, SqlSelectOptions):
        raise TypeError("options must be a SqlSelectOptions or None.")

    plan = _parse_select(sql, schema=schema, options=options)
    logical_database = _prepare_database_tensor(database, schema=schema)
    subring_layout = _subring_layout_for_active_slots(
        int(logical_database.shape[1]),
        ring_dimension=hom_params.n,
    )
    parameter_layout = _parameter_layout_for_bindings(
        len(plan.parameter_bindings),
        subring_layout=subring_layout,
    )
    subring_database = _pad_last_dimension(
        logical_database,
        size=subring_layout.subring_slots,
    )
    prepared_database = _repeat_subring_block(
        subring_database,
        repetitions=subring_layout.repetitions,
    )
    pipeline = HomSqlPipeline(
        plan=plan,
        schema=schema,
        options=options,
        subring_layout=subring_layout,
        parameter_layout=parameter_layout,
        log_n_subring=subring_layout.log_n_subring,
        bootstrap_target_output_scale=hom_params.pt_scale,
    )
    pipeline = HomomorphicPipeline(
        hom=pipeline,
        client_pre=[Repeat(dim=1)],
        input_shape={
            CompiledSqlSelect.PARAMETERS_INPUT_NAME: parameter_layout.input_shape,
            CompiledSqlSelect.DATABASE_INPUT_NAME: tuple(prepared_database.shape),
        },
        n_axis=-1,
    )

    parameter_names = tuple(dict.fromkeys(binding.name for binding in plan.parameter_bindings))
    projected_columns = tuple(schema.columns[index] for index in plan.projected_column_indices)
    return CompiledSqlSelect(
        sql=sql,
        schema=schema,
        hom_params=hom_params,
        options=options,
        pipeline=pipeline,
        prepared_database=prepared_database,
        parameter_names=parameter_names,
        projected_columns=projected_columns,
        _parameter_bindings=plan.parameter_bindings,
        _predicate=plan.predicate,
        _projected_column_indices=plan.projected_column_indices,
        _subring_layout=subring_layout,
        _parameter_layout=parameter_layout,
    )

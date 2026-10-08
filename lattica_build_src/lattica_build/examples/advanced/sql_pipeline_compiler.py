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

import heapq
import json
import math
import re
from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, replace
from functools import lru_cache
from numbers import Real
from typing import ClassVar, Literal, TypeAlias

import numpy as np
import sqlglot
import torch
from sqlglot import exp
from sqlglot.errors import ParseError

from lattica_build.base_classes.hom_op import HomOp
from lattica_build.base_classes.hom_pipeline import HomomorphicPipeline
from lattica_build.base_classes.hom_value import HomValue
from lattica_build.operators.arithmetic.h_const_mul import HomConstMul
from lattica_build.operators.composite.module_list import ModuleListHomOp
from lattica_build.operators.fhe.h_bootstrap import Bootstrap
from lattica_build.operators.polynomials.h_poly_eval_base import HomPolyEvalBase
from lattica_build.operators.polynomials.remez_utils import sign_minimax
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
    comparison_resolution: float | None = None

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
        if not math.isfinite(maximum - minimum):
            raise ValueError("Column range must be representable in float64.")
        if self.kind not in ("real", "integer"):
            raise ValueError(
                f"{self.name}.kind must be 'real' or 'integer'; got {self.kind!r}."
            )
        object.__setattr__(self, "min_value", minimum)
        object.__setattr__(self, "max_value", maximum)
        if self.comparison_resolution is not None:
            resolution = _validate_real(
                self.comparison_resolution, label="comparison_resolution"
            )
            if resolution <= 0:
                raise ValueError("comparison_resolution must be positive.")
            object.__setattr__(self, "comparison_resolution", resolution)
        if self.kind == "integer" and max(abs(minimum), abs(maximum)) >= 2**52:
            raise ValueError(
                "Integer bounds must leave representable half-integer boundaries in float64 (absolute value < 2**52)."
            )


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
    error_tolerance: float = 2**-8
    rows_per_block: int | None = None

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
        tolerance = _validate_real(self.error_tolerance, label="error_tolerance")
        if not 0 < tolerance < min(threshold, 1 - threshold):
            raise ValueError(
                "error_tolerance must be positive and smaller than the selection margin."
            )
        if self.rows_per_block is not None and (
            isinstance(self.rows_per_block, bool)
            or not isinstance(self.rows_per_block, int)
            or self.rows_per_block <= 0
            or self.rows_per_block & (self.rows_per_block - 1)
        ):
            raise ValueError("rows_per_block must be a positive power of two.")


@dataclass(frozen=True)
class _ParameterBinding:
    names: tuple[str, ...]
    column_index: int
    column_is_greater: bool
    reduction: Literal["min", "max"] | None = None


@dataclass(frozen=True)
class _Predicate:
    # Child indices refer only to earlier nodes: hashing and traversal never recurse.
    operator: Literal["compare", "and", "or"]
    children: tuple[int, ...] = ()
    binding_index: int = -1


@dataclass(frozen=True)
class _SelectPlan:
    nodes: tuple[_Predicate, ...]
    root: int
    parameter_bindings: tuple[_ParameterBinding, ...]
    projected_column_indices: tuple[int, ...]
    parameter_names: tuple[str, ...]


@dataclass(frozen=True, kw_only=True)
class SqlPackingPlan:
    """Fixed row capacity, independent row blocks, and packed column layouts."""

    row_count: int
    ring_slots: int
    row_slots: int
    row_blocks: int
    binding_count: int
    pack_width: int
    pack_count: int
    working_slots: int
    output_channels: int

    @property
    def parameter_shape(self) -> tuple[int, int, int]:
        # The same encrypted thresholds broadcast over every row block.
        return (1, self.pack_count, self.working_slots)

    @property
    def comparison_shape(self) -> tuple[int, int, int]:
        return (self.row_blocks, self.pack_count, self.working_slots)

    @property
    def output_pack_count(self) -> int:
        return (self.output_channels + self.pack_width - 1) // self.pack_width

    @property
    def output_shape(self) -> tuple[int, int, int]:
        return (self.row_blocks, self.output_pack_count, self.working_slots)


def _pow2(value: int) -> int:
    return 1 << (value - 1).bit_length()


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
        2.0 * ((values - column.min_value) / (column.max_value - column.min_value))
        - 1.0
    )


def _denormalize(values: torch.Tensor, column: SqlColumn) -> torch.Tensor:
    return ((values + 1.0) / 2.0) * (
        column.max_value - column.min_value
    ) + column.min_value


def _as_real_tensor(value: object, *, label: str) -> torch.Tensor:
    try:
        if isinstance(value, torch.Tensor):
            if value.dtype == torch.bool or value.is_complex():
                raise TypeError("bool and complex inputs are not supported")
            tensor = value.detach().to(dtype=torch.float64, device="cpu")
        else:
            # Infer the input type without passing Python floats through float32.
            array = np.asarray(value)
            if array.dtype.kind not in "iuf":
                raise TypeError("expected real numeric values")
            if not array.flags.c_contiguous:
                array = np.ascontiguousarray(array)
            tensor = torch.as_tensor(array, dtype=torch.float64)
    except (TypeError, ValueError, RuntimeError, OverflowError) as exc:
        raise TypeError(
            f"{label} must contain real numeric values, excluding bool and complex."
        ) from exc
    if not bool(torch.all(torch.isfinite(tensor))):
        raise ValueError(f"{label} must contain only finite values.")
    return tensor


def _validate_in_bounds(values, column, *, label):
    if bool(torch.any((values < column.min_value) | (values > column.max_value))):
        raise ValueError(
            f"{label} must be within [{column.min_value}, {column.max_value}]."
        )


def _validate_database(data, *, schema):
    """Convert and validate once; normalization is only done for used columns."""
    if not isinstance(data, Mapping):
        raise TypeError("database must be a mapping of column names to values.")
    names = {column.name for column in schema.columns}
    if set(data) != names:
        raise ValueError(
            f"Database columns must match the schema exactly. Missing: {sorted(names - set(data))}; unknown: {sorted(set(data) - names)}."
        )
    converted = {}
    row_count = None
    for column in schema.columns:
        values = _as_real_tensor(data[column.name], label=f"column {column.name!r}")
        if values.ndim != 1:
            raise ValueError(f"column {column.name!r} must be one-dimensional.")
        if row_count is None:
            row_count = len(values)
        elif len(values) != row_count:
            raise ValueError("All database columns must have the same row count.")
        _validate_in_bounds(values, column, label=f"column {column.name!r}")
        if column.kind == "integer" and bool(torch.any(values != torch.round(values))):
            raise ValueError(f"column {column.name!r} must contain integer values.")
        converted[column.name] = values
    return converted, row_count


def _comparison_padding(column, options):
    if column.kind == "integer":
        return 0.5
    return column.comparison_resolution or (
        (column.max_value - column.min_value) * 2.0**-options.x_accuracy
    )


def _normalize_comparison(values, column, options):
    padding = _comparison_padding(column, options)
    return (values - column.min_value + padding) / (
        column.max_value - column.min_value + 2 * padding
    ) - 0.5


@dataclass(frozen=True, kw_only=True)
class CompiledSqlSelect:
    """A public plan with codecs for runtime private inputs and bounded batches."""

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
    _layout: "_Layout" = field(repr=False)
    prepared_database: dict[str, torch.Tensor] | None = field(
        default=None, repr=False, compare=False
    )

    @property
    def row_count(self) -> int:
        """Maximum rows per prepared batch; shorter and empty tables are accepted."""
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
    def parameter_shape(self) -> tuple[int, int, int]:
        return self.packing.parameter_shape

    @property
    def database_shape(self) -> tuple[int, int, int]:
        return self.packing.output_shape

    @property
    def output_shape(self) -> tuple[int, int, int]:
        return self.packing.output_shape

    @property
    def bootstrap_count(self) -> int:
        return self.pipeline.hom.bootstrap_count

    def prepare_database(self, data: Mapping[str, object]) -> dict[str, torch.Tensor]:
        """Pack up to row_count rows into reusable encrypted database inputs."""
        columns, count = _validate_database(data, schema=self.schema)
        if count > self.row_count:
            raise ValueError(
                f"Database has {count} rows but compiled capacity is {self.row_count}; use iter_database_batches or compile a larger capacity."
            )
        return self._prepare_validated(columns, count)

    def iter_database_batches(
        self, data: Mapping[str, object]
    ) -> Iterator[tuple[int, dict[str, torch.Tensor]]]:
        """Yield (row offset, input mapping), retaining only one prepared batch.

        Reuse the deployed pipeline, key and encrypted parameters. Upload each
        mapping before querying it, then decode with row_offset=offset.
        """
        columns, count = _validate_database(data, schema=self.schema)
        for start in range(0, count, self.row_count):
            stop = min(count, start + self.row_count)
            yield (
                start,
                self._prepare_validated(
                    {name: values[start:stop] for name, values in columns.items()},
                    stop - start,
                ),
            )

    def _prepare_validated(self, columns, count):
        p = self.packing
        output = torch.zeros(p.output_shape, dtype=torch.float64)
        output_blocks = output.view(
            p.row_blocks, p.output_pack_count, p.pack_width, p.row_slots
        )
        validity = torch.zeros((p.row_blocks, p.row_slots), dtype=torch.float64)
        validity.view(-1)[:count] = 1
        output_blocks[:, 0, 0] = validity
        for channel, column in enumerate(self.projected_columns, start=1):
            values = torch.zeros_like(validity)
            values.view(-1)[:count] = _normalize(columns[column.name], column)
            pack, lane = divmod(channel, p.pack_width)
            output_blocks[:, pack, lane] = values
        comparisons = torch.zeros(p.comparison_shape, dtype=torch.float64)
        comparison_blocks = comparisons.view(
            p.row_blocks, p.pack_count, p.pack_width, p.row_slots
        )
        normalized = {}
        uses = Counter(
            binding.column_index for binding in self._plan.parameter_bindings
        )
        for index, binding in enumerate(self._plan.parameter_bindings):
            column = self.schema.columns[binding.column_index]
            if binding.column_index not in normalized:
                values = torch.zeros_like(validity)
                values.view(-1)[:count] = _normalize_comparison(
                    columns[column.name], column, self.options
                )
                normalized[binding.column_index] = values
            values = normalized[binding.column_index]
            if not binding.column_is_greater:
                values = -values
            pack, lane = self._layout.positions[index]
            if self._layout.replicate_single:
                comparison_blocks[:, pack] = values.unsqueeze(1)
            else:
                comparison_blocks[:, pack, lane] = values
            uses[binding.column_index] -= 1
            if uses[binding.column_index] == 0:
                del normalized[binding.column_index]
        return {
            self.COMPARISONS_INPUT_NAME: comparisons,
            self.DATABASE_INPUT_NAME: output,
        }

    def prepare_parameters(self, values: Mapping[str, object]) -> torch.Tensor:
        """Privately fold thresholds, then pack one input shared by all row blocks."""
        if not isinstance(values, Mapping):
            raise TypeError(
                "prepare_parameters expects a mapping of parameter names to values."
            )
        if set(values) != set(self.parameter_names):
            raise ValueError(
                f"Query parameters must match SQL placeholders {self.parameter_names}."
            )
        scalars = {}
        for name in self.parameter_names:
            value = _as_real_tensor(values[name], label=f"parameter :{name}")
            if value.numel() != 1:
                raise ValueError(f"parameter :{name} must be a scalar.")
            scalars[name] = float(value.item())
        p = self.packing
        prepared = torch.zeros(p.parameter_shape, dtype=torch.float64)
        blocks = prepared.view(1, p.pack_count, p.pack_width, p.row_slots)
        for index, binding in enumerate(self._plan.parameter_bindings):
            column = self.schema.columns[binding.column_index]
            thresholds = [scalars[name] for name in binding.names]
            value = min(thresholds) if binding.reduction == "min" else max(thresholds)
            padding = _comparison_padding(column, self.options)
            # Out-of-range thresholds have constant SQL truth values. Saturate
            # locally, without publishing their values or changing the graph.
            value = min(
                column.max_value + padding, max(column.min_value - padding, value)
            )
            if column.kind == "integer":
                value = (
                    math.floor(value) + 0.5
                    if binding.column_is_greater
                    else math.ceil(value) - 0.5
                )
                value = min(
                    column.max_value + padding, max(column.min_value - padding, value)
                )
            value = _normalize_comparison(value, column, self.options)
            if not binding.column_is_greater:
                value = -value
            pack, lane = self._layout.positions[index]
            if self._layout.replicate_single:
                blocks[:, pack].fill_(value)
            else:
                blocks[:, pack, lane].fill_(value)
        return prepared

    def apply_clear(
        self,
        prepared_parameters: torch.Tensor,
        *,
        prepared_database: Mapping[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Exact predicate reference on the prepared inputs (no polynomial ops)."""
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
        if set(data) != {self.COMPARISONS_INPUT_NAME, self.DATABASE_INPUT_NAME}:
            raise ValueError(
                "Prepared database must contain both compiled database views."
            )
        p = self.packing
        comparisons = _validate_prepared_tensor(
            data[self.COMPARISONS_INPUT_NAME],
            expected_shape=p.comparison_shape,
            label="prepared comparison database",
        )
        output = _validate_prepared_tensor(
            data[self.DATABASE_INPUT_NAME],
            expected_shape=p.output_shape,
            label="prepared output database",
        )
        predicates = []
        for node in self._plan.nodes:
            if node.operator == "compare":
                pack, lane = self._layout.positions[node.binding_index]
                slots = slice(lane * p.row_slots, (lane + 1) * p.row_slots)
                value = comparisons[:, pack, slots] > parameters[:, pack, slots]
            else:
                value = predicates[node.children[0]]
                for child in node.children[1:]:
                    value = (
                        value & predicates[child]
                        if node.operator == "and"
                        else value | predicates[child]
                    )
            predicates.append(value)
        selected = predicates[self._plan.root].repeat(1, p.pack_width).unsqueeze(1)
        return output * selected.to(output.dtype)

    def decode_result(
        self, result: torch.Tensor, *, row_offset: int = 0
    ) -> SqlSelectResult:
        """Unpack columns and compact rows after decryption, preserving source order."""
        if (
            isinstance(row_offset, bool)
            or not isinstance(row_offset, int)
            or row_offset < 0
        ):
            raise ValueError("row_offset must be a non-negative integer.")
        p = self.packing
        decoded = _validate_prepared_tensor(
            result, expected_shape=p.output_shape, label="decrypted SQL result"
        )
        blocks = decoded.view(
            p.row_blocks, p.output_pack_count, p.pack_width, p.row_slots
        )
        validity = blocks[:, 0, 0].reshape(-1)[: self.row_count]
        selected = validity > self.options.selection_threshold
        columns = {}
        for channel, column in enumerate(self.projected_columns, start=1):
            pack, lane = divmod(channel, p.pack_width)
            values = blocks[:, pack, lane].reshape(-1)[: self.row_count][selected]
            # The validity channel carries the same approximate predicate. This
            # removes its attenuation after decryption without encrypted division.
            columns[column.name] = _denormalize(values / validity[selected], column)
        return SqlSelectResult(
            row_indices=torch.nonzero(selected).flatten() + row_offset, columns=columns
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


@dataclass(frozen=True)
class _Layout:
    positions: tuple[tuple[int, int], ...]
    groups: dict[int, tuple[tuple[int, int, int, int], ...]]
    pack_count: int
    replicate_single: bool


def _make_layout(plan, width, fuse):
    positions = [None] * len(plan.parameter_bindings)
    groups = {}
    reserved = set()
    cursor = 0
    candidates = []
    if fuse and width > 1:
        for node_id, node in enumerate(plan.nodes):
            if node.operator != "compare" and all(
                plan.nodes[c].operator == "compare" for c in node.children
            ):
                candidates.append(
                    (node_id, tuple(plan.nodes[c].binding_index for c in node.children))
                )
    for node_id, bindings in sorted(
        candidates, key=lambda item: (-len(item[1]), item[0])
    ):
        if any(positions[i] is not None for i in bindings):
            continue
        chunks = []
        for offset in range(0, len(bindings), width):
            chunk = bindings[offset : offset + width]
            span = _pow2(len(chunk))
            cursor = ((cursor + span - 1) // span) * span
            pack, lane = divmod(cursor, width)
            chunks.append((pack, lane, span, len(chunk)))
            for j, binding in enumerate(chunk):
                positions[binding] = (pack, lane + j)
            reserved.update(range(cursor, cursor + span))
            cursor += span
        groups[node_id] = tuple(chunks)
    cursor = 0
    for binding, position in enumerate(positions):
        if position is None:
            while cursor in reserved:
                cursor += 1
            positions[binding] = divmod(cursor, width)
            reserved.add(cursor)
    return _Layout(
        tuple(positions),
        groups,
        max(pack for pack, _ in positions) + 1,
        len(positions) == 1,
    )


@lru_cache(maxsize=64)
def _fit_comparison(x_accuracy, y_accuracy):
    polynomial = sign_minimax(x_accuracy=x_accuracy, y_accuracy=y_accuracy - 1)
    coefs = tuple(polynomial.coefs())
    # Error measured by the fitter, including its acceptance tolerance.
    return coefs, 1.1 * 2.0 ** (-polynomial.bits - 1)


@dataclass(frozen=True)
class _ComparisonPlan:
    coefficients: tuple
    refresh_before: frozenset[int]
    remaining: int
    full_budget: int
    depth: int
    x_accuracy: int
    y_accuracy: int
    error_bound: float
    reduced_initial: bool


def _plan_comparison(plan, schema, options, params):
    x_accuracy = options.x_accuracy
    for index in {binding.column_index for binding in plan.parameter_bindings}:
        column = schema.columns[index]
        padding = _comparison_padding(column, options)
        width = column.max_value - column.min_value + 2 * padding
        if not all(
            math.isfinite(value)
            for value in (width, column.min_value - padding, column.max_value + padding)
        ):
            raise SqlCompileError(
                f"Column {column.name!r} needs finite comparison normalization bounds."
            )
        resolution = 0.5 if column.kind == "integer" else padding
        if resolution < 2 * max(math.ulp(column.min_value), math.ulp(column.max_value)):
            raise SqlCompileError(
                f"Column {column.name!r} requires a comparison resolution representable in float64."
            )
        x_accuracy = max(
            x_accuracy, math.ceil(math.log2(width) - math.log2(resolution)) + 1
        )
    weights = []
    for node in plan.nodes:
        weights.append(
            1 if node.operator == "compare" else sum(weights[c] for c in node.children)
        )
    log_weight = math.log2(weights[plan.root])
    y_accuracy = options.y_accuracy
    target = options.error_tolerance * 0.75
    while True:
        if y_accuracy > 48:
            raise SqlCompileError(
                "Requested expression accuracy exceeds the float64 polynomial fitter; use a larger error_tolerance."
            )
        coefs, error = _fit_comparison(x_accuracy, y_accuracy)
        exponent = log_weight + math.log2(math.log1p(error))
        bound = math.expm1(2.0**exponent) if exponent < 9 else math.inf
        if bound <= target:
            break
        y_accuracy = max(y_accuracy + 1, math.ceil(log_weight - math.log2(target)))
    full = replace(params, num_init_rows=None)
    full.mod_chain = ModulusChain(full)
    rows, cols = init_active_rows_cols(full)
    available = sum(map(sum, cols))
    full_budget = available - 2
    initial = replace(params)
    initial.mod_chain = ModulusChain(initial)
    _, initial_cols = init_active_rows_cols(initial)
    remaining = sum(map(sum, initial_cols)) - 2
    reduced_initial = params.num_init_rows is not None and params.num_init_rows < len(
        params.full_q_list_precision
    )
    if reduced_initial:
        remaining = (
            sum(
                len(row) for row in params.full_q_list_precision[: params.num_init_rows]
            )
            - 2
        )
    refresh_before = set()
    depth = 0
    for index, coef in enumerate(coefs):
        value = HomValue(
            id="probe",
            tensor_shape=(1, 1, 1),
            n_axis=2,
            n_slots=1,
            active_rows=rows,
            active_cols=cols,
            pt_scale=params.pt_scale,
        )
        op = HomPolyEvalBase(coefs=coef)
        try:
            result = op.infer_output_level_and_scale(value, hom_params=full)
            consumed = available - sum(map(sum, result.active_cols))
        except (ValueError, RuntimeError, IndexError) as exc:
            raise SqlCompileError(
                "A comparison polynomial stage exceeds the supplied modulus budget."
            ) from exc
        if consumed > full_budget:
            raise SqlCompileError(
                "A comparison polynomial stage leaves insufficient modulus budget; increase the supplied HE budget."
            )
        if remaining < consumed:
            if options.bootstrap == "never":
                raise SqlCompileError(
                    "The comparison needs bootstrapping (bootstrap='never') or a larger modulus budget."
                )
            refresh_before.add(index)
            remaining = full_budget
        remaining -= consumed
        depth += consumed
    return _ComparisonPlan(
        coefs,
        frozenset(refresh_before),
        remaining,
        full_budget,
        depth,
        x_accuracy,
        y_accuracy,
        bound,
        reduced_initial,
    )


class _SqlBootstrap(HomOp):
    """Refresh bounded SQL values with headroom for periodic/near-constant inputs.

    Encode the scalar 1/16 at scale 16: its integer encoding is exactly 1, so
    shrinking the message consumes no modulus level. Restoring its magnitude
    after bootstrap suppresses the uncorrected sine's cubic error without
    changing the shared bootstrap configuration.
    """

    def __init__(self, target_output_scale):
        super().__init__()
        self.shrink = HomConstMul(dims=(), with_modswitch=False, pt_scale=16).set_data(
            torch.tensor(1 / 16, dtype=torch.float64)
        )
        self.refresh = Bootstrap(target_output_scale=target_output_scale)

    def forward(self, value):
        return self.refresh(self.shrink(value)) * 16


class _SqlCompare(HomOp):
    def __init__(self, comparison, params):
        super().__init__()
        self.stages = ModuleListHomOp()
        for index, coef in enumerate(comparison.coefficients):
            if index in comparison.refresh_before:
                self.stages.append(_SqlBootstrap(target_output_scale=params.pt_scale))
            if index == len(comparison.coefficients) - 1:
                coef = coef * 0.5
            self.stages.append(HomPolyEvalBase(coefs=coef))

    def forward(self, database, parameters):
        value = database - parameters
        for stage in self.stages:
            value = stage(value)
        return value + 0.5


@dataclass(frozen=True)
class _Instruction:
    kind: str
    inputs: tuple[int, ...]
    argument: object = None


@dataclass(frozen=True)
class _Circuit:
    instructions: tuple[_Instruction, ...]
    root: int
    refresh_packed: bool
    refresh_database: bool
    bootstraps: int
    rotations: int
    multiplies: int
    masks: int
    cost: float


def _plan_circuit(plan, packing, layout, comparison, options, refresh_packed):
    p = packing
    instructions = []
    remaining = []
    full_state = []
    refreshed = {}
    packs = {}
    masks = rotations = multiplies = boots = 0
    initial_remaining = (
        comparison.full_budget if refresh_packed else comparison.remaining
    )

    initial_full = (
        not comparison.reduced_initial
        or bool(comparison.refresh_before)
        or refresh_packed
    )

    def emit(kind, inputs, argument=None, budget=None, full=None):
        instructions.append(_Instruction(kind, tuple(inputs), argument))
        remaining.append(initial_remaining if budget is None else budget)
        if full is None:
            full = (
                initial_full if inputs == (-1,) else all(full_state[i] for i in inputs)
            )
        full_state.append(full)
        return len(instructions) - 1

    def refresh(value):
        nonlocal boots
        if options.bootstrap == "never":
            raise SqlCompileError(
                "The query needs bootstrapping (bootstrap='never') or a larger modulus budget."
            )
        if value not in refreshed:
            refreshed[value] = emit(
                "bootstrap", (value,), budget=comparison.full_budget, full=True
            )
            boots += 1
        return refreshed[value]

    def ensure(value, consumed):
        if consumed > comparison.full_budget:
            raise SqlCompileError(
                "Insufficient modulus budget for a Boolean operation."
            )
        return value if remaining[value] >= consumed else refresh(value)

    def pack_value(pack):
        if pack not in packs:
            packs[pack] = emit("slice", (-1,), pack)
        return packs[pack]

    def complement(value):
        return emit("complement", (value,), budget=remaining[value])

    def mask(value, lane, count, neutral=False):
        nonlocal masks
        value = ensure(value, 1)
        masks += 1
        return emit("mask", (value,), (lane, count, neutral), remaining[value] - 1)

    def extract(value, lane):
        nonlocal rotations
        if p.pack_width == 1:
            return value
        value = mask(value, lane, 1)
        for stage in range(p.pack_width.bit_length() - 1):
            value = emit(
                "rotate_add", (value,), p.row_slots * (1 << stage), remaining[value]
            )
            rotations += 1
        return value

    def combine(operator, values):
        nonlocal multiplies
        queue = [(-remaining[v], v) for v in values]
        heapq.heapify(queue)
        while len(queue) > 1:
            _, left = heapq.heappop(queue)
            _, right = heapq.heappop(queue)
            left, right = ensure(left, 1), ensure(right, 1)
            if full_state[left] != full_state[right]:
                left = left if full_state[left] else refresh(left)
                right = right if full_state[right] else refresh(right)
            value = emit(
                operator,
                (left, right),
                budget=min(remaining[left], remaining[right]) - 1,
            )
            multiplies += 1
            heapq.heappush(queue, (-remaining[value], value))
        return queue[0][1]

    needed = set()
    pending = [plan.root]
    while pending:
        node_id = pending.pop()
        if node_id not in needed:
            needed.add(node_id)
            if node_id not in layout.groups:
                pending.extend(plan.nodes[node_id].children)
    evaluated = {}
    for node_id, node in enumerate(plan.nodes):
        if node_id not in needed:
            continue
        if node.operator == "compare":
            pack, lane = layout.positions[node.binding_index]
            value = pack_value(pack)
            if not layout.replicate_single:
                value = extract(value, lane)
        elif node_id in layout.groups:
            chunks = []
            for pack, lane, width, count in layout.groups[node_id]:
                value = pack_value(pack)
                if node.operator == "or":
                    value = complement(value)
                if count < width:
                    value = mask(value, lane, count, neutral=True)
                for stage in range(width.bit_length() - 1):
                    # Refresh before rotating, so both factors share the refresh.
                    value = ensure(value, 1)
                    rotated = emit(
                        "rotate", (value,), p.row_slots * (1 << stage), remaining[value]
                    )
                    value = emit("and", (value, rotated), budget=remaining[value] - 1)
                    rotations += 1
                    multiplies += 1
                if width < p.pack_width:
                    value = extract(value, lane)
                if node.operator == "or":
                    value = complement(value)
                chunks.append(value)
            value = combine(node.operator, chunks)
        else:
            value = combine(
                node.operator, [evaluated[child] for child in node.children]
            )
        evaluated[node_id] = value
    root = ensure(evaluated[plan.root], 1)  # Projected output multiplication.
    if options.bootstrap == "always":
        root = emit("bootstrap", (root,), budget=comparison.full_budget, full=True)
        boots += 1
    refresh_database = comparison.reduced_initial and full_state[root]
    total_boots = p.row_blocks * (
        boots
        + (p.output_pack_count if refresh_database else 0)
        + p.pack_count * (len(comparison.refresh_before) + int(refresh_packed))
    )
    # Relative ciphertext costs, including comparison expansion and the public
    # bootstrap period. The final graph is checked by the real level tracer.
    cost = p.row_blocks * (
        p.pack_count * comparison.depth * 3
        + rotations * 3
        + multiplies * 2
        + masks
        + p.output_pack_count * 2
    )
    cost += total_boots * (100 + 10 * math.log2(p.working_slots))
    return _Circuit(
        tuple(instructions),
        root,
        refresh_packed,
        refresh_database,
        total_boots,
        rotations * p.row_blocks,
        (multiplies + p.output_pack_count) * p.row_blocks,
        masks * p.row_blocks,
        cost,
    )


def _plan_execution(plan, row_count, params, options, comparison):
    if isinstance(row_count, bool) or not isinstance(row_count, int) or row_count <= 0:
        raise ValueError("row_count must be a positive integer capacity.")
    slots = params.internal_n
    if options.rows_per_block is not None and options.rows_per_block > slots:
        raise ValueError("rows_per_block exceeds the ring's slot capacity.")
    row_sizes = (
        [options.rows_per_block]
        if options.rows_per_block
        else [1 << i for i in range(min(_pow2(row_count), slots).bit_length())]
    )
    best = None
    failure = None
    binding_count = len(plan.parameter_bindings)
    for rows in row_sizes:
        max_width = min(
            slots // rows,
            _pow2(max(binding_count, len(plan.projected_column_indices) + 1)),
        )
        for power in range(max_width.bit_length()):
            width = 1 << power
            for fuse in (False, True) if width > 1 and binding_count > 1 else (False,):
                layout = _make_layout(plan, width, fuse)
                packing = SqlPackingPlan(
                    row_count=row_count,
                    ring_slots=slots,
                    row_slots=rows,
                    row_blocks=(row_count + rows - 1) // rows,
                    binding_count=binding_count,
                    pack_width=width,
                    pack_count=layout.pack_count,
                    working_slots=rows * width,
                    output_channels=len(plan.projected_column_indices) + 1,
                )
                refresh_options = (
                    (False, True)
                    if options.bootstrap != "never"
                    and comparison.remaining < comparison.full_budget
                    else (False,)
                )
                for refresh in refresh_options:
                    try:
                        circuit = _plan_circuit(
                            plan, packing, layout, comparison, options, refresh
                        )
                    except SqlCompileError as exc:
                        failure = exc
                        continue
                    key = (
                        circuit.cost,
                        circuit.bootstraps,
                        packing.row_blocks,
                        packing.working_slots,
                    )
                    if best is None or key < best[0]:
                        best = (key, packing, layout, circuit)
    if best is None:
        raise failure or SqlCompileError(
            "No valid execution plan for the supplied HE parameters."
        )
    return best[1:]


class _CircuitStep(HomOp):
    def __init__(self, instruction, packing, params):
        super().__init__()
        self.kind = instruction.kind
        if self.kind == "slice":
            self.operation = HomSlice(dim=1, key=instruction.argument)
        elif self.kind in ("rotate", "rotate_add"):
            self.operation = HomRotateSum(
                rotations=(instruction.argument,), perform_sum=False
            )
            self.squeeze = HomSqueeze(dim=0)
        elif self.kind == "bootstrap":
            self.operation = _SqlBootstrap(target_output_scale=params.pt_scale)
        elif self.kind == "mask":
            lane, count, self.neutral = instruction.argument
            mask = torch.zeros((1, packing.working_slots), dtype=torch.float64)
            mask[:, lane * packing.row_slots : (lane + count) * packing.row_slots] = 1
            self.operation = HomConstMul(dims=tuple(mask.shape)).set_data(mask)
            self.neutral_value = 1 - mask

    def forward(self, left):
        if self.kind in ("slice", "bootstrap"):
            return self.operation(left)
        if self.kind in ("rotate", "rotate_add"):
            rotated = self.squeeze(self.operation(left))
            return left + rotated if self.kind == "rotate_add" else rotated
        if self.kind == "mask":
            result = self.operation(left)
            return result + self.neutral_value if self.neutral else result
        if self.kind == "complement":
            return 1 - left
        raise RuntimeError(f"Unknown unary SQL instruction: {self.kind}")


class _CircuitBinaryStep(_CircuitStep):
    def forward(self, left, right):
        product = left * right
        return product if self.kind == "and" else left + right - product


class HomSqlPipeline(HomOp):
    def __init__(self, packing, circuit, comparison, params):
        super().__init__()
        self.parameters_shape = HomReshape(packing.parameter_shape)
        self.compare = _SqlCompare(comparison, params)
        self.refresh_packed = (
            _SqlBootstrap(target_output_scale=params.pt_scale)
            if circuit.refresh_packed
            else None
        )
        self.refresh_database = (
            _SqlBootstrap(target_output_scale=params.pt_scale)
            if circuit.refresh_database
            else None
        )
        self.steps = ModuleListHomOp(
            (_CircuitBinaryStep if len(step.inputs) == 2 else _CircuitStep)(
                step, packing, params
            )
            for step in circuit.instructions
        )
        self.instructions = circuit.instructions
        self.root = circuit.root
        self.output_axis = HomUnsqueeze(dim=1)
        self.bootstrap_count = circuit.bootstraps
        self.rotation_count = circuit.rotations
        self.multiplication_count = circuit.multiplies
        self.comparison_x_accuracy = comparison.x_accuracy
        self.comparison_y_accuracy = comparison.y_accuracy
        self.predicate_error_bound = comparison.error_bound
        # Release clear intermediates as soon as their final consumer runs.
        uses = [0] * len(circuit.instructions)
        for instruction in circuit.instructions:
            for source in instruction.inputs:
                if source >= 0:
                    uses[source] += 1
        uses[self.root] += 1
        self.uses = tuple(uses)

    def forward(self, parameters, comparison_database, database):
        parameters = self.parameters_shape(parameters)
        compared = self.compare(comparison_database, parameters)
        if self.refresh_packed is not None:
            compared = self.refresh_packed(compared)
        values = {}
        uses = list(self.uses)
        for index, (instruction, operation) in enumerate(
            zip(self.instructions, self.steps)
        ):
            arguments = [
                compared if source == -1 else values[source]
                for source in instruction.inputs
            ]
            values[index] = operation(*arguments)
            for source in instruction.inputs:
                if source >= 0:
                    uses[source] -= 1
                    if uses[source] == 0:
                        del values[source]
        if self.refresh_database is not None:
            database = self.refresh_database(database)
        return database * self.output_axis(values[self.root])


def _matches_identifier(identifier: exp.Identifier, expected: str) -> bool:
    actual = identifier.name
    if identifier.args.get("quoted"):
        return actual == expected
    return actual.casefold() == expected.casefold()


def _resolve_column(expression: exp.Column, schema: SqlTableSchema, lookup=None) -> int:
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
    if lookup is None:
        lookup = {
            column.name.casefold(): index for index, column in enumerate(schema.columns)
        }
    index = lookup.get(identifier.name.casefold())
    if index is not None and _matches_identifier(
        identifier, schema.columns[index].name
    ):
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


def _compile_predicate(expression, *, schema):
    """Build a canonical, topologically ordered DAG without walking recursively."""
    nodes: list[_Predicate] = []
    node_ids: dict[_Predicate, int] = {}
    bindings: list[_ParameterBinding] = []
    binding_ids: dict[_ParameterBinding, int] = {}
    parameter_names = {}
    columns = {c.name.casefold(): i for i, c in enumerate(schema.columns)}

    def intern(node):
        if node not in node_ids:
            node_ids[node] = len(nodes)
            nodes.append(node)
        return node_ids[node]

    def comparison(binding):
        if binding not in binding_ids:
            binding_ids[binding] = len(bindings)
            bindings.append(binding)
        return intern(_Predicate("compare", binding_index=binding_ids[binding]))

    def combine(operator, children):
        children = set(children)
        opposite = "or" if operator == "and" else "and"
        # Absorption: A AND (A OR B) = A, and the dual rule.
        children = {
            child
            for child in children
            if not (
                nodes[child].operator == opposite
                and children.intersection(nodes[child].children)
            )
        }
        grouped = {}
        remaining = []
        for child in sorted(children):
            node = nodes[child]
            if node.operator != "compare":
                remaining.append(child)
                continue
            binding = bindings[node.binding_index]
            reduction = (
                "max" if ((operator == "and") == binding.column_is_greater) else "min"
            )
            if binding.reduction not in (None, reduction):
                remaining.append(child)
                continue
            grouped.setdefault(
                (binding.column_index, binding.column_is_greater, reduction), []
            ).append(child)
        for (column, greater, reduction), group in grouped.items():
            if len(group) == 1:
                remaining.extend(group)
            else:
                names = tuple(
                    sorted(
                        {
                            name
                            for child in group
                            for name in bindings[nodes[child].binding_index].names
                        }
                    )
                )
                remaining.append(
                    comparison(_ParameterBinding(names, column, greater, reduction))
                )
        children = tuple(sorted(set(remaining)))
        return (
            children[0]
            if len(children) == 1
            else intern(_Predicate(operator, children))
        )

    results = {}
    tasks = [(expression, None)]
    while tasks:
        current, children = tasks.pop()
        if children == "alias":
            inner = current.this
            while isinstance(inner, exp.Paren):
                inner = inner.this
            results[id(current)] = results[id(inner)]
            continue
        if children is not None:
            operator = "and" if isinstance(current, exp.And) else "or"
            results[id(current)] = combine(
                operator, [results[id(child)] for child in children]
            )
            continue
        if isinstance(current, exp.Paren):
            # Parentheses are transparent; flatten without growing the Python stack.
            inner = current.this
            while isinstance(inner, exp.Paren):
                inner = inner.this
            tasks.append((current, "alias"))
            tasks.append((inner, None))
            continue
        if isinstance(current, (exp.And, exp.Or)):
            children = []
            pending = [current]
            operator_type = type(current)
            while pending:
                item = pending.pop()
                while isinstance(item, exp.Paren):
                    item = item.this
                if isinstance(item, operator_type):
                    pending.extend((item.expression, item.this))
                else:
                    children.append(item)
            tasks.append((current, children))
            tasks.extend((child, None) for child in reversed(children))
            continue
        if not isinstance(current, (exp.GT, exp.LT)):
            raise SqlCompileError("WHERE supports only >, <, AND, OR, and parentheses.")
        left, right = current.this, current.expression
        if isinstance(left, exp.Column) and isinstance(right, exp.Placeholder):
            column, parameter = left, right
            greater = isinstance(current, exp.GT)
        elif isinstance(left, exp.Placeholder) and isinstance(right, exp.Column):
            column, parameter = right, left
            greater = isinstance(current, exp.LT)
        else:
            raise SqlCompileError(
                "Each comparison must contain one column and one named parameter."
            )
        column_index = _resolve_column(column, schema, columns)
        name = _placeholder_name(parameter)
        parameter_names[name] = None
        results[id(current)] = comparison(
            _ParameterBinding((name,), column_index, greater)
        )
    root = results[id(expression)]
    # Drop branches eliminated by Boolean simplification and parameter folding.
    live = set()
    pending = [root]
    while pending:
        node_id = pending.pop()
        if node_id not in live:
            live.add(node_id)
            pending.extend(nodes[node_id].children)
    remap = {old: new for new, old in enumerate(sorted(live))}
    used_bindings = sorted(
        {nodes[i].binding_index for i in live if nodes[i].operator == "compare"}
    )
    binding_remap = {old: new for new, old in enumerate(used_bindings)}
    compact = tuple(
        _Predicate(
            nodes[i].operator,
            tuple(remap[c] for c in nodes[i].children),
            binding_remap[nodes[i].binding_index]
            if nodes[i].operator == "compare"
            else -1,
        )
        for i in sorted(live)
    )
    return (
        compact,
        remap[root],
        tuple(bindings[i] for i in used_bindings),
        tuple(parameter_names),
    )


def _parse_where_tokens(tokens):
    """Shunting-yard parser for the existing predicate grammar, without recursion."""
    from sqlglot.tokens import TokenType as T

    values, operators = [], []
    index = 0
    expect_value = True
    precedence = {T.OR: 1, T.AND: 2}

    def fail():
        raise SqlCompileError(
            "WHERE supports column/parameter comparisons with >, <, AND, OR, and balanced parentheses."
        )

    def reduce_operator():
        if len(values) < 2:
            fail()
        right, left = values.pop(), values.pop()
        operator = operators.pop()
        values.append(
            (exp.And if operator == T.AND else exp.Or)(this=left, expression=right)
        )

    def identifier(token):
        if token.token_type not in (T.VAR, T.IDENTIFIER):
            fail()
        return exp.Identifier(this=token.text, quoted=token.token_type == T.IDENTIFIER)

    def operand():
        nonlocal index
        if index >= len(tokens):
            fail()
        token = tokens[index]
        index += 1
        if token.token_type == T.COLON:
            if index >= len(tokens) or not _IDENTIFIER_PATTERN.fullmatch(
                tokens[index].text
            ):
                fail()
            name = tokens[index].text
            index += 1
            return exp.Placeholder(this=name)
        column = identifier(token)
        if index < len(tokens) and tokens[index].token_type == T.DOT:
            index += 1
            if index >= len(tokens):
                fail()
            name = identifier(tokens[index])
            index += 1
            return exp.Column(this=name, table=column)
        return exp.Column(this=column)

    while index < len(tokens):
        token = tokens[index].token_type
        if expect_value:
            if token == T.L_PAREN:
                operators.append(token)
                index += 1
                continue
            left = operand()
            if index >= len(tokens) or tokens[index].token_type not in (T.GT, T.LT):
                fail()
            comparison = exp.GT if tokens[index].token_type == T.GT else exp.LT
            index += 1
            values.append(comparison(this=left, expression=operand()))
            expect_value = False
        elif token == T.R_PAREN:
            while operators and operators[-1] != T.L_PAREN:
                reduce_operator()
            if not operators:
                fail()
            operators.pop()
            index += 1
        elif token in precedence:
            while (
                operators
                and operators[-1] in precedence
                and precedence[operators[-1]] >= precedence[token]
            ):
                reduce_operator()
            operators.append(token)
            expect_value = True
            index += 1
        else:
            fail()
    if expect_value:
        fail()
    while operators:
        if operators[-1] == T.L_PAREN:
            fail()
        reduce_operator()
    if len(values) != 1:
        fail()
    return values[0]


def _parse_sql_structure(sql, dialect):
    """Let sqlglot validate SELECT/FROM; parse the unbounded WHERE iteratively."""
    from sqlglot.tokens import TokenType as T

    tokens = sqlglot.Dialect.get_or_raise(dialect).tokenize(sql)
    if tokens and tokens[-1].token_type == T.SEMICOLON:
        tokens = tokens[:-1]
    if any(token.token_type == T.SEMICOLON for token in tokens):
        raise SqlCompileError("Exactly one SQL statement is required.")
    if not tokens or tokens[0].token_type != T.SELECT:
        raise SqlCompileError("Only SELECT statements are supported.")
    where_index = next(
        (i for i, token in enumerate(tokens) if token.token_type == T.WHERE), None
    )
    if where_index is None:
        return sqlglot.parse(sql, read=dialect)
    prefix = tokens[:where_index]
    if any(token.token_type in (T.L_PAREN, T.R_PAREN) for token in prefix):
        raise SqlCompileError(
            "SELECT projections and FROM must be simple columns and one named table."
        )
    statements = sqlglot.parse(
        sql[: tokens[where_index].start] + " WHERE 1 > :_shape", read=dialect
    )
    if len(statements) == 1 and isinstance(statements[0], exp.Select):
        statements[0].set(
            "where", exp.Where(this=_parse_where_tokens(tokens[where_index + 1 :]))
        )
    return statements


def _parse_select(
    sql: str,
    *,
    schema: SqlTableSchema,
    options: SqlSelectOptions,
) -> _SelectPlan:
    if not isinstance(sql, str) or not sql.strip():
        raise SqlCompileError("SQL must be a non-empty string.")
    try:
        statements = _parse_sql_structure(sql, options.dialect)
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
        column_lookup = {
            column.name.casefold(): index for index, column in enumerate(schema.columns)
        }
        for projection in projections:
            if not isinstance(projection, exp.Column):
                raise SqlCompileError(
                    "SELECT projections must be simple columns without aliases or expressions; "
                    f"got {projection.sql()!r}."
                )
            projected_indices_list.append(
                _resolve_column(projection, schema, column_lookup)
            )
        if len(set(projected_indices_list)) != len(projected_indices_list):
            raise SqlCompileError("Duplicate projected columns are not supported.")
        projected_indices = tuple(projected_indices_list)

    where = statement.args.get("where")
    if not isinstance(where, exp.Where):
        raise SqlCompileError("SELECT requires a WHERE clause.")
    nodes, root, bindings, parameter_names = _compile_predicate(
        where.this, schema=schema
    )
    return _SelectPlan(nodes, root, bindings, projected_indices, parameter_names)


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

    Supply either row_count (capacity per batch) or database. The latter also
    prepares the two database views; neither view is embedded in the graph.
    Larger tables can stream through iter_database_batches without recompiling.
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
    columns = None
    if database is not None:
        columns, count = _validate_database(database, schema=schema)
        row_count = max(1, count)
    comparison = _plan_comparison(plan, schema, options, hom_params)
    packing, layout, circuit = _plan_execution(
        plan, row_count, hom_params, options, comparison
    )
    params = replace(hom_params, n_slots=packing.working_slots)
    hom = HomSqlPipeline(packing, circuit, comparison, params)
    pipeline = HomomorphicPipeline(
        hom=hom,
        input_shape={
            CompiledSqlSelect.PARAMETERS_INPUT_NAME: packing.parameter_shape,
            CompiledSqlSelect.COMPARISONS_INPUT_NAME: packing.comparison_shape,
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
            f"Cannot compile SQL with the supplied HE parameters: {exc}"
        ) from exc
    compiled = CompiledSqlSelect(
        sql=sql,
        schema=schema,
        hom_params=params,
        options=options,
        pipeline=pipeline,
        packing=packing,
        parameter_names=plan.parameter_names,
        projected_columns=tuple(
            schema.columns[i] for i in plan.projected_column_indices
        ),
        _plan=plan,
        _layout=layout,
    )
    if database is not None:
        compiled = replace(
            compiled, prepared_database=compiled._prepare_validated(columns, count)
        )
    return compiled

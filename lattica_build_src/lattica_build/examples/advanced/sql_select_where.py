"""Reusable encrypted SQL selection pipeline builder.

Edit ``SQL_QUERY``, ``DATA_SEED``, or the HE constants to experiment. Changing
the query or database dimensions requires redeployment. Runtime concerns
(query parameters, input preparation, and result verification/display) live in
the client demo adapter ``lattica_internal_demos.sql_select_where``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from lattica_build.base_classes.hom_pipeline import HomomorphicPipeline
from lattica_build.params.params import (
    DecompositionType,
    HomParams,
)
from lattica_build.examples.advanced.sql_pipeline_compiler import (
    CompiledSqlSelect,
    SqlColumn,
    SqlSelectOptions,
    SqlTableSchema,
    compile_sql_select,
)

NUM_ROWS = 100
DATA_SEED = 2023_1446
VALUE_MIN = 40
VALUE_MAX = 100

N = 2**14
Q_LIST_PRECISION = ((60, 30),) * 9
PT_SCALE = 2**30
SK_HW = 192
NUM_SPECIAL_PRIMES = 9

X_ACCURACY = 9
Y_ACCURACY = 10

SQL_QUERY = """
    SELECT *
    FROM table1
    WHERE col1 > :threshold1
       OR (col2 > :threshold2
           AND col3 > :threshold3)
"""

# These non-semantic aliases do not reveal what the private columns represent.
# Database values and SQL parameters are encrypted client-side before upload;
# the remote service receives the compiled pipeline and tensor dimensions.
TABLE_SCHEMA = SqlTableSchema(
    name="table1",
    columns=(
        SqlColumn(name="col0", min_value=1, max_value=NUM_ROWS, kind="integer"),
        SqlColumn(name="col1", min_value=0, max_value=100, kind="integer"),
        SqlColumn(name="col2", min_value=0, max_value=100, kind="integer"),
        SqlColumn(name="col3", min_value=0, max_value=100, kind="integer"),
    ),
)


@dataclass(frozen=True)
class ExampleTable:
    col0: torch.Tensor
    col1: torch.Tensor
    col2: torch.Tensor
    col3: torch.Tensor

    def as_columns(self) -> dict[str, torch.Tensor]:
        return {
            "col0": self.col0,
            "col1": self.col1,
            "col2": self.col2,
            "col3": self.col3,
        }


def generate_example_table(seed: int = DATA_SEED) -> ExampleTable:
    generator = torch.Generator().manual_seed(seed)
    values = torch.randint(
        VALUE_MIN,
        VALUE_MAX + 1,
        (3, NUM_ROWS),
        generator=generator,
        dtype=torch.int64,
    )
    return ExampleTable(
        col0=torch.arange(1, NUM_ROWS + 1, dtype=torch.int64),
        col1=values[0],
        col2=values[1],
        col3=values[2],
    )


def example_hom_params() -> HomParams:
    return HomParams(
        full_q_list_precision=Q_LIST_PRECISION,
        n=N,
        pt_scale=PT_SCALE,
        sk_hw=SK_HW,
        num_special_primes=NUM_SPECIAL_PRIMES,
        decomposition_type=DecompositionType.HYBRID,
    )


def compile_example(
    database: ExampleTable,
) -> CompiledSqlSelect:
    return compile_sql_select(
        SQL_QUERY,
        schema=TABLE_SCHEMA,
        database=database.as_columns(),
        hom_params=example_hom_params(),
        options=SqlSelectOptions(
            x_accuracy=X_ACCURACY,
            y_accuracy=Y_ACCURACY,
        ),
    )


def build_pipeline() -> HomomorphicPipeline:
    """Build the deterministic example pipeline for the Lattica Build CLI."""
    return compile_example(generate_example_table()).pipeline


def build_params() -> HomParams:
    """Build the HE parameters paired with the deterministic example pipeline."""
    return example_hom_params()

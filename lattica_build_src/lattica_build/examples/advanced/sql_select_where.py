"""Reusable encrypted SQL selection pipeline builder.

Edit ``SQL_QUERY``, ``DATA_SEED``, or the HE constants to experiment. Changing
the query or database dimensions requires redeployment. Threshold values and
replacement rows are prepared at runtime by ``CompiledSqlSelect``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from lattica_build.base_classes.hom_pipeline import HomomorphicPipeline
from lattica_build.base_classes.pipeline_wrapper import PipelineWrapper
from lattica_build.examples.advanced.sql_pipeline_compiler import (
    CompiledSqlSelect,
    SqlColumn,
    SqlSelectOptions,
    SqlTableSchema,
    compile_sql_select,
)
from lattica_build.params.params import (
    DecompositionType,
    HomParams,
)

NUM_ROWS = 100
DATA_SEED = 2023_1446
VALUE_MIN = 40
VALUE_MAX = 100

N = 2**14
Q_LIST_PRECISION = ((60, 30),) * 10
PT_SCALE = 2**30
SK_HW = 192
NUM_SPECIAL_PRIMES = 9

X_ACCURACY = 9
Y_ACCURACY = 10
EXAMPLE_PARAMETERS = {"threshold1": 90.0, "threshold2": 80.0, "threshold3": 85.0}

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


class Pipeline(PipelineWrapper):
    """CLI and local-runner adapter sharing one compiled graph and HE config."""

    def __init__(self) -> None:
        self.compiled = compile_example(generate_example_table())

    def build_pipeline(self) -> HomomorphicPipeline:
        pipeline = self.compiled.pipeline
        parameters = self._set_example_pt()
        pipeline.verification_data = {
            self.compiled.PARAMETERS_INPUT_NAME: parameters,
            **self.compiled.prepared_database,
            "expected_output": self.compiled.apply_clear(parameters),
            "accuracy": 2**-8,
        }
        return pipeline

    def build_params(self) -> HomParams:
        return self.compiled.hom_params

    def _set_example_pt(self) -> torch.Tensor:
        return self.compiled.prepare_parameters(EXAMPLE_PARAMETERS)

    def _set_custom_data(self) -> None:
        self.custom_data = self.compiled.prepared_database

    def compute_expected(self, example_pt: torch.Tensor) -> torch.Tensor:
        return self.compiled.apply_clear(example_pt)

    def verify_results(self, actual: torch.Tensor, expected: torch.Tensor) -> None:
        reference = self.compute_expected(self.exmpl_pt)
        actual_rows = self.compiled.decode_result(actual).row_indices
        expected_rows = self.compiled.decode_result(reference).row_indices
        torch.testing.assert_close(actual_rows, expected_rows, rtol=0, atol=0)
        torch.testing.assert_close(actual, reference, rtol=0, atol=2**-8)


def build_pipeline() -> HomomorphicPipeline:
    """Compatibility entry point for callers using the module-level API."""
    return Pipeline().build_pipeline()


def build_params() -> HomParams:
    """Build the HE parameters paired with the deterministic example pipeline."""
    return Pipeline().build_params()

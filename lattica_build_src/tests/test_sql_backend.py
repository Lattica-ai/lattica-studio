"""Optional encrypted regression checks using the locally installed toolkit."""

from dataclasses import replace

import pytest
import torch

pytest.importorskip("latticabe", reason="Encrypted SQL checks require lattice-toolkit")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")


@pytest.mark.parametrize("bootstrap", ["auto", "always"])
def test_encrypted_sql_roundtrip(bootstrap):
    from lattica_build.examples.advanced import sql_select_where as example
    from lattica_build.examples.advanced.sql_pipeline_compiler import compile_sql_select
    from latticabe.inference.common import InferenceRunner

    wrapper = example.Pipeline()
    if bootstrap == "always":
        wrapper.compiled = compile_sql_select(
            example.SQL_QUERY,
            schema=example.TABLE_SCHEMA,
            database=example.generate_example_table().as_columns(),
            hom_params=example.example_hom_params(),
            options=replace(wrapper.compiled.options, bootstrap=bootstrap),
        )
    runner = InferenceRunner(wrapper, client_on_gpu=True)
    runner.run_example()
    assert wrapper.compiled.bootstrap_count == (bootstrap == "always")
    # In particular, both custom encryption states must have been initialized
    # before the client encrypts the two database views.
    assert all(
        metadata.ct_state is not None
        for metadata in runner.be_context.custom_ct_metadata.values()
    )


@pytest.mark.parametrize(
    "case",
    [
        "packed_and",
        "packed_or",
        "mixed",
        "wide",
        "rows",
        "reduced_chain",
        "cpu_client",
        "many",
    ],
)
def test_encrypted_sql_plans(case):
    from lattica_build.base_classes.pipeline_wrapper import PipelineWrapper
    from lattica_build.examples.advanced import sql_select_where as example
    from lattica_build.examples.advanced.sql_pipeline_compiler import (
        SqlColumn,
        SqlSelectOptions,
        SqlTableSchema,
        compile_sql_select,
    )
    from latticabe.inference.common import InferenceRunner

    params = example.example_hom_params()
    options = SqlSelectOptions()
    if case == "wide":
        schema = SqlTableSchema(
            name="t",
            columns=tuple(
                SqlColumn(name=name, min_value=0, max_value=1_000_000, kind="integer")
                for name in ("a", "b")
            ),
        )
        data = {
            name: torch.tensor([500000, 500001, 500002, 500100]) for name in ("a", "b")
        }
        query = "SELECT a FROM t WHERE a>:p AND b>:q"
        thresholds = {"p": 500000, "q": 500000}
        selected = torch.tensor([1, 2, 3])
    elif case == "rows":
        schema = SqlTableSchema(
            name="t",
            columns=(SqlColumn(name="a", min_value=0, max_value=100, kind="integer"),),
        )
        data = {"a": torch.arange(9001) % 100}
        query, thresholds = "SELECT a FROM t WHERE a>:p", {"p": 50}
        selected = torch.nonzero(data["a"] > 50).flatten()
    else:
        count = (
            1024
            if case == "many"
            else (64 if case in ("packed_and", "reduced_chain") else 5)
        )
        schema = SqlTableSchema(
            name="t",
            columns=tuple(
                SqlColumn(name=f"c{i}", min_value=0, max_value=100, kind="integer")
                for i in range(count)
            ),
        )
        generator = torch.Generator().manual_seed(101)
        matrix = torch.randint(
            0, 101, (count, 3 if case == "many" else 17), generator=generator
        )
        matrix[:, 0] = 90
        matrix[:, 1] = 10
        if case == "many":
            matrix[:, 2] = 50
        data = {column.name: matrix[i] for i, column in enumerate(schema.columns)}
        thresholds = {f"p{i}": 40 for i in range(count)}
        if case == "mixed":
            query = "SELECT c4, c0 FROM t WHERE (c0>:p0 AND c1<:p1) OR (c2>:p2 AND c3<:p3) OR c4>:p4"
            truth = (
                ((matrix[0] > 40) & (matrix[1] < 40))
                | ((matrix[2] > 40) & (matrix[3] < 40))
                | (matrix[4] > 40)
            )
            options = SqlSelectOptions(rows_per_block=8)
        else:
            operator = (
                "AND" if case in ("packed_and", "reduced_chain", "many") else "OR"
            )
            query = "SELECT c0 FROM t WHERE " + f" {operator} ".join(
                f"c{i}>:p{i}" for i in range(count)
            )
            truth = (matrix > 40).all(0) if operator == "AND" else (matrix > 40).any(0)
            if case == "packed_or":
                options = SqlSelectOptions(rows_per_block=8)
        selected = torch.nonzero(truth).flatten()
        if case == "reduced_chain":
            params = replace(params, num_init_rows=8)
    compiled = compile_sql_select(
        query, schema=schema, database=data, hom_params=params, options=options
    )

    class Wrapper(PipelineWrapper):
        def build_pipeline(self):
            return compiled.pipeline

        def build_params(self):
            return compiled.hom_params

        def _set_example_pt(self):
            return compiled.prepare_parameters(thresholds)

        def _set_custom_data(self):
            self.custom_data = compiled.prepared_database

        def compute_expected(self, pt):
            return compiled.apply_clear(pt)

        def verify_results(self, actual, expected):
            result = compiled.decode_result(actual)
            torch.testing.assert_close(result.row_indices, selected, rtol=0, atol=0)
            torch.testing.assert_close(
                actual,
                self.compute_expected(self.exmpl_pt),
                rtol=0,
                atol=options.error_tolerance,
            )

        def display_results(self, actual, expected):
            pass

    runner = InferenceRunner(Wrapper(), client_on_gpu=case != "cpu_client")
    runner.run_example()

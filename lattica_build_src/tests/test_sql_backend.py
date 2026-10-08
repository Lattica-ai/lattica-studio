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

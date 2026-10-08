"""SQL semantics, packing boundaries, budgets, and the build/codec contract."""

import json
from dataclasses import replace

import pytest
import torch
from lattica_build.build import build_module
from lattica_build.examples.advanced import sql_select_where as example
from lattica_build.examples.advanced.sql_pipeline_compiler import (
    SqlColumn,
    SqlCompileError,
    SqlSelectOptions,
    SqlTableSchema,
    compile_sql_select,
)
from lattica_build.serialization.hom_op_pb2 import HomOpType

SCHEMA = SqlTableSchema(
    name="t",
    columns=(
        SqlColumn(name="id", min_value=-1, max_value=1000, kind="integer"),
        SqlColumn(name="a", min_value=-10, max_value=10, kind="integer"),
        SqlColumn(name="b", min_value=-10, max_value=10, kind="integer"),
    ),
)
DATA = {
    "id": torch.arange(7),
    "a": torch.tensor([-10, -2, -1, 0, 1, 2, 10]),
    "b": torch.tensor([10, 2, 1, 0, -1, -2, -10]),
}


def compile_query(query, *, data=DATA, rows=10, **kwargs):
    params = replace(
        example.example_hom_params(), full_q_list_precision=((60, 30),) * rows
    )
    return compile_sql_select(
        query, schema=SCHEMA, database=data, hom_params=params, **kwargs
    )


def graph_leaves(node):
    if node.get("op") is not None:
        yield node
    for child in node.get("child_ops", []):
        yield from graph_leaves(child)


@pytest.mark.parametrize(
    "query,parameters,selected,projection",
    [
        ("SELECT id FROM t WHERE a > :v", {"v": 0}, DATA["a"] > 0, ("id",)),
        ("SELECT b, id FROM t WHERE :v > a", {"v": 0}, DATA["a"] < 0, ("b", "id")),
        (
            "SELECT * FROM t WHERE a > :v OR b > :v",
            {"v": 0},
            (DATA["a"] > 0) | (DATA["b"] > 0),
            ("id", "a", "b"),
        ),
        (
            "SELECT id FROM t WHERE a > :lo AND a < :hi",
            {"lo": -1, "hi": 2},
            (DATA["a"] > -1) & (DATA["a"] < 2),
            ("id",),
        ),
        ("SELECT id FROM t WHERE a > :v", {"v": 0.9}, DATA["a"] > 0.9, ("id",)),
        ("SELECT id FROM t WHERE a < :v", {"v": -0.9}, DATA["a"] < -0.9, ("id",)),
        ("SELECT id FROM t WHERE a > :v", {"v": 10}, DATA["a"] > 10, ("id",)),
        ("SELECT id FROM t WHERE a < :v", {"v": -10}, DATA["a"] < -10, ("id",)),
        (
            "SELECT id FROM t WHERE a > :lo OR a < :hi",
            {"lo": -10, "hi": 10},
            torch.ones(7, dtype=torch.bool),
            ("id",),
        ),
    ],
)
def test_integer_sql_matches_torch(query, parameters, selected, projection):
    compiled = compile_query(query)
    prepared = compiled.prepare_parameters(parameters)
    actual = compiled.pipeline.forward_clear(
        parameters=prepared, **compiled.prepared_database
    )
    expected = compiled.apply_clear(prepared)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0.004)
    result = compiled.decode_result(actual)
    torch.testing.assert_close(result.row_indices, torch.nonzero(selected).flatten())
    assert tuple(result.columns) == projection
    for name in projection:
        # Check the exact reference codec separately from CKKS approximation error.
        decoded_reference = compiled.decode_result(expected)
        torch.testing.assert_close(
            decoded_reference[name], DATA[name][selected].double(), rtol=0, atol=1e-10
        )
    assert actual.shape[0] == 1 + len(projection)
    assert not actual[0, compiled.row_count : compiled.subring_slots].any()


def test_multiple_packs_and_unused_lanes():
    data = {
        "id": torch.arange(3),
        "a": torch.tensor([-5, 0, 5]),
        "b": torch.tensor([5, 0, -5]),
    }
    # Eight physical slots: two four-row blocks per ciphertext, five bindings.
    params = replace(example.example_hom_params(), n=16, n_slots=None)
    query = "SELECT b FROM t WHERE " + " OR ".join(f"a > :p{i}" for i in range(5))
    compiled = compile_sql_select(
        query, schema=SCHEMA, database=data, hom_params=params
    )
    assert compiled.parameter_shape == (3, 8)
    assert compiled.parameter_binding_count == 5
    prepared = compiled.prepare_parameters({f"p{i}": i for i in range(5)})
    actual = compiled.pipeline.forward_clear(
        parameters=prepared, **compiled.prepared_database
    )
    expected = compiled.apply_clear(prepared)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0.004)
    torch.testing.assert_close(
        compiled.decode_result(actual).row_indices, torch.tensor([2])
    )


@pytest.mark.parametrize("row_count", [1, 8, 9, 100])
def test_public_shape_compilation_and_database_updates(row_count):
    params = example.example_hom_params()
    compiled = compile_sql_select(
        "SELECT id FROM t WHERE a > :v",
        schema=SCHEMA,
        row_count=row_count,
        hom_params=params,
    )
    assert compiled.prepared_database is None
    assert params.n_slots == params.internal_n
    assert compiled.hom_params.n_slots == compiled.packing.working_slots
    prepared = compiled.prepare_parameters({"v": 0})
    for value in (-5, 5):
        data = {
            "id": torch.arange(row_count),
            "a": torch.full((row_count,), value),
            "b": torch.zeros(row_count),
        }
        views = compiled.prepare_database(data)
        result = compiled.pipeline.forward_clear(parameters=prepared, **views)
        expected = compiled.apply_clear(prepared, prepared_database=views)
        torch.testing.assert_close(result, expected, rtol=0, atol=0.004)
        assert len(compiled.decode_result(result)) == (row_count if value > 0 else 0)
    with pytest.raises(ValueError, match="prepared_database"):
        compiled.apply_clear(prepared)


def test_repeated_comparisons_share_bindings_and_graph_results():
    compiled = compile_query(
        "SELECT id FROM t WHERE (a > :v AND b < :v) OR (a > :v AND b < :v)"
    )
    assert compiled.parameter_binding_count == 2
    assert compiled.parameter_names == ("v",)
    graph, _ = compiled.pipeline.serialize(compiled.hom_params)
    leaves = list(graph_leaves(json.loads(graph)["pipeline_sections"]["hom"]))
    # One shared AND and one output multiply; A OR A simplifies to A.
    assert sum(op["op"] == HomOpType.Mul for op in leaves) == 2


def test_example_graph_has_no_repeat_bootstrap_or_channel_sums():
    compiled = example.compile_example(example.generate_example_table())
    assert compiled.packing.working_slots == 512
    assert compiled.bootstrap_count == 0
    assert compiled.output_shape == (5, 512)
    graph, _ = compiled.pipeline.serialize(compiled.hom_params)
    graph = json.loads(graph)
    assert "client_pre" not in graph["pipeline_sections"]
    leaves = list(graph_leaves(graph["pipeline_sections"]["hom"]))
    names = [HomOpType.Name(op["op"]) for op in leaves]
    assert names.count("ConstMul") == 1
    assert "Bootstrap" not in names
    assert "AxisSum" not in names
    assert compiled.hom_params.boot_params is None
    output = graph["pipeline_sections"]["hom"]["body_output"]
    assert sum(map(sum, output["active_cols"])) >= 2


@pytest.mark.parametrize(
    "policy,rows,expected_bootstraps",
    [("auto", 10, 0), ("auto", 9, 1), ("always", 10, 1)],
)
def test_bootstrap_policy(policy, rows, expected_bootstraps):
    compiled = compile_query(
        "SELECT * FROM t WHERE a > :p OR (b > :q AND a < :r)",
        rows=rows,
        options=SqlSelectOptions(bootstrap=policy),
    )
    assert compiled.bootstrap_count == expected_bootstraps
    p = compiled.prepare_parameters({"p": 0, "q": 0, "r": 5})
    actual = compiled.pipeline.forward_clear(parameters=p, **compiled.prepared_database)
    torch.testing.assert_close(actual, compiled.apply_clear(p), rtol=0, atol=0.004)


def test_deeper_boolean_tree_refreshes_before_exhaustion():
    compiled = compile_query(
        "SELECT id FROM t WHERE " + " AND ".join(f"a > :p{i}" for i in range(16)),
        rows=9,
    )
    assert compiled.bootstrap_count > 1
    p = compiled.prepare_parameters({f"p{i}": 0 for i in range(16)})
    actual = compiled.pipeline.forward_clear(parameters=p, **compiled.prepared_database)
    torch.testing.assert_close(
        compiled.decode_result(actual).row_indices, torch.tensor([4, 5, 6])
    )


def test_insufficient_budget_is_reported_at_compile_time():
    with pytest.raises(SqlCompileError, match="budget"):
        compile_query(
            "SELECT * FROM t WHERE a > :p OR (b > :q AND a < :r)",
            rows=9,
            options=SqlSelectOptions(bootstrap="never"),
        )
    with pytest.raises(SqlCompileError, match="budget"):
        compile_query("SELECT * FROM t WHERE a > :p", rows=2)
    with pytest.raises(ValueError, match="bootstrap"):
        SqlSelectOptions(bootstrap="invalid")


def test_private_database_values_are_not_serialized():
    a = compile_query("SELECT id FROM t WHERE a > :v")
    b = compile_query("SELECT id FROM t WHERE a > :v", data={**DATA, "a": -DATA["a"]})
    assert a.pipeline.serialize(a.hom_params) == b.pipeline.serialize(b.hom_params)


@pytest.mark.parametrize(
    "query",
    [
        "DELETE FROM t",
        "SELECT * FROM t",
        "SELECT id FROM t WHERE a = :v",
        "SELECT id FROM t WHERE a > :v LIMIT 1",
        "SELECT id FROM t JOIN t u ON t.a=u.a WHERE t.a>:v",
    ],
)
def test_unsupported_sql_is_rejected(query):
    with pytest.raises(SqlCompileError):
        compile_query(query)


def test_input_validation():
    c = compile_query("SELECT id FROM t WHERE a > :v")
    with pytest.raises(ValueError, match="placeholders"):
        c.prepare_parameters({"other": 0})
    with pytest.raises(ValueError, match="within"):
        c.prepare_parameters({"v": 11})
    with pytest.raises(ValueError, match="scalar"):
        c.prepare_parameters({"v": [1, 2]})
    with pytest.raises(ValueError, match="shape"):
        c.decode_result(torch.zeros(1, 1))
    with pytest.raises(ValueError, match="exactly one"):
        compile_sql_select(
            "SELECT * FROM t WHERE a>:v",
            schema=SCHEMA,
            hom_params=example.example_hom_params(),
        )


def test_cli_wrapper_builds_matching_pipeline_and_params(tmp_path):
    wrapper = example.Pipeline()
    assert wrapper.build_params() is wrapper.compiled.hom_params
    assert wrapper.build_params().n_slots == 512
    artifact = build_module(example, tmp_path / "sql.zip")
    assert artifact.path.is_file()
    assert artifact.init_context_params["n_slots"] == 512
    assert not artifact.init_context_params["bootstrapping"]

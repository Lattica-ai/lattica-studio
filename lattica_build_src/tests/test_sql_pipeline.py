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
    assert tuple(actual.shape) == compiled.output_shape
    assert compiled.packing.output_channels == 1 + len(projection)


def test_multiple_packs_and_unused_lanes():
    columns = tuple(
        SqlColumn(name=f"c{i}", min_value=-10, max_value=10, kind="integer")
        for i in range(5)
    )
    schema = SqlTableSchema(name="t", columns=columns)
    data = {column.name: torch.tensor([-5, 0, 5]) for column in columns}
    params = replace(example.example_hom_params(), n=16, n_slots=None)
    query = "SELECT c0 FROM t WHERE " + " OR ".join(f"c{i} > :p{i}" for i in range(5))
    compiled = compile_sql_select(
        query,
        schema=schema,
        database=data,
        hom_params=params,
        options=SqlSelectOptions(rows_per_block=4),
    )
    assert compiled.parameter_ciphertext_count > 1
    assert compiled.parameter_binding_count == 5
    prepared = compiled.prepare_parameters({f"p{i}": i for i in range(5)})
    actual = compiled.pipeline.forward_clear(
        parameters=prepared, **compiled.prepared_database
    )
    torch.testing.assert_close(
        actual,
        compiled.apply_clear(prepared),
        rtol=0,
        atol=compiled.options.error_tolerance,
    )
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
    assert compiled.output_shape == (1, 2, 512)
    graph, _ = compiled.pipeline.serialize(compiled.hom_params)
    graph = json.loads(graph)
    assert "client_pre" not in graph["pipeline_sections"]
    leaves = list(graph_leaves(graph["pipeline_sections"]["hom"]))
    names = [HomOpType.Name(op["op"]) for op in leaves]
    assert "Bootstrap" not in names
    assert "AxisSum" not in names
    assert compiled.hom_params.boot_params is None
    output = graph["pipeline_sections"]["hom"]["body_output"]
    assert sum(map(sum, output["active_cols"])) >= 2


@pytest.mark.parametrize(
    "policy,rows,expected_bootstraps",
    [("auto", 10, 0), ("auto", 9, 0), ("always", 10, 1)],
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
    schema = SqlTableSchema(
        name="t",
        columns=tuple(
            SqlColumn(name=f"c{i}", min_value=-10, max_value=10, kind="integer")
            for i in range(16)
        ),
    )
    data = {column.name: DATA["a"] for column in schema.columns}
    compiled = compile_sql_select(
        "SELECT c0 FROM t WHERE " + " AND ".join(f"c{i} > :p{i}" for i in range(16)),
        schema=schema,
        database=data,
        hom_params=replace(
            example.example_hom_params(), full_q_list_precision=((60, 30),) * 9
        ),
    )
    assert compiled.bootstrap_count >= 1
    p = compiled.prepare_parameters({f"p{i}": 0 for i in range(16)})
    actual = compiled.pipeline.forward_clear(parameters=p, **compiled.prepared_database)
    torch.testing.assert_close(
        actual, compiled.apply_clear(p), rtol=0, atol=compiled.options.error_tolerance
    )
    torch.testing.assert_close(
        compiled.decode_result(actual).row_indices, torch.tensor([4, 5, 6])
    )


def test_insufficient_budget_is_reported_at_compile_time():
    with pytest.raises(SqlCompileError, match="budget"):
        compile_query(
            "SELECT * FROM t WHERE a > :p OR (b > :q AND a < :r)",
            rows=7,
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
    with pytest.raises(ValueError, match="finite"):
        c.prepare_parameters({"v": float("nan")})
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


def test_wide_integer_range_and_comparison_stage_bootstrap():
    schema = SqlTableSchema(
        name="t",
        columns=tuple(
            SqlColumn(name=name, min_value=0, max_value=1_000_000, kind="integer")
            for name in ("a", "b")
        ),
    )
    data = {name: torch.tensor([500000, 500001, 500002, 500100]) for name in ("a", "b")}
    c = compile_sql_select(
        "SELECT a FROM t WHERE a>:p AND b>:q",
        schema=schema,
        database=data,
        hom_params=example.example_hom_params(),
    )
    p = c.prepare_parameters({"p": 500000, "q": 500000})
    actual = c.pipeline.forward_clear(parameters=p, **c.prepared_database)
    assert c.pipeline.hom.comparison_x_accuracy >= 21
    assert c.bootstrap_count > 0
    torch.testing.assert_close(
        c.decode_result(actual).row_indices, torch.tensor([1, 2, 3])
    )
    torch.testing.assert_close(
        actual, c.apply_clear(p), rtol=0, atol=c.options.error_tolerance
    )


@pytest.mark.parametrize("operator", ["AND", "OR"])
def test_long_query_folds_private_thresholds_without_recursion(operator):
    count = 3000
    c = compile_query(
        "SELECT id FROM t WHERE "
        + f" {operator} ".join(f"a>:p{i}" for i in range(count))
    )
    assert len(c.parameter_names) == count
    assert c.parameter_binding_count == 1
    thresholds = {f"p{i}": (i % 7) - 3 for i in range(count)}
    prepared = c.prepare_parameters(thresholds)
    actual = c.pipeline.forward_clear(parameters=prepared, **c.prepared_database)
    effective = (
        max(thresholds.values()) if operator == "AND" else min(thresholds.values())
    )
    torch.testing.assert_close(
        c.decode_result(actual).row_indices,
        torch.nonzero(DATA["a"] > effective).flatten(),
    )


@pytest.mark.parametrize(
    "query,bindings",
    [
        ("SELECT id FROM t WHERE (a>:p AND b>:q) OR (b>:q AND a>:p)", 2),
        ("SELECT id FROM t WHERE a>:p OR (a>:p AND b>:q)", 1),
        ("SELECT id FROM t WHERE a>:p AND (a>:p OR b>:q)", 1),
        ("SELECT id FROM t WHERE ((a>:p))", 1),
    ],
)
def test_canonical_boolean_expressions(query, bindings):
    c = compile_query(query)
    assert c.parameter_binding_count == bindings
    p = c.prepare_parameters({name: 0 for name in c.parameter_names})
    actual = c.pipeline.forward_clear(parameters=p, **c.prepared_database)
    torch.testing.assert_close(
        c.decode_result(actual).row_indices,
        c.decode_result(c.apply_clear(p)).row_indices,
    )


@pytest.mark.parametrize("operator", ["AND", "OR"])
@pytest.mark.parametrize("count", [3, 5, 16, 64])
def test_packed_reductions_match_independent_torch_reference(operator, count):
    schema = SqlTableSchema(
        name="t",
        columns=tuple(
            SqlColumn(name=f"c{i}", min_value=-10, max_value=10, kind="integer")
            for i in range(count)
        ),
    )
    generator = torch.Generator().manual_seed(10 + count)
    matrix = torch.randint(-10, 11, (count, 37), generator=generator)
    matrix[:, 0] = 10
    matrix[:, 1] = -10
    data = {column.name: matrix[i] for i, column in enumerate(schema.columns)}
    c = compile_sql_select(
        "SELECT c0 FROM t WHERE "
        + f" {operator} ".join(f"c{i}>:p{i}" for i in range(count)),
        schema=schema,
        database=data,
        hom_params=example.example_hom_params(),
    )
    p = c.prepare_parameters({f"p{i}": i % 3 for i in range(count)})
    truth = matrix > torch.tensor([i % 3 for i in range(count)])[:, None]
    selected = truth.all(0) if operator == "AND" else truth.any(0)
    actual = c.pipeline.forward_clear(parameters=p, **c.prepared_database)
    torch.testing.assert_close(
        c.decode_result(actual).row_indices, torch.nonzero(selected).flatten()
    )
    torch.testing.assert_close(
        actual, c.apply_clear(p), rtol=0, atol=c.options.error_tolerance
    )
    if count == 64:
        assert c.pipeline.hom.rotation_count < 64
        assert c.bootstrap_count < 4


def test_multiple_row_blocks_smaller_updates_and_streaming():
    schema = SqlTableSchema(
        name="t",
        columns=(SqlColumn(name="a", min_value=0, max_value=100, kind="integer"),),
    )
    c = compile_sql_select(
        "SELECT a FROM t WHERE a>:p",
        schema=schema,
        row_count=9001,
        hom_params=example.example_hom_params(),
    )
    assert c.packing.row_blocks > 1
    p = c.prepare_parameters({"p": 50})
    for count in (0, 1, 8192, 9001):
        data = {"a": torch.arange(count) % 100}
        views = c.prepare_database(data)
        out = c.pipeline.forward_clear(parameters=p, **views)
        expected = torch.nonzero(data["a"] > 50).flatten()
        torch.testing.assert_close(c.decode_result(out).row_indices, expected)
    data = {"a": torch.arange(20005) % 100}
    indices = []
    for offset, views in c.iter_database_batches(data):
        out = c.pipeline.forward_clear(parameters=p, **views)
        indices.append(c.decode_result(out, row_offset=offset).row_indices)
    torch.testing.assert_close(
        torch.cat(indices), torch.nonzero(data["a"] > 50).flatten()
    )


def test_float_inputs_keep_float64_precision():
    schema = SqlTableSchema(
        name="t",
        columns=(
            SqlColumn(
                name="a", min_value=499999, max_value=500001, comparison_resolution=0.01
            ),
        ),
    )
    c = compile_sql_select(
        "SELECT a FROM t WHERE a>:p",
        schema=schema,
        database={"a": [500000.01, 500000.02]},
        hom_params=example.example_hom_params(),
    )
    p = c.prepare_parameters({"p": 500000.005})
    out = c.pipeline.forward_clear(parameters=p, **c.prepared_database)
    result = c.decode_result(out)
    torch.testing.assert_close(result.row_indices, torch.tensor([0, 1]))
    torch.testing.assert_close(
        result["a"],
        torch.tensor([500000.01, 500000.02], dtype=torch.float64),
        rtol=0,
        atol=1e-8,
    )


@pytest.mark.parametrize(
    "value,greater", [(-100, True), (100, True), (-100, False), (100, False)]
)
def test_out_of_range_thresholds_preserve_sql_membership(value, greater):
    c = compile_query(f"SELECT id FROM t WHERE a {'>' if greater else '<'} :p")
    p = c.prepare_parameters({"p": value})
    actual = c.pipeline.forward_clear(parameters=p, **c.prepared_database)
    selected = DATA["a"] > value if greater else DATA["a"] < value
    torch.testing.assert_close(
        c.decode_result(actual).row_indices, torch.nonzero(selected).flatten()
    )


def test_database_normalizes_only_projected_columns_once(monkeypatch):
    from lattica_build.examples.advanced import sql_pipeline_compiler as sql

    calls = []
    original = sql._normalize

    def track(values, column):
        calls.append(column.name)
        return original(values, column)

    monkeypatch.setattr(sql, "_normalize", track)
    compile_query("SELECT id FROM t WHERE a>:p")
    assert calls == ["id"]


def test_deep_parentheses_do_not_use_python_recursion():
    expression = "a>:p"
    for i in range(1500):
        expression = f"b>:q {'AND' if i % 2 else 'OR'} ({expression})"
    c = compile_query("SELECT id FROM t WHERE " + expression)
    p = c.prepare_parameters({"p": 0, "q": 0})
    actual = c.pipeline.forward_clear(parameters=p, **c.prepared_database)
    torch.testing.assert_close(
        c.decode_result(actual).row_indices,
        c.decode_result(c.apply_clear(p)).row_indices,
    )


@pytest.mark.parametrize(
    "predicate",
    [
        "a>:p AND",
        "(a>:p",
        "a>:p)",
        "a>:p b>:q",
        "a>:p OR ()",
        "a>:p; SELECT id FROM t WHERE b>:q",
    ],
)
def test_invalid_predicate_structure_is_rejected(predicate):
    with pytest.raises(SqlCompileError):
        compile_query("SELECT id FROM t WHERE " + predicate)


def test_reduced_initial_budget_is_planned_separately():
    schema = SqlTableSchema(
        name="t",
        columns=tuple(
            SqlColumn(name=f"c{i}", min_value=0, max_value=100, kind="integer")
            for i in range(16)
        ),
    )
    c = compile_sql_select(
        "SELECT c0 FROM t WHERE " + " AND ".join(f"c{i}>:p{i}" for i in range(16)),
        schema=schema,
        row_count=5,
        hom_params=replace(example.example_hom_params(), num_init_rows=7),
    )
    assert c.hom_params.num_init_rows == 7
    assert c.bootstrap_count > 0


@pytest.mark.parametrize("seed", range(12))
def test_mixed_predicates_against_independent_reference(seed):
    import random

    rng = random.Random(seed)
    generator = torch.Generator().manual_seed(seed)
    schema = SqlTableSchema(
        name="t",
        columns=tuple(
            SqlColumn(name=f"c{i}", min_value=-10, max_value=10, kind="integer")
            for i in range(6)
        ),
    )
    matrix = torch.randint(-10, 11, (6, 31), generator=generator)
    data = {column.name: matrix[i] for i, column in enumerate(schema.columns)}
    thresholds = {}

    def expression(depth):
        if depth == 0 or rng.random() < 0.25:
            col = rng.randrange(6)
            name = f"p{len(thresholds)}"
            threshold = rng.randrange(-15, 16) + 0.25
            thresholds[name] = threshold
            greater = rng.choice((True, False))
            sql = f"c{col}{'>' if greater else '<'}:{name}"
            truth = matrix[col] > threshold if greater else matrix[col] < threshold
            return sql, truth
        left_sql, left = expression(depth - 1)
        right_sql, right = expression(depth - 1)
        use_and = rng.choice((True, False))
        return (
            f"({left_sql} {'AND' if use_and else 'OR'} {right_sql})",
            left & right if use_and else left | right,
        )

    predicate, selected = expression(5)
    c = compile_sql_select(
        "SELECT c5, c2 FROM t WHERE " + predicate,
        schema=schema,
        database=data,
        hom_params=example.example_hom_params(),
        options=SqlSelectOptions(rows_per_block=8),
    )
    p = c.prepare_parameters(thresholds)
    actual = c.pipeline.forward_clear(parameters=p, **c.prepared_database)
    result = c.decode_result(actual)
    torch.testing.assert_close(result.row_indices, torch.nonzero(selected).flatten())
    for column in ("c5", "c2"):
        torch.testing.assert_close(
            result[column], data[column][selected].double(), rtol=0, atol=1e-8
        )
    torch.testing.assert_close(
        actual, c.apply_clear(p), rtol=0, atol=c.options.error_tolerance
    )


def test_many_independent_comparisons():
    count = 1024
    schema = SqlTableSchema(
        name="t",
        columns=tuple(
            SqlColumn(name=f"c{i}", min_value=0, max_value=100, kind="integer")
            for i in range(count)
        ),
    )
    c = compile_sql_select(
        "SELECT c0 FROM t WHERE " + " AND ".join(f"c{i}>:p{i}" for i in range(count)),
        schema=schema,
        row_count=3,
        hom_params=example.example_hom_params(),
    )
    assert c.parameter_binding_count == count
    data = {column.name: torch.tensor([10, 50, 90]) for column in schema.columns}
    p = c.prepare_parameters({name: 40 for name in c.parameter_names})
    views = c.prepare_database(data)
    actual = c.pipeline.forward_clear(parameters=p, **views)
    torch.testing.assert_close(
        c.decode_result(actual).row_indices, torch.tensor([1, 2])
    )
    torch.testing.assert_close(
        actual,
        c.apply_clear(p, prepared_database=views),
        rtol=0,
        atol=c.options.error_tolerance,
    )


def test_large_finite_real_range_does_not_overflow_normalization():
    from lattica_build.examples.advanced.sql_pipeline_compiler import (
        _denormalize,
        _normalize,
    )

    column = SqlColumn(name="a", min_value=0, max_value=1e308)
    values = torch.tensor([0, 5e307, 1e308], dtype=torch.float64)
    encoded = _normalize(values, column)
    assert torch.isfinite(encoded).all()
    torch.testing.assert_close(encoded, torch.tensor([-1, 0, 1], dtype=torch.float64))
    torch.testing.assert_close(_denormalize(encoded, column), values)

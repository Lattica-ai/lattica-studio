"""See `base_classes/README.md` for usage details."""

from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import lattica_build.serialization.hom_op_pb2 as hom_op_pb2


_GRAPH_FILENAME = "hom_pipeline.json"

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


# ---------------------------------------------------------------------------
# Terminal styling
# ---------------------------------------------------------------------------


class Style:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    BRIGHT_WHITE = "\033[97m"
    BRIGHT_BLACK = "\033[90m"

    BRIGHT_CYAN = "\033[96m"
    BLUE = "\033[34m"
    YELLOW = "\033[33m"
    BRIGHT_GREEN = "\033[92m"

    MAGENTA = "\033[35m"
    BRIGHT_MAGENTA = "\033[95m"


@dataclass(frozen=True)
class Renderer:
    color: bool = True
    verbose: bool = False
    hom_metadata: bool = True

    def style(self, text: str, *styles: str) -> str:
        if not self.color:
            return text

        return "".join(styles) + text + Style.RESET


# ---------------------------------------------------------------------------
# Graph model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValueInfo:
    id: str
    shape: tuple[Any, ...] | None
    n_axis: int | None
    active_rows: tuple[int, ...]
    active_cols: tuple[tuple[bool, ...], ...]
    scale: Any | None


@dataclass(frozen=True)
class ModulusChainInfo:
    full_q_list_precision: tuple[tuple[int, ...], ...]
    pipeline_rows: tuple[int, ...]


@dataclass
class Scope:
    values: dict[str, ValueInfo] = field(default_factory=dict)
    producers: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Generic terminal helpers
# ---------------------------------------------------------------------------


def _visible_len(text: str) -> int:
    return len(_ANSI_RE.sub("", text))


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _load_graph_json(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))

    with zipfile.ZipFile(path, "r") as zf:
        return json.loads(
            zf.read(_GRAPH_FILENAME).decode("utf-8")
        )


# ---------------------------------------------------------------------------
# IR parsing
# ---------------------------------------------------------------------------


def _op_name(op_code: int | None) -> str:
    if op_code is None:
        return "Composite"

    enum_wrapper = getattr(hom_op_pb2, "HomOpType", None)

    if enum_wrapper is not None:
        try:
            return enum_wrapper.Name(int(op_code))
        except ValueError:
            pass

    return f"UnknownOp({op_code})"


def _value_info(
    value: dict[str, Any] | None,
) -> ValueInfo | None:
    if not isinstance(value, dict):
        return None

    value_id = value.get("id")

    if value_id is None:
        return None

    raw_shape = value.get("tensor_shape")
    shape = (
        tuple(raw_shape)
        if raw_shape is not None
        else None
    )

    raw_n_axis = value.get("n_axis")
    n_axis = (
        int(raw_n_axis)
        if raw_n_axis is not None
        else None
    )

    raw_active_rows = value.get("active_rows", [])
    active_rows = tuple(
        int(row)
        for row in raw_active_rows
    )

    raw_active_cols = value.get("active_cols", [])
    active_cols = tuple(
        tuple(bool(x) for x in row)
        for row in raw_active_cols
    )

    return ValueInfo(
        id=str(value_id),
        shape=shape,
        n_axis=n_axis,
        active_rows=active_rows,
        active_cols=active_cols,
        scale=value.get("pt_scale"),
    )


def _modulus_chain_info(
    graph: dict[str, Any],
) -> ModulusChainInfo | None:
    modulus_chain = graph.get("modulus_chain")

    if not isinstance(modulus_chain, dict):
        return None

    raw_q_list = modulus_chain.get("full_q_list_precision")

    if not isinstance(raw_q_list, list):
        return None

    full_q_list_precision = tuple(
        tuple(int(precision) for precision in row)
        for row in raw_q_list
    )

    section_rows = modulus_chain.get("section_rows", {})

    raw_pipeline_rows = (
        section_rows.get("pipeline")
        if isinstance(section_rows, dict)
        else None
    )

    if isinstance(raw_pipeline_rows, list):
        pipeline_rows = tuple(
            int(row)
            for row in raw_pipeline_rows
        )
    else:
        pipeline_rows = tuple(
            range(len(full_q_list_precision))
        )

    return ModulusChainInfo(
        full_q_list_precision=full_q_list_precision,
        pipeline_rows=pipeline_rows,
    )


# ---------------------------------------------------------------------------
# Value formatting
# ---------------------------------------------------------------------------


def _format_scale(scale: Any) -> str:
    try:
        value = float(scale)
    except (TypeError, ValueError, OverflowError):
        return str(scale)

    if value.is_integer():
        return str(int(value))

    # 10 significant digits total.
    return f"{value:.10g}"


def _format_external_shape(
    shape: tuple[Any, ...] | None,
    *,
    renderer: Renderer,
) -> str:
    if shape is None:
        return renderer.style(
            "?",
            Style.DIM,
        )

    text = "(" + ", ".join(
        str(dim)
        for dim in shape
    ) + ")"

    return renderer.style(
        text,
        Style.BLUE,
    )


def _format_hom_shape(
    value: ValueInfo,
    *,
    renderer: Renderer,
) -> str:
    if value.shape is None:
        return renderer.style(
            "?",
            Style.DIM,
        )

    return renderer.style(
        "(" + ", ".join(
            str(dim)
            for dim in value.shape
        ) + ")",
        Style.BLUE,
    )


def _active_col_lookup(
    value: ValueInfo,
) -> dict[int, tuple[bool, ...]]:
    return {
        row_index: cols
        for row_index, cols in zip(
            value.active_rows,
            value.active_cols,
        )
    }


def _format_q_chain(
    value: ValueInfo,
    *,
    modulus_chain: ModulusChainInfo | None,
    renderer: Renderer,
) -> str | None:
    if modulus_chain is None:
        return None

    active_by_row = _active_col_lookup(value)

    rendered_rows: list[str] = []

    for row_index in modulus_chain.pipeline_rows:
        if not (
            0
            <= row_index
            < len(modulus_chain.full_q_list_precision)
        ):
            continue

        precisions = (
            modulus_chain.full_q_list_precision[row_index]
        )

        active_cols = active_by_row.get(row_index)

        rendered_cols: list[str] = []

        for col_index, precision in enumerate(precisions):
            active = (
                active_cols is not None
                and col_index < len(active_cols)
                and active_cols[col_index]
            )

            text = str(precision)

            if active:
                text = renderer.style(
                    text,
                    Style.BRIGHT_MAGENTA,
                )
            else:
                text = renderer.style(
                    text,
                    Style.DIM,
                    Style.BRIGHT_BLACK,
                )

            rendered_cols.append(text)

        rendered_rows.append(
            "[" + " ".join(rendered_cols) + "]"
        )

    if not rendered_rows:
        return None

    return " ".join(rendered_rows)


def _format_hom_value_main(
    value: ValueInfo,
    *,
    renderer: Renderer,
) -> str:
    parts = [
        renderer.style(
            value.id,
            Style.BOLD,
            Style.BRIGHT_CYAN,
        )
    ]

    parts.append(
        "shape="
        + _format_hom_shape(
            value,
            renderer=renderer,
        )
    )

    if renderer.hom_metadata:
        rendered_n_axis = (
            renderer.style(
                str(value.n_axis),
                Style.BLUE,
            )
            if value.n_axis is not None
            else renderer.style("—", Style.DIM)
        )

        parts.append(
            "n_axis="
            + rendered_n_axis
        )

        rendered_scale = (
            renderer.style(
                _format_scale(value.scale),
                Style.BRIGHT_GREEN,
            )
            if value.scale is not None
            else renderer.style("—", Style.DIM)
        )

        parts.append(
            "scale="
            + rendered_scale
        )

    return "  ".join(parts)


def _format_hom_value(
    value: ValueInfo,
    *,
    modulus_chain: ModulusChainInfo | None,
    renderer: Renderer,
) -> tuple[str, str | None]:
    main = _format_hom_value_main(
        value,
        renderer=renderer,
    )

    q_chain = (
        _format_q_chain(
            value,
            modulus_chain=modulus_chain,
            renderer=renderer,
        )
        if renderer.hom_metadata
        else None
    )

    if q_chain is None:
        return main, None

    q_label = renderer.style(
        "q=",
        Style.BOLD,
        Style.MAGENTA,
    )

    return main, q_label + q_chain


def _format_scalar(
    value: Any,
    *,
    renderer: Renderer,
) -> str:
    return renderer.style(
        repr(value),
        Style.BOLD,
        Style.YELLOW,
    )


def _format_attr_value(
    value: Any,
    max_len: int = 90,
) -> str:
    rendered = repr(value)

    if len(rendered) > max_len:
        rendered = (
            rendered[: max_len - 3]
            + "..."
        )

    return rendered


# ---------------------------------------------------------------------------
# Rendering primitives
# ---------------------------------------------------------------------------


def _print_section_header(
    name: str,
    *,
    renderer: Renderer,
) -> None:
    print()

    print(
        renderer.style(
            name,
            Style.BOLD,
            Style.BRIGHT_WHITE,
        )
    )

    print(
        renderer.style(
            "─" * len(name),
            Style.BRIGHT_BLACK,
        )
    )


def _print_attrs(
    attrs: dict[str, Any],
    *,
    prefix: str,
    renderer: Renderer,
) -> None:
    for name in sorted(attrs):
        print(
            f"{prefix}"
            + renderer.style(
                name,
                Style.DIM,
            )
            + "="
            + renderer.style(
                _format_attr_value(attrs[name]),
                Style.DIM,
            )
        )


def _print_input(
    value_ref: Any,
    *,
    prefix: str,
    scope: Scope,
    modulus_chain: ModulusChainInfo | None,
    renderer: Renderer,
    label: str | None = None,
) -> None:
    line_prefix = f"{prefix}├─ "

    if label is not None:
        rendered_label = (
            renderer.style(
                label,
                Style.DIM,
            )
            + ": "
        )
    else:
        rendered_label = ""

    value_id = str(value_ref)
    value = scope.values.get(value_id)

    if value is not None:
        rendered_value, rendered_q = _format_hom_value(
            value,
            modulus_chain=modulus_chain,
            renderer=renderer,
        )
    else:
        rendered_value = _format_scalar(
            value_ref,
            renderer=renderer,
        )
        rendered_q = None

    print(
        line_prefix
        + rendered_label
        + rendered_value
    )

    if rendered_q is not None:
        print(
            f"{prefix}│  "
            + " " * _visible_len(rendered_label)
            + rendered_q
        )

    if renderer.verbose:
        producer = scope.producers.get(value_id)

        if producer is not None:
            print(
                f"{prefix}│  "
                + renderer.style(
                    f"from {producer}",
                    Style.BRIGHT_BLACK,
                )
            )


def _print_output(
    value: ValueInfo | None,
    *,
    prefix: str,
    modulus_chain: ModulusChainInfo | None,
    renderer: Renderer,
) -> None:
    if value is None:
        return

    arrow = renderer.style(
        "→ ",
        Style.BRIGHT_GREEN,
    )

    line_prefix = (
        f"{prefix}└─ "
        + arrow
    )

    rendered_value, rendered_q = _format_hom_value(
        value,
        modulus_chain=modulus_chain,
        renderer=renderer,
    )

    print(
        line_prefix
        + rendered_value
    )

    if rendered_q is not None:
        value_indent = (
            _visible_len(line_prefix)
            - _visible_len(prefix)
        )

        print(
            prefix
            + " " * value_indent
            + rendered_q
        )


# ---------------------------------------------------------------------------
# Scope handling
# ---------------------------------------------------------------------------


def _value_with_id(
    value: ValueInfo,
    value_id: Any,
) -> ValueInfo:
    return ValueInfo(
        id=str(value_id),
        shape=value.shape,
        n_axis=value.n_axis,
        active_rows=value.active_rows,
        active_cols=value.active_cols,
        scale=value.scale,
    )


def _seed_section_inputs(
    section: Any,
    *,
    scope: Scope,
) -> None:
    if not isinstance(section, dict):
        return

    raw_values = section.get("input_values")

    if not isinstance(raw_values, list):
        return

    caller_inputs = section.get("inputs", [])
    body_inputs = section.get("body_inputs", [])

    aliases = [
        values
        if isinstance(values, list)
        else []
        for values in (
            caller_inputs,
            body_inputs,
        )
    ]

    for index, raw_value in enumerate(raw_values):
        value = _value_info(raw_value)

        if value is None:
            continue

        scope.values[value.id] = value

        for names in aliases:
            if index < len(names):
                alias = _value_with_id(
                    value,
                    names[index],
                )
                scope.values[alias.id] = alias


def _make_local_composite_scope(
    op_ir: dict[str, Any],
    *,
    parent_scope: Scope,
) -> Scope:
    local_scope = Scope()

    body_inputs = op_ir.get(
        "body_inputs",
        [],
    )

    caller_inputs = op_ir.get(
        "inputs",
        [],
    )

    if (
        isinstance(body_inputs, list)
        and isinstance(caller_inputs, list)
        and caller_inputs
    ):
        for local_name, caller_ref in zip(
            body_inputs,
            caller_inputs,
        ):
            source = parent_scope.values.get(
                str(caller_ref)
            )

            if source is None:
                continue

            local_scope.values[str(local_name)] = ValueInfo(
                id=str(local_name),
                shape=source.shape,
                n_axis=source.n_axis,
                active_rows=source.active_rows,
                active_cols=source.active_cols,
                scale=source.scale,
            )

            producer = parent_scope.producers.get(
                str(caller_ref)
            )

            if producer is not None:
                local_scope.producers[
                    str(local_name)
                ] = producer

    else:
        # Top-level composite.
        if isinstance(body_inputs, list):
            for name in body_inputs:
                source = parent_scope.values.get(
                    str(name)
                )

                if source is not None:
                    local_scope.values[
                        str(name)
                    ] = source

    return local_scope


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------


def _print_regular_op_body(
    op_ir: dict[str, Any],
    *,
    node_id: str,
    prefix: str,
    scope: Scope,
    modulus_chain: ModulusChainInfo | None,
    renderer: Renderer,
) -> ValueInfo | None:
    attrs = op_ir.get(
        "op_attrs",
        {},
    )

    if isinstance(attrs, dict):
        _print_attrs(
            attrs,
            prefix=f"{prefix}   ",
            renderer=renderer,
        )

    inputs = op_ir.get(
        "inputs",
        [],
    )

    if isinstance(inputs, list):
        for value_ref in inputs:
            _print_input(
                value_ref,
                prefix=prefix,
                scope=scope,
                modulus_chain=modulus_chain,
                renderer=renderer,
            )

    custom_inputs = op_ir.get(
        "custom_inputs",
        {},
    )

    if isinstance(custom_inputs, dict):
        for name, value_ref in custom_inputs.items():
            _print_input(
                value_ref,
                prefix=prefix,
                scope=scope,
                modulus_chain=modulus_chain,
                renderer=renderer,
                label=name,
            )

    if renderer.verbose:
        data_ref = op_ir.get(
            "data_ref"
        )

        if data_ref:
            print(
                f"{prefix}├─ "
                + renderer.style(
                    (
                        "data_ref="
                        + _format_attr_value(data_ref)
                    ),
                    Style.DIM,
                )
            )

    output = _value_info(
        op_ir.get("output")
    )

    _print_output(
        output,
        prefix=prefix,
        modulus_chain=modulus_chain,
        renderer=renderer,
    )

    if output is not None:
        scope.values[output.id] = output

        scope.producers[output.id] = (
            f"[{node_id}] "
            f"{_op_name(int(op_ir['op']))}"
        )

    return output


def _print_composite_body(
    op_ir: dict[str, Any],
    *,
    node_id: str,
    prefix: str,
    parent_scope: Scope,
    modulus_chain: ModulusChainInfo | None,
    renderer: Renderer,
) -> ValueInfo | None:
    local_scope = _make_local_composite_scope(
        op_ir,
        parent_scope=parent_scope,
    )

    body_inputs = op_ir.get(
        "body_inputs",
        [],
    )

    if isinstance(body_inputs, list):
        for value_ref in body_inputs:
            _print_input(
                value_ref,
                prefix=prefix,
                scope=local_scope,
                modulus_chain=modulus_chain,
                renderer=renderer,
            )

    children = [
        child
        for child in op_ir.get(
            "child_ops",
            [],
        )
        if isinstance(child, dict)
    ]

    if children:
        print(
            f"{prefix}"
            + renderer.style(
                "│",
                Style.BRIGHT_BLACK,
            )
        )

    for index, child in enumerate(children):
        child_output = _print_op(
            child,
            node_id=f"{node_id}.{index}",
            prefix=prefix,
            is_last=(
                index
                == len(children) - 1
            ),
            scope=local_scope,
            modulus_chain=modulus_chain,
            renderer=renderer,
        )

        if child_output is not None:
            local_scope.values[
                child_output.id
            ] = child_output

    body_output = _value_info(
        op_ir.get("body_output")
    )

    if body_output is not None:
        if children:
            print(
                f"{prefix}"
                + renderer.style(
                    "│",
                    Style.BRIGHT_BLACK,
                )
            )

        _print_output(
            body_output,
            prefix=prefix,
            modulus_chain=modulus_chain,
            renderer=renderer,
        )

    caller_output = _value_info(
        op_ir.get("output")
    )

    output = (
        caller_output
        or body_output
    )

    if output is not None:
        parent_scope.values[
            output.id
        ] = output

        parent_scope.producers[
            output.id
        ] = f"[{node_id}] Composite"

    return output


def _print_op(
    op_ir: dict[str, Any],
    *,
    node_id: str,
    prefix: str,
    is_last: bool,
    scope: Scope,
    modulus_chain: ModulusChainInfo | None,
    renderer: Renderer,
) -> ValueInfo | None:
    connector = (
        "└─"
        if is_last
        else "├─"
    )

    continuation = (
        "   "
        if is_last
        else "│  "
    )

    body_prefix = (
        prefix
        + continuation
    )

    op_code = op_ir.get("op")

    op_name = (
        "Composite"
        if op_code is None
        else _op_name(int(op_code))
    )

    rendered_node = renderer.style(
        f"[{node_id}]",
        Style.BRIGHT_BLACK,
    )

    rendered_name = renderer.style(
        op_name,
        Style.BOLD,
        Style.BRIGHT_WHITE,
    )

    print(
        f"{prefix}{connector} "
        f"{rendered_node} "
        f"{rendered_name}"
    )

    if op_code is None:
        return _print_composite_body(
            op_ir,
            node_id=node_id,
            prefix=body_prefix,
            parent_scope=scope,
            modulus_chain=modulus_chain,
            renderer=renderer,
        )

    return _print_regular_op_body(
        op_ir,
        node_id=node_id,
        prefix=body_prefix,
        scope=scope,
        modulus_chain=modulus_chain,
        renderer=renderer,
    )


# ---------------------------------------------------------------------------
# Initial inputs
# ---------------------------------------------------------------------------


def _initial_hom_scope(
    graph: dict[str, Any],
    *,
    modulus_chain: ModulusChainInfo | None,
) -> Scope:
    scope = Scope()

    primary_input = graph.get(
        "primary_input_name"
    )

    input_shapes = graph.get(
        "input_shape",
        {},
    )

    if not isinstance(
        input_shapes,
        dict,
    ):
        return scope

    sections = graph.get(
        "pipeline_sections",
        {},
    )

    hom_shape = (
        sections.get(
            "input_shape_to_hom_section"
        )
        if isinstance(sections, dict)
        else None
    )

    modulus_chain_json = graph.get(
        "modulus_chain",
        {},
    )

    if isinstance(
        modulus_chain_json,
        dict,
    ):
        raw_active_rows = modulus_chain_json.get(
            "full_active_rows",
            [],
        )

        raw_active_cols = modulus_chain_json.get(
            "full_active_cols",
            [],
        )
    else:
        raw_active_rows = []
        raw_active_cols = []

    full_active_rows = tuple(
        int(row)
        for row in raw_active_rows
    )

    full_active_cols = tuple(
        tuple(bool(x) for x in row)
        for row in raw_active_cols
    )

    if modulus_chain is not None:
        active_by_row = {
            row: cols
            for row, cols in zip(
                full_active_rows,
                full_active_cols,
            )
        }

        initial_rows: list[int] = []
        initial_cols: list[
            tuple[bool, ...]
        ] = []

        for row in modulus_chain.pipeline_rows:
            cols = active_by_row.get(row)

            if cols is None:
                continue

            initial_rows.append(row)
            initial_cols.append(cols)

        active_rows = tuple(
            initial_rows
        )

        active_cols = tuple(
            initial_cols
        )

    else:
        active_rows = full_active_rows
        active_cols = full_active_cols

    graph_n_axis = graph.get(
        "n_axis"
    )

    n_axis = (
        int(graph_n_axis)
        if graph_n_axis is not None
        else None
    )

    for name, external_shape in input_shapes.items():
        if (
            name == primary_input
            and isinstance(
                hom_shape,
                list,
            )
        ):
            shape = tuple(hom_shape)
        else:
            shape = tuple(external_shape)

        scope.values[str(name)] = ValueInfo(
            id=str(name),
            shape=shape,
            n_axis=n_axis,
            active_rows=active_rows,
            active_cols=active_cols,
            scale=None,
        )

    return scope


# ---------------------------------------------------------------------------
# Pipeline rendering
# ---------------------------------------------------------------------------


def _print_pipeline_header(
    graph: dict[str, Any],
    *,
    scope: Scope,
    modulus_chain: ModulusChainInfo | None,
    renderer: Renderer,
) -> None:
    print(
        renderer.style(
            "Lattica Pipeline",
            Style.BOLD,
            Style.BRIGHT_WHITE,
        )
    )

    print(
        renderer.style(
            "════════════════",
            Style.BRIGHT_BLACK,
        )
    )

    primary_input = graph.get(
        "primary_input_name"
    )

    input_shapes = graph.get(
        "input_shape",
        {},
    )

    if not isinstance(
        input_shapes,
        dict,
    ):
        return

    ordered_names: list[str] = []

    if primary_input in input_shapes:
        ordered_names.append(
            primary_input
        )

    ordered_names.extend(
        name
        for name in input_shapes
        if name != primary_input
    )

    for name in ordered_names:
        marker = renderer.style(
            (
                "●"
                if name == primary_input
                else "○"
            ),
            Style.BRIGHT_CYAN,
        )

        line_prefix = f"{marker} "

        value = scope.values.get(
            str(name)
        )

        if value is not None:
            rendered_value, rendered_q = _format_hom_value(
                value,
                modulus_chain=modulus_chain,
                renderer=renderer,
            )
        else:
            rendered_value = (
                renderer.style(
                    str(name),
                    Style.BOLD,
                    Style.BRIGHT_CYAN,
                )
                + "  shape="
                + _format_external_shape(
                    tuple(
                        input_shapes[name]
                    ),
                    renderer=renderer,
                )
            )
            rendered_q = None

        role = (
            "primary"
            if name == primary_input
            else "input"
        )

        print(
            line_prefix
            + rendered_value
            + "  "
            + renderer.style(
                role,
                Style.DIM,
            )
        )

        if rendered_q is not None:
            print(
                " " * _visible_len(line_prefix)
                + rendered_q
            )


def _print_section(
    *,
    section_name: str,
    section: Any,
    scope: Scope,
    modulus_chain: ModulusChainInfo | None,
    renderer: Renderer,
) -> None:
    section_renderer = Renderer(
        color=renderer.color,
        verbose=renderer.verbose,
        hom_metadata=(section_name == "hom"),
    )

    _print_section_header(
        section_name,
        renderer=renderer,
    )

    if not isinstance(
        section,
        dict,
    ):
        print(
            renderer.style(
                (
                    "Unsupported section format: "
                    f"{type(section).__name__}"
                ),
                Style.DIM,
            )
        )
        return

    _print_op(
        section,
        node_id="0",
        prefix="",
        is_last=True,
        scope=scope,
        modulus_chain=modulus_chain,
        renderer=section_renderer,
    )


def print_graph(
    graph: dict[str, Any],
    *,
    color: bool,
    verbose: bool,
) -> None:
    print()

    renderer = Renderer(
        color=color,
        verbose=verbose,
    )

    modulus_chain = (
        _modulus_chain_info(graph)
    )

    scope = _initial_hom_scope(
        graph,
        modulus_chain=modulus_chain,
    )

    sections = graph.get(
        "pipeline_sections",
        {},
    )

    if isinstance(sections, dict):
        _seed_section_inputs(
            sections.get("hom"),
            scope=scope,
        )

    _print_pipeline_header(
        graph,
        scope=scope,
        modulus_chain=modulus_chain,
        renderer=renderer,
    )

    if not isinstance(
        sections,
        dict,
    ):
        print()
        return

    for section_name in (
        "client_pre",
        "hom",
        "client_post",
    ):
        section = sections.get(
            section_name
        )

        if section is None:
            continue

        _seed_section_inputs(
            section,
            scope=scope,
        )

        _print_section(
            section_name=section_name,
            section=section,
            scope=scope,
            modulus_chain=modulus_chain,
            renderer=renderer,
        )

    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize the structure of a compiled "
            "Lattica pipeline from hom_pipeline.json "
            "or a pipeline artifact."
        )
    )

    parser.add_argument(
        "input_path",
        type=Path,
        help=(
            "Path to hom_pipeline.json or a pipeline "
            f"artifact containing {_GRAPH_FILENAME}."
        ),
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help=(
            "Show debugging metadata such as value "
            "producers and data_ref."
        ),
    )

    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI terminal colors.",
    )

    args = parser.parse_args()

    graph = _load_graph_json(
        args.input_path
    )

    print_graph(
        graph,
        color=(
            sys.stdout.isatty()
            and not args.no_color
        ),
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()

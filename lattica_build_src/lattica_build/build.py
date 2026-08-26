"""Build deployable Lattica artifacts.

This module supports both:
- Python API usage via `build(...)`
- CLI usage via `python -m lattica_build.build ...`
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from lattica_build.base_classes.print_graph_structure import (
    _load_graph_json,
    print_graph,
)


@dataclass(frozen=True)
class BuildArtifact:
    path: Path
    init_context_params: dict[str, Any]

    @property
    def size_bytes(self) -> int:
        return self.path.stat().st_size


def build(
    pipeline,
    params,
    out: str | Path,
    *,
    display_graph: bool = False,
) -> BuildArtifact:
    """Build a deployable artifact from a homomorphic pipeline and its params."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    pipeline.save(out, params)

    if display_graph:
        graph = _load_graph_json(out)
        print_graph(
            graph,
            color=True,
            verbose=False,
        )

    return BuildArtifact(
        path=out,
        init_context_params=params.to_dict(),
    )


def build_module(
    module: ModuleType,
    out: str | Path,
    *,
    display_graph: bool = False,
) -> BuildArtifact:
    """Build an artifact from a module defining build_pipeline() and build_params()."""
    build_pipeline = getattr(module, "build_pipeline", None)
    build_params = getattr(module, "build_params", None)

    if not callable(build_pipeline) or not callable(build_params):
        raise ValueError(
            "Pipeline module must define callable "
            "build_pipeline() and build_params()"
        )

    return build(
        build_pipeline(),
        build_params(),
        out,
        display_graph=display_graph,
    )


def build_file(
    pipeline_file: str | Path,
    out: str | Path,
    *,
    display_graph: bool = False,
) -> BuildArtifact:
    """Build an artifact from a standalone Python pipeline file."""
    module = _load_pipeline_module(Path(pipeline_file))

    return build_module(
        module,
        out,
        display_graph=display_graph,
    )


def build_module_name(
    module_name: str,
    out: str | Path,
    *,
    display_graph: bool = False,
) -> BuildArtifact:
    """Build an artifact from an importable pipeline-definition module."""
    module = importlib.import_module(module_name)

    return build_module(
        module,
        out,
        display_graph=display_graph,
    )


def summarize_artifact(
    artifact: BuildArtifact | str | Path,
) -> dict[str, object]:
    """Return human-readable metadata about a built artifact."""
    path = (
        artifact.path
        if isinstance(artifact, BuildArtifact)
        else Path(artifact)
    )

    with zipfile.ZipFile(path, "r") as zf:
        graph_bytes = zf.read("hom_pipeline.json")
        tensor_bytes = zf.read("hom_pipeline.safetensors")
        members = sorted(zf.namelist())

    graph = json.loads(graph_bytes.decode("utf-8"))

    return {
        "artifact": str(path),
        "size_bytes": path.stat().st_size,
        "graph_bytes": len(graph_bytes),
        "tensor_bytes": len(tensor_bytes),
        "members": members,
        "pipeline_sections": list(
            graph["pipeline_sections"].keys()
        ),
    }


def _load_pipeline_module(
    pipeline_file: Path,
) -> ModuleType:
    resolved = pipeline_file.resolve()

    spec = importlib.util.spec_from_file_location(
        f"build_pipeline_{resolved.stem}",
        resolved,
    )

    if spec is None or spec.loader is None:
        raise ValueError(
            f"Cannot import pipeline file: {resolved}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a homomorphic pipeline from a pipeline-definition "
            "module or file."
        )
    )

    parser.add_argument(
        "pipeline_file",
        nargs="?",
        type=Path,
        default=None,
        help=(
            "Path to a Python file defining build_pipeline() and "
            "build_params()."
        ),
    )

    parser.add_argument(
        "--pipeline-module",
        default=None,
        help=(
            "Import path of a module defining build_pipeline() and "
            "build_params()."
        ),
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=Path("quickstart.zip"),
        help="Output artifact path.",
    )

    parser.add_argument(
        "--print_graph",
        action="store_true",
        help="Print the compiled graph structure after build.",
    )

    args = parser.parse_args()

    if (
        args.pipeline_module is not None
        and args.pipeline_file is not None
    ):
        parser.error(
            "Provide either pipeline_file or --pipeline-module, not both"
        )

    if args.pipeline_module is not None:
        artifact = build_module_name(
            args.pipeline_module,
            args.out,
            display_graph=args.print_graph,
        )
    else:
        pipeline_file = (
            args.pipeline_file
            or Path(
                "lattica_build/examples/"
                "advanced/branching.py"
            )
        )

        artifact = build_file(
            pipeline_file,
            args.out,
            display_graph=args.print_graph,
        )

    print(
        json.dumps(
            summarize_artifact(artifact),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

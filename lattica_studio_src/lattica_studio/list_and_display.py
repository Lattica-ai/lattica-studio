"""Small CLI utility to print Lattica Studio resource tables."""

from __future__ import annotations

import argparse
import os

from lattica_studio import LatticaStudio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="List and display resources as formatted tables.",
    )
    parser.add_argument(
        "resource",
        nargs="?",
        choices=("models", "workers", "tokens", "all"),
        default="models",
        help="Resource to display (default: models).",
    )
    parser.add_argument(
        "--license-key",
        default=None,
        help="Lattica account license key. Defaults to LATTICA_LICENSE_KEY.",
    )
    return parser.parse_args()


def resolve_license_key(cli_value: str | None) -> str:
    key = cli_value or os.getenv("LATTICA_LICENSE_KEY", "")
    if not key:
        raise SystemExit(
            "Missing license key. Set LATTICA_LICENSE_KEY or pass --license-key."
        )
    return key


def main() -> None:
    args = parse_args()
    studio = LatticaStudio(resolve_license_key(args.license_key))

    if args.resource in ("models", "all"):
        print("\nModels")
        studio.models.display(studio.models.list())

    if args.resource in ("workers", "all"):
        print("\nWorker sessions")
        studio.workers.display(studio.workers.list_sessions())

    if args.resource in ("tokens", "all"):
        print("\nTokens")
        studio.tokens.display(studio.tokens.list())


if __name__ == "__main__":
    main()


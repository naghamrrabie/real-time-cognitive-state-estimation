"""Command-line placeholder for the cognitive state MVP scaffold."""

from __future__ import annotations

import argparse

from cognitive_state import __version__


def build_parser() -> argparse.ArgumentParser:
    """Build the scaffold CLI parser."""

    parser = argparse.ArgumentParser(
        prog="cognitive-state",
        description="Scaffold CLI for cognitive state estimation.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the scaffold CLI."""

    parser = build_parser()
    parser.parse_args(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

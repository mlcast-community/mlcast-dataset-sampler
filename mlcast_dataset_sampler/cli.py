"""Command-line interface for the MLCast dataset sampler."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from loguru import logger

from . import __version__
from .commands import stats, validate_stats


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="mlcast.sample_dataset",
        description="MLCast dataset sampler - utilities for sampling training data from source datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"mlcast-dataset-sampler {__version__}",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity level (-v for INFO, -vv for DEBUG).",
    )

    subparsers = parser.add_subparsers(
        title="commands",
        dest="command",
        description="Available sampling commands",
    )

    # stats subcommand
    stats_parser = subparsers.add_parser(
        "stats",
        help="Compute per-datacube stats (nan_count, sum, mean, frac_wet) via cumsum windows.",
        description="Scan a Zarr dataset for valid datacube candidates and write per-window stats to parquet.",
    )
    stats.add_arguments(stats_parser)
    stats_parser.set_defaults(func=stats.run)

    # validate-stats subcommand
    validate_stats_parser = subparsers.add_parser(
        "validate-stats",
        help="Validate a stats parquet file against the canonical contract.",
        description="Check column schema, metadata payload, and per-row value invariants of a stats parquet.",
    )
    validate_stats.add_arguments(validate_stats_parser)
    validate_stats_parser.set_defaults(func=validate_stats.run)

    return parser


def configure_logging(verbosity: int) -> None:
    """Configure loguru based on verbosity level."""
    logger.remove()

    if verbosity == 0:
        level = "WARNING"
    elif verbosity == 1:
        level = "INFO"
    else:
        level = "DEBUG"

    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )


@logger.catch
def main(argv: Sequence[str] | None = None) -> int:
    """Main entry point for the CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    configure_logging(args.verbose)

    logger.info(f"mlcast-dataset-sampler {__version__}")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

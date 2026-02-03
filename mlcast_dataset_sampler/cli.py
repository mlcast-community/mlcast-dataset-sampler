"""Command-line interface for the MLCast dataset sampler."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from loguru import logger

from . import __version__
from .commands import filter_nan, sample


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

    # filter-nan subcommand
    filter_nan_parser = subparsers.add_parser(
        "filter-nan",
        help="Filter datacubes based on NaN count.",
        description="Process a Zarr dataset and output valid datacube coordinates with NaN count below threshold.",
    )
    filter_nan.add_arguments(filter_nan_parser)
    filter_nan_parser.set_defaults(func=filter_nan.run)

    # sample subcommand
    sample_parser = subparsers.add_parser(
        "sample",
        help="Importance sampling of valid datacubes.",
        description="Perform importance sampling on filtered datacubes based on rain rate intensity.",
    )
    sample.add_arguments(sample_parser)
    sample_parser.set_defaults(func=sample.run)

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

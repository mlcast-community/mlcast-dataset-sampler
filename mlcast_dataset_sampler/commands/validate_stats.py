"""Validate a stats parquet file against the canonical contract.

Thin CLI wrapper over :func:`stats_spec.validate_stats_parquet`. Checks the
column schema and metadata payload, and (unless ``--no-data-checks``) the
per-row value invariants.
"""

from __future__ import annotations

import argparse

from loguru import logger

from ..stats_spec import validate_stats_parquet


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add validate-stats specific arguments to the parser."""
    parser.add_argument("parquet_path", type=str, help="Path to the stats parquet file.")
    parser.add_argument(
        "--no-data-checks",
        action="store_true",
        help="Only check the schema and metadata (read the footer, not the rows).",
    )


def run(args: argparse.Namespace) -> int:
    """Execute the validate-stats command."""
    report = validate_stats_parquet(args.parquet_path, check_data=not args.no_data_checks)

    for w in report.warnings:
        logger.warning(w)
    for e in report.errors:
        logger.error(e)

    if report.ok:
        logger.success(f"{args.parquet_path}: valid stats parquet")
        return 0
    logger.error(f"{args.parquet_path}: {len(report.errors)} contract violation(s)")
    return 1

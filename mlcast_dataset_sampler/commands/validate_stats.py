"""Validate a stats parquet file against the canonical contract.

Thin CLI wrapper over :func:`stats_spec.validate_stats_parquet`. Checks the
column schema and metadata payload, and (unless ``--no-data-checks``) the
per-row value invariants.
"""

from __future__ import annotations

import argparse

import pyarrow.parquet as pq
from rich.panel import Panel
from rich.table import Table

from ..console import console
from ..stats_spec import read_metadata, validate_stats_parquet


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add validate-stats specific arguments to the parser."""
    parser.add_argument("parquet_path", type=str, help="Path to the stats parquet file.")
    parser.add_argument(
        "--no-data-checks",
        action="store_true",
        help="Only check the schema and metadata (read the footer, not the rows).",
    )


def _summary_grid(path: str) -> Table:
    """Compact grid of the file's key parameters, shown on success."""
    meta = read_metadata(path)
    grid = Table.grid(padding=(0, 2))
    grid.add_column(justify="right", style="bold cyan")
    grid.add_column()
    grid.add_row("Datacube", f"{meta.time_depth} × {meta.width} × {meta.height}   "
                             f"stride {meta.step_t} × {meta.step_x} × {meta.step_y}")
    grid.add_row("Range", f"{meta.start_date:%Y-%m-%d} → {meta.end_date:%Y-%m-%d}")
    grid.add_row("Data kind", f"{meta.data_kind}   (wet > {meta.wet_threshold:g} {meta.units or '?'})")
    grid.add_row("Rows", f"{pq.read_metadata(path).num_rows:,}")
    return grid


def run(args: argparse.Namespace) -> int:
    """Execute the validate-stats command."""
    path = args.parquet_path
    report = validate_stats_parquet(path, check_data=not args.no_data_checks)

    if report.ok and not report.warnings:
        console.print(
            Panel(
                _summary_grid(path),
                title="[bold green]✓ valid stats parquet[/]",
                subtitle=f"[dim]{path}[/]",
                border_style="green",
                expand=False,
            )
        )
        return 0

    issues = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    issues.add_column("")
    issues.add_column("message")
    for w in report.warnings:
        issues.add_row("[yellow]warn[/]", w)
    for e in report.errors:
        issues.add_row("[red]error[/]", e)

    if report.ok:
        title = f"[bold yellow]valid — {len(report.warnings)} warning(s)[/]"
        border = "yellow"
    else:
        title = f"[bold red]invalid — {len(report.errors)} violation(s)[/]"
        border = "red"
    console.print(Panel(issues, title=title, subtitle=f"[dim]{path}[/]", border_style=border, expand=False))

    return 0 if report.ok else 1

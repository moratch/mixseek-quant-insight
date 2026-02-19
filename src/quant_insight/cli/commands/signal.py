"""Signal generation CLI commands."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated

import typer

signal_app = typer.Typer(
    name="signal",
    help="Generate ensemble portfolio signals",
)


@signal_app.command("generate")
def generate_signal(
    workspace: Annotated[
        str,
        typer.Option(
            "--workspace",
            "-w",
            envvar="MIXSEEK_WORKSPACE",
            help="Workspace path (or set MIXSEEK_WORKSPACE env var)",
        ),
    ],
    date: Annotated[
        str | None,
        typer.Option("--date", "-d", help="Target date (YYYY-MM-DD). Default: latest available"),
    ] = None,
    top_n: Annotated[
        int,
        typer.Option("--top-n", "-n", help="Number of top long/short positions to output"),
    ] = 50,
    output: Annotated[
        str | None,
        typer.Option("--output", "-o", help="Output JSON path (default: stdout)"),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose logging"),
    ] = False,
) -> None:
    """Generate ensemble portfolio signals for a given date.

    Runs all 5 validated strategies, applies quantile thresholds,
    and outputs weighted ensemble positions.

    Examples:
        quant-insight signal generate --date 2025-12-30
        quant-insight signal generate --date 2025-12-30 --top-n 20 -o signals.json
        quant-insight signal generate  # latest date
    """
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    from quant_insight.signal.generator import EnsembleSignalGenerator

    ws = Path(workspace)
    typer.echo("Loading data...")
    gen = EnsembleSignalGenerator(workspace=ws)
    gen.load_data()

    if date is None:
        # Use the latest available date
        typer.echo("Generating signals for latest date...")
    else:
        typer.echo(f"Generating signals for {date}...")

    outputs = gen.generate(date=date, top_n=top_n)

    if not outputs:
        typer.echo("No signals generated (date not found in data).", err=True)
        raise typer.Exit(1)

    # If no date specified, take only the last one
    if date is None:
        outputs = [outputs[-1]]

    # Format output
    result = []
    for out in outputs:
        entry = {
            "date": out.date,
            "n_long": out.n_long,
            "n_short": out.n_short,
            "n_neutral": out.n_neutral,
            "n_total": out.n_total,
            "positions": out.positions,
        }
        result.append(entry)

    json_str = json.dumps(result, ensure_ascii=False, indent=2, default=str)

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json_str, encoding="utf-8")
        typer.echo(f"Saved to {out_path}")
    else:
        typer.echo(json_str)

    # Summary
    for out in outputs:
        typer.echo(
            f"\n{out.date}: {out.n_long} long, {out.n_short} short, {out.n_neutral} neutral (total {out.n_total})",
            err=True,
        )

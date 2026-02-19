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


@signal_app.command("cost-analysis")
def cost_analysis(
    workspace: Annotated[
        str,
        typer.Option(
            "--workspace",
            "-w",
            envvar="MIXSEEK_WORKSPACE",
            help="Workspace path (or set MIXSEEK_WORKSPACE env var)",
        ),
    ],
    output: Annotated[
        str | None,
        typer.Option("--output", "-o", help="Output JSON path (default: stdout)"),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose logging"),
    ] = False,
) -> None:
    """Analyze transaction cost impact on ensemble Sharpe ratio.

    Calculates daily turnover from position changes and estimates
    net Sharpe ratios at round-trip cost levels of 10, 20, 30, 50 bps.

    Examples:
        quant-insight signal cost-analysis
        quant-insight signal cost-analysis -o cost_report.json -v
    """
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    import polars as pl

    from quant_insight.analysis.cost import (
        analyze_ensemble_costs,
        calculate_turnover,
    )
    from quant_insight.signal.generator import EnsembleSignalGenerator

    ws = Path(workspace)
    typer.echo("Loading data and generating signals across all dates...")

    gen = EnsembleSignalGenerator(workspace=ws)
    gen.load_data()

    # Generate ensemble signals for all dates (needed for turnover)
    all_outputs = gen.generate()
    if not all_outputs:
        typer.echo("No signals generated.", err=True)
        raise typer.Exit(1)

    typer.echo(f"Generated signals for {len(all_outputs)} dates")

    # Reconstruct full ensemble positions DataFrame
    # Re-run internal logic to get combined positions
    all_positions: dict[str, pl.DataFrame] = {}
    for spec in gen.strategies:
        all_positions[spec.name] = gen._generate_strategy_signals(spec)

    # Build combined ensemble signal per (datetime, symbol)
    all_pairs: list[pl.DataFrame] = []
    for df in all_positions.values():
        all_pairs.append(df.select(["datetime", "symbol"]))
    combined = pl.concat(all_pairs).unique(subset=["datetime", "symbol"])
    combined = combined.with_columns(pl.lit(0.0).alias("ensemble_signal"))
    for spec in gen.strategies:
        strat_df = all_positions[spec.name].select(
            ["datetime", "symbol", pl.col("position").alias(f"pos_{spec.name}")]
        )
        combined = combined.join(strat_df, on=["datetime", "symbol"], how="left")
        combined = combined.with_columns(
            (pl.col("ensemble_signal") + pl.col(f"pos_{spec.name}").fill_null(0) * spec.weight).alias(
                "ensemble_signal"
            )
        )
        combined = combined.drop(f"pos_{spec.name}")

    # Calculate turnover
    turnover = calculate_turnover(combined)
    typer.echo(
        f"Turnover: mean={turnover.mean_daily_turnover:.3f}, "
        f"median={turnover.median_daily_turnover:.3f}, "
        f"max={turnover.max_daily_turnover:.3f}"
    )

    # Calculate daily portfolio returns (equal-weighted long/short)
    # Load OHLCV for forward returns
    ohlcv = gen._ohlcv
    assert ohlcv is not None

    # Forward return: next-day close / today close - 1
    ohlcv_ret = ohlcv.sort(["symbol", "datetime"]).with_columns(
        (pl.col("close").shift(-1).over("symbol") / pl.col("close") - 1).alias("fwd_return")
    )

    # Join ensemble signal with forward returns
    port = combined.join(ohlcv_ret.select(["datetime", "symbol", "fwd_return"]), on=["datetime", "symbol"], how="left")

    # Daily portfolio return: mean of (position * forward_return) per date
    daily_ret = (
        port.with_columns((pl.col("ensemble_signal") * pl.col("fwd_return").fill_null(0)).alias("weighted_ret"))
        .group_by("datetime")
        .agg(pl.col("weighted_ret").mean().alias("port_return"))
        .sort("datetime")
    )

    # Run cost analysis
    result = analyze_ensemble_costs(
        ensemble_positions=combined,
        daily_returns=daily_ret["port_return"],
    )

    # Format output
    report: dict[str, object] = {
        "turnover": {
            "mean_daily": round(result.turnover.mean_daily_turnover, 4),
            "median_daily": round(result.turnover.median_daily_turnover, 4),
            "max_daily": round(result.turnover.max_daily_turnover, 4),
            "n_dates": result.turnover.n_dates,
        },
        "scenarios": [],
    }

    typer.echo("\n--- Cost Impact Analysis ---")
    typer.echo(f"{'Cost (bps)':>12} {'Gross SR':>10} {'Net SR':>10} {'Degrad%':>10} {'Ann.Cost%':>10}")
    typer.echo("-" * 56)

    for s in result.scenarios:
        typer.echo(
            f"{s.round_trip_bps:>10.0f}bp {s.gross_sharpe:>10.2f} {s.net_sharpe:>10.2f} "
            f"{s.sharpe_degradation_pct:>9.1f}% {s.annual_cost_pct:>9.2f}%"
        )
        scenario_dict: dict[str, object] = {
            "round_trip_bps": s.round_trip_bps,
            "gross_sharpe": round(s.gross_sharpe, 4),
            "net_sharpe": round(s.net_sharpe, 4),
            "sharpe_degradation_pct": round(s.sharpe_degradation_pct, 2),
            "annual_cost_pct": round(s.annual_cost_pct, 4),
        }
        scenarios_list = report["scenarios"]
        assert isinstance(scenarios_list, list)
        scenarios_list.append(scenario_dict)

    json_str = json.dumps(report, ensure_ascii=False, indent=2, default=str)

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json_str, encoding="utf-8")
        typer.echo(f"\nSaved to {out_path}")
    else:
        typer.echo(f"\n{json_str}")

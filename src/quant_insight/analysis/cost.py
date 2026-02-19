"""Transaction cost analysis for ensemble signals.

Calculates turnover, cost drag, and net Sharpe ratios across
multiple round-trip cost scenarios.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import polars as pl

logger = logging.getLogger(__name__)


@dataclass
class CostScenario:
    """Result for a single cost scenario."""

    round_trip_bps: float  # round-trip cost in basis points
    daily_cost_drag: float  # average daily cost (return units)
    gross_sharpe: float
    net_sharpe: float
    sharpe_degradation_pct: float  # percentage reduction
    annual_cost_pct: float  # annualized total cost as % of capital


@dataclass
class TurnoverStats:
    """Turnover statistics for an ensemble signal."""

    mean_daily_turnover: float  # fraction of positions changing per day
    median_daily_turnover: float
    max_daily_turnover: float
    n_dates: int


@dataclass
class CostAnalysisResult:
    """Complete cost analysis result."""

    turnover: TurnoverStats
    scenarios: list[CostScenario] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def calculate_turnover(positions: pl.DataFrame) -> TurnoverStats:
    """Calculate daily turnover from position data.

    Turnover = fraction of symbols whose position (long/short/neutral)
    changes from one day to the next.

    Args:
        positions: DataFrame with columns [datetime, symbol, position].
            position should be {-1, 0, 1} or continuous signal.

    Returns:
        TurnoverStats with daily turnover metrics.
    """
    dates = positions.select("datetime").unique().sort("datetime").to_series()
    if len(dates) < 2:
        return TurnoverStats(
            mean_daily_turnover=0.0,
            median_daily_turnover=0.0,
            max_daily_turnover=0.0,
            n_dates=len(dates),
        )

    # Discretize position: > 0 → 1, < 0 → -1, else 0
    pos_col = "position" if "position" in positions.columns else "ensemble_signal"
    df = positions.select(["datetime", "symbol", pos_col]).with_columns(
        pl.when(pl.col(pos_col) > 0).then(1).when(pl.col(pos_col) < 0).then(-1).otherwise(0).alias("discrete_pos")
    )

    # Sort for consistent join
    df = df.sort(["symbol", "datetime"])

    # Self-join: compare each date with the previous date
    # Create lagged position per symbol
    df = df.with_columns(pl.col("discrete_pos").shift(1).over("symbol").alias("prev_pos"))

    # Filter to rows that have a previous date
    df_valid = df.filter(pl.col("prev_pos").is_not_null())

    # Per date: fraction of symbols with changed position
    turnover_per_date = (
        df_valid.group_by("datetime")
        .agg(
            (pl.col("discrete_pos") != pl.col("prev_pos")).mean().alias("turnover"),
        )
        .sort("datetime")
    )

    turnover_series = turnover_per_date["turnover"]

    return TurnoverStats(
        mean_daily_turnover=float(turnover_series.mean()),  # type: ignore[arg-type]
        median_daily_turnover=float(turnover_series.median()),  # type: ignore[arg-type]
        max_daily_turnover=float(turnover_series.max()),  # type: ignore[arg-type]
        n_dates=len(dates),
    )


def calculate_cost_scenarios(
    daily_returns: pl.Series,
    turnover_rate: float,
    cost_levels_bps: list[float] | None = None,
) -> list[CostScenario]:
    """Calculate net Sharpe at various cost levels.

    Args:
        daily_returns: Series of daily portfolio returns.
        turnover_rate: Average daily turnover fraction (0-1).
        cost_levels_bps: Round-trip cost levels in basis points.
            Default: [10, 20, 30, 50] (0.1% to 0.5%).

    Returns:
        List of CostScenario for each cost level.
    """
    if cost_levels_bps is None:
        cost_levels_bps = [10.0, 20.0, 30.0, 50.0]

    mean_ret = float(daily_returns.mean())  # type: ignore[arg-type]
    std_ret = float(daily_returns.std())  # type: ignore[arg-type]

    if std_ret == 0:
        gross_sharpe = 0.0
    else:
        gross_sharpe = (mean_ret / std_ret) * (252**0.5)

    scenarios: list[CostScenario] = []
    for bps in cost_levels_bps:
        # Daily cost = round_trip_cost * turnover_rate
        # Only the fraction that actually trades incurs cost
        round_trip_frac = bps / 10000.0
        daily_cost = round_trip_frac * turnover_rate

        net_mean = mean_ret - daily_cost
        if std_ret == 0:
            net_sharpe = 0.0
        else:
            net_sharpe = (net_mean / std_ret) * (252**0.5)

        degradation = 0.0
        if gross_sharpe != 0:
            degradation = (1 - net_sharpe / gross_sharpe) * 100

        annual_cost = daily_cost * 252 * 100  # as percentage

        scenarios.append(
            CostScenario(
                round_trip_bps=bps,
                daily_cost_drag=daily_cost,
                gross_sharpe=gross_sharpe,
                net_sharpe=net_sharpe,
                sharpe_degradation_pct=degradation,
                annual_cost_pct=annual_cost,
            )
        )

    return scenarios


def analyze_ensemble_costs(
    ensemble_positions: pl.DataFrame,
    daily_returns: pl.Series,
    cost_levels_bps: list[float] | None = None,
) -> CostAnalysisResult:
    """Run full cost analysis on ensemble positions.

    Args:
        ensemble_positions: DataFrame with [datetime, symbol, ensemble_signal].
        daily_returns: Series of daily portfolio returns.
        cost_levels_bps: Round-trip cost levels in basis points.

    Returns:
        Complete CostAnalysisResult.
    """
    turnover = calculate_turnover(ensemble_positions)
    logger.info(
        "Turnover: mean=%.3f, median=%.3f, max=%.3f (%d dates)",
        turnover.mean_daily_turnover,
        turnover.median_daily_turnover,
        turnover.max_daily_turnover,
        turnover.n_dates,
    )

    scenarios = calculate_cost_scenarios(
        daily_returns=daily_returns,
        turnover_rate=turnover.mean_daily_turnover,
        cost_levels_bps=cost_levels_bps,
    )

    return CostAnalysisResult(
        turnover=turnover,
        scenarios=scenarios,
        metadata={
            "n_dates": turnover.n_dates,
            "mean_turnover": turnover.mean_daily_turnover,
        },
    )

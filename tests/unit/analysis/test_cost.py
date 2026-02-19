"""Tests for transaction cost analysis."""

from __future__ import annotations

import polars as pl
import pytest

from quant_insight.analysis.cost import (
    CostAnalysisResult,
    TurnoverStats,
    analyze_ensemble_costs,
    calculate_cost_scenarios,
    calculate_turnover,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def position_data() -> pl.DataFrame:
    """Position data with known turnover pattern."""
    # 3 dates, 4 symbols
    # Day 1→2: SYM0 changes (1→-1), SYM1 same (1→1), SYM2 changes (0→1), SYM3 same (-1→-1)
    # Turnover day 2 = 2/4 = 0.5
    # Day 2→3: SYM0 same (-1→-1), SYM1 changes (1→0), SYM2 same (1→1), SYM3 changes (-1→1)
    # Turnover day 3 = 2/4 = 0.5
    return pl.DataFrame(
        {
            "datetime": [
                "2025-01-01",
                "2025-01-01",
                "2025-01-01",
                "2025-01-01",
                "2025-01-02",
                "2025-01-02",
                "2025-01-02",
                "2025-01-02",
                "2025-01-03",
                "2025-01-03",
                "2025-01-03",
                "2025-01-03",
            ],
            "symbol": ["SYM0", "SYM1", "SYM2", "SYM3"] * 3,
            "ensemble_signal": [
                1.0,
                1.0,
                0.0,
                -1.0,  # Day 1
                -1.0,
                1.0,
                1.0,
                -1.0,  # Day 2
                -1.0,
                0.0,
                1.0,
                1.0,  # Day 3
            ],
        }
    ).with_columns(pl.col("datetime").str.strptime(pl.Datetime, "%Y-%m-%d"))


@pytest.fixture()
def daily_returns() -> pl.Series:
    """Synthetic daily returns."""
    import random

    random.seed(42)
    returns = [0.001 * (i % 3 - 1) + 0.0005 for i in range(100)]
    return pl.Series("port_return", returns)


# ---------------------------------------------------------------------------
# Tests: TurnoverStats
# ---------------------------------------------------------------------------


class TestTurnoverStats:
    def test_fields(self) -> None:
        ts = TurnoverStats(mean_daily_turnover=0.3, median_daily_turnover=0.25, max_daily_turnover=0.8, n_dates=100)
        assert ts.mean_daily_turnover == 0.3
        assert ts.n_dates == 100


# ---------------------------------------------------------------------------
# Tests: calculate_turnover
# ---------------------------------------------------------------------------


class TestCalculateTurnover:
    def test_known_turnover(self, position_data: pl.DataFrame) -> None:
        stats = calculate_turnover(position_data)
        # Both days have 2/4 = 0.5 turnover
        assert abs(stats.mean_daily_turnover - 0.5) < 1e-6
        assert abs(stats.median_daily_turnover - 0.5) < 1e-6
        assert stats.n_dates == 3

    def test_zero_turnover(self) -> None:
        # Same positions every day → 0 turnover
        df = pl.DataFrame(
            {
                "datetime": ["2025-01-01", "2025-01-02", "2025-01-03"] * 2,
                "symbol": ["A", "A", "A", "B", "B", "B"],
                "ensemble_signal": [1.0, 1.0, 1.0, -1.0, -1.0, -1.0],
            }
        ).with_columns(pl.col("datetime").str.strptime(pl.Datetime, "%Y-%m-%d"))
        stats = calculate_turnover(df)
        assert stats.mean_daily_turnover == 0.0

    def test_full_turnover(self) -> None:
        # All positions flip every day
        df = pl.DataFrame(
            {
                "datetime": ["2025-01-01", "2025-01-02"] * 2,
                "symbol": ["A", "A", "B", "B"],
                "ensemble_signal": [1.0, -1.0, -1.0, 1.0],
            }
        ).with_columns(pl.col("datetime").str.strptime(pl.Datetime, "%Y-%m-%d"))
        stats = calculate_turnover(df)
        assert abs(stats.mean_daily_turnover - 1.0) < 1e-6

    def test_single_date(self) -> None:
        df = pl.DataFrame(
            {
                "datetime": ["2025-01-01"],
                "symbol": ["A"],
                "ensemble_signal": [1.0],
            }
        ).with_columns(pl.col("datetime").str.strptime(pl.Datetime, "%Y-%m-%d"))
        stats = calculate_turnover(df)
        assert stats.mean_daily_turnover == 0.0
        assert stats.n_dates == 1


# ---------------------------------------------------------------------------
# Tests: calculate_cost_scenarios
# ---------------------------------------------------------------------------


class TestCalculateCostScenarios:
    def test_default_levels(self, daily_returns: pl.Series) -> None:
        scenarios = calculate_cost_scenarios(daily_returns, turnover_rate=0.3)
        assert len(scenarios) == 4  # 10, 20, 30, 50 bps
        assert scenarios[0].round_trip_bps == 10.0

    def test_custom_levels(self, daily_returns: pl.Series) -> None:
        scenarios = calculate_cost_scenarios(daily_returns, turnover_rate=0.3, cost_levels_bps=[5.0, 100.0])
        assert len(scenarios) == 2

    def test_net_sharpe_decreases(self, daily_returns: pl.Series) -> None:
        scenarios = calculate_cost_scenarios(daily_returns, turnover_rate=0.5)
        # Net Sharpe should decrease as cost increases
        for i in range(len(scenarios) - 1):
            assert scenarios[i].net_sharpe >= scenarios[i + 1].net_sharpe

    def test_zero_cost_equals_gross(self, daily_returns: pl.Series) -> None:
        scenarios = calculate_cost_scenarios(daily_returns, turnover_rate=0.5, cost_levels_bps=[0.0])
        assert abs(scenarios[0].net_sharpe - scenarios[0].gross_sharpe) < 1e-6

    def test_zero_turnover_no_cost(self, daily_returns: pl.Series) -> None:
        # If no turnover, cost should not affect Sharpe
        scenarios = calculate_cost_scenarios(daily_returns, turnover_rate=0.0, cost_levels_bps=[10.0, 50.0])
        for s in scenarios:
            assert abs(s.net_sharpe - s.gross_sharpe) < 1e-6
            assert s.annual_cost_pct == 0.0


# ---------------------------------------------------------------------------
# Tests: analyze_ensemble_costs
# ---------------------------------------------------------------------------


class TestAnalyzeEnsembleCosts:
    def test_full_analysis(self, position_data: pl.DataFrame, daily_returns: pl.Series) -> None:
        result = analyze_ensemble_costs(position_data, daily_returns)
        assert isinstance(result, CostAnalysisResult)
        assert result.turnover.n_dates == 3
        assert len(result.scenarios) == 4

    def test_result_metadata(self, position_data: pl.DataFrame, daily_returns: pl.Series) -> None:
        result = analyze_ensemble_costs(position_data, daily_returns)
        assert "n_dates" in result.metadata
        assert "mean_turnover" in result.metadata

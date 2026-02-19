"""Tests for the production signal pipeline."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from quant_insight.pipeline.production import (
    DailySignal,
    LiquidityStats,
    Position,
    ProductionConfig,
    ProductionPipeline,
    ProductionResult,
    _daily_signal_to_dict,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    """Create a minimal workspace with test data."""
    raw_dir = tmp_path / "data" / "inputs" / "raw"
    raw_dir.mkdir(parents=True)

    # OHLCV data (5 dates × 20 symbols)
    # First 10 symbols have high volume (liquid), last 10 have low volume (illiquid)
    dates = ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"]
    symbols = [f"SYM{i:04d}" for i in range(20)]
    rows = []
    for d in dates:
        for i, s in enumerate(symbols):
            # Liquid: volume=500000 * close~100 = 50M+ JPY turnover
            # Illiquid: volume=100 * close~100 = 10K JPY turnover
            vol = 500_000.0 if i < 10 else 100.0
            rows.append(
                {
                    "datetime": d,
                    "symbol": s,
                    "open": 100.0 + i,
                    "high": 105.0 + i,
                    "low": 95.0 + i,
                    "close": 102.0 + i * (1 if d != "2025-01-03" else -1),
                    "volume": vol,
                }
            )

    ohlcv = pl.DataFrame(rows).with_columns(
        pl.col("datetime").str.strptime(pl.Datetime, "%Y-%m-%d")
    )
    ohlcv.write_parquet(raw_dir / "ohlcv.parquet")

    return tmp_path


@pytest.fixture()
def config(workspace: Path) -> ProductionConfig:
    """Create a production config with test workspace (no liquidity filter)."""
    return ProductionConfig(
        workspace=workspace,
        strategy_name="I5_cp_gap3d",
        top_n=5,
        min_avg_turnover_yen=0,  # disable for basic tests
    )


@pytest.fixture()
def pipeline(config: ProductionConfig) -> ProductionPipeline:
    """Create a production pipeline."""
    return ProductionPipeline(config)


# ---------------------------------------------------------------------------
# Tests: ProductionConfig
# ---------------------------------------------------------------------------


class TestProductionConfig:
    def test_defaults(self, workspace: Path) -> None:
        config = ProductionConfig(workspace=workspace)
        assert config.strategy_name == "I5_cp_gap3d"
        assert config.long_quantile == 0.85
        assert config.short_quantile == 0.15
        assert config.top_n == 50
        assert config.holding_period_days == 10

    def test_custom_config(self, workspace: Path) -> None:
        config = ProductionConfig(
            workspace=workspace,
            strategy_name="I5_cp_gap3d",
            long_quantile=0.9,
            short_quantile=0.1,
            top_n=30,
        )
        assert config.long_quantile == 0.9
        assert config.top_n == 30


# ---------------------------------------------------------------------------
# Tests: ProductionPipeline
# ---------------------------------------------------------------------------


class TestProductionPipeline:
    def test_run_latest_date(self, pipeline: ProductionPipeline) -> None:
        result = pipeline.run()
        assert isinstance(result, ProductionResult)
        assert len(result.signals) == 1
        assert result.signals[0].strategy == "I5_cp_gap3d"
        assert result.signals[0].n_universe > 0

    def test_run_specific_date(self, pipeline: ProductionPipeline) -> None:
        result = pipeline.run(date="2025-01-02")
        assert len(result.signals) == 1
        assert "2025-01-02" in result.signals[0].date

    def test_run_nonexistent_date(self, pipeline: ProductionPipeline) -> None:
        result = pipeline.run(date="2099-01-01")
        assert len(result.signals) == 0

    def test_run_has_positions(self, pipeline: ProductionPipeline) -> None:
        result = pipeline.run()
        sig = result.signals[0]
        assert sig.n_long >= 0
        assert sig.n_short >= 0
        # With 20 symbols and 85/15 quantile, we should have some longs and shorts
        assert sig.n_long + sig.n_short > 0

    def test_positions_have_correct_direction(self, pipeline: ProductionPipeline) -> None:
        result = pipeline.run()
        sig = result.signals[0]
        for p in sig.long_positions:
            assert p.direction == "long"
        for p in sig.short_positions:
            assert p.direction == "short"

    def test_top_n_limits_output(self, pipeline: ProductionPipeline) -> None:
        result = pipeline.run()
        sig = result.signals[0]
        assert len(sig.long_positions) <= pipeline.config.top_n
        assert len(sig.short_positions) <= pipeline.config.top_n

    def test_result_metadata(self, pipeline: ProductionPipeline) -> None:
        result = pipeline.run()
        assert result.generated_at  # not empty
        assert result.data_range  # not empty
        assert result.config["strategy"] == "I5_cp_gap3d"
        assert result.config["long_quantile"] == 0.85
        assert result.config["holding_period_days"] == 10

    def test_invalid_strategy_raises(self, workspace: Path) -> None:
        config = ProductionConfig(workspace=workspace, strategy_name="NONEXISTENT")
        pipeline = ProductionPipeline(config)
        with pytest.raises(ValueError, match="not found in validated registry"):
            pipeline.run()


# ---------------------------------------------------------------------------
# Tests: Save result
# ---------------------------------------------------------------------------


class TestSaveResult:
    def test_save_creates_file(self, pipeline: ProductionPipeline, tmp_path: Path) -> None:
        result = pipeline.run()
        out_dir = tmp_path / "output"
        saved = pipeline.save_result(result, output_dir=out_dir)
        assert saved.exists()
        assert saved.suffix == ".json"
        assert "I5_cp_gap3d" in saved.name

    def test_save_default_dir(self, pipeline: ProductionPipeline) -> None:
        result = pipeline.run()
        saved = pipeline.save_result(result)
        assert saved.exists()
        assert "signals" in str(saved.parent)

    def test_save_json_parseable(self, pipeline: ProductionPipeline, tmp_path: Path) -> None:
        import json

        result = pipeline.run()
        saved = pipeline.save_result(result, output_dir=tmp_path)
        data = json.loads(saved.read_text(encoding="utf-8"))
        assert "signals" in data
        assert "config" in data
        assert "generated_at" in data


# ---------------------------------------------------------------------------
# Tests: Helper functions
# ---------------------------------------------------------------------------


class TestLiquidityFilter:
    def test_filter_reduces_universe(self, workspace: Path) -> None:
        config = ProductionConfig(
            workspace=workspace,
            strategy_name="I5_cp_gap3d",
            top_n=5,
            min_avg_turnover_yen=1e6,  # 1M JPY - should exclude low-volume symbols
        )
        pipeline = ProductionPipeline(config)
        result = pipeline.run()
        assert len(result.signals) == 1
        # With filter, universe should be smaller than 20
        assert result.signals[0].n_universe < 20
        assert result.signals[0].n_universe == 10  # only liquid symbols pass

    def test_no_filter_keeps_all(self, workspace: Path) -> None:
        config = ProductionConfig(
            workspace=workspace,
            strategy_name="I5_cp_gap3d",
            top_n=5,
            min_avg_turnover_yen=0,  # disabled
        )
        pipeline = ProductionPipeline(config)
        result = pipeline.run()
        assert result.signals[0].n_universe == 20

    def test_liquidity_stats_in_result(self, workspace: Path) -> None:
        config = ProductionConfig(
            workspace=workspace,
            strategy_name="I5_cp_gap3d",
            top_n=5,
            min_avg_turnover_yen=1e6,
        )
        pipeline = ProductionPipeline(config)
        result = pipeline.run()
        liq = result.config.get("liquidity")
        assert liq is not None
        assert liq["total_symbols"] == 20
        assert liq["filtered_symbols"] == 10
        assert liq["excluded_symbols"] == 10

    def test_high_threshold_filters_all(self, workspace: Path) -> None:
        config = ProductionConfig(
            workspace=workspace,
            strategy_name="I5_cp_gap3d",
            top_n=5,
            min_avg_turnover_yen=1e15,  # impossibly high
        )
        pipeline = ProductionPipeline(config)
        result = pipeline.run()
        assert len(result.signals) == 0  # no data after filter


class TestHelpers:
    def test_daily_signal_to_dict(self) -> None:
        sig = DailySignal(
            date="2025-01-01",
            strategy="test",
            n_long=3,
            n_short=2,
            n_universe=100,
            long_positions=[
                Position(symbol="A", direction="long", signal=0.95, pct_rank=0.98),
            ],
            short_positions=[
                Position(symbol="B", direction="short", signal=0.05, pct_rank=0.02),
            ],
        )
        d = _daily_signal_to_dict(sig)
        assert d["date"] == "2025-01-01"
        assert d["n_long"] == 3
        assert len(d["long_positions"]) == 1
        assert d["long_positions"][0]["symbol"] == "A"
        assert d["short_positions"][0]["direction"] == "short"

    def test_position_dataclass(self) -> None:
        p = Position(symbol="X", direction="long", signal=0.8, pct_rank=0.95)
        assert p.symbol == "X"
        assert p.direction == "long"

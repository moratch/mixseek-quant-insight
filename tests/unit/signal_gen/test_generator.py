"""Tests for signal generator."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from quant_insight.signal.generator import (
    DEFAULT_STRATEGIES,
    DEFAULT_WEIGHTS,
    EnsembleSignalGenerator,
    SignalOutput,
    StrategySpec,
    load_weights_from_json,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    """Create a minimal workspace with test data."""
    raw_dir = tmp_path / "data" / "inputs" / "raw"
    raw_dir.mkdir(parents=True)

    # OHLCV data (3 dates × 10 symbols)
    dates = ["2025-01-01", "2025-01-02", "2025-01-03"]
    symbols = [f"SYM{i:04d}" for i in range(10)]
    rows = []
    for d in dates:
        for i, s in enumerate(symbols):
            rows.append(
                {
                    "datetime": d,
                    "symbol": s,
                    "open": 100.0 + i,
                    "high": 105.0 + i,
                    "low": 95.0 + i,
                    "close": 102.0 + i * (1 if d != "2025-01-03" else -1),
                    "volume": 1000.0 * (i + 1),
                }
            )

    ohlcv = pl.DataFrame(rows).with_columns(pl.col("datetime").str.strptime(pl.Datetime, "%Y-%m-%d"))
    ohlcv.write_parquet(raw_dir / "ohlcv.parquet")

    # Indicators data (matching dates × symbols)
    ind_rows = []
    for d in dates:
        for i, s in enumerate(symbols):
            ind_rows.append(
                {
                    "datetime": d,
                    "symbol": s,
                    "INTRADAY_VOL": 0.1 * (i + 1),
                    "RETURN_2D": -0.05 + 0.01 * i,
                    "STOCH_K_5": 20.0 + 5.0 * i,
                }
            )
    indicators = pl.DataFrame(ind_rows).with_columns(pl.col("datetime").str.strptime(pl.Datetime, "%Y-%m-%d"))
    indicators.write_parquet(raw_dir / "indicators.parquet")

    return tmp_path


@pytest.fixture()
def simple_strategy() -> StrategySpec:
    """A simple strategy that uses OHLCV close price as signal."""
    code = """```python
import polars as pl
def generate_signal(ohlcv, additional_data):
    df = ohlcv.select(["datetime", "symbol", "close"])
    df = df.with_columns(
        pl.col("close").rank().over("datetime").alias("signal")
    )
    return df.select(["datetime", "symbol", "signal"])
```"""
    return StrategySpec(name="test_close", code=code, weight=1.0, long_quantile=0.8, short_quantile=0.2)


# ---------------------------------------------------------------------------
# Tests: StrategySpec / SignalOutput
# ---------------------------------------------------------------------------


class TestStrategySpec:
    def test_defaults(self) -> None:
        spec = StrategySpec(name="test", code="```python\n```", weight=0.5)
        assert spec.long_quantile == 0.9
        assert spec.short_quantile == 0.1

    def test_custom_quantiles(self) -> None:
        spec = StrategySpec(name="test", code="", weight=0.3, long_quantile=0.85, short_quantile=0.15)
        assert spec.long_quantile == 0.85
        assert spec.short_quantile == 0.15


class TestSignalOutput:
    def test_fields(self) -> None:
        out = SignalOutput(date="2025-01-01", n_long=10, n_short=5, n_neutral=85, n_total=100)
        assert out.positions == []
        assert out.n_total == 100


# ---------------------------------------------------------------------------
# Tests: DEFAULT_WEIGHTS / DEFAULT_STRATEGIES
# ---------------------------------------------------------------------------


class TestDefaults:
    def test_weights_sum_to_one(self) -> None:
        total = sum(DEFAULT_WEIGHTS.values())
        assert abs(total - 1.0) < 1e-6, f"Weights sum to {total}, expected 1.0"

    def test_five_strategies(self) -> None:
        assert len(DEFAULT_STRATEGIES) == 5

    def test_strategy_names_match_weights(self) -> None:
        names = {s.name for s in DEFAULT_STRATEGIES}
        assert names == set(DEFAULT_WEIGHTS.keys())


# ---------------------------------------------------------------------------
# Tests: EnsembleSignalGenerator
# ---------------------------------------------------------------------------


class TestEnsembleSignalGenerator:
    def test_load_data(self, workspace: Path) -> None:
        gen = EnsembleSignalGenerator(workspace=workspace, strategies=[])
        gen.load_data()
        assert gen._ohlcv is not None
        assert "indicators" in gen._additional_data  # type: ignore[operator]

    def test_load_data_no_indicators(self, tmp_path: Path) -> None:
        raw_dir = tmp_path / "data" / "inputs" / "raw"
        raw_dir.mkdir(parents=True)
        ohlcv = pl.DataFrame(
            {
                "datetime": ["2025-01-01"],
                "symbol": ["SYM0001"],
                "open": [100.0],
                "high": [105.0],
                "low": [95.0],
                "close": [102.0],
                "volume": [1000.0],
            }
        ).with_columns(pl.col("datetime").str.strptime(pl.Datetime, "%Y-%m-%d"))
        ohlcv.write_parquet(raw_dir / "ohlcv.parquet")

        gen = EnsembleSignalGenerator(workspace=tmp_path, strategies=[])
        gen.load_data()
        assert "indicators" not in gen._additional_data  # type: ignore[operator]

    def test_generate_single_strategy(self, workspace: Path, simple_strategy: StrategySpec) -> None:
        gen = EnsembleSignalGenerator(workspace=workspace, strategies=[simple_strategy])
        gen.load_data()
        outputs = gen.generate(date="2025-01-02")
        assert len(outputs) == 1
        out = outputs[0]
        assert "2025-01-02" in out.date
        assert out.n_total == 10
        assert out.n_long + out.n_short + out.n_neutral == out.n_total

    def test_generate_all_dates(self, workspace: Path, simple_strategy: StrategySpec) -> None:
        gen = EnsembleSignalGenerator(workspace=workspace, strategies=[simple_strategy])
        gen.load_data()
        outputs = gen.generate()
        assert len(outputs) == 3  # 3 dates

    def test_generate_invalid_date(self, workspace: Path, simple_strategy: StrategySpec) -> None:
        gen = EnsembleSignalGenerator(workspace=workspace, strategies=[simple_strategy])
        gen.load_data()
        outputs = gen.generate(date="2099-01-01")
        assert outputs == []

    def test_positions_top_n(self, workspace: Path, simple_strategy: StrategySpec) -> None:
        gen = EnsembleSignalGenerator(workspace=workspace, strategies=[simple_strategy])
        gen.load_data()
        outputs = gen.generate(date="2025-01-02", top_n=3)
        assert len(outputs) == 1
        # Positions should include top_n long + top_n short (minus neutrals)
        for pos in outputs[0].positions:
            assert pos["ensemble_signal"] != 0

    def test_auto_load_data(self, workspace: Path, simple_strategy: StrategySpec) -> None:
        gen = EnsembleSignalGenerator(workspace=workspace, strategies=[simple_strategy])
        # Don't call load_data() explicitly - generate() should auto-load
        outputs = gen.generate(date="2025-01-02")
        assert len(outputs) == 1


# ---------------------------------------------------------------------------
# Tests: load_weights_from_json
# ---------------------------------------------------------------------------


class TestLoadWeightsFromJson:
    def test_load_default(self, tmp_path: Path) -> None:
        import json

        data = {
            "best": "5_strat_super",
            "option_a_5strat": {
                "weights": {"A": 0.4, "B": 0.6},
            },
        }
        path = tmp_path / "ensemble.json"
        path.write_text(json.dumps(data), encoding="utf-8")

        weights = load_weights_from_json(path)
        assert weights == {"A": 0.4, "B": 0.6}

    def test_load_unknown_key(self, tmp_path: Path) -> None:
        import json

        data = {
            "best": "option_a_5strat",
            "option_a_5strat": {
                "weights": {"X": 1.0},
            },
        }
        path = tmp_path / "ensemble.json"
        path.write_text(json.dumps(data), encoding="utf-8")

        weights = load_weights_from_json(path)
        assert weights == {"X": 1.0}

"""Unit tests for SignalToStrategyAdapter (P0)."""

from datetime import datetime

import pandas as pd
import polars as pl
import pytest

from quant_insight.adapter.signal_to_strategy import (
    AdapterConfig,
    SignalToStrategyAdapter,
    ThresholdMethod,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_dates() -> list[datetime]:
    """Two trading dates."""
    return [datetime(2025, 1, 6), datetime(2025, 1, 7)]


@pytest.fixture()
def signal_df(sample_dates: list[datetime]) -> pl.DataFrame:
    """10 symbols x 2 dates with linearly spaced signals."""
    symbols = [f"SYM{i:02d}" for i in range(10)]
    rows: list[dict[str, object]] = []
    for dt in sample_dates:
        for i, sym in enumerate(symbols):
            rows.append({"datetime": dt, "symbol": sym, "signal": float(i * 10)})
    return pl.DataFrame(rows).cast({"datetime": pl.Datetime("us")})


@pytest.fixture()
def returns_df(sample_dates: list[datetime]) -> pl.DataFrame:
    """Returns data matching signal_df."""
    symbols = [f"SYM{i:02d}" for i in range(10)]
    rows: list[dict[str, object]] = []
    for dt in sample_dates:
        for i, sym in enumerate(symbols):
            rows.append({"datetime": dt, "symbol": sym, "return_value": (i - 5) * 0.01})
    return pl.DataFrame(rows).cast({"datetime": pl.Datetime("us")})


@pytest.fixture()
def ohlcv_df(sample_dates: list[datetime]) -> pl.DataFrame:
    """OHLCV data matching signal_df."""
    symbols = [f"SYM{i:02d}" for i in range(10)]
    rows: list[dict[str, object]] = []
    for dt in sample_dates:
        for sym in symbols:
            rows.append(
                {
                    "datetime": dt,
                    "symbol": sym,
                    "open": 100.0,
                    "high": 105.0,
                    "low": 95.0,
                    "close": 102.0,
                    "volume": 10000,
                }
            )
    return pl.DataFrame(rows).cast({"datetime": pl.Datetime("us")})


# ---------------------------------------------------------------------------
# P0-d: 10 test cases
# ---------------------------------------------------------------------------


@pytest.mark.unit()
class TestColumnMapping:
    """Test column renaming from MixSeek to quant-alpha-lab format."""

    def test_column_mapping(
        self,
        signal_df: pl.DataFrame,
        returns_df: pl.DataFrame,
        ohlcv_df: pl.DataFrame,
    ) -> None:
        """datetime→DATE, symbol→CODE, return_value→FORWARD_RETURN, close→CLOSE."""
        adapter = SignalToStrategyAdapter()
        result = adapter.convert(signal_df, returns_df, ohlcv_df)

        assert "CODE" in result.columns
        assert "DATE" in result.columns
        assert "FORWARD_RETURN" in result.columns
        assert "CLOSE" in result.columns
        assert "signal" in result.columns
        assert "raw_signal" in result.columns

        # Original MixSeek column names should not be present
        assert "symbol" not in result.columns
        assert "datetime" not in result.columns
        assert "return_value" not in result.columns

    def test_column_mapping_without_ohlcv(
        self,
        signal_df: pl.DataFrame,
        returns_df: pl.DataFrame,
    ) -> None:
        """CLOSE is absent when ohlcv_df is None."""
        adapter = SignalToStrategyAdapter()
        result = adapter.convert(signal_df, returns_df)

        assert "CLOSE" not in result.columns
        assert "CODE" in result.columns
        assert "FORWARD_RETURN" in result.columns


@pytest.mark.unit()
class TestQuantileDiscretization:
    """Test quantile-based discretization."""

    def test_quantile_discretization(
        self,
        signal_df: pl.DataFrame,
        returns_df: pl.DataFrame,
    ) -> None:
        """Top 10% → 1, bottom 10% → -1, others → 0."""
        config = AdapterConfig(
            threshold_method=ThresholdMethod.QUANTILE,
            long_quantile=0.9,
            short_quantile=0.1,
        )
        adapter = SignalToStrategyAdapter(config)
        result = adapter.convert(signal_df, returns_df)

        # 10 symbols per date: top 10% = 1 symbol (SYM09), bottom 10% = 1 symbol (SYM00)
        day1 = result[result["DATE"] == pd.Timestamp("2025-01-06")]

        long_signals = day1[day1["signal"] == 1]
        short_signals = day1[day1["signal"] == -1]
        neutral_signals = day1[day1["signal"] == 0]

        assert len(long_signals) == 1
        assert long_signals.iloc[0]["CODE"] == "SYM09"
        assert len(short_signals) == 1
        assert short_signals.iloc[0]["CODE"] == "SYM00"
        assert len(neutral_signals) == 8

    def test_quantile_handles_skewed_distribution(self) -> None:
        """Works correctly when 95% of signals are negative (like Focused strategy)."""
        dt = datetime(2025, 1, 6)
        symbols = [f"SYM{i:02d}" for i in range(100)]
        # 95 negative, 5 positive (skewed like Focused strategy)
        signals = [-50.0 + i * 0.5 for i in range(95)] + [1.0, 2.0, 3.0, 4.0, 5.0]

        signal_df = pl.DataFrame(
            {
                "datetime": [dt] * 100,
                "symbol": symbols,
                "signal": signals,
            }
        ).cast({"datetime": pl.Datetime("us")})

        returns_df = pl.DataFrame(
            {
                "datetime": [dt] * 100,
                "symbol": symbols,
                "return_value": [0.01] * 100,
            }
        ).cast({"datetime": pl.Datetime("us")})

        config = AdapterConfig(long_quantile=0.9, short_quantile=0.1)
        adapter = SignalToStrategyAdapter(config)
        result = adapter.convert(signal_df, returns_df)

        # Top 10% = 10 symbols, bottom 10% = 10 symbols
        assert (result["signal"] == 1).sum() == 10
        assert (result["signal"] == -1).sum() == 10
        assert (result["signal"] == 0).sum() == 80


@pytest.mark.unit()
class TestFixedDiscretization:
    """Test fixed-threshold discretization."""

    def test_fixed_threshold_discretization(
        self,
        signal_df: pl.DataFrame,
        returns_df: pl.DataFrame,
    ) -> None:
        """Signals >= 70 → 1, signals <= 10 → -1, others → 0."""
        config = AdapterConfig(
            threshold_method=ThresholdMethod.FIXED,
            fixed_long_threshold=70.0,
            fixed_short_threshold=10.0,
        )
        adapter = SignalToStrategyAdapter(config)
        result = adapter.convert(signal_df, returns_df)

        day1 = result[result["DATE"] == pd.Timestamp("2025-01-06")]
        # Signals: 0, 10, 20, 30, 40, 50, 60, 70, 80, 90
        # >= 70: SYM07(70), SYM08(80), SYM09(90) → 3 longs
        # <= 10: SYM00(0), SYM01(10) → 2 shorts
        assert (day1["signal"] == 1).sum() == 3
        assert (day1["signal"] == -1).sum() == 2
        assert (day1["signal"] == 0).sum() == 5


@pytest.mark.unit()
class TestZeroDiscretization:
    """Test zero-threshold discretization."""

    def test_zero_threshold_discretization(self) -> None:
        """signal >= 0 → 1, signal < 0 → -1."""
        dt = datetime(2025, 1, 6)
        signal_df = pl.DataFrame(
            {
                "datetime": [dt] * 5,
                "symbol": ["A", "B", "C", "D", "E"],
                "signal": [-2.0, -0.5, 0.0, 0.5, 2.0],
            }
        ).cast({"datetime": pl.Datetime("us")})

        returns_df = pl.DataFrame(
            {
                "datetime": [dt] * 5,
                "symbol": ["A", "B", "C", "D", "E"],
                "return_value": [0.01] * 5,
            }
        ).cast({"datetime": pl.Datetime("us")})

        config = AdapterConfig(threshold_method=ThresholdMethod.ZERO)
        adapter = SignalToStrategyAdapter(config)
        result = adapter.convert(signal_df, returns_df)

        assert (result["signal"] == -1).sum() == 2  # A(-2), B(-0.5)
        assert (result["signal"] == 1).sum() == 3  # C(0), D(0.5), E(2)


@pytest.mark.unit()
class TestStrategyFunc:
    """Test strategy_func closure generation."""

    def test_strategy_func_returns_code_date_signal(
        self,
        signal_df: pl.DataFrame,
    ) -> None:
        """strategy_func returns DataFrame with CODE, DATE, signal columns."""
        adapter = SignalToStrategyAdapter()
        func = adapter.make_strategy_func(signal_df)

        # Simulate quant-alpha-lab input
        input_data = pd.DataFrame(
            {
                "CODE": ["SYM00", "SYM05", "SYM09"],
                "DATE": pd.to_datetime(["2025-01-06"] * 3),
                "FORWARD_RETURN": [0.01, -0.02, 0.03],
            }
        )

        result = func(input_data)
        assert list(result.columns) == ["CODE", "DATE", "signal"]
        assert len(result) == 3

    def test_strategy_func_signal_domain(
        self,
        signal_df: pl.DataFrame,
    ) -> None:
        """strategy_func output signal values are in {-1, 0, 1}."""
        adapter = SignalToStrategyAdapter()
        func = adapter.make_strategy_func(signal_df)

        input_data = pd.DataFrame(
            {
                "CODE": [f"SYM{i:02d}" for i in range(10)],
                "DATE": pd.to_datetime(["2025-01-06"] * 10),
                "FORWARD_RETURN": [0.01] * 10,
            }
        )

        result = func(input_data)
        unique_signals = set(result["signal"].unique())
        assert unique_signals.issubset({-1, 0, 1})


@pytest.mark.unit()
class TestEdgeCases:
    """Test edge cases and robustness."""

    def test_nan_signal_becomes_zero(self) -> None:
        """NaN signals in strategy_func lookup are filled with 0."""
        dt = datetime(2025, 1, 6)
        signal_df = pl.DataFrame(
            {
                "datetime": [dt, dt],
                "symbol": ["A", "B"],
                "signal": [1.0, 2.0],
            }
        ).cast({"datetime": pl.Datetime("us")})

        adapter = SignalToStrategyAdapter()
        func = adapter.make_strategy_func(signal_df)

        # "C" does not exist in signal_df → should get signal=0
        input_data = pd.DataFrame(
            {
                "CODE": ["A", "C"],
                "DATE": pd.to_datetime([dt, dt]),
                "FORWARD_RETURN": [0.01, 0.02],
            }
        )

        result = func(input_data)
        assert result.loc[result["CODE"] == "C", "signal"].iloc[0] == 0

    def test_empty_dataframe(self) -> None:
        """Empty input does not raise errors."""
        signal_df = pl.DataFrame(schema={"datetime": pl.Datetime("us"), "symbol": pl.Utf8, "signal": pl.Float64})
        returns_df = pl.DataFrame(
            schema={"datetime": pl.Datetime("us"), "symbol": pl.Utf8, "return_value": pl.Float64}
        )

        adapter = SignalToStrategyAdapter()
        result = adapter.convert(signal_df, returns_df)
        assert len(result) == 0
        assert "CODE" in result.columns

    def test_deterministic(
        self,
        signal_df: pl.DataFrame,
        returns_df: pl.DataFrame,
    ) -> None:
        """Same input produces identical output."""
        adapter = SignalToStrategyAdapter()
        result1 = adapter.convert(signal_df, returns_df)
        result2 = adapter.convert(signal_df, returns_df)

        pd.testing.assert_frame_equal(result1, result2)

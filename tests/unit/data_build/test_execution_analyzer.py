"""Unit tests for ExecutionAnalyzer.

Tests cover:
1. daytrade_open_limit (long/short)
2. daytrade_intraday_limit (long/short)
3. Boundary NaN handling
4. Multiple symbols
5. Input validation (invalid method, position_side, negative offset)
6. Execution rate calculation
"""

from datetime import UTC, datetime

import polars as pl
import pytest

from quant_insight.data_build.execution_analyzer import ExecutionAnalyzer, ExecutionResult


class TestExecutionAnalyzer:
    """ExecutionAnalyzerの単体テスト."""

    @pytest.fixture
    def sample_ohlcv(self) -> pl.DataFrame:
        """サンプルOHLCVデータ（単一銘柄）.

        Day 1: O=100, H=102, L=99,  C=101
        Day 2: O=102, H=104, L=101, C=103
        Day 3: O=101, H=103, L=100, C=102
        Day 4: O=103, H=105, L=102, C=104
        Day 5: O=104, H=106, L=103, C=105
        """
        return pl.DataFrame(
            {
                "datetime": [
                    datetime(2023, 1, 1, tzinfo=UTC),
                    datetime(2023, 1, 2, tzinfo=UTC),
                    datetime(2023, 1, 3, tzinfo=UTC),
                    datetime(2023, 1, 4, tzinfo=UTC),
                    datetime(2023, 1, 5, tzinfo=UTC),
                ],
                "symbol": ["AAPL"] * 5,
                "open": [100.0, 102.0, 101.0, 103.0, 104.0],
                "high": [102.0, 104.0, 103.0, 105.0, 106.0],
                "low": [99.0, 101.0, 100.0, 102.0, 103.0],
                "close": [101.0, 103.0, 102.0, 104.0, 105.0],
                "volume": [1000, 1100, 1050, 1200, 1150],
            }
        )

    @pytest.fixture
    def multi_symbol_ohlcv(self) -> pl.DataFrame:
        """サンプルOHLCVデータ（複数銘柄）."""
        return pl.DataFrame(
            {
                "datetime": [
                    datetime(2023, 1, 1, tzinfo=UTC),
                    datetime(2023, 1, 1, tzinfo=UTC),
                    datetime(2023, 1, 2, tzinfo=UTC),
                    datetime(2023, 1, 2, tzinfo=UTC),
                    datetime(2023, 1, 3, tzinfo=UTC),
                    datetime(2023, 1, 3, tzinfo=UTC),
                ],
                "symbol": ["AAPL", "GOOGL", "AAPL", "GOOGL", "AAPL", "GOOGL"],
                "open": [100.0, 200.0, 102.0, 204.0, 101.0, 202.0],
                "high": [102.0, 202.0, 104.0, 206.0, 103.0, 204.0],
                "low": [99.0, 199.0, 101.0, 203.0, 100.0, 201.0],
                "close": [101.0, 201.0, 103.0, 205.0, 102.0, 203.0],
                "volume": [1000, 2000, 1100, 2100, 1050, 2050],
            }
        )

    def test_result_type(self, sample_ohlcv: pl.DataFrame) -> None:
        """Returns ExecutionResult with correct structure."""
        analyzer = ExecutionAnalyzer()
        result = analyzer.analyze(sample_ohlcv)

        assert isinstance(result, ExecutionResult)
        assert isinstance(result.data, pl.DataFrame)
        assert set(result.data.columns) == {"datetime", "symbol", "is_executed", "entry_price", "limit_return"}
        assert result.method == "daytrade_open_limit"
        assert result.position_side == "long"
        assert result.limit_offset_pct == 1.0

    def test_open_limit_long(self, sample_ohlcv: pl.DataFrame) -> None:
        """daytrade_open_limit long: fill if next_open <= close[t] * (1 - offset).

        offset=1%:
        Day 1: limit = 101 * 0.99 = 99.99. next_open = 102.0 > 99.99 → NOT filled
        Day 2: limit = 103 * 0.99 = 101.97. next_open = 101.0 <= 101.97 → FILLED
        Day 3: limit = 102 * 0.99 = 100.98. next_open = 103.0 > 100.98 → NOT filled
        Day 4: limit = 104 * 0.99 = 102.96. next_open = 104.0 > 102.96 → NOT filled
        Day 5: boundary → NOT filled
        """
        analyzer = ExecutionAnalyzer()
        result = analyzer.analyze(
            sample_ohlcv, method="daytrade_open_limit", position_side="long", limit_offset_pct=1.0
        )

        executed = result.data["is_executed"].to_list()
        assert executed == [False, True, False, False, False]

        # Day 2 filled: entry at open[1/3]=101, return = (close[1/3] - 101) / 101 = (102 - 101) / 101
        day2 = result.data.filter(pl.col("datetime") == datetime(2023, 1, 2, tzinfo=UTC))
        assert day2["entry_price"][0] == pytest.approx(101.0)
        assert day2["limit_return"][0] == pytest.approx((102.0 - 101.0) / 101.0, rel=1e-4)

    def test_open_limit_short(self, sample_ohlcv: pl.DataFrame) -> None:
        """daytrade_open_limit short: fill if next_open >= close[t] * (1 + offset).

        offset=1%:
        Day 1: limit = 101 * 1.01 = 102.01. next_open = 102.0 < 102.01 → NOT filled
        Day 2: limit = 103 * 1.01 = 104.03. next_open = 101.0 < 104.03 → NOT filled
        Day 3: limit = 102 * 1.01 = 103.02. next_open = 103.0 < 103.02 → NOT filled
        Day 4: limit = 104 * 1.01 = 105.04. next_open = 104.0 < 105.04 → NOT filled
        Day 5: boundary → NOT filled
        """
        analyzer = ExecutionAnalyzer()
        result = analyzer.analyze(
            sample_ohlcv, method="daytrade_open_limit", position_side="short", limit_offset_pct=1.0
        )

        executed = result.data["is_executed"].to_list()
        # With these prices and 1% offset, no shorts fill
        assert all(not x for x in executed)

    def test_intraday_limit_long(self, sample_ohlcv: pl.DataFrame) -> None:
        """daytrade_intraday_limit long: fill if next_low <= limit.

        offset=1%:
        Day 1: limit = 101 * 0.99 = 99.99. next_low = 101.0 > 99.99 → NOT filled
        Day 2: limit = 103 * 0.99 = 101.97. next_low = 100.0 <= 101.97 → FILLED
        Day 3: limit = 102 * 0.99 = 100.98. next_low = 102.0 > 100.98 → NOT filled
        Day 4: limit = 104 * 0.99 = 102.96. next_low = 103.0 > 102.96 → NOT filled
        Day 5: boundary → NOT filled
        """
        analyzer = ExecutionAnalyzer()
        result = analyzer.analyze(
            sample_ohlcv, method="daytrade_intraday_limit", position_side="long", limit_offset_pct=1.0
        )

        executed = result.data["is_executed"].to_list()
        assert executed == [False, True, False, False, False]

        # Day 2 filled: entry = min(next_open=101, limit=101.97) = 101.0
        day2 = result.data.filter(pl.col("datetime") == datetime(2023, 1, 2, tzinfo=UTC))
        assert day2["entry_price"][0] == pytest.approx(101.0)

    def test_intraday_limit_short(self, sample_ohlcv: pl.DataFrame) -> None:
        """daytrade_intraday_limit short: fill if next_high >= limit.

        offset=1%:
        Day 1: limit = 101 * 1.01 = 102.01. next_high = 104.0 >= 102.01 → FILLED
        Day 2: limit = 103 * 1.01 = 104.03. next_high = 103.0 < 104.03 → NOT filled
        Day 3: limit = 102 * 1.01 = 103.02. next_high = 105.0 >= 103.02 → FILLED
        Day 4: limit = 104 * 1.01 = 105.04. next_high = 106.0 >= 105.04 → FILLED
        Day 5: boundary → NOT filled
        """
        analyzer = ExecutionAnalyzer()
        result = analyzer.analyze(
            sample_ohlcv, method="daytrade_intraday_limit", position_side="short", limit_offset_pct=1.0
        )

        executed = result.data["is_executed"].to_list()
        assert executed == [True, False, True, True, False]

        # Day 1 filled: entry = max(next_open=102, limit=102.01) = 102.01
        day1 = result.data.filter(pl.col("datetime") == datetime(2023, 1, 1, tzinfo=UTC))
        assert day1["entry_price"][0] == pytest.approx(102.01, rel=1e-4)
        # Short return: (entry - next_close) / entry = (102.01 - 103) / 102.01
        assert day1["limit_return"][0] == pytest.approx((102.01 - 103.0) / 102.01, rel=1e-3)

    def test_boundary_not_executed(self, sample_ohlcv: pl.DataFrame) -> None:
        """Last row per symbol is never executed (no next-day data)."""
        analyzer = ExecutionAnalyzer()
        result = analyzer.analyze(sample_ohlcv)

        last_row = result.data.filter(pl.col("datetime") == datetime(2023, 1, 5, tzinfo=UTC))
        assert last_row["is_executed"][0] is False

    def test_non_executed_has_null_return(self, sample_ohlcv: pl.DataFrame) -> None:
        """Non-executed rows have null limit_return."""
        analyzer = ExecutionAnalyzer()
        result = analyzer.analyze(sample_ohlcv, method="daytrade_open_limit", position_side="long")

        non_executed = result.data.filter(~pl.col("is_executed"))
        assert non_executed["limit_return"].null_count() == len(non_executed)

    def test_execution_rate(self, sample_ohlcv: pl.DataFrame) -> None:
        """Execution rate is calculated correctly (excluding boundary rows)."""
        analyzer = ExecutionAnalyzer()
        result = analyzer.analyze(
            sample_ohlcv, method="daytrade_open_limit", position_side="long", limit_offset_pct=1.0
        )

        # 1 fill out of 4 non-boundary rows = 0.25
        assert result.execution_rate == pytest.approx(0.25)

    def test_multiple_symbols(self, multi_symbol_ohlcv: pl.DataFrame) -> None:
        """Multiple symbols: each symbol analyzed independently."""
        analyzer = ExecutionAnalyzer()
        result = analyzer.analyze(
            multi_symbol_ohlcv, method="daytrade_open_limit", position_side="long", limit_offset_pct=0.5
        )

        assert len(result.data) == 6
        # Verify per-symbol boundary handling
        aapl_last = result.data.filter(
            (pl.col("symbol") == "AAPL") & (pl.col("datetime") == datetime(2023, 1, 3, tzinfo=UTC))
        )
        assert aapl_last["is_executed"][0] is False

        googl_last = result.data.filter(
            (pl.col("symbol") == "GOOGL") & (pl.col("datetime") == datetime(2023, 1, 3, tzinfo=UTC))
        )
        assert googl_last["is_executed"][0] is False

    def test_invalid_method(self, sample_ohlcv: pl.DataFrame) -> None:
        """Invalid method raises ValueError."""
        analyzer = ExecutionAnalyzer()
        with pytest.raises(ValueError, match="Invalid method"):
            analyzer.analyze(sample_ohlcv, method="invalid_method")

    def test_invalid_position_side(self, sample_ohlcv: pl.DataFrame) -> None:
        """Invalid position_side raises ValueError."""
        analyzer = ExecutionAnalyzer()
        with pytest.raises(ValueError, match="Invalid position_side"):
            analyzer.analyze(sample_ohlcv, position_side="neutral")

    def test_negative_offset(self, sample_ohlcv: pl.DataFrame) -> None:
        """Negative limit_offset_pct raises ValueError."""
        analyzer = ExecutionAnalyzer()
        with pytest.raises(ValueError, match="limit_offset_pct must be >= 0"):
            analyzer.analyze(sample_ohlcv, limit_offset_pct=-1.0)

    def test_zero_offset(self, sample_ohlcv: pl.DataFrame) -> None:
        """Zero offset is valid (market order equivalent)."""
        analyzer = ExecutionAnalyzer()
        result = analyzer.analyze(sample_ohlcv, limit_offset_pct=0.0)
        assert isinstance(result, ExecutionResult)
        assert result.limit_offset_pct == 0.0

    def test_non_executed_has_null_entry_price_open_limit_long(self, sample_ohlcv: pl.DataFrame) -> None:
        """Non-executed rows have null entry_price (open_limit, long)."""
        analyzer = ExecutionAnalyzer()
        result = analyzer.analyze(sample_ohlcv, method="daytrade_open_limit", position_side="long")
        non_exec = result.data.filter(~pl.col("is_executed"))
        assert non_exec["entry_price"].null_count() == len(non_exec)

    def test_non_executed_has_null_entry_price_open_limit_short(self, sample_ohlcv: pl.DataFrame) -> None:
        """Non-executed rows have null entry_price (open_limit, short)."""
        analyzer = ExecutionAnalyzer()
        result = analyzer.analyze(sample_ohlcv, method="daytrade_open_limit", position_side="short")
        non_exec = result.data.filter(~pl.col("is_executed"))
        assert non_exec["entry_price"].null_count() == len(non_exec)

    def test_non_executed_has_null_entry_price_intraday_limit_long(self, sample_ohlcv: pl.DataFrame) -> None:
        """Non-executed rows have null entry_price (intraday_limit, long)."""
        analyzer = ExecutionAnalyzer()
        result = analyzer.analyze(sample_ohlcv, method="daytrade_intraday_limit", position_side="long")
        non_exec = result.data.filter(~pl.col("is_executed"))
        assert non_exec["entry_price"].null_count() == len(non_exec)

    def test_non_executed_has_null_entry_price_intraday_limit_short(self, sample_ohlcv: pl.DataFrame) -> None:
        """Non-executed rows have null entry_price (intraday_limit, short)."""
        analyzer = ExecutionAnalyzer()
        result = analyzer.analyze(sample_ohlcv, method="daytrade_intraday_limit", position_side="short")
        non_exec = result.data.filter(~pl.col("is_executed"))
        assert non_exec["entry_price"].null_count() == len(non_exec)

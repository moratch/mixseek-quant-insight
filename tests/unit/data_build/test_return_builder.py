"""Unit tests for ReturnBuilder.

Test Cases:
1. close2close method with window=1
2. close2close method with window=5
3. open2close method
4. Invalid method raises ValueError
5. Multiple symbols handling
6. NaN handling (first rows after window)
"""

from datetime import UTC, datetime

import polars as pl
import pytest

from quant_insight.data_build.return_builder import ReturnBuilder


class TestReturnBuilder:
    """ReturnBuilderの単体テスト."""

    @pytest.fixture
    def sample_ohlcv(self) -> pl.DataFrame:
        """サンプルOHLCVデータ（単一銘柄）."""
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

    def test_close2close_window_1(self, sample_ohlcv: pl.DataFrame) -> None:
        """close2close method with window=1 (future returns).

        未来リターン計算: 各日付における翌日のリターン
        1/1: (103 - 101) / 101 ≈ 0.0198 (翌日のclose)
        1/2: (102 - 103) / 103 ≈ -0.0097 (翌日のclose)
        1/5: NaN (翌日がない)
        """
        builder = ReturnBuilder()
        result = builder.calculate_returns(sample_ohlcv, window=1, method="close2close")

        assert isinstance(result, pl.DataFrame)
        assert set(result.columns) == {"datetime", "symbol", "return_value"}
        assert len(result) == 5

        # 1日目: 翌日(1/2)のリターン = (103 - 101) / 101 ≈ 0.0198
        first_return = result.filter(pl.col("datetime") == datetime(2023, 1, 1, tzinfo=UTC))["return_value"][0]
        assert first_return == pytest.approx((103.0 - 101.0) / 101.0, rel=1e-4)

        # 2日目: 翌日(1/3)のリターン = (102 - 103) / 103 ≈ -0.0097
        second_return = result.filter(pl.col("datetime") == datetime(2023, 1, 2, tzinfo=UTC))["return_value"][0]
        assert second_return == pytest.approx((102.0 - 103.0) / 103.0, rel=1e-4)

        # 5日目: 翌日がないのでNaN
        last_return = result.filter(pl.col("datetime") == datetime(2023, 1, 5, tzinfo=UTC))["return_value"][0]
        assert last_return is None or last_return != last_return  # NaN check

    def test_close2close_window_5(self, sample_ohlcv: pl.DataFrame) -> None:
        """close2close method with window=5.

        Expected: すべての行がNaN（5行しかないため、window=5では計算できない）
        pct_change(5)は6行目以降でのみ計算可能
        """
        builder = ReturnBuilder()
        result = builder.calculate_returns(sample_ohlcv, window=5, method="close2close")

        assert len(result) == 5

        # すべての行がNaN（5行しかないのでwindow=5では計算不可）
        all_returns = result["return_value"].to_list()
        for val in all_returns:
            assert val is None or val != val  # NaN check

    def test_open2close(self, sample_ohlcv: pl.DataFrame) -> None:
        """open2close method (future returns).

        未来リターン計算: 各日付における翌open→翌closeリターン
        window=1: 翌日のopen→翌日のclose
        1/1: 翌open(102) → 翌close(103) = (103 - 102) / 102
        1/5: NaN (翌日がない)
        """
        builder = ReturnBuilder()
        result = builder.calculate_returns(sample_ohlcv, method="open2close")

        assert len(result) == 5

        # 1日目: 翌open(102) → 翌close(103) = (103 - 102) / 102
        first_return = result.filter(pl.col("datetime") == datetime(2023, 1, 1, tzinfo=UTC))["return_value"][0]
        assert first_return == pytest.approx((103.0 - 102.0) / 102.0, rel=1e-4)

        # 2日目: 翌open(101) → 翌close(102) = (102 - 101) / 101
        second_return = result.filter(pl.col("datetime") == datetime(2023, 1, 2, tzinfo=UTC))["return_value"][0]
        assert second_return == pytest.approx((102.0 - 101.0) / 101.0, rel=1e-4)

        # 5日目: 翌日がないのでNaN
        last_return = result.filter(pl.col("datetime") == datetime(2023, 1, 5, tzinfo=UTC))["return_value"][0]
        assert last_return is None or last_return != last_return  # NaN check

    def test_open2close_window_2(self, sample_ohlcv: pl.DataFrame) -> None:
        """open2close method with window=2 (future returns).

        window=2: 翌日のopen→2日後のclose
        1/1: 翌open(102) → 2日後close(102) = (102 - 102) / 102 = 0
        1/2: 翌open(101) → 2日後close(104) = (104 - 101) / 101
        1/4, 1/5: NaN (2日後がない)
        """
        builder = ReturnBuilder()
        result = builder.calculate_returns(sample_ohlcv, window=2, method="open2close")

        assert len(result) == 5

        # 1日目: 翌open(102) → 2日後close(102) = 0
        first_return = result.filter(pl.col("datetime") == datetime(2023, 1, 1, tzinfo=UTC))["return_value"][0]
        assert first_return == pytest.approx(0.0, abs=1e-6)

        # 2日目: 翌open(101) → 2日後close(104) = (104 - 101) / 101
        second_return = result.filter(pl.col("datetime") == datetime(2023, 1, 2, tzinfo=UTC))["return_value"][0]
        assert second_return == pytest.approx((104.0 - 101.0) / 101.0, rel=1e-4)

        # 4日目: 2日後がないのでNaN
        fourth_return = result.filter(pl.col("datetime") == datetime(2023, 1, 4, tzinfo=UTC))["return_value"][0]
        assert fourth_return is None or fourth_return != fourth_return  # NaN check

        # 5日目: 2日後がないのでNaN
        fifth_return = result.filter(pl.col("datetime") == datetime(2023, 1, 5, tzinfo=UTC))["return_value"][0]
        assert fifth_return is None or fifth_return != fifth_return  # NaN check

    def test_invalid_method_raises_value_error(self, sample_ohlcv: pl.DataFrame) -> None:
        """Invalid method raises ValueError."""
        builder = ReturnBuilder()

        with pytest.raises(ValueError, match="Unknown method"):
            builder.calculate_returns(sample_ohlcv, method="invalid_method")

    def test_multiple_symbols_handling(self, multi_symbol_ohlcv: pl.DataFrame) -> None:
        """Multiple symbols handling with close2close method (future returns).

        未来リターン計算はsymbol単位で独立して行われる
        AAPL 1/1 → 翌日(1/2): (103 - 101) / 101
        GOOGL 1/1 → 翌日(1/2): (205 - 201) / 201
        """
        builder = ReturnBuilder()
        result = builder.calculate_returns(multi_symbol_ohlcv, window=1, method="close2close")

        assert len(result) == 6

        # AAPL: 1日目の未来リターン = 翌日(1/2)のリターン = (103 - 101) / 101
        aapl_day1 = result.filter(
            (pl.col("symbol") == "AAPL") & (pl.col("datetime") == datetime(2023, 1, 1, tzinfo=UTC))
        )["return_value"][0]
        assert aapl_day1 == pytest.approx((103.0 - 101.0) / 101.0, rel=1e-4)

        # GOOGL: 1日目の未来リターン = 翌日(1/2)のリターン = (205 - 201) / 201
        googl_day1 = result.filter(
            (pl.col("symbol") == "GOOGL") & (pl.col("datetime") == datetime(2023, 1, 1, tzinfo=UTC))
        )["return_value"][0]
        assert googl_day1 == pytest.approx((205.0 - 201.0) / 201.0, rel=1e-4)

    def test_nan_handling_first_rows(self, sample_ohlcv: pl.DataFrame) -> None:
        """NaN handling for last rows when future data is unavailable.

        window=2の場合、未来2日後のデータがない行はNaN
        1/1 → 1/3のリターン = (102 - 101) / 101
        1/4, 1/5 → NaN（2日後のデータがない）
        """
        builder = ReturnBuilder()
        result = builder.calculate_returns(sample_ohlcv, window=2, method="close2close")

        # 1日目: 2日後(1/3)のリターン = (102 - 101) / 101
        first_return = result.filter(pl.col("datetime") == datetime(2023, 1, 1, tzinfo=UTC))["return_value"][0]
        assert first_return == pytest.approx((102.0 - 101.0) / 101.0, rel=1e-4)

        # 2日目: 2日後(1/4)のリターン = (104 - 103) / 103
        second_return = result.filter(pl.col("datetime") == datetime(2023, 1, 2, tzinfo=UTC))["return_value"][0]
        assert second_return == pytest.approx((104.0 - 103.0) / 103.0, rel=1e-4)

        # 4日目: 2日後がないのでNaN
        fourth_return = result.filter(pl.col("datetime") == datetime(2023, 1, 4, tzinfo=UTC))["return_value"][0]
        assert fourth_return is None or fourth_return != fourth_return

        # 5日目: 2日後がないのでNaN
        fifth_return = result.filter(pl.col("datetime") == datetime(2023, 1, 5, tzinfo=UTC))["return_value"][0]
        assert fifth_return is None or fifth_return != fifth_return

    def test_daytrade_market(self, sample_ohlcv: pl.DataFrame) -> None:
        """daytrade_market method: 翌日の日中リターン (open[t+1] → close[t+1]).

        翌日の日中リターン = (close[t+1] - open[t+1]) / open[t+1]
        1/1: (close[1/2] - open[1/2]) / open[1/2] = (103 - 102) / 102
        1/2: (close[1/3] - open[1/3]) / open[1/3] = (102 - 101) / 101
        1/3: (close[1/4] - open[1/4]) / open[1/4] = (104 - 103) / 103
        1/4: (close[1/5] - open[1/5]) / open[1/5] = (105 - 104) / 104
        1/5: NaN (翌日がない)
        """
        builder = ReturnBuilder()
        result = builder.calculate_returns(sample_ohlcv, window=1, method="daytrade_market")

        assert isinstance(result, pl.DataFrame)
        assert set(result.columns) == {"datetime", "symbol", "return_value"}
        assert len(result) == 5

        # 1日目: 翌日の日中リターン = (103 - 102) / 102
        first_return = result.filter(pl.col("datetime") == datetime(2023, 1, 1, tzinfo=UTC))["return_value"][0]
        assert first_return == pytest.approx((103.0 - 102.0) / 102.0, rel=1e-4)

        # 2日目: 翌日の日中リターン = (102 - 101) / 101
        second_return = result.filter(pl.col("datetime") == datetime(2023, 1, 2, tzinfo=UTC))["return_value"][0]
        assert second_return == pytest.approx((102.0 - 101.0) / 101.0, rel=1e-4)

        # 4日目: 翌日の日中リターン = (105 - 104) / 104
        fourth_return = result.filter(pl.col("datetime") == datetime(2023, 1, 4, tzinfo=UTC))["return_value"][0]
        assert fourth_return == pytest.approx((105.0 - 104.0) / 104.0, rel=1e-4)

        # 5日目: 翌日がないのでNaN（境界NaN）
        last_return = result.filter(pl.col("datetime") == datetime(2023, 1, 5, tzinfo=UTC))["return_value"][0]
        assert last_return is None or last_return != last_return  # NaN check

    def test_daytrade_market_boundary_nan(self, sample_ohlcv: pl.DataFrame) -> None:
        """daytrade_market boundary NaN: 最終行のみNaN（全銘柄一律、close2closeと同じ挙動）."""
        builder = ReturnBuilder()
        result = builder.calculate_returns(sample_ohlcv, window=1, method="daytrade_market")

        # 最終行以外はNaNでない
        non_last = result.filter(pl.col("datetime") != datetime(2023, 1, 5, tzinfo=UTC))
        assert non_last["return_value"].null_count() == 0

        # 最終行はNaN
        last = result.filter(pl.col("datetime") == datetime(2023, 1, 5, tzinfo=UTC))
        assert last["return_value"].null_count() == 1

    def test_daytrade_multiple_symbols(self, multi_symbol_ohlcv: pl.DataFrame) -> None:
        """daytrade_market: 複数銘柄でsymbol独立計算.

        AAPL 1/1: (close[1/2] - open[1/2]) / open[1/2] = (103 - 102) / 102
        GOOGL 1/1: (close[1/2] - open[1/2]) / open[1/2] = (205 - 204) / 204
        """
        builder = ReturnBuilder()
        result = builder.calculate_returns(multi_symbol_ohlcv, window=1, method="daytrade_market")

        assert len(result) == 6

        # AAPL 1/1: (103 - 102) / 102
        aapl_day1 = result.filter(
            (pl.col("symbol") == "AAPL") & (pl.col("datetime") == datetime(2023, 1, 1, tzinfo=UTC))
        )["return_value"][0]
        assert aapl_day1 == pytest.approx((103.0 - 102.0) / 102.0, rel=1e-4)

        # GOOGL 1/1: (205 - 204) / 204
        googl_day1 = result.filter(
            (pl.col("symbol") == "GOOGL") & (pl.col("datetime") == datetime(2023, 1, 1, tzinfo=UTC))
        )["return_value"][0]
        assert googl_day1 == pytest.approx((205.0 - 204.0) / 204.0, rel=1e-4)

        # 各銘柄の最終行はNaN
        aapl_last = result.filter(
            (pl.col("symbol") == "AAPL") & (pl.col("datetime") == datetime(2023, 1, 3, tzinfo=UTC))
        )["return_value"][0]
        assert aapl_last is None or aapl_last != aapl_last

        googl_last = result.filter(
            (pl.col("symbol") == "GOOGL") & (pl.col("datetime") == datetime(2023, 1, 3, tzinfo=UTC))
        )["return_value"][0]
        assert googl_last is None or googl_last != googl_last

    def test_daytrade_market_window_guard(self, sample_ohlcv: pl.DataFrame) -> None:
        """daytrade_market with window≠1 raises ValueError（直接呼び出し経路の回帰テスト）."""
        builder = ReturnBuilder()
        with pytest.raises(ValueError, match="daytrade_market requires window=1"):
            builder.calculate_returns(sample_ohlcv, method="daytrade_market", window=5)

"""Unit tests for backtest_loop module."""

from datetime import datetime

import pandas as pd
import polars as pl
import pytest

from quant_insight.evaluator.backtest_loop import BacktestLoop
from quant_insight.exceptions import SubmissionFailedError, SubmissionInvalidError


@pytest.fixture
def sample_ohlcv():
    """Create sample OHLCV data for testing with 4 symbols for valid correlation."""
    return pl.DataFrame(
        {
            "datetime": [
                datetime(2023, 1, 1),
                datetime(2023, 1, 1),
                datetime(2023, 1, 1),
                datetime(2023, 1, 1),
                datetime(2023, 1, 2),
                datetime(2023, 1, 2),
                datetime(2023, 1, 2),
                datetime(2023, 1, 2),
            ],
            "symbol": ["AAPL", "GOOGL", "MSFT", "AMZN", "AAPL", "GOOGL", "MSFT", "AMZN"],
            "open": [100.0, 200.0, 150.0, 300.0, 105.0, 210.0, 155.0, 310.0],
            "high": [102.0, 205.0, 155.0, 310.0, 108.0, 215.0, 160.0, 320.0],
            "low": [99.0, 198.0, 148.0, 295.0, 104.0, 208.0, 153.0, 305.0],
            "close": [101.0, 203.0, 152.0, 305.0, 107.0, 212.0, 157.0, 315.0],
            "volume": [1000, 2000, 1500, 2500, 1100, 2100, 1600, 2600],
        }
    )


@pytest.fixture
def sample_returns():
    """Create sample returns data for testing with 4 symbols."""
    return pl.DataFrame(
        {
            "datetime": [
                datetime(2023, 1, 1),
                datetime(2023, 1, 1),
                datetime(2023, 1, 1),
                datetime(2023, 1, 1),
                datetime(2023, 1, 2),
                datetime(2023, 1, 2),
                datetime(2023, 1, 2),
                datetime(2023, 1, 2),
            ],
            "symbol": ["AAPL", "GOOGL", "MSFT", "AMZN", "AAPL", "GOOGL", "MSFT", "AMZN"],
            "return_value": [0.01, 0.015, 0.012, 0.018, 0.02, 0.018, 0.022, 0.016],
        }
    )


@pytest.mark.unit
def test_backtest_loop_iteration(sample_ohlcv, sample_returns):
    """Test that backtest loop iterates over unique datetime values."""

    def simple_signal(ohlcv, additional_data):
        # Return varying signal based on close price for valid correlation
        return (
            ohlcv.select(["datetime", "symbol", "close"])
            .with_columns(pl.col("close").alias("signal"))
            .select(["datetime", "symbol", "signal"])
        )

    loop = BacktestLoop(sample_ohlcv, sample_returns)
    result = loop.run(simple_signal)

    assert result.status == "completed"
    assert result.total_iterations == 2  # Two unique datetimes
    assert len(result.iteration_results) == 2


@pytest.mark.unit
def test_backtest_loop_data_filtering(sample_ohlcv, sample_returns):
    """Test that signal function receives only data up to current datetime."""
    received_data_lengths = []

    def tracking_signal(ohlcv, additional_data):
        received_data_lengths.append(len(ohlcv))
        # Use close price as signal for valid correlation
        return (
            ohlcv.select(["datetime", "symbol", "close"])
            .with_columns(pl.col("close").alias("signal"))
            .select(["datetime", "symbol", "signal"])
        )

    loop = BacktestLoop(sample_ohlcv, sample_returns)
    loop.run(tracking_signal)

    # First iteration should have 4 rows (4 symbols), second should have 8 rows
    assert received_data_lengths == [4, 8]


@pytest.mark.unit
def test_signal_validation_missing_columns(sample_ohlcv, sample_returns):
    """Test that signals are validated for required columns."""

    def invalid_signal(ohlcv, additional_data):
        # Missing 'signal' column
        return ohlcv.select(["datetime", "symbol"])

    loop = BacktestLoop(sample_ohlcv, sample_returns)
    with pytest.raises(SubmissionInvalidError, match="カラム"):
        loop.run(invalid_signal)


@pytest.mark.unit
def test_signal_with_nan_handling(sample_ohlcv, sample_returns):
    """Test NaN handling in signals (fill with mean)."""

    def signal_with_nan(ohlcv, additional_data):
        # Create signal with NaN for one symbol and varying signals for others
        # AAPL=NaN, GOOGL=2.0, MSFT=1.5, AMZN=2.5 → AAPL will be filled with mean
        return ohlcv.select(["datetime", "symbol"]).with_columns(
            pl.when(pl.col("symbol") == "AAPL")
            .then(float("nan"))
            .when(pl.col("symbol") == "GOOGL")
            .then(2.0)
            .when(pl.col("symbol") == "MSFT")
            .then(1.5)
            .otherwise(2.5)
            .alias("signal")
        )

    loop = BacktestLoop(sample_ohlcv, sample_returns)
    result = loop.run(signal_with_nan)

    # Should complete successfully with NaN filled
    # NaN in AAPL is filled with mean of [NaN, 2.0, 1.5, 2.5] = 2.0
    assert result.status == "completed"
    assert result.valid_iterations > 0  # Correlation should be calculable


@pytest.mark.unit
def test_return_with_nan_handling(sample_ohlcv, sample_returns):
    """Test NaN handling in returns (exclude those symbols)."""
    # Create returns with NaN for first AAPL entry
    returns = pl.DataFrame(
        {
            "datetime": [
                datetime(2023, 1, 1),
                datetime(2023, 1, 1),
                datetime(2023, 1, 1),
                datetime(2023, 1, 1),
                datetime(2023, 1, 2),
                datetime(2023, 1, 2),
                datetime(2023, 1, 2),
                datetime(2023, 1, 2),
            ],
            "symbol": ["AAPL", "GOOGL", "MSFT", "AMZN", "AAPL", "GOOGL", "MSFT", "AMZN"],
            "return_value": [float("nan"), 0.015, 0.012, 0.018, 0.02, 0.018, 0.022, 0.016],
        }
    )

    def varying_signal(ohlcv, additional_data):
        # Use close price as signal for varying values
        return (
            ohlcv.select(["datetime", "symbol", "close"])
            .with_columns(pl.col("close").alias("signal"))
            .select(["datetime", "symbol", "signal"])
        )

    loop = BacktestLoop(sample_ohlcv, returns)
    result = loop.run(varying_signal)

    # Should complete successfully (3 valid points per iteration after NaN removal)
    assert result.status == "completed"
    assert result.valid_iterations > 0


@pytest.mark.unit
def test_spearman_correlation_calculation():
    """Test Spearman rank correlation calculation."""
    ohlcv = pl.DataFrame(
        {
            "datetime": [datetime(2023, 1, 1)] * 3,
            "symbol": ["A", "B", "C"],
            "open": [100.0, 200.0, 300.0],
            "high": [100.0, 200.0, 300.0],
            "low": [100.0, 200.0, 300.0],
            "close": [100.0, 200.0, 300.0],
            "volume": [1000, 1000, 1000],
        }
    )

    returns = pl.DataFrame(
        {
            "datetime": [datetime(2023, 1, 1)] * 3,
            "symbol": ["A", "B", "C"],
            "return_value": [0.01, 0.02, 0.03],  # Perfectly correlated with signal
        }
    )

    def perfect_signal(ohlcv, additional_data):
        # Perfect positive correlation with returns
        return pl.DataFrame(
            {
                "datetime": [datetime(2023, 1, 1)] * 3,
                "symbol": ["A", "B", "C"],
                "signal": [1.0, 2.0, 3.0],
            }
        )

    loop = BacktestLoop(ohlcv, returns)
    result = loop.run(perfect_signal)

    # Should have perfect correlation (1.0)
    assert result.iteration_results[0].rank_correlation == pytest.approx(1.0)


@pytest.mark.unit
def test_spearman_correlation_edge_case_less_than_2_points():
    """Test that SubmissionFailedError is raised when all iterations have < 2 data points."""
    ohlcv = pl.DataFrame(
        {
            "datetime": [datetime(2023, 1, 1)],
            "symbol": ["A"],
            "open": [100.0],
            "high": [100.0],
            "low": [100.0],
            "close": [100.0],
            "volume": [1000],
        }
    )

    returns = pl.DataFrame(
        {
            "datetime": [datetime(2023, 1, 1)],
            "symbol": ["A"],
            "return_value": [0.01],
        }
    )

    def simple_signal(ohlcv, additional_data):
        return ohlcv.select(["datetime", "symbol"]).with_columns(pl.lit(1.0).alias("signal"))

    loop = BacktestLoop(ohlcv, returns)

    # Should raise SubmissionFailedError when no valid correlations can be computed
    with pytest.raises(SubmissionFailedError, match="有効な相関を計算できるイテレーションがありません"):
        loop.run(simple_signal)


@pytest.mark.unit
def test_sharpe_ratio_calculation():
    """Test Sharpe ratio calculation from correlation series."""
    # Create data with 3 symbols and 3 iterations for valid Sharpe ratio calculation
    ohlcv = pl.DataFrame(
        {
            "datetime": ([datetime(2023, 1, 1)] * 3 + [datetime(2023, 1, 2)] * 3 + [datetime(2023, 1, 3)] * 3),
            "symbol": ["A", "B", "C"] * 3,
            "open": [100.0, 200.0, 300.0, 101.0, 201.0, 301.0, 102.0, 202.0, 302.0],
            "high": [100.0, 200.0, 300.0, 101.0, 201.0, 301.0, 102.0, 202.0, 302.0],
            "low": [100.0, 200.0, 300.0, 101.0, 201.0, 301.0, 102.0, 202.0, 302.0],
            "close": [100.0, 200.0, 300.0, 101.0, 201.0, 301.0, 102.0, 202.0, 302.0],
            "volume": [1000] * 9,
        }
    )

    # Create returns that will give positive correlations with close price
    returns = pl.DataFrame(
        {
            "datetime": ([datetime(2023, 1, 1)] * 3 + [datetime(2023, 1, 2)] * 3 + [datetime(2023, 1, 3)] * 3),
            "symbol": ["A", "B", "C"] * 3,
            "return_value": [0.01, 0.02, 0.03, 0.011, 0.021, 0.031, 0.012, 0.022, 0.032],
        }
    )

    def consistent_signal(ohlcv, additional_data):
        # Use close price as signal - perfectly correlated with returns
        return (
            ohlcv.select(["datetime", "symbol", "close"])
            .with_columns(pl.col("close").alias("signal"))
            .select(["datetime", "symbol", "signal"])
        )

    loop = BacktestLoop(ohlcv, returns)
    result = loop.run(consistent_signal)

    # Should have valid iterations
    assert result.valid_iterations == 3
    # With perfect correlation (1.0) every iteration, std is 0, so sharpe is 0.0
    assert result.mean_correlation == pytest.approx(1.0)
    assert result.std_correlation == pytest.approx(0.0)
    assert result.sharpe_ratio == 0.0


@pytest.mark.unit
def test_sharpe_ratio_edge_case_zero_std():
    """Test Sharpe ratio when standard deviation is zero."""
    # Create scenario with multiple symbols for valid correlations
    ohlcv = pl.DataFrame(
        {
            "datetime": [datetime(2023, 1, 1)] * 3 + [datetime(2023, 1, 2)] * 3,
            "symbol": ["A", "B", "C", "A", "B", "C"],
            "open": [100.0] * 6,
            "high": [100.0] * 6,
            "low": [100.0] * 6,
            "close": [100.0] * 6,
            "volume": [1000] * 6,
        }
    )

    returns = pl.DataFrame(
        {
            "datetime": [datetime(2023, 1, 1)] * 3 + [datetime(2023, 1, 2)] * 3,
            "symbol": ["A", "B", "C", "A", "B", "C"],
            "return_value": [0.01, 0.02, 0.03, 0.01, 0.02, 0.03],
        }
    )

    def identical_signal(ohlcv, additional_data):
        # Same signal every time - will produce identical correlations
        return pl.DataFrame(
            {
                "datetime": ohlcv["datetime"],
                "symbol": ohlcv["symbol"],
                "signal": [1.0, 2.0, 3.0] * (len(ohlcv) // 3),
            }
        )

    loop = BacktestLoop(ohlcv, returns)
    result = loop.run(identical_signal)

    # When std is 0 (all correlations identical), sharpe ratio should be 0.0
    assert result.sharpe_ratio == 0.0


@pytest.mark.unit
def test_sharpe_ratio_edge_case_no_valid_iterations():
    """Test that SubmissionFailedError is raised when there are no valid iterations."""
    ohlcv = pl.DataFrame(
        {
            "datetime": [datetime(2023, 1, 1)],
            "symbol": ["A"],
            "open": [100.0],
            "high": [100.0],
            "low": [100.0],
            "close": [100.0],
            "volume": [1000],
        }
    )

    returns = pl.DataFrame(
        {
            "datetime": [datetime(2023, 1, 1)],
            "symbol": ["A"],
            "return_value": [0.01],
        }
    )

    def simple_signal(ohlcv, additional_data):
        return ohlcv.select(["datetime", "symbol"]).with_columns(pl.lit(1.0).alias("signal"))

    loop = BacktestLoop(ohlcv, returns)

    # No valid iterations (need >= 2 points for correlation) should raise SubmissionFailedError
    with pytest.raises(SubmissionFailedError, match="有効な相関を計算できるイテレーションがありません"):
        loop.run(simple_signal)


@pytest.mark.unit
def test_signal_function_raises_exception(sample_ohlcv, sample_returns):
    """Test that exceptions in signal function are caught and reported."""

    def failing_signal(ohlcv, additional_data):
        raise ValueError("Intentional failure")

    loop = BacktestLoop(sample_ohlcv, sample_returns)
    with pytest.raises(SubmissionFailedError, match="シグナル関数が例外を発生させました"):
        loop.run(failing_signal)


@pytest.mark.unit
def test_additional_data_filtering(sample_ohlcv, sample_returns):
    """Test that additional data is also filtered by datetime."""
    additional_data = {
        "fundamentals": pl.DataFrame(
            {
                "datetime": [datetime(2023, 1, 1), datetime(2023, 1, 2)],
                "symbol": ["AAPL", "AAPL"],
                "pe_ratio": [15.0, 16.0],
            }
        )
    }

    received_additional_lengths = []

    def tracking_signal(ohlcv, additional_data):
        received_additional_lengths.append(len(additional_data.get("fundamentals", pl.DataFrame())))
        # Use close price as signal for valid correlation
        return (
            ohlcv.select(["datetime", "symbol", "close"])
            .with_columns(pl.col("close").alias("signal"))
            .select(["datetime", "symbol", "signal"])
        )

    loop = BacktestLoop(sample_ohlcv, sample_returns, additional_data)
    loop.run(tracking_signal)

    # First iteration: 1 row, second iteration: 2 rows
    assert received_additional_lengths == [1, 2]


@pytest.mark.unit
def test_backtest_result_statistics(sample_ohlcv, sample_returns):
    """Test that BacktestResult includes correct statistics."""

    def simple_signal(ohlcv, additional_data):
        # Use close price as signal for valid correlation
        return (
            ohlcv.select(["datetime", "symbol", "close"])
            .with_columns(pl.col("close").alias("signal"))
            .select(["datetime", "symbol", "signal"])
        )

    loop = BacktestLoop(sample_ohlcv, sample_returns)
    result = loop.run(simple_signal)

    assert result.total_iterations == 2
    assert result.evaluation_started_at is not None
    assert result.evaluation_completed_at is not None
    assert result.evaluation_started_at <= result.evaluation_completed_at


@pytest.mark.unit
def test_backtest_loop_with_pandas_dataframe(sample_ohlcv, sample_returns):
    """Test that backtest loop accepts pd.DataFrame as signal output."""

    def pandas_signal(ohlcv, additional_data):
        # Return pandas DataFrame instead of polars
        df = ohlcv.to_pandas()
        return pd.DataFrame(
            {
                "datetime": df["datetime"],
                "symbol": df["symbol"],
                "signal": df["close"],
            }
        )

    loop = BacktestLoop(sample_ohlcv, sample_returns)
    result = loop.run(pandas_signal)

    assert result.status == "completed"
    assert result.total_iterations == 2
    assert result.valid_iterations > 0


@pytest.mark.unit
def test_backtest_loop_rejects_invalid_return_type(sample_ohlcv, sample_returns):
    """Test that backtest loop rejects non-DataFrame return types."""

    def invalid_signal(ohlcv, additional_data):
        return {"datetime": [], "symbol": [], "signal": []}  # dict, not DataFrame

    loop = BacktestLoop(sample_ohlcv, sample_returns)
    with pytest.raises(SubmissionInvalidError, match="pl.DataFrameまたはpd.DataFrame"):
        loop.run(invalid_signal)


@pytest.mark.unit
def test_backtest_loop_normalizes_nanosecond_datetime_to_microsecond():
    """Test that nanosecond datetime precision is normalized to microsecond."""
    # Create data with explicit nanosecond precision (pandas default)
    ohlcv_ns = pl.DataFrame(
        {
            "datetime": [datetime(2023, 1, 1)] * 3,
            "symbol": ["A", "B", "C"],
            "open": [100.0, 200.0, 300.0],
            "high": [100.0, 200.0, 300.0],
            "low": [100.0, 200.0, 300.0],
            "close": [100.0, 200.0, 300.0],
            "volume": [1000, 1000, 1000],
        }
    ).with_columns(pl.col("datetime").dt.cast_time_unit("ns"))

    returns_ns = pl.DataFrame(
        {
            "datetime": [datetime(2023, 1, 1)] * 3,
            "symbol": ["A", "B", "C"],
            "return_value": [0.01, 0.02, 0.03],
        }
    ).with_columns(pl.col("datetime").dt.cast_time_unit("ns"))

    # Verify input is nanosecond precision
    assert ohlcv_ns.schema["datetime"] == pl.Datetime("ns")
    assert returns_ns.schema["datetime"] == pl.Datetime("ns")

    loop = BacktestLoop(ohlcv_ns, returns_ns)

    # Verify internal data is normalized to microsecond precision
    assert loop.ohlcv.schema["datetime"] == pl.Datetime("us")
    assert loop.returns.schema["datetime"] == pl.Datetime("us")


@pytest.mark.unit
def test_backtest_loop_handles_mixed_datetime_precision():
    """Test that backtest loop handles mixed ns/us precision between input and signal."""
    # Create OHLCV with nanosecond precision (simulating pandas-created parquet)
    ohlcv_ns = pl.DataFrame(
        {
            "datetime": [datetime(2023, 1, 1)] * 3,
            "symbol": ["A", "B", "C"],
            "open": [100.0, 200.0, 300.0],
            "high": [100.0, 200.0, 300.0],
            "low": [100.0, 200.0, 300.0],
            "close": [100.0, 200.0, 300.0],
            "volume": [1000, 1000, 1000],
        }
    ).with_columns(pl.col("datetime").dt.cast_time_unit("ns"))

    # Create returns with nanosecond precision
    returns_ns = pl.DataFrame(
        {
            "datetime": [datetime(2023, 1, 1)] * 3,
            "symbol": ["A", "B", "C"],
            "return_value": [0.01, 0.02, 0.03],
        }
    ).with_columns(pl.col("datetime").dt.cast_time_unit("ns"))

    def signal_with_us_precision(ohlcv, additional_data):
        # Signal function returns microsecond precision (polars default)
        return pl.DataFrame(
            {
                "datetime": [datetime(2023, 1, 1)] * 3,
                "symbol": ["A", "B", "C"],
                "signal": [1.0, 2.0, 3.0],
            }
        )

    loop = BacktestLoop(ohlcv_ns, returns_ns)
    result = loop.run(signal_with_us_precision)

    # Should complete successfully without join errors
    assert result.status == "completed"
    assert result.valid_iterations == 1
    assert result.iteration_results[0].rank_correlation == pytest.approx(1.0)


@pytest.mark.unit
def test_backtest_loop_handles_pandas_signal_with_ns_datetime():
    """Test that pandas DataFrame signals with ns datetime are normalized."""
    # Create OHLCV and returns with microsecond precision
    ohlcv = pl.DataFrame(
        {
            "datetime": [datetime(2023, 1, 1)] * 3,
            "symbol": ["A", "B", "C"],
            "open": [100.0, 200.0, 300.0],
            "high": [100.0, 200.0, 300.0],
            "low": [100.0, 200.0, 300.0],
            "close": [100.0, 200.0, 300.0],
            "volume": [1000, 1000, 1000],
        }
    )

    returns = pl.DataFrame(
        {
            "datetime": [datetime(2023, 1, 1)] * 3,
            "symbol": ["A", "B", "C"],
            "return_value": [0.01, 0.02, 0.03],
        }
    )

    def pandas_signal(ohlcv, additional_data):
        # pandas DataFrame will have nanosecond precision by default
        return pd.DataFrame(
            {
                "datetime": [datetime(2023, 1, 1)] * 3,
                "symbol": ["A", "B", "C"],
                "signal": [1.0, 2.0, 3.0],
            }
        )

    loop = BacktestLoop(ohlcv, returns)
    result = loop.run(pandas_signal)

    # Should complete successfully without join errors
    assert result.status == "completed"
    assert result.valid_iterations == 1
    assert result.iteration_results[0].rank_correlation == pytest.approx(1.0)


@pytest.mark.unit
def test_backtest_loop_preserves_microsecond_datetime():
    """Test that microsecond datetime precision is preserved (no unnecessary conversion)."""
    # Create data with explicit microsecond precision
    ohlcv_us = pl.DataFrame(
        {
            "datetime": [datetime(2023, 1, 1)] * 3,
            "symbol": ["A", "B", "C"],
            "open": [100.0, 200.0, 300.0],
            "high": [100.0, 200.0, 300.0],
            "low": [100.0, 200.0, 300.0],
            "close": [100.0, 200.0, 300.0],
            "volume": [1000, 1000, 1000],
        }
    )  # Default is microsecond

    returns_us = pl.DataFrame(
        {
            "datetime": [datetime(2023, 1, 1)] * 3,
            "symbol": ["A", "B", "C"],
            "return_value": [0.01, 0.02, 0.03],
        }
    )

    # Verify input is already microsecond precision
    assert ohlcv_us.schema["datetime"] == pl.Datetime("us")
    assert returns_us.schema["datetime"] == pl.Datetime("us")

    loop = BacktestLoop(ohlcv_us, returns_us)

    # Verify data remains microsecond precision
    assert loop.ohlcv.schema["datetime"] == pl.Datetime("us")
    assert loop.returns.schema["datetime"] == pl.Datetime("us")


@pytest.mark.unit
def test_signal_validation_datetime_type_error(sample_ohlcv, sample_returns):
    """Test that signal datetime column must be Datetime type."""

    def invalid_datetime_signal(ohlcv, additional_data):
        # Return DataFrame with datetime as string instead of Datetime
        return pl.DataFrame(
            {
                "datetime": ["2023-01-01"] * 4,  # String, not Datetime
                "symbol": ["AAPL", "GOOGL", "MSFT", "AMZN"],
                "signal": [1.0, 2.0, 3.0, 4.0],
            }
        )

    loop = BacktestLoop(sample_ohlcv, sample_returns)
    with pytest.raises(SubmissionInvalidError, match="'datetime'カラムの型はDatetimeである必要があります"):
        loop.run(invalid_datetime_signal)


@pytest.mark.unit
def test_signal_validation_symbol_type_error(sample_ohlcv, sample_returns):
    """Test that signal symbol column must be Utf8/String type."""

    def invalid_symbol_signal(ohlcv, additional_data):
        # Return DataFrame with symbol as int instead of String
        return pl.DataFrame(
            {
                "datetime": [datetime(2023, 1, 1)] * 4,
                "symbol": [1, 2, 3, 4],  # Int, not String
                "signal": [1.0, 2.0, 3.0, 4.0],
            }
        )

    loop = BacktestLoop(sample_ohlcv, sample_returns)
    with pytest.raises(SubmissionInvalidError, match="'symbol'カラムの型はUtf8/Stringである必要があります"):
        loop.run(invalid_symbol_signal)


@pytest.mark.unit
def test_signal_validation_signal_type_error(sample_ohlcv, sample_returns):
    """Test that signal column must be numeric type."""

    def invalid_signal_type(ohlcv, additional_data):
        # Return DataFrame with signal as string instead of numeric
        return pl.DataFrame(
            {
                "datetime": [datetime(2023, 1, 1)] * 4,
                "symbol": ["AAPL", "GOOGL", "MSFT", "AMZN"],
                "signal": ["high", "low", "medium", "high"],  # String, not numeric
            }
        )

    loop = BacktestLoop(sample_ohlcv, sample_returns)
    with pytest.raises(SubmissionInvalidError, match="'signal'カラムの型は数値型である必要があります"):
        loop.run(invalid_signal_type)

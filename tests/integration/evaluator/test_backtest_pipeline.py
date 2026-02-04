"""Integration tests for the complete backtest pipeline."""

from datetime import datetime

import polars as pl
import pytest

from quant_insight.evaluator.correlation_sharpe_ratio import CorrelationSharpeRatio


@pytest.fixture
def workspace_with_momentum_data(tmp_path, monkeypatch):
    """Create workspace with momentum test data (10 days × 3 symbols)."""
    # Set MIXSEEK_WORKSPACE environment variable
    monkeypatch.setenv("MIXSEEK_WORKSPACE", str(tmp_path))

    # Create directory structure
    ohlcv_dir = tmp_path / "data" / "inputs" / "ohlcv"
    returns_dir = tmp_path / "data" / "inputs" / "returns"
    ohlcv_dir.mkdir(parents=True)
    returns_dir.mkdir(parents=True)

    # Create sample OHLCV data with multiple symbols and datetimes
    # Generate 10 days × 3 symbols = 30 rows
    dates = [datetime(2023, 1, i) for i in range(1, 11)] * 3
    dates.sort()  # Sort so each symbol has all dates

    ohlcv = pl.DataFrame(
        {
            "datetime": dates,
            "symbol": ["AAPL"] * 10 + ["GOOGL"] * 10 + ["MSFT"] * 10,
            "open": list(range(100, 130)),
            "high": list(range(102, 132)),
            "low": list(range(99, 129)),
            "close": list(range(101, 131)),
            "volume": [1000 + i * 100 for i in range(30)],
        }
    )

    # Create returns with some correlation to close prices
    returns = ohlcv.select(["datetime", "symbol"]).with_columns(
        [
            # Returns correlated with position in sequence (simulating trend)
            pl.Series("return_value", [(i % 10) * 0.001 + 0.01 for i in range(30)])
        ]
    )

    # Save to expected paths
    ohlcv.write_parquet(ohlcv_dir / "test.parquet")
    returns.write_parquet(returns_dir / "test.parquet")

    return tmp_path


@pytest.mark.integration
async def test_end_to_end_backtest_pipeline(workspace_with_momentum_data):
    """Test complete pipeline: submission string → backtest → sharpe ratio score.

    This integration test validates the entire workflow:
    1. Parse submission string
    2. Load test data
    3. Run backtest loop with Time Series API format
    4. Calculate Spearman correlation per iteration
    5. Compute Sharpe ratio
    6. Return MetricScore
    """
    # Create a momentum-based signal submission
    submission = """```python
import polars as pl

def generate_signal(ohlcv: pl.DataFrame, additional_data: dict) -> pl.DataFrame:
    # Simple momentum signal: use 5-day price change
    signal_df = ohlcv.sort(["symbol", "datetime"]).with_columns([
        (pl.col("close") - pl.col("close").shift(5).over("symbol")).alias("signal")
    ])

    # Fill NaN with 0 for early periods
    signal_df = signal_df.with_columns([
        pl.col("signal").fill_null(0.0)
    ])

    return signal_df.select(["datetime", "symbol", "signal"])
```"""

    # Run evaluation
    metric = CorrelationSharpeRatio()
    result = await metric.evaluate(
        user_query="Generate momentum signals",
        submission=submission,
    )

    # Verify result structure
    assert result.metric_name == "CorrelationSharpeRatio"
    assert isinstance(result.score, float)
    assert isinstance(result.evaluator_comment, str)

    # Verify score is a valid Sharpe ratio
    # (can be any float, including negative)
    assert not pl.Series([result.score]).is_null()[0]

    # Verify comment contains expected information (Japanese)
    assert any(keyword in result.evaluator_comment for keyword in ["イテレーション", "相関", "完了", "バックテスト"])


@pytest.fixture
def workspace_with_fundamentals(tmp_path, monkeypatch):
    """Create workspace with OHLCV, returns, and fundamentals data."""
    # Set MIXSEEK_WORKSPACE environment variable
    monkeypatch.setenv("MIXSEEK_WORKSPACE", str(tmp_path))

    # Create directory structure
    ohlcv_dir = tmp_path / "data" / "inputs" / "ohlcv"
    returns_dir = tmp_path / "data" / "inputs" / "returns"
    fundamentals_dir = tmp_path / "data" / "inputs" / "fundamentals"
    configs_dir = tmp_path / "configs"
    ohlcv_dir.mkdir(parents=True)
    returns_dir.mkdir(parents=True)
    fundamentals_dir.mkdir(parents=True)
    configs_dir.mkdir(parents=True)

    # Create OHLCV
    ohlcv = pl.DataFrame(
        {
            "datetime": [datetime(2023, 1, 1), datetime(2023, 1, 1), datetime(2023, 1, 2), datetime(2023, 1, 2)],
            "symbol": ["AAPL", "GOOGL", "AAPL", "GOOGL"],
            "open": [100.0, 200.0, 105.0, 210.0],
            "high": [102.0, 205.0, 108.0, 215.0],
            "low": [99.0, 198.0, 104.0, 208.0],
            "close": [101.0, 203.0, 107.0, 212.0],
            "volume": [1000, 2000, 1500, 2500],
        }
    )

    # Create returns
    returns = pl.DataFrame(
        {
            "datetime": [datetime(2023, 1, 1), datetime(2023, 1, 1), datetime(2023, 1, 2), datetime(2023, 1, 2)],
            "symbol": ["AAPL", "GOOGL", "AAPL", "GOOGL"],
            "return_value": [0.01, 0.02, 0.015, 0.018],
        }
    )

    # Create additional fundamental data
    fundamentals = pl.DataFrame(
        {
            "datetime": [datetime(2023, 1, 1), datetime(2023, 1, 1), datetime(2023, 1, 2), datetime(2023, 1, 2)],
            "symbol": ["AAPL", "GOOGL", "AAPL", "GOOGL"],
            "pe_ratio": [15.0, 20.0, 15.5, 20.5],
        }
    )

    # Save to expected paths
    ohlcv.write_parquet(ohlcv_dir / "test.parquet")
    returns.write_parquet(returns_dir / "test.parquet")
    fundamentals.write_parquet(fundamentals_dir / "test.parquet")

    # Create competition.toml with fundamentals defined
    config_content = """
[competition]
name = "Test Competition"
description = "Test"

[[competition.data]]
name = "ohlcv"
datetime_column = "datetime"

[[competition.data]]
name = "returns"
datetime_column = "datetime"

[[competition.data]]
name = "fundamentals"
datetime_column = "datetime"

[competition.data_split]
train_end = "2022-12-31T23:59:59"
valid_end = "2023-06-30T23:59:59"
"""
    (configs_dir / "competition.toml").write_text(config_content)

    return tmp_path


@pytest.mark.integration
async def test_backtest_with_additional_data(workspace_with_fundamentals):
    """Test backtest pipeline with additional data sources."""
    # Submission that uses additional data
    submission = """```python
import polars as pl

def generate_signal(ohlcv: pl.DataFrame, additional_data: dict) -> pl.DataFrame:
    # Use fundamental data if available
    if "fundamentals" in additional_data:
        fundamentals = additional_data["fundamentals"]
        # Join with fundamentals and use PE ratio as signal
        signal_df = ohlcv.join(fundamentals, on=["datetime", "symbol"], how="left")
        signal_df = signal_df.with_columns([
            pl.col("pe_ratio").fill_null(15.0).alias("signal")
        ])
        return signal_df.select(["datetime", "symbol", "signal"])
    else:
        # Fallback to simple signal
        return ohlcv.select(["datetime", "symbol"]).with_columns([
            pl.lit(1.0).alias("signal")
        ])
```"""

    # Run evaluation with additional data
    metric = CorrelationSharpeRatio()
    result = await metric.evaluate(
        user_query="Generate fundamental signals",
        submission=submission,
    )

    # Verify successful execution
    assert result.metric_name == "CorrelationSharpeRatio"
    assert isinstance(result.score, float)
    assert "完了" in result.evaluator_comment or "バックテスト" in result.evaluator_comment


@pytest.fixture
def workspace_with_poor_signal_data(tmp_path, monkeypatch):
    """Create workspace for testing poor signal handling."""
    # Set MIXSEEK_WORKSPACE environment variable
    monkeypatch.setenv("MIXSEEK_WORKSPACE", str(tmp_path))

    # Create directory structure
    ohlcv_dir = tmp_path / "data" / "inputs" / "ohlcv"
    returns_dir = tmp_path / "data" / "inputs" / "returns"
    ohlcv_dir.mkdir(parents=True)
    returns_dir.mkdir(parents=True)

    # Create data
    ohlcv = pl.DataFrame(
        {
            "datetime": [datetime(2023, 1, i) for i in range(1, 6)] * 2,
            "symbol": ["AAPL"] * 5 + ["GOOGL"] * 5,
            "open": list(range(100, 110)),
            "high": list(range(102, 112)),
            "low": list(range(99, 109)),
            "close": list(range(101, 111)),
            "volume": [1000] * 10,
        }
    )

    # Returns positively correlated with time
    returns = ohlcv.select(["datetime", "symbol"]).with_columns(
        [pl.Series("return_value", [0.01 * i for i in range(10)])]
    )

    # Save
    ohlcv.write_parquet(ohlcv_dir / "test.parquet")
    returns.write_parquet(returns_dir / "test.parquet")

    return tmp_path


@pytest.mark.integration
async def test_backtest_handles_poor_signals(workspace_with_poor_signal_data):
    """Test that backtest correctly handles signals with poor performance."""
    # Submission with anti-correlated signals (should get negative Sharpe)
    submission = """```python
import polars as pl

def generate_signal(ohlcv: pl.DataFrame, additional_data: dict) -> pl.DataFrame:
    # Reverse signal: higher values for earlier dates
    return ohlcv.select(["datetime", "symbol"]).with_columns([
        (10.0 - pl.col("datetime").rank("dense")).alias("signal")
    ])
```"""

    metric = CorrelationSharpeRatio()
    result = await metric.evaluate(
        user_query="Generate signals",
        submission=submission,
    )

    # Should complete successfully even with poor signals
    assert result.metric_name == "CorrelationSharpeRatio"
    assert isinstance(result.score, float)
    # Negative Sharpe ratios are allowed (no normalization)

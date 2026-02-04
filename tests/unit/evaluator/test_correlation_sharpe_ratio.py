"""Unit tests for correlation_sharpe_ratio module."""

from datetime import datetime

import polars as pl
import pytest

from quant_insight.evaluator.correlation_sharpe_ratio import CorrelationSharpeRatio


@pytest.fixture
def temp_workspace(tmp_path, monkeypatch):
    """Create temporary workspace with test data files.

    Sets MIXSEEK_WORKSPACE environment variable to the temporary directory
    and creates the expected directory structure with test data.
    """
    # Set MIXSEEK_WORKSPACE environment variable
    monkeypatch.setenv("MIXSEEK_WORKSPACE", str(tmp_path))

    # Create directory structure
    ohlcv_dir = tmp_path / "data" / "inputs" / "ohlcv"
    returns_dir = tmp_path / "data" / "inputs" / "returns"
    ohlcv_dir.mkdir(parents=True)
    returns_dir.mkdir(parents=True)

    # Create test OHLCV data
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

    # Create test returns data
    returns = pl.DataFrame(
        {
            "datetime": [datetime(2023, 1, 1), datetime(2023, 1, 1), datetime(2023, 1, 2), datetime(2023, 1, 2)],
            "symbol": ["AAPL", "GOOGL", "AAPL", "GOOGL"],
            "return_value": [0.01, 0.015, 0.02, 0.018],
        }
    )

    # Save to expected paths
    ohlcv.write_parquet(ohlcv_dir / "test.parquet")
    returns.write_parquet(returns_dir / "test.parquet")

    return tmp_path


@pytest.mark.unit
async def test_correlation_sharpe_ratio_returns_metric_score(temp_workspace):
    """Test that evaluate() returns a MetricScore."""
    submission = """```python
import polars as pl

def generate_signal(ohlcv: pl.DataFrame, additional_data: dict) -> pl.DataFrame:
    return ohlcv.select(["datetime", "symbol"]).with_columns(
        pl.when(pl.col("symbol") == "AAPL").then(1.0).otherwise(2.0).alias("signal")
    )
```"""

    metric = CorrelationSharpeRatio()
    result = await metric.evaluate(
        user_query="Generate signals",  # Ignored in backtest
        submission=submission,
    )

    assert result.metric_name == "CorrelationSharpeRatio"
    assert isinstance(result.score, float)
    assert isinstance(result.evaluator_comment, str)


@pytest.mark.unit
async def test_sharpe_ratio_used_as_score_no_normalization(temp_workspace):
    """Test that sharpe_ratio is used directly as score without normalization."""
    submission = """```python
import polars as pl

def generate_signal(ohlcv: pl.DataFrame, additional_data: dict) -> pl.DataFrame:
    return ohlcv.select(["datetime", "symbol"]).with_columns(
        pl.when(pl.col("symbol") == "AAPL").then(1.0).otherwise(2.0).alias("signal")
    )
```"""

    metric = CorrelationSharpeRatio()
    result = await metric.evaluate(
        user_query="Generate signals",
        submission=submission,
    )

    # Sharpe ratio can be any float, including negative values
    assert isinstance(result.score, float)
    # Score should be the sharpe ratio directly (not normalized to 0-100)


@pytest.mark.unit
async def test_failed_submission_returns_invalid_score(temp_workspace):
    """Test that failed submissions return INVALID_SUBMISSION_SCORE (-100.0)."""
    submission = """```python
import polars as pl

def generate_signal(ohlcv: pl.DataFrame, additional_data: dict) -> pl.DataFrame:
    raise ValueError("Intentional failure")
```"""

    metric = CorrelationSharpeRatio()
    result = await metric.evaluate(
        user_query="Generate signals",
        submission=submission,
    )

    assert result.score == CorrelationSharpeRatio.INVALID_SUBMISSION_SCORE
    # 日本語エラーメッセージのチェック
    assert "無効な評価結果" in result.evaluator_comment or "エラー" in result.evaluator_comment


@pytest.mark.unit
async def test_invalid_submission_returns_invalid_score(temp_workspace):
    """Test that invalid submissions (parse error) return INVALID_SUBMISSION_SCORE."""
    submission = """```python
def wrong_function_name(ohlcv, additional_data):
    pass
```"""

    metric = CorrelationSharpeRatio()
    result = await metric.evaluate(
        user_query="Generate signals",
        submission=submission,
    )

    assert result.score == CorrelationSharpeRatio.INVALID_SUBMISSION_SCORE
    # 日本語エラーメッセージのチェック
    assert "パースエラー" in result.evaluator_comment or "無効な評価結果" in result.evaluator_comment


@pytest.mark.unit
async def test_syntax_error_returns_invalid_score(temp_workspace):
    """Test that submissions with syntax errors return INVALID_SUBMISSION_SCORE."""
    submission = """```python
def generate_signal(ohlcv, additional_data)
    # Missing colon
    pass
```"""

    metric = CorrelationSharpeRatio()
    result = await metric.evaluate(
        user_query="Generate signals",
        submission=submission,
    )

    assert result.score == CorrelationSharpeRatio.INVALID_SUBMISSION_SCORE
    # 日本語エラーメッセージのチェック
    assert "構文エラー" in result.evaluator_comment or "無効な評価結果" in result.evaluator_comment


@pytest.mark.unit
async def test_evaluator_comment_includes_statistics(temp_workspace):
    """Test that evaluator_comment includes iteration statistics."""
    submission = """```python
import polars as pl
from typing import Dict

def generate_signal(ohlcv: pl.DataFrame, additional_data: Dict[str, pl.DataFrame]) -> pl.DataFrame:
    return ohlcv.select(["datetime", "symbol"]).with_columns(
        pl.when(pl.col("symbol") == "AAPL").then(1.0).otherwise(2.0).alias("signal")
    )
```"""

    metric = CorrelationSharpeRatio()
    result = await metric.evaluate(
        user_query="Generate signals",
        submission=submission,
    )

    comment = result.evaluator_comment
    # 日本語コメントのチェック
    assert "イテレーション" in comment or "バックテスト完了" in comment
    assert "有効" in comment or "完了" in comment


@pytest.mark.unit
async def test_user_query_is_ignored(temp_workspace):
    """Test that user_query parameter is ignored in backtest evaluation."""
    submission = """```python
import polars as pl
from typing import Dict

def generate_signal(ohlcv: pl.DataFrame, additional_data: Dict[str, pl.DataFrame]) -> pl.DataFrame:
    return ohlcv.select(["datetime", "symbol"]).with_columns(
        pl.when(pl.col("symbol") == "AAPL").then(1.0).otherwise(2.0).alias("signal")
    )
```"""

    metric = CorrelationSharpeRatio()

    # Run with different user queries - should produce same results
    result1 = await metric.evaluate(
        user_query="Query 1",
        submission=submission,
    )

    # Create a new metric instance to reset cached data paths
    metric2 = CorrelationSharpeRatio()
    result2 = await metric2.evaluate(
        user_query="Completely different query",
        submission=submission,
    )

    # Results should be identical (user_query is ignored)
    assert result1.score == result2.score
    assert result1.evaluator_comment == result2.evaluator_comment

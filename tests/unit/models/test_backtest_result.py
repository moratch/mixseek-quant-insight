"""Unit tests for backtest result models."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from quant_insight.models.backtest_result import BacktestResult, IterationResult


class TestIterationResult:
    """Test IterationResult model."""

    def test_valid_iteration_with_correlation(self) -> None:
        """Test IterationResult with valid correlation."""
        result = IterationResult(
            datetime=datetime(2023, 1, 1),
            rank_correlation=0.5,
        )
        assert result.datetime == datetime(2023, 1, 1)
        assert result.rank_correlation == 0.5
        assert result.error_message is None

    def test_iteration_with_nan_correlation(self) -> None:
        """Test IterationResult with None correlation (NaN)."""
        result = IterationResult(
            datetime=datetime(2023, 1, 1),
            rank_correlation=None,
        )
        assert result.rank_correlation is None

    def test_iteration_with_error(self) -> None:
        """Test IterationResult with error message."""
        result = IterationResult(
            datetime=datetime(2023, 1, 1),
            rank_correlation=None,
            error_message="Division by zero",
        )
        assert result.error_message == "Division by zero"


class TestBacktestResultValidation:
    """Test BacktestResult validation."""

    def test_valid_completed_result(self) -> None:
        """Test valid completed BacktestResult."""
        result = BacktestResult(
            status="completed",
            iteration_results=[
                IterationResult(datetime=datetime(2023, 1, 1), rank_correlation=0.5),
                IterationResult(datetime=datetime(2023, 1, 2), rank_correlation=0.6),
            ],
            sharpe_ratio=2.5,
            total_iterations=2,
            valid_iterations=2,
            mean_correlation=0.55,
            std_correlation=0.05,
        )
        assert result.status == "completed"
        assert result.sharpe_ratio == 2.5
        assert result.is_valid is True

    def test_valid_failed_result(self) -> None:
        """Test valid failed BacktestResult."""
        result = BacktestResult(
            status="failed",
            error_message="Submission raised exception",
            iteration_results=[],
            sharpe_ratio=0.0,
            total_iterations=0,
            valid_iterations=0,
        )
        assert result.status == "failed"
        assert result.error_message == "Submission raised exception"
        assert result.is_valid is False

    def test_status_pattern_validation(self) -> None:
        """Test status must be 'completed' or 'failed'."""
        with pytest.raises(ValidationError) as exc_info:
            BacktestResult(
                status="invalid_status",
                total_iterations=0,
                valid_iterations=0,
            )
        assert "String should match pattern" in str(exc_info.value)

    def test_default_values(self) -> None:
        """Test default values for BacktestResult."""
        result = BacktestResult(
            total_iterations=0,
            valid_iterations=0,
        )
        assert result.status == "completed"
        assert result.error_message is None
        assert result.iteration_results == []
        assert result.sharpe_ratio == 0.0
        assert result.mean_correlation is None
        assert result.std_correlation is None


class TestBacktestResultIsValid:
    """Test BacktestResult.is_valid computed field."""

    def test_is_valid_when_completed_with_valid_iterations(self) -> None:
        """Test is_valid returns True when completed with valid_iterations > 0."""
        result = BacktestResult(
            status="completed",
            total_iterations=5,
            valid_iterations=5,
        )
        assert result.is_valid is True

    def test_is_not_valid_when_failed(self) -> None:
        """Test is_valid returns False when status is failed."""
        result = BacktestResult(
            status="failed",
            total_iterations=0,
            valid_iterations=0,
        )
        assert result.is_valid is False

    def test_is_not_valid_when_zero_valid_iterations(self) -> None:
        """Test is_valid returns False when valid_iterations is 0."""
        result = BacktestResult(
            status="completed",
            total_iterations=5,
            valid_iterations=0,
        )
        assert result.is_valid is False

    def test_is_not_valid_when_completed_but_all_nan(self) -> None:
        """Test is_valid returns False when all iterations are NaN."""
        result = BacktestResult(
            status="completed",
            iteration_results=[
                IterationResult(datetime=datetime(2023, 1, 1), rank_correlation=None),
                IterationResult(datetime=datetime(2023, 1, 2), rank_correlation=None),
            ],
            sharpe_ratio=0.0,
            total_iterations=2,
            valid_iterations=0,
        )
        assert result.is_valid is False


class TestBacktestResultSerialization:
    """Test BacktestResult serialization."""

    def test_model_dump(self) -> None:
        """Test model_dump returns correct dict."""
        result = BacktestResult(
            status="completed",
            iteration_results=[
                IterationResult(datetime=datetime(2023, 1, 1), rank_correlation=0.5),
            ],
            sharpe_ratio=1.5,
            total_iterations=1,
            valid_iterations=1,
            mean_correlation=0.5,
            std_correlation=0.0,
            evaluation_started_at=datetime(2023, 1, 1, 10, 0, 0),
            evaluation_completed_at=datetime(2023, 1, 1, 10, 5, 0),
        )
        data = result.model_dump()
        assert data["status"] == "completed"
        assert data["sharpe_ratio"] == 1.5
        assert data["is_valid"] is True
        assert len(data["iteration_results"]) == 1

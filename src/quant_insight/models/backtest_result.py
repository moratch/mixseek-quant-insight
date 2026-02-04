"""Backtest result models for quant-insight."""

from datetime import datetime as dt

from pydantic import BaseModel, Field, computed_field


class IterationResult(BaseModel):
    """Result for a single iteration in the backtest (FR-004).

    Each iteration represents one time step in the Time Series API backtest loop.
    """

    datetime: dt = Field(
        ...,
        description="Iteration datetime",
    )
    rank_correlation: float | None = Field(
        ...,
        description="Spearman rank correlation for this iteration. None if NaN.",
    )
    error_message: str | None = Field(
        default=None,
        description="Error message if iteration failed",
    )


class BacktestResult(BaseModel):
    """Backtest execution result (FR-004, FR-005).

    Contains all iteration results and aggregate metrics.
    """

    status: str = Field(
        default="completed",
        pattern=r"^(completed|failed)$",
        description="Backtest status: completed (normal) or failed (submission disqualified)",
    )
    error_message: str | None = Field(
        default=None,
        description="Error message if status is failed",
    )
    iteration_results: list[IterationResult] = Field(
        default_factory=list,
        description="Results for each iteration (partial results if failed)",
    )
    sharpe_ratio: float = Field(
        default=0.0,
        description="Sharpe ratio of rank correlation series (0.0 if failed)",
    )
    total_iterations: int = Field(
        ...,
        ge=0,
        description="Total number of iterations",
    )
    valid_iterations: int = Field(
        ...,
        ge=0,
        description="Number of valid iterations (excluding NaN)",
    )
    mean_correlation: float | None = Field(
        default=None,
        description="Mean of rank correlations",
    )
    std_correlation: float | None = Field(
        default=None,
        description="Standard deviation of rank correlations",
    )

    # Metadata
    evaluation_started_at: dt | None = Field(
        default=None,
        description="Evaluation start datetime",
    )
    evaluation_completed_at: dt | None = Field(
        default=None,
        description="Evaluation completion datetime",
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_valid(self) -> bool:
        """Check if backtest result is valid.

        Returns:
            True if status is completed and has at least one valid iteration
        """
        return self.status == "completed" and self.valid_iterations > 0

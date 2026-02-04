"""Exception classes for quant-insight."""

from datetime import datetime


class QuantInsightError(Exception):
    """Base exception for quant-insight."""


# Submission-related exceptions


class SubmissionError(QuantInsightError):
    """Base exception for submission-related errors."""


class SubmissionFailedError(SubmissionError):
    """Exception raised when submission execution fails during backtest."""

    def __init__(
        self,
        message: str,
        iteration: int | None = None,
        datetime: datetime | None = None,
    ) -> None:
        """SubmissionFailedErrorを初期化します。

        Args:
            message: エラーメッセージ
            iteration: エラーが発生したイテレーション番号
            datetime: エラーが発生した日時
        """
        self.iteration = iteration
        self.datetime = datetime
        super().__init__(message)


class SubmissionInvalidError(SubmissionError):
    """Exception raised when submission format is invalid."""


# Data build-related exceptions


class DataBuildError(QuantInsightError):
    """Base exception for data build-related errors."""


class AuthenticationError(DataBuildError):
    """Exception raised when API authentication fails."""


class DataFetchError(DataBuildError):
    """Exception raised when data fetching fails."""


class DataSplitError(DataBuildError):
    """Exception raised when data splitting fails."""

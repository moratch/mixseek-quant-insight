"""Unit tests for quant_insight exception hierarchy."""

from datetime import datetime

import pytest

from quant_insight.exceptions import (
    AuthenticationError,
    DataBuildError,
    DataFetchError,
    DataSplitError,
    QuantInsightError,
    SubmissionError,
    SubmissionFailedError,
    SubmissionInvalidError,
)


class TestExceptionHierarchy:
    """Test exception class hierarchy and relationships."""

    def test_base_exception(self) -> None:
        """Test QuantInsightError base exception."""
        exc = QuantInsightError("test error")
        assert str(exc) == "test error"
        assert isinstance(exc, Exception)

    def test_submission_error_hierarchy(self) -> None:
        """Test SubmissionError hierarchy."""
        exc = SubmissionError("submission error")
        assert isinstance(exc, QuantInsightError)
        assert str(exc) == "submission error"

    def test_submission_failed_error(self) -> None:
        """Test SubmissionFailedError with metadata."""
        test_datetime = datetime(2024, 1, 1, 12, 0, 0)
        exc = SubmissionFailedError("execution failed", iteration=5, datetime=test_datetime)
        assert isinstance(exc, SubmissionError)
        assert isinstance(exc, QuantInsightError)
        assert str(exc) == "execution failed"
        assert exc.iteration == 5
        assert exc.datetime == test_datetime

    def test_submission_failed_error_without_metadata(self) -> None:
        """Test SubmissionFailedError without optional metadata."""
        exc = SubmissionFailedError("execution failed")
        assert isinstance(exc, SubmissionError)
        assert exc.iteration is None
        assert exc.datetime is None

    def test_submission_invalid_error(self) -> None:
        """Test SubmissionInvalidError."""
        exc = SubmissionInvalidError("invalid format")
        assert isinstance(exc, SubmissionError)
        assert isinstance(exc, QuantInsightError)
        assert str(exc) == "invalid format"

    def test_data_build_error_hierarchy(self) -> None:
        """Test DataBuildError hierarchy."""
        exc = DataBuildError("data build error")
        assert isinstance(exc, QuantInsightError)
        assert str(exc) == "data build error"

    def test_authentication_error(self) -> None:
        """Test AuthenticationError."""
        exc = AuthenticationError("auth failed")
        assert isinstance(exc, DataBuildError)
        assert isinstance(exc, QuantInsightError)
        assert str(exc) == "auth failed"

    def test_data_fetch_error(self) -> None:
        """Test DataFetchError."""
        exc = DataFetchError("fetch failed")
        assert isinstance(exc, DataBuildError)
        assert isinstance(exc, QuantInsightError)
        assert str(exc) == "fetch failed"

    def test_data_split_error(self) -> None:
        """Test DataSplitError."""
        exc = DataSplitError("split failed")
        assert isinstance(exc, DataBuildError)
        assert isinstance(exc, QuantInsightError)
        assert str(exc) == "split failed"


class TestExceptionRaising:
    """Test exception raising and catching patterns."""

    def test_catch_submission_error_base(self) -> None:
        """Test catching SubmissionError catches all submission exceptions."""
        with pytest.raises(SubmissionError):
            raise SubmissionFailedError("test")

        with pytest.raises(SubmissionError):
            raise SubmissionInvalidError("test")

    def test_catch_data_build_error_base(self) -> None:
        """Test catching DataBuildError catches all data build exceptions."""
        with pytest.raises(DataBuildError):
            raise AuthenticationError("test")

        with pytest.raises(DataBuildError):
            raise DataFetchError("test")

        with pytest.raises(DataBuildError):
            raise DataSplitError("test")

    def test_catch_quant_insight_error_base(self) -> None:
        """Test catching QuantInsightError catches all custom exceptions."""
        with pytest.raises(QuantInsightError):
            raise SubmissionError("test")

        with pytest.raises(QuantInsightError):
            raise DataBuildError("test")

        with pytest.raises(QuantInsightError):
            raise SubmissionFailedError("test")

        with pytest.raises(QuantInsightError):
            raise AuthenticationError("test")

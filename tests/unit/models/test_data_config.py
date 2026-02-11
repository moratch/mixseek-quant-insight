"""Unit tests for DataConfig model."""

import pytest
from pydantic import ValidationError

from quant_insight.models.data_config import DataConfig


class TestDataConfigValidation:
    """Test DataConfig validation."""

    def test_valid_config(self) -> None:
        """Test valid DataConfig."""
        config = DataConfig(name="ohlcv", datetime_column="datetime")
        assert config.name == "ohlcv"
        assert config.datetime_column == "datetime"

    def test_default_datetime_column(self) -> None:
        """Test default datetime_column is 'datetime'."""
        config = DataConfig(name="returns")
        assert config.name == "returns"
        assert config.datetime_column == "datetime"

    def test_name_required(self) -> None:
        """Test name is required."""
        with pytest.raises(ValidationError) as exc_info:
            DataConfig()  # type: ignore
        assert "name" in str(exc_info.value).lower()

    def test_custom_datetime_column(self) -> None:
        """Test custom datetime_column."""
        config = DataConfig(name="fundamentals", datetime_column="timestamp")
        assert config.datetime_column == "timestamp"


class TestDataConfigSerialization:
    """Test DataConfig serialization."""

    def test_model_dump(self) -> None:
        """Test model_dump returns correct dict."""
        config = DataConfig(name="sentiment", datetime_column="date")
        data = config.model_dump()
        assert data == {"name": "sentiment", "datetime_column": "date", "required": True}

    def test_model_dump_with_defaults(self) -> None:
        """Test model_dump with default values."""
        config = DataConfig(name="ohlcv")
        data = config.model_dump()
        assert data == {"name": "ohlcv", "datetime_column": "datetime", "required": True}

    def test_model_dump_optional_dataset(self) -> None:
        """Test model_dump for optional dataset."""
        config = DataConfig(name="indicators", required=False)
        data = config.model_dump()
        assert data == {"name": "indicators", "datetime_column": "datetime", "required": False}

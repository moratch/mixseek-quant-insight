"""Unit tests for DataSplitConfig model."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from quant_insight.models.data_split_config import DataSplitConfig


class TestDataSplitConfigValidation:
    """Test DataSplitConfig validation."""

    def test_valid_config(self) -> None:
        """Test valid DataSplitConfig."""
        config = DataSplitConfig(
            train_end=datetime(2023, 1, 31),
            valid_end=datetime(2023, 6, 30),
            purge_rows=1,
        )
        assert config.train_end == datetime(2023, 1, 31)
        assert config.valid_end == datetime(2023, 6, 30)
        assert config.purge_rows == 1

    def test_default_purge_rows(self) -> None:
        """Test default purge_rows is 0."""
        config = DataSplitConfig(
            train_end=datetime(2023, 1, 31),
            valid_end=datetime(2023, 6, 30),
        )
        assert config.purge_rows == 0

    def test_train_end_before_valid_end(self) -> None:
        """Test train_end must be before valid_end."""
        with pytest.raises(ValidationError) as exc_info:
            DataSplitConfig(
                train_end=datetime(2023, 6, 30),
                valid_end=datetime(2023, 1, 31),
            )
        assert "train_end must be before valid_end" in str(exc_info.value)

    def test_train_end_equal_valid_end(self) -> None:
        """Test train_end cannot equal valid_end."""
        with pytest.raises(ValidationError) as exc_info:
            DataSplitConfig(
                train_end=datetime(2023, 6, 30),
                valid_end=datetime(2023, 6, 30),
            )
        assert "train_end must be before valid_end" in str(exc_info.value)

    def test_purge_rows_must_be_non_negative(self) -> None:
        """Test purge_rows must be >= 0."""
        with pytest.raises(ValidationError) as exc_info:
            DataSplitConfig(
                train_end=datetime(2023, 1, 31),
                valid_end=datetime(2023, 6, 30),
                purge_rows=-1,
            )
        assert "greater than or equal to 0" in str(exc_info.value)

    def test_purge_rows_zero_allowed(self) -> None:
        """Test purge_rows can be 0."""
        config = DataSplitConfig(
            train_end=datetime(2023, 1, 31),
            valid_end=datetime(2023, 6, 30),
            purge_rows=0,
        )
        assert config.purge_rows == 0


class TestDataSplitConfigSerialization:
    """Test DataSplitConfig serialization."""

    def test_model_dump(self) -> None:
        """Test model_dump returns correct dict."""
        config = DataSplitConfig(
            train_end=datetime(2023, 1, 31, 12, 0, 0),
            valid_end=datetime(2023, 6, 30, 12, 0, 0),
            purge_rows=2,
        )
        data = config.model_dump()
        assert data["train_end"] == datetime(2023, 1, 31, 12, 0, 0)
        assert data["valid_end"] == datetime(2023, 6, 30, 12, 0, 0)
        assert data["purge_rows"] == 2

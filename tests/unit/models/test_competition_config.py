"""Unit tests for CompetitionConfig model."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from quant_insight.models.competition_config import CompetitionConfig
from quant_insight.models.data_config import DataConfig
from quant_insight.models.data_split_config import DataSplitConfig
from quant_insight.models.return_definition import ReturnDefinition


class TestCompetitionConfigValidation:
    """Test CompetitionConfig validation."""

    def test_valid_config_minimal(self) -> None:
        """Test valid CompetitionConfig with minimal required datasets."""
        config = CompetitionConfig(
            name="Test Competition",
            data=[
                DataConfig(name="ohlcv"),
                DataConfig(name="returns"),
            ],
            data_split=DataSplitConfig(
                train_end=datetime(2023, 1, 31),
                valid_end=datetime(2023, 6, 30),
            ),
        )
        assert config.name == "Test Competition"
        assert len(config.data) == 2
        assert config.data[0].name == "ohlcv"
        assert config.data[1].name == "returns"

    def test_valid_config_with_additional_data(self) -> None:
        """Test CompetitionConfig with additional datasets."""
        config = CompetitionConfig(
            name="Test Competition",
            description="Test description",
            data=[
                DataConfig(name="ohlcv"),
                DataConfig(name="returns"),
                DataConfig(name="fundamentals"),
                DataConfig(name="sentiment"),
            ],
            data_split=DataSplitConfig(
                train_end=datetime(2023, 1, 31),
                valid_end=datetime(2023, 6, 30),
                purge_rows=1,
            ),
            return_definition=ReturnDefinition(window=5, method="open2close"),
        )
        assert len(config.data) == 4
        assert config.return_definition.window == 5

    def test_default_return_definition(self) -> None:
        """Test default ReturnDefinition is created."""
        config = CompetitionConfig(
            name="Test",
            data=[
                DataConfig(name="ohlcv"),
                DataConfig(name="returns"),
            ],
            data_split=DataSplitConfig(
                train_end=datetime(2023, 1, 31),
                valid_end=datetime(2023, 6, 30),
            ),
        )
        assert config.return_definition.window == 1
        assert config.return_definition.method == "close2close"

    def test_missing_ohlcv_dataset(self) -> None:
        """Test validation fails when ohlcv dataset is missing."""
        with pytest.raises(ValidationError) as exc_info:
            CompetitionConfig(
                name="Test",
                data=[
                    DataConfig(name="returns"),
                    DataConfig(name="sentiment"),
                ],
                data_split=DataSplitConfig(
                    train_end=datetime(2023, 1, 31),
                    valid_end=datetime(2023, 6, 30),
                ),
            )
        assert "ohlcv dataset is required" in str(exc_info.value)

    def test_missing_returns_dataset(self) -> None:
        """Test validation fails when returns dataset is missing."""
        with pytest.raises(ValidationError) as exc_info:
            CompetitionConfig(
                name="Test",
                data=[
                    DataConfig(name="ohlcv"),
                    DataConfig(name="sentiment"),
                ],
                data_split=DataSplitConfig(
                    train_end=datetime(2023, 1, 31),
                    valid_end=datetime(2023, 6, 30),
                ),
            )
        assert "returns dataset is required" in str(exc_info.value)

    def test_data_min_length_validation(self) -> None:
        """Test data list must have at least 2 items."""
        with pytest.raises(ValidationError) as exc_info:
            CompetitionConfig(
                name="Test",
                data=[
                    DataConfig(name="ohlcv"),
                ],
                data_split=DataSplitConfig(
                    train_end=datetime(2023, 1, 31),
                    valid_end=datetime(2023, 6, 30),
                ),
            )
        # Will fail on min_length constraint
        assert "at least 2 items" in str(exc_info.value)


class TestCompetitionConfigIntegration:
    """Test CompetitionConfig integration with nested models."""

    def test_data_split_validation_propagates(self) -> None:
        """Test DataSplitConfig validation (train_end < valid_end) propagates."""
        with pytest.raises(ValidationError) as exc_info:
            CompetitionConfig(
                name="Test",
                data=[
                    DataConfig(name="ohlcv"),
                    DataConfig(name="returns"),
                ],
                data_split=DataSplitConfig(
                    train_end=datetime(2023, 6, 30),
                    valid_end=datetime(2023, 1, 31),  # Invalid: before train_end
                ),
            )
        assert "train_end must be before valid_end" in str(exc_info.value)

    def test_return_definition_validation_propagates(self) -> None:
        """Test ReturnDefinition validation propagates."""
        with pytest.raises(ValidationError) as exc_info:
            CompetitionConfig(
                name="Test",
                data=[
                    DataConfig(name="ohlcv"),
                    DataConfig(name="returns"),
                ],
                data_split=DataSplitConfig(
                    train_end=datetime(2023, 1, 31),
                    valid_end=datetime(2023, 6, 30),
                ),
                return_definition=ReturnDefinition(window=0),  # Invalid
            )
        assert "greater than or equal to 1" in str(exc_info.value)


class TestCompetitionConfigSerialization:
    """Test CompetitionConfig serialization."""

    def test_model_dump(self) -> None:
        """Test model_dump returns correct structure."""
        config = CompetitionConfig(
            name="Test Competition",
            description="Test desc",
            data=[
                DataConfig(name="ohlcv"),
                DataConfig(name="returns"),
            ],
            data_split=DataSplitConfig(
                train_end=datetime(2023, 1, 31),
                valid_end=datetime(2023, 6, 30),
                purge_rows=1,
            ),
            return_definition=ReturnDefinition(window=5, method="open2close"),
        )
        data = config.model_dump()
        assert data["name"] == "Test Competition"
        assert data["description"] == "Test desc"
        assert len(data["data"]) == 2
        assert data["data_split"]["purge_rows"] == 1
        assert data["return_definition"]["window"] == 5

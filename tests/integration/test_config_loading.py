"""Integration test for loading competition config from TOML."""

import tomllib
from datetime import datetime
from pathlib import Path

import pytest

from quant_insight.models import CompetitionConfig


class TestConfigLoading:
    """Test loading CompetitionConfig from TOML file."""

    @pytest.fixture
    def competition_toml_path(self) -> Path:
        """Path to test competition.toml fixture."""
        return Path(__file__).parent.parent / "fixtures" / "competition.toml"

    def test_load_competition_config_from_toml(
        self,
        competition_toml_path: Path,
    ) -> None:
        """Test loading CompetitionConfig from TOML file."""
        # Read TOML file
        with open(competition_toml_path, "rb") as f:
            config_dict = tomllib.load(f)

        # Parse into CompetitionConfig
        config = CompetitionConfig(**config_dict["competition"])

        # Verify basic fields
        assert config.name == "Test Quant Competition"
        assert config.description == "Test competition for unit testing"

        # Verify datasets
        assert len(config.data) == 3
        dataset_names = {d.name for d in config.data}
        assert "ohlcv" in dataset_names
        assert "returns" in dataset_names
        assert "fundamentals" in dataset_names

        # Verify data split
        assert config.data_split.train_end == datetime(2023, 1, 31, 23, 59, 59)
        assert config.data_split.valid_end == datetime(2023, 6, 30, 23, 59, 59)
        assert config.data_split.purge_rows == 1

        # Verify return definition
        assert config.return_definition.window == 5
        assert config.return_definition.method == "open2close"

    def test_competition_config_validates_required_datasets(
        self,
        competition_toml_path: Path,
    ) -> None:
        """Test that CompetitionConfig validates required datasets."""
        with open(competition_toml_path, "rb") as f:
            config_dict = tomllib.load(f)

        # This should succeed because config has ohlcv and returns
        config = CompetitionConfig(**config_dict["competition"])
        assert config is not None

    def test_competition_config_model_dump_roundtrip(
        self,
        competition_toml_path: Path,
    ) -> None:
        """Test that CompetitionConfig can be dumped and reloaded."""
        # Load from TOML
        with open(competition_toml_path, "rb") as f:
            config_dict = tomllib.load(f)

        # Parse into model
        config1 = CompetitionConfig(**config_dict["competition"])

        # Dump and reload
        data = config1.model_dump()
        config2 = CompetitionConfig(**data)

        # Verify equality
        assert config2.name == config1.name
        assert config2.data_split.train_end == config1.data_split.train_end
        assert config2.return_definition.window == config1.return_definition.window

    def test_competition_config_daytrade(self, tmp_path: Path) -> None:
        """Test loading daytrade_market + window=1 from TOML."""
        toml_content = """\
[competition]
name = "daytrade-test"
description = "Day-trade competition"

[[competition.data]]
name = "ohlcv"
datetime_column = "datetime"

[[competition.data]]
name = "returns"
datetime_column = "datetime"

[competition.data_split]
train_end = "2024-12-31T23:59:59"
valid_end = "2025-06-30T23:59:59"
purge_rows = 1

[competition.return_definition]
method = "daytrade_market"
window = 1
"""
        toml_file = tmp_path / "competition_daytrade.toml"
        toml_file.write_text(toml_content, encoding="utf-8")

        with open(toml_file, "rb") as f:
            config_dict = tomllib.load(f)

        config = CompetitionConfig(**config_dict["competition"])
        assert config.return_definition.method == "daytrade_market"
        assert config.return_definition.window == 1

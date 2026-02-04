"""Configuration loading utilities for quant-insight."""

import tomllib
from pathlib import Path

from quant_insight.models.competition_config import CompetitionConfig
from quant_insight.utils.env import get_workspace


def load_competition_config(config_path: Path | None = None) -> CompetitionConfig | None:
    """Load CompetitionConfig from TOML file.

    Args:
        config_path: Optional path to competition.toml.
            If None, uses default path: $MIXSEEK_WORKSPACE/configs/competition.toml

    Returns:
        CompetitionConfig if file exists and is valid, None otherwise.

    Note:
        This function does not raise exceptions for missing files or invalid config.
        It returns None to allow callers to fall back to default behavior.
    """
    if config_path is None:
        try:
            workspace = get_workspace()
            config_path = workspace / "configs" / "competition.toml"
        except ValueError:
            # MIXSEEK_WORKSPACE not set
            return None

    if not config_path.exists():
        return None

    try:
        with open(config_path, "rb") as f:
            data = tomllib.load(f)

        # Extract competition section
        competition_data = data.get("competition")
        if competition_data is None:
            return None

        return CompetitionConfig.model_validate(competition_data)
    except Exception:
        # Invalid TOML or schema mismatch
        return None


def get_test_data_paths(
    config: CompetitionConfig | None = None,
) -> tuple[Path, Path, dict[str, Path]]:
    """Get test data paths from config or default locations.

    Args:
        config: Optional CompetitionConfig. If None, attempts to load from default location.

    Returns:
        Tuple of (ohlcv_path, returns_path, additional_data_paths)

    Raises:
        ValueError: If MIXSEEK_WORKSPACE is not set.
    """
    workspace = get_workspace()
    data_dir = workspace / "data" / "inputs"

    # Default paths
    ohlcv_path = data_dir / "ohlcv" / "test.parquet"
    returns_path = data_dir / "returns" / "test.parquet"
    additional_data_paths: dict[str, Path] = {}

    # If config is provided, extract additional data names
    if config is None:
        config = load_competition_config()

    if config is not None:
        # Build additional data paths from config
        # ohlcv and returns are always present, additional data is everything else
        for data_config in config.data:
            if data_config.name not in ("ohlcv", "returns"):
                additional_data_paths[data_config.name] = data_dir / data_config.name / "test.parquet"

    return ohlcv_path, returns_path, additional_data_paths

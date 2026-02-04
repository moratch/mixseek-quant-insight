"""Environment variable utilities for quant-insight."""

import os
from pathlib import Path


def get_workspace() -> Path:
    """Get workspace directory from MIXSEEK_WORKSPACE environment variable.

    Returns:
        Path to workspace directory

    Raises:
        ValueError: If MIXSEEK_WORKSPACE environment variable is not set or empty
    """
    workspace = os.environ.get("MIXSEEK_WORKSPACE", "").strip()
    if not workspace:
        raise ValueError("MIXSEEK_WORKSPACE environment variable is not set")
    return Path(workspace)


def get_data_inputs_dir() -> Path:
    """Get data inputs directory path.

    Returns:
        Path to {workspace}/data/inputs/ directory

    Raises:
        ValueError: If MIXSEEK_WORKSPACE environment variable is not set or empty
    """
    workspace = get_workspace()
    return workspace / "data" / "inputs"


def get_raw_data_dir() -> Path:
    """Get raw data directory path.

    Returns:
        Path to {workspace}/data/inputs/raw/ directory

    Raises:
        ValueError: If MIXSEEK_WORKSPACE environment variable is not set or empty
    """
    workspace = get_workspace()
    return workspace / "data" / "inputs" / "raw"

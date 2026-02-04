"""Unit tests for environment utilities."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from quant_insight.utils.env import get_data_inputs_dir, get_raw_data_dir, get_workspace


class TestGetWorkspace:
    """Test get_workspace function."""

    def test_get_workspace_success(self) -> None:
        """Test get_workspace returns Path when MIXSEEK_WORKSPACE is set."""
        with patch.dict(os.environ, {"MIXSEEK_WORKSPACE": "/path/to/workspace"}):
            result = get_workspace()
            assert result == Path("/path/to/workspace")
            assert isinstance(result, Path)

    def test_get_workspace_missing_env_var(self) -> None:
        """Test get_workspace raises ValueError when MIXSEEK_WORKSPACE is not set."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="MIXSEEK_WORKSPACE environment variable is not set"):
                get_workspace()

    def test_get_workspace_empty_env_var(self) -> None:
        """Test get_workspace raises ValueError when MIXSEEK_WORKSPACE is empty."""
        with patch.dict(os.environ, {"MIXSEEK_WORKSPACE": ""}):
            with pytest.raises(ValueError, match="MIXSEEK_WORKSPACE environment variable is not set"):
                get_workspace()


class TestGetDataInputsDir:
    """Test get_data_inputs_dir function."""

    def test_get_data_inputs_dir_success(self) -> None:
        """Test get_data_inputs_dir returns correct path."""
        with patch.dict(os.environ, {"MIXSEEK_WORKSPACE": "/workspace"}):
            result = get_data_inputs_dir()
            assert result == Path("/workspace/data/inputs")
            assert isinstance(result, Path)

    def test_get_data_inputs_dir_missing_workspace(self) -> None:
        """Test get_data_inputs_dir raises ValueError when workspace not set."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="MIXSEEK_WORKSPACE environment variable is not set"):
                get_data_inputs_dir()


class TestGetRawDataDir:
    """Test get_raw_data_dir function."""

    def test_get_raw_data_dir_success(self) -> None:
        """Test get_raw_data_dir returns correct path."""
        with patch.dict(os.environ, {"MIXSEEK_WORKSPACE": "/workspace"}):
            result = get_raw_data_dir()
            assert result == Path("/workspace/data/inputs/raw")
            assert isinstance(result, Path)

    def test_get_raw_data_dir_missing_workspace(self) -> None:
        """Test get_raw_data_dir raises ValueError when workspace not set."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="MIXSEEK_WORKSPACE environment variable is not set"):
                get_raw_data_dir()


class TestPathHierarchy:
    """Test path hierarchy consistency."""

    def test_path_hierarchy(self) -> None:
        """Test that path hierarchy matches plan.md structure."""
        with patch.dict(os.environ, {"MIXSEEK_WORKSPACE": "/workspace"}):
            workspace = get_workspace()
            data_inputs = get_data_inputs_dir()
            raw_data = get_raw_data_dir()

            # Verify hierarchy
            assert data_inputs == workspace / "data" / "inputs"
            assert raw_data == workspace / "data" / "inputs" / "raw"
            assert raw_data.parent == data_inputs
            assert data_inputs.parent.parent == workspace

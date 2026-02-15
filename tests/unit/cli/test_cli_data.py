"""Unit tests for CLI data commands (analyze-execution)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest
from typer.testing import CliRunner

from quant_insight.cli.main import app

runner = CliRunner()


@pytest.fixture
def mock_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Set up a mock workspace with sample OHLCV data."""
    workspace = tmp_path / "workspace"
    raw_dir = workspace / "data" / "inputs" / "raw"
    raw_dir.mkdir(parents=True)
    reports_dir = workspace / "reports"
    reports_dir.mkdir(parents=True)

    # Create sample OHLCV parquet
    ohlcv = pl.DataFrame(
        {
            "datetime": [
                datetime(2023, 1, 1, tzinfo=UTC),
                datetime(2023, 1, 2, tzinfo=UTC),
                datetime(2023, 1, 3, tzinfo=UTC),
            ],
            "symbol": ["AAPL"] * 3,
            "open": [100.0, 102.0, 101.0],
            "high": [102.0, 104.0, 103.0],
            "low": [99.0, 101.0, 100.0],
            "close": [101.0, 103.0, 102.0],
            "volume": [1000, 1100, 1050],
        }
    )
    ohlcv.write_parquet(raw_dir / "ohlcv.parquet")

    monkeypatch.setenv("MIXSEEK_WORKSPACE", str(workspace))
    return workspace


@pytest.mark.unit
class TestAnalyzeExecutionCli:
    """Tests for analyze-execution CLI command."""

    def test_analyze_execution_help(self) -> None:
        """--help exits successfully."""
        result = runner.invoke(app, ["data", "analyze-execution", "--help"])
        assert result.exit_code == 0
        assert "analyze-execution" in result.output.lower() or "execution" in result.output.lower()

    def test_analyze_execution_invalid_method(self, mock_workspace: Path) -> None:
        """Invalid method produces error."""
        result = runner.invoke(app, ["data", "analyze-execution", "--method", "invalid_method"])
        assert result.exit_code != 0
        assert "Invalid method" in result.output

    def test_analyze_execution_invalid_position_side(self, mock_workspace: Path) -> None:
        """Invalid position_side produces error."""
        result = runner.invoke(app, ["data", "analyze-execution", "--position-side", "neutral"])
        assert result.exit_code != 0
        assert "Invalid position_side" in result.output

    def test_analyze_execution_negative_offset(self, mock_workspace: Path) -> None:
        """Negative limit_offset_pct produces error."""
        result = runner.invoke(app, ["data", "analyze-execution", "--limit-offset-pct", "-1.0"])
        assert result.exit_code != 0
        assert "limit_offset_pct must be >= 0" in result.output

    def test_analyze_execution_invalid_config_path(self, mock_workspace: Path) -> None:
        """Non-existent --config path produces error."""
        result = runner.invoke(app, ["data", "analyze-execution", "--config", "/nonexistent/path.toml"])
        assert result.exit_code != 0
        assert "Failed to load config" in result.output

    def test_analyze_execution_no_config(self, mock_workspace: Path) -> None:
        """--config omitted: metadata skipped, runs normally."""
        result = runner.invoke(app, ["data", "analyze-execution"])
        assert result.exit_code == 0
        assert "Execution rate" in result.output
        assert "Done!" in result.output

    def test_analyze_execution_no_workspace(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """MIXSEEK_WORKSPACE not set produces error."""
        monkeypatch.delenv("MIXSEEK_WORKSPACE", raising=False)
        result = runner.invoke(app, ["data", "analyze-execution"])
        assert result.exit_code != 0

    def test_analyze_execution_output_saved(self, mock_workspace: Path) -> None:
        """Execution result is saved to workspace/reports/."""
        result = runner.invoke(
            app,
            [
                "data",
                "analyze-execution",
                "--method",
                "daytrade_open_limit",
                "--position-side",
                "long",
                "--limit-offset-pct",
                "1.0",
            ],
        )
        assert result.exit_code == 0

        reports_dir = mock_workspace / "reports"
        output_files = list(reports_dir.glob("execution_analysis_*.parquet"))
        assert len(output_files) == 1
        assert "daytrade_open_limit_long_1.0pct" in output_files[0].name

        # Verify the saved file has correct columns
        saved = pl.read_parquet(output_files[0])
        assert set(saved.columns) == {"datetime", "symbol", "is_executed", "entry_price", "limit_return"}

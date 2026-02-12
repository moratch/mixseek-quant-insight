"""Unit tests for ScreeningPipeline (P1)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import polars as pl
import pytest

from quant_insight.pipeline.result_models import (
    CPCVSummary,
    WFASummary,
)
from quant_insight.pipeline.screening import ScreeningConfig, ScreeningPipeline

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

WORKSPACE = Path("C:/Dev/Mixseek/workspace")


@pytest.fixture()
def screening_config(tmp_path: Path) -> ScreeningConfig:
    """ScreeningConfig pointing to tmp workspace."""
    return ScreeningConfig(workspace=tmp_path)


@pytest.fixture()
def sample_ohlcv() -> pl.DataFrame:
    """Minimal OHLCV data for 2 dates x 5 symbols."""
    dates = [datetime(2025, 1, 6), datetime(2025, 1, 7)]
    symbols = [f"SYM{i:02d}" for i in range(5)]
    rows: list[dict[str, object]] = []
    for dt in dates:
        for sym in symbols:
            rows.append(
                {
                    "datetime": dt,
                    "symbol": sym,
                    "open": 100.0,
                    "high": 105.0,
                    "low": 95.0,
                    "close": 102.0,
                    "volume": 10000,
                }
            )
    return pl.DataFrame(rows).cast({"datetime": pl.Datetime("us")})


@pytest.fixture()
def sample_returns() -> pl.DataFrame:
    """Returns data matching sample_ohlcv."""
    dates = [datetime(2025, 1, 6), datetime(2025, 1, 7)]
    symbols = [f"SYM{i:02d}" for i in range(5)]
    rows: list[dict[str, object]] = []
    for dt in dates:
        for i, sym in enumerate(symbols):
            rows.append({"datetime": dt, "symbol": sym, "return_value": (i - 2) * 0.01})
    return pl.DataFrame(rows).cast({"datetime": pl.Datetime("us")})


@pytest.fixture()
def _setup_workspace_data(
    tmp_path: Path,
    sample_ohlcv: pl.DataFrame,
    sample_returns: pl.DataFrame,
) -> Path:
    """Create minimal workspace data files in tmp_path."""
    raw_dir = tmp_path / "data" / "inputs" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    sample_ohlcv.write_parquet(raw_dir / "ohlcv.parquet")
    sample_returns.write_parquet(raw_dir / "returns.parquet")

    # Indicators
    indicators = sample_ohlcv.select(["datetime", "symbol"]).with_columns(
        pl.lit(0.5).alias("IND_A"),
    )
    indicators.write_parquet(raw_dir / "indicators.parquet")

    # competition.toml
    config_dir = tmp_path / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "competition.toml").write_text(
        """
[competition]
name = "test"
description = "Test competition"

[competition.return_definition]
window = 5
method = "close2close"

[competition.data_split]
train_end = "2025-01-06"
valid_end = "2025-01-07"
purge_rows = 0

[[competition.data]]
name = "ohlcv"
file_name = "ohlcv.parquet"
required = true

[[competition.data]]
name = "returns"
file_name = "returns.parquet"
required = true

[[competition.data]]
name = "indicators"
file_name = "indicators.parquet"
required = false
""",
        encoding="utf-8",
    )

    return tmp_path


def _make_wfa_summary(**overrides: Any) -> WFASummary:
    """Helper to create WFASummary with defaults."""
    defaults: dict[str, Any] = {
        "n_cycles": 6,
        "mean_oos_sharpe": 0.5,
        "std_oos_sharpe": 0.1,
        "mean_wfe": 0.7,
        "std_wfe": 0.05,
        "consistency_score": 0.8,
        "degradation_rate": 0.0,
        "degradation_pvalue": 0.5,
        "trend_direction": "stable",
        "cycles": [],
        "alerts": [],
    }
    defaults.update(overrides)
    return WFASummary(**defaults)


def _make_cpcv_summary(**overrides: Any) -> CPCVSummary:
    """Helper to create CPCVSummary with defaults."""
    defaults: dict[str, Any] = {
        "n_splits": 6,
        "purge_length": 5,
        "embargo_pct": 0.01,
        "mean_test_sharpe": 0.4,
        "std_test_sharpe": 0.1,
        "pbo": 0.2,
        "pbo_pvalue": 0.1,
        "deflated_sharpe": 0.35,
        "sharpe_haircut": 0.1,
        "consistency_ratio": 0.75,
        "rank_correlation": 0.6,
        "alerts": [],
    }
    defaults.update(overrides)
    return CPCVSummary(**defaults)


# ---------------------------------------------------------------------------
# P1-k: 12 test cases
# ---------------------------------------------------------------------------


@pytest.mark.unit()
class TestLoadFullPeriodData:
    """Test data loading from workspace."""

    @pytest.mark.usefixtures("_setup_workspace_data")
    def test_load_full_period_data_returns_all(self, tmp_path: Path) -> None:
        """OHLCV and returns are loaded with correct shapes."""
        config = ScreeningConfig(workspace=tmp_path)
        pipeline = ScreeningPipeline(config)

        ohlcv, returns, additional = pipeline.load_full_period_data()

        assert len(ohlcv) == 10  # 2 dates x 5 symbols
        assert len(returns) == 10
        assert "datetime" in ohlcv.columns
        assert "return_value" in returns.columns

    @pytest.mark.usefixtures("_setup_workspace_data")
    def test_load_full_period_data_includes_indicators(self, tmp_path: Path) -> None:
        """additional_data includes indicators when configured."""
        config = ScreeningConfig(workspace=tmp_path)
        pipeline = ScreeningPipeline(config)

        _, _, additional = pipeline.load_full_period_data()

        assert "indicators" in additional
        assert "IND_A" in additional["indicators"].columns


@pytest.mark.unit()
class TestExtractCandidates:
    """Test DuckDB candidate extraction."""

    def test_candidate_query_prefers_final_submission(self, tmp_path: Path) -> None:
        """final_submission=true is preferred over latest round."""
        import duckdb

        db_path = tmp_path / "mixseek.db"
        conn = duckdb.connect(str(db_path))
        conn.execute("CREATE SEQUENCE IF NOT EXISTS leader_board_id_seq")
        conn.execute(
            """
            CREATE TABLE leader_board (
                id INTEGER PRIMARY KEY DEFAULT nextval('leader_board_id_seq'),
                execution_id VARCHAR NOT NULL,
                team_id VARCHAR NOT NULL,
                team_name VARCHAR NOT NULL,
                round_number INTEGER NOT NULL,
                submission_content TEXT NOT NULL,
                submission_format VARCHAR DEFAULT 'md',
                score FLOAT NOT NULL,
                score_details JSON NOT NULL DEFAULT '{}',
                final_submission BOOLEAN NOT NULL DEFAULT FALSE,
                exit_reason VARCHAR NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        # Round 2 has higher round_number but final_submission=false
        conn.execute(
            "INSERT INTO leader_board (execution_id, team_id, team_name, round_number, "
            "submission_content, score, final_submission) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ["exec1", "team1", "TeamA", 2, "code_r2", 2.0, False],
        )
        # Round 1 has final_submission=true
        conn.execute(
            "INSERT INTO leader_board (execution_id, team_id, team_name, round_number, "
            "submission_content, score, final_submission) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ["exec1", "team1", "TeamA", 1, "code_r1_final", 1.5, True],
        )
        conn.close()

        config = ScreeningConfig(workspace=tmp_path)
        pipeline = ScreeningPipeline(config)
        candidates = pipeline.extract_candidates("exec1")

        assert len(candidates) == 1
        assert candidates[0]["submission_content"] == "code_r1_final"
        assert candidates[0]["round_number"] == 1

    def test_candidate_query_latest_round_fallback(self, tmp_path: Path) -> None:
        """Without final_submission, falls back to latest round_number."""
        import duckdb

        db_path = tmp_path / "mixseek.db"
        conn = duckdb.connect(str(db_path))
        conn.execute("CREATE SEQUENCE IF NOT EXISTS leader_board_id_seq")
        conn.execute(
            """
            CREATE TABLE leader_board (
                id INTEGER PRIMARY KEY DEFAULT nextval('leader_board_id_seq'),
                execution_id VARCHAR, team_id VARCHAR, team_name VARCHAR,
                round_number INTEGER, submission_content TEXT, score FLOAT,
                score_details JSON DEFAULT '{}',
                final_submission BOOLEAN DEFAULT FALSE,
                exit_reason VARCHAR, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            "INSERT INTO leader_board (execution_id, team_id, team_name, round_number, "
            "submission_content, score) VALUES (?, ?, ?, ?, ?, ?)",
            ["exec1", "team1", "TeamA", 1, "code_r1", 1.0],
        )
        conn.execute(
            "INSERT INTO leader_board (execution_id, team_id, team_name, round_number, "
            "submission_content, score) VALUES (?, ?, ?, ?, ?, ?)",
            ["exec1", "team1", "TeamA", 2, "code_r2", 2.0],
        )
        conn.close()

        config = ScreeningConfig(workspace=tmp_path)
        pipeline = ScreeningPipeline(config)
        candidates = pipeline.extract_candidates("exec1")

        assert len(candidates) == 1
        assert candidates[0]["submission_content"] == "code_r2"
        assert candidates[0]["round_number"] == 2


@pytest.mark.unit()
class TestGenerateFullSignals:
    """Test signal generation (fast and strict modes)."""

    @pytest.mark.usefixtures("_setup_workspace_data")
    def test_generate_full_signals_fast_schema(self, tmp_path: Path, sample_ohlcv: pl.DataFrame) -> None:
        """Fast mode returns DataFrame with datetime, symbol, signal."""
        submission_code = """```python
import polars as pl

def generate_signal(ohlcv: pl.DataFrame, additional_data: dict[str, pl.DataFrame]):
    return ohlcv.select([
        pl.col("datetime"),
        pl.col("symbol"),
        pl.col("close").alias("signal"),
    ])
```"""
        config = ScreeningConfig(workspace=tmp_path)
        pipeline = ScreeningPipeline(config)

        result = pipeline.generate_full_signals(submission_code, sample_ohlcv, {}, mode="fast")

        assert isinstance(result, pl.DataFrame)
        assert set(result.columns) >= {"datetime", "symbol", "signal"}
        assert len(result) == len(sample_ohlcv)

    @pytest.mark.usefixtures("_setup_workspace_data")
    def test_generate_full_signals_strict_schema(self, tmp_path: Path, sample_ohlcv: pl.DataFrame) -> None:
        """Strict mode returns same schema as fast mode."""
        submission_code = """```python
import polars as pl

def generate_signal(ohlcv: pl.DataFrame, additional_data: dict[str, pl.DataFrame]):
    return ohlcv.select([
        pl.col("datetime"),
        pl.col("symbol"),
        pl.col("close").alias("signal"),
    ])
```"""
        config = ScreeningConfig(workspace=tmp_path)
        pipeline = ScreeningPipeline(config)

        result = pipeline.generate_full_signals(submission_code, sample_ohlcv, {}, mode="strict")

        assert isinstance(result, pl.DataFrame)
        assert set(result.columns) >= {"datetime", "symbol", "signal"}
        assert len(result) == len(sample_ohlcv)


@pytest.mark.unit()
class TestWFAWrapper:
    """Test WFA wrapper (mocked qal)."""

    def test_run_wfa_returns_summary(self, tmp_path: Path) -> None:
        """WFA returns WFASummary with n_cycles cycles."""
        config = ScreeningConfig(workspace=tmp_path)
        pipeline = ScreeningPipeline(config)

        # Mock WalkForwardAnalyzer
        mock_cycle = MagicMock()
        mock_cycle.cycle_id = 1
        mock_cycle.is_sharpe = 0.5
        mock_cycle.oos_sharpe = 0.3
        mock_cycle.wfe = 0.6
        mock_cycle.is_return = 0.05
        mock_cycle.oos_return = 0.03

        mock_result = MagicMock()
        mock_result.n_cycles = 6
        mock_result.mean_oos_sharpe = 0.35
        mock_result.std_oos_sharpe = 0.1
        mock_result.mean_wfe = 0.65
        mock_result.std_wfe = 0.08
        mock_result.consistency_score = 0.83
        mock_result.degradation_rate = -0.01
        mock_result.degradation_pvalue = 0.4
        mock_result.trend_direction = "stable"
        mock_result.cycles = [mock_cycle] * 6
        mock_result.alerts = []

        mock_analyzer_cls = MagicMock()
        mock_analyzer_cls.return_value.run_wfa.return_value = mock_result

        mock_wfa_module = MagicMock()
        mock_wfa_module.WalkForwardAnalyzer = mock_analyzer_cls

        with patch.dict(
            "sys.modules",
            {"core": MagicMock(), "core.walk_forward_analysis": mock_wfa_module},
        ):
            summary = pipeline.run_wfa(
                strategy_data=pd.DataFrame({"CODE": [], "DATE": [], "FORWARD_RETURN": []}),
                strategy_func=lambda df: df,
            )

        assert isinstance(summary, WFASummary)
        assert summary.n_cycles == 6
        assert summary.mean_oos_sharpe == 0.35
        assert len(summary.cycles) == 6


@pytest.mark.unit()
class TestCPCVWrapper:
    """Test CPCV wrapper (mocked qal)."""

    def test_run_cpcv_returns_pbo_and_dsr(self, tmp_path: Path) -> None:
        """CPCV returns CPCVSummary with PBO and DSR."""
        config = ScreeningConfig(workspace=tmp_path)
        pipeline = ScreeningPipeline(config)

        mock_result = MagicMock()
        mock_result.n_splits = 6
        mock_result.purge_length = 5
        mock_result.mean_test_sharpe = 0.4
        mock_result.std_test_sharpe = 0.12
        mock_result.pbo = 0.25
        mock_result.pbo_pvalue = 0.08
        mock_result.deflated_sharpe = 0.32
        mock_result.sharpe_haircut = 0.08
        mock_result.consistency_ratio = 0.7
        mock_result.rank_correlation = 0.55
        mock_result.alerts = []

        mock_analyzer_cls = MagicMock()
        mock_analyzer_cls.return_value.run_cpcv.return_value = mock_result

        mock_cpcv_module = MagicMock()
        mock_cpcv_module.CPCVAnalyzer = mock_analyzer_cls

        with patch.dict(
            "sys.modules",
            {"core": MagicMock(), "core.combinatorial_purged_cv": mock_cpcv_module},
        ):
            summary = pipeline.run_cpcv(
                strategy_data=pd.DataFrame({"CODE": [], "DATE": [], "FORWARD_RETURN": []}),
                strategy_func=lambda df: df,
            )

        assert isinstance(summary, CPCVSummary)
        assert summary.pbo == 0.25
        assert summary.deflated_sharpe == 0.32


@pytest.mark.unit()
class TestVerdict:
    """Test screening verdict evaluation."""

    def test_verdict_pass_when_all_criteria_met(self, tmp_path: Path) -> None:
        """All criteria met → passed=True."""
        config = ScreeningConfig(workspace=tmp_path)
        pipeline = ScreeningPipeline(config)

        wfa = _make_wfa_summary(
            mean_oos_sharpe=0.5,
            mean_wfe=0.7,
            consistency_score=0.8,
            degradation_pvalue=0.5,
        )
        cpcv = _make_cpcv_summary(pbo=0.2, deflated_sharpe=0.35)

        verdict = pipeline.evaluate_verdict(wfa, cpcv)

        assert verdict.passed is True
        assert all(verdict.criteria.values())
        assert verdict.reasoning == "All screening criteria passed."

    def test_verdict_fail_when_pbo_high(self, tmp_path: Path) -> None:
        """PBO > max_pbo → passed=False."""
        config = ScreeningConfig(workspace=tmp_path)
        pipeline = ScreeningPipeline(config)

        wfa = _make_wfa_summary()
        cpcv = _make_cpcv_summary(pbo=0.6)  # > 0.4 threshold

        verdict = pipeline.evaluate_verdict(wfa, cpcv)

        assert verdict.passed is False
        assert verdict.criteria["max_pbo"] is False
        assert "max_pbo" in verdict.reasoning

    def test_verdict_fail_when_degrading(self, tmp_path: Path) -> None:
        """Significant degradation (pvalue < 0.05) → passed=False."""
        config = ScreeningConfig(workspace=tmp_path)
        pipeline = ScreeningPipeline(config)

        wfa = _make_wfa_summary(degradation_pvalue=0.01)  # < 0.05
        cpcv = _make_cpcv_summary()

        verdict = pipeline.evaluate_verdict(wfa, cpcv)

        assert verdict.passed is False
        assert verdict.criteria["min_degradation_pvalue"] is False

    def test_deterministic_verdict(self, tmp_path: Path) -> None:
        """Same inputs produce identical verdict."""
        config = ScreeningConfig(workspace=tmp_path)
        pipeline = ScreeningPipeline(config)

        wfa = _make_wfa_summary()
        cpcv = _make_cpcv_summary()

        v1 = pipeline.evaluate_verdict(wfa, cpcv)
        v2 = pipeline.evaluate_verdict(wfa, cpcv)

        assert v1.passed == v2.passed
        assert v1.criteria == v2.criteria
        assert v1.reasoning == v2.reasoning

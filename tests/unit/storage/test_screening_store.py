"""Unit tests for ScreeningResultStore (P3-b)."""

from __future__ import annotations

from datetime import datetime

import pytest

from quant_insight.pipeline.result_models import (
    CPCVSummary,
    ScreeningResult,
    ScreeningVerdict,
    WFASummary,
)
from quant_insight.storage.screening_store import ScreeningResultStore

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_wfa(**overrides: object) -> WFASummary:
    defaults = {
        "n_cycles": 6,
        "mean_oos_sharpe": 1.5,
        "std_oos_sharpe": 0.3,
        "mean_wfe": 0.6,
        "std_wfe": 0.1,
        "consistency_score": 0.8,
        "degradation_rate": -0.01,
        "degradation_pvalue": 0.45,
        "trend_direction": "stable",
        "cycles": [{"sharpe": 1.2}],
        "alerts": [],
    }
    defaults.update(overrides)
    return WFASummary(**defaults)  # type: ignore[arg-type]


def _make_cpcv(**overrides: object) -> CPCVSummary:
    defaults = {
        "n_splits": 6,
        "purge_length": 5,
        "embargo_pct": 0.01,
        "mean_test_sharpe": 1.0,
        "std_test_sharpe": 0.5,
        "pbo": 0.3,
        "pbo_pvalue": 0.02,
        "deflated_sharpe": 0.8,
        "sharpe_haircut": 0.2,
        "consistency_ratio": 0.75,
        "rank_correlation": 0.6,
    }
    defaults.update(overrides)
    return CPCVSummary(**defaults)  # type: ignore[arg-type]


def _make_result(**overrides: object) -> ScreeningResult:
    defaults: dict[str, object] = {
        "execution_id": "exec-001",
        "team_id": "team-a",
        "team_name": "Alpha Team",
        "round_number": 1,
        "strategy_name": "mean_reversion_v1",
        "screened_at": datetime(2026, 2, 12, 10, 0, 0),
        "mode": "fast",
        "mixseek_score": 1.73,
        "wfa": _make_wfa(),
        "cpcv": _make_cpcv(),
        "verdict": ScreeningVerdict(passed=True, criteria={"pbo": True, "dsr": True}, reasoning="All passed"),
        "adapter_config": {"threshold_method": "quantile", "long_quantile": 0.9},
        "wfa_config": {"n_cycles": 6},
        "cpcv_config": {"n_splits": 6},
    }
    defaults.update(overrides)
    return ScreeningResult(**defaults)  # type: ignore[arg-type]


@pytest.fixture()
def store(tmp_path):
    """Create a ScreeningResultStore with a temp DuckDB."""
    s = ScreeningResultStore(workspace=tmp_path, db_path=tmp_path / "test.db")
    s.initialize_schema()
    return s


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSaveAndGet:
    """Test #5: save -> get roundtrip."""

    def test_save_and_get(self, store):
        result = _make_result()
        store.save(result)

        retrieved = store.get_by_execution("exec-001")
        assert len(retrieved) == 1

        r = retrieved[0]
        assert r.execution_id == "exec-001"
        assert r.team_name == "Alpha Team"
        assert r.mixseek_score == 1.73
        assert r.wfa.mean_oos_sharpe == 1.5
        assert r.cpcv.pbo == 0.3
        assert r.verdict.passed is True


@pytest.mark.unit
class TestGetPassed:
    """Test #6: filter by passed=True."""

    def test_get_passed(self, store):
        r_pass = _make_result(strategy_name="good_strat")
        r_fail = _make_result(
            strategy_name="bad_strat",
            verdict=ScreeningVerdict(passed=False, criteria={"pbo": False}, reasoning="Fail"),
        )
        store.save(r_pass)
        store.save(r_fail)

        passed = store.get_passed(execution_id="exec-001")
        assert len(passed) == 1
        assert passed[0].strategy_name == "good_strat"


@pytest.mark.unit
class TestUpsertSameConfig:
    """Test #7: same config_hash overwrites (UPSERT)."""

    def test_upsert_same_config(self, store):
        r1 = _make_result(mixseek_score=1.0)
        store.save(r1)

        r2 = _make_result(mixseek_score=2.0)
        store.save(r2)

        retrieved = store.get_by_execution("exec-001")
        assert len(retrieved) == 1
        assert retrieved[0].mixseek_score == 2.0


@pytest.mark.unit
class TestEmptyResult:
    """Test #8: non-existent execution returns empty list."""

    def test_empty_result(self, store):
        results = store.get_by_execution("nonexistent")
        assert results == []

    def test_empty_before_table_created(self, tmp_path):
        """Returns empty list even when table has not been initialized."""
        s = ScreeningResultStore(workspace=tmp_path, db_path=tmp_path / "empty.db")
        results = s.get_by_execution("any-id")
        assert results == []

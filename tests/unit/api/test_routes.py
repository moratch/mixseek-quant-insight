"""Unit tests for API route endpoints (P3-d).

These tests require fastapi[testclient] — skipped if not installed.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from quant_insight.api.app import create_app  # noqa: E402
from quant_insight.pipeline.result_models import (  # noqa: E402
    CPCVSummary,
    ScreeningResult,
    ScreeningVerdict,
    WFASummary,
)
from quant_insight.storage.screening_store import ScreeningResultStore  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_wfa() -> WFASummary:
    return WFASummary(
        n_cycles=6,
        mean_oos_sharpe=1.5,
        std_oos_sharpe=0.3,
        mean_wfe=0.6,
        std_wfe=0.1,
        consistency_score=0.8,
        degradation_rate=-0.01,
        degradation_pvalue=0.45,
        trend_direction="stable",
        cycles=[{"sharpe": 1.2}],
    )


def _make_cpcv() -> CPCVSummary:
    return CPCVSummary(
        n_splits=6,
        purge_length=5,
        embargo_pct=0.01,
        mean_test_sharpe=1.0,
        std_test_sharpe=0.5,
        pbo=0.3,
        pbo_pvalue=0.02,
        deflated_sharpe=0.8,
        sharpe_haircut=0.2,
        consistency_ratio=0.75,
        rank_correlation=0.6,
    )


def _make_result(execution_id: str = "exec-001", passed: bool = True) -> ScreeningResult:
    return ScreeningResult(
        execution_id=execution_id,
        team_id="team-a",
        team_name="Alpha Team",
        round_number=1,
        strategy_name="mean_reversion_v1",
        screened_at=datetime(2026, 2, 12, 10, 0, 0),
        mode="fast",
        mixseek_score=1.73,
        wfa=_make_wfa(),
        cpcv=_make_cpcv(),
        verdict=ScreeningVerdict(passed=passed, criteria={"pbo": True}, reasoning="OK"),
        adapter_config={"threshold_method": "quantile"},
        wfa_config={"n_cycles": 6},
        cpcv_config={"n_splits": 6},
    )


def _write_ensemble_json(reports_dir: Path) -> None:
    """Write a minimal super-ensemble JSON for testing."""
    import json

    reports_dir.mkdir(parents=True, exist_ok=True)
    ensemble = {
        "individual_sharpe": {"StratA": 5.0, "StratB": 8.0},
        "option_a_5strat": {
            "weights": {"StratA": 0.4, "StratB": 0.6},
            "expected_return": 0.08,
            "volatility": 0.007,
            "sharpe_ratio": 11.4,
            "max_drawdown": -0.005,
            "diversification_ratio": 1.5,
            "component_strategies": ["StratA", "StratB"],
            "risk_contributions": {"StratA": 0.35, "StratB": 0.65},
            "optimization_method": "sharpe",
            "n_strategies_input": 2,
            "correlation_matrix": {
                "StratA": {"StratA": 1.0, "StratB": -0.3},
                "StratB": {"StratA": -0.3, "StratB": 1.0},
            },
        },
        "best": "5_strat_super",
    }
    (reports_dir / "super_ensemble_20260219.json").write_text(json.dumps(ensemble), encoding="utf-8")

    regime = {
        "regime_sharpe": {"bull": 6.0, "sideways": 12.0, "bear": 17.0},
        "annual_sharpe": {"2024": 16.0, "2025": 9.0},
        "quarterly_sharpe": {"2024Q1": 12.0},
        "monthly_win_rate": 0.958,
        "drawdown": {"max_dd": -0.005, "max_dd_date": "2025-08-22"},
        "contribution": {"StratA": 0.04, "StratB": 0.06},
    }
    (reports_dir / "regime_analysis_20260219.json").write_text(json.dumps(regime), encoding="utf-8")


@pytest.fixture()
def client(tmp_path: Path) -> TestClient:
    """Create a TestClient with a temp workspace and pre-populated data."""
    _write_ensemble_json(tmp_path / "reports")
    app = create_app(workspace=tmp_path)
    store: ScreeningResultStore = app.state.screening_store
    store.initialize_schema()
    store.save(_make_result(execution_id="exec-001"))
    return TestClient(app)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHealthEndpoint:
    """Test #9: health check."""

    def test_health_endpoint(self, client: TestClient):
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


@pytest.mark.unit
class TestScreeningByExecution:
    """Test #10: retrieve by execution_id."""

    def test_screening_by_execution(self, client: TestClient):
        response = client.get("/api/v1/screening/by-execution/exec-001")
        assert response.status_code == 200

        data = response.json()
        assert len(data) == 1
        assert data[0]["execution_id"] == "exec-001"
        assert data[0]["mixseek_score"] == 1.73
        assert data[0]["verdict"]["passed"] is True


@pytest.mark.unit
class TestScreeningNotFound:
    """Test #11: non-existent execution returns empty list."""

    def test_screening_not_found(self, client: TestClient):
        response = client.get("/api/v1/screening/by-execution/nonexistent")
        assert response.status_code == 200
        assert response.json() == []


@pytest.mark.unit
class TestEnsembleLatest:
    """Test #13: ensemble latest endpoint with regime data."""

    def test_ensemble_latest(self, client: TestClient):
        response = client.get("/api/v1/ensemble/latest")
        assert response.status_code == 200

        data = response.json()
        assert data["ensemble"]["sharpe_ratio"] == 11.4
        assert data["ensemble"]["weights"]["StratA"] == 0.4
        assert data["individual_sharpe"]["StratB"] == 8.0
        assert data["regime_sharpe"]["bear"] == 17.0
        assert data["monthly_win_rate"] == pytest.approx(0.958)
        assert data["contribution"]["StratA"] == 0.04

    def test_ensemble_latest_no_data(self, tmp_path: Path):
        app = create_app(workspace=tmp_path)
        empty_client = TestClient(app)
        response = empty_client.get("/api/v1/ensemble/latest")
        assert response.status_code == 404


@pytest.mark.unit
class TestEnsembleByExecution:
    """Test #14: ensemble by execution_id."""

    def test_ensemble_by_execution(self, client: TestClient):
        response = client.get("/api/v1/ensemble/by-execution/super-ensemble-v1")
        assert response.status_code == 200

        data = response.json()
        assert data["sharpe_ratio"] == 11.4
        assert "StratA" in data["weights"]

    def test_ensemble_not_found(self, client: TestClient):
        response = client.get("/api/v1/ensemble/by-execution/nonexistent")
        assert response.status_code == 200
        assert response.json()["status"] == "not_found"

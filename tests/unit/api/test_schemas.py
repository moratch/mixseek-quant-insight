"""Unit tests for API Pydantic schemas and conversion functions (P3-a)."""

from __future__ import annotations

from datetime import datetime

import pytest

from quant_insight.api.schemas import (
    BatchScreeningResponse,
    EnsembleResultResponse,
    ScreeningResultResponse,
    batch_screening_to_response,
    ensemble_result_to_response,
    screening_result_to_response,
)
from quant_insight.ensemble.builder import EnsembleResult
from quant_insight.pipeline.result_models import (
    BatchScreeningResult,
    CPCVSummary,
    ScreeningResult,
    ScreeningVerdict,
    WFASummary,
)

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
        "deflated_sharpe_source": "cpcv_builtin",
        "adjusted_deflated_sharpe": None,
        "alerts": [],
    }
    defaults.update(overrides)
    return CPCVSummary(**defaults)  # type: ignore[arg-type]


def _make_screening_result(**overrides: object) -> ScreeningResult:
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestScreeningResultToResponse:
    """Test #1: dataclass -> Pydantic conversion."""

    def test_screening_result_to_response(self):
        result = _make_screening_result()
        response = screening_result_to_response(result)

        assert isinstance(response, ScreeningResultResponse)
        assert response.execution_id == "exec-001"
        assert response.team_name == "Alpha Team"
        assert response.mixseek_score == 1.73
        assert response.wfa.mean_oos_sharpe == 1.5
        assert response.cpcv.pbo == 0.3
        assert response.verdict.passed is True
        assert response.mode == "fast"


@pytest.mark.unit
class TestBatchScreeningToResponse:
    """Test #2: batch conversion with counts."""

    def test_batch_screening_to_response(self):
        r1 = _make_screening_result(strategy_name="strat_a")
        r2 = _make_screening_result(
            strategy_name="strat_b",
            verdict=ScreeningVerdict(passed=False, criteria={"pbo": False}, reasoning="PBO too high"),
        )
        batch = BatchScreeningResult(
            screened_at=datetime(2026, 2, 12),
            n_candidates=2,
            n_passed=1,
            n_failed=1,
            results=[r1, r2],
            screening_criteria={"max_pbo": 0.4},
        )
        response = batch_screening_to_response(batch)

        assert isinstance(response, BatchScreeningResponse)
        assert response.n_passed == 1
        assert response.n_failed == 1
        assert len(response.results) == 2
        assert response.results[0].verdict.passed is True
        assert response.results[1].verdict.passed is False


@pytest.mark.unit
class TestEnsembleResultToResponse:
    """Test #3: EnsembleResult -> Pydantic conversion."""

    def test_ensemble_result_to_response(self):
        ensemble = EnsembleResult(
            weights={"strat_a": 0.6, "strat_b": 0.4},
            expected_return=0.12,
            volatility=0.15,
            sharpe_ratio=0.8,
            max_drawdown=-0.1,
            diversification_ratio=1.2,
            component_strategies=["strat_a", "strat_b"],
            risk_contributions={"strat_a": 0.55, "strat_b": 0.45},
            optimization_method="sharpe",
            n_strategies_input=2,
            correlation_matrix={"strat_a": {"strat_a": 1.0, "strat_b": 0.3}},
        )
        response = ensemble_result_to_response(ensemble)

        assert isinstance(response, EnsembleResultResponse)
        assert response.weights["strat_a"] == 0.6
        assert response.sharpe_ratio == 0.8
        assert response.n_strategies_input == 2


@pytest.mark.unit
class TestResponseJsonRoundtrip:
    """Test #4: Pydantic -> JSON -> Pydantic roundtrip."""

    def test_response_json_roundtrip(self):
        result = _make_screening_result()
        response = screening_result_to_response(result)

        json_str = response.model_dump_json()
        restored = ScreeningResultResponse.model_validate_json(json_str)

        assert restored.execution_id == response.execution_id
        assert restored.mixseek_score == response.mixseek_score
        assert restored.wfa.mean_oos_sharpe == response.wfa.mean_oos_sharpe
        assert restored.cpcv.pbo == response.cpcv.pbo
        assert restored.verdict.passed == response.verdict.passed

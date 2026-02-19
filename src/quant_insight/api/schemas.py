"""Pydantic response models for the screening/ensemble API.

Converts existing dataclass models (result_models.py, builder.py) to
Pydantic models for FastAPI response serialization and OpenAPI doc generation.

Note: This module depends only on pydantic (not fastapi), so tests can run
without the [api] extra installed.
"""

from __future__ import annotations

from datetime import datetime as dt
from typing import Any

from pydantic import BaseModel

from quant_insight.ensemble.builder import EnsembleResult
from quant_insight.pipeline.result_models import (
    BatchScreeningResult,
    CPCVSummary,
    ScreeningResult,
    ScreeningVerdict,
    WFASummary,
)

# ---------------------------------------------------------------------------
# Response models (1:1 with dataclasses)
# ---------------------------------------------------------------------------


class WFASummaryResponse(BaseModel):
    """Walk-Forward Analysis result summary."""

    n_cycles: int
    mean_oos_sharpe: float
    std_oos_sharpe: float
    mean_wfe: float
    std_wfe: float
    consistency_score: float
    degradation_rate: float
    degradation_pvalue: float
    trend_direction: str
    cycles: list[dict[str, float]]
    alerts: list[str] = []


class CPCVSummaryResponse(BaseModel):
    """Combinatorial Purged Cross-Validation result summary."""

    n_splits: int
    purge_length: int
    embargo_pct: float
    mean_test_sharpe: float
    std_test_sharpe: float
    pbo: float
    pbo_pvalue: float
    deflated_sharpe: float
    sharpe_haircut: float
    consistency_ratio: float
    rank_correlation: float
    deflated_sharpe_source: str = "cpcv_builtin"
    adjusted_deflated_sharpe: float | None = None
    alerts: list[str] = []


class ScreeningVerdictResponse(BaseModel):
    """Pass/fail verdict with per-criterion details."""

    passed: bool
    criteria: dict[str, bool]
    reasoning: str


class ScreeningResultResponse(BaseModel):
    """Complete screening result for one signal function."""

    execution_id: str
    team_id: str
    team_name: str
    round_number: int
    strategy_name: str
    screened_at: dt
    mode: str
    mixseek_score: float
    wfa: WFASummaryResponse
    cpcv: CPCVSummaryResponse
    verdict: ScreeningVerdictResponse
    adapter_config: dict[str, Any]
    wfa_config: dict[str, Any]
    cpcv_config: dict[str, Any]


class BatchScreeningResponse(BaseModel):
    """Batch screening result for multiple signal functions."""

    screened_at: dt
    n_candidates: int
    n_passed: int
    n_failed: int
    results: list[ScreeningResultResponse]
    screening_criteria: dict[str, float]


class EnsembleResultResponse(BaseModel):
    """Result of ensemble portfolio optimization."""

    weights: dict[str, float]
    expected_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    diversification_ratio: float
    component_strategies: list[str]
    risk_contributions: dict[str, float]
    optimization_method: str
    n_strategies_input: int
    correlation_matrix: dict[str, dict[str, float]]


class EnsembleOverviewResponse(BaseModel):
    """Full ensemble overview including regime analysis."""

    ensemble: EnsembleResultResponse
    individual_sharpe: dict[str, float]
    regime_sharpe: dict[str, float] | None = None
    annual_sharpe: dict[str, float] | None = None
    quarterly_sharpe: dict[str, float] | None = None
    monthly_win_rate: float | None = None
    max_drawdown_date: str | None = None
    contribution: dict[str, float] | None = None


# ---------------------------------------------------------------------------
# Conversion functions (dataclass -> Pydantic)
# ---------------------------------------------------------------------------


def _wfa_to_response(wfa: WFASummary) -> WFASummaryResponse:
    return WFASummaryResponse(
        n_cycles=wfa.n_cycles,
        mean_oos_sharpe=wfa.mean_oos_sharpe,
        std_oos_sharpe=wfa.std_oos_sharpe,
        mean_wfe=wfa.mean_wfe,
        std_wfe=wfa.std_wfe,
        consistency_score=wfa.consistency_score,
        degradation_rate=wfa.degradation_rate,
        degradation_pvalue=wfa.degradation_pvalue,
        trend_direction=wfa.trend_direction,
        cycles=wfa.cycles,
        alerts=wfa.alerts,
    )


def _cpcv_to_response(cpcv: CPCVSummary) -> CPCVSummaryResponse:
    return CPCVSummaryResponse(
        n_splits=cpcv.n_splits,
        purge_length=cpcv.purge_length,
        embargo_pct=cpcv.embargo_pct,
        mean_test_sharpe=cpcv.mean_test_sharpe,
        std_test_sharpe=cpcv.std_test_sharpe,
        pbo=cpcv.pbo,
        pbo_pvalue=cpcv.pbo_pvalue,
        deflated_sharpe=cpcv.deflated_sharpe,
        sharpe_haircut=cpcv.sharpe_haircut,
        consistency_ratio=cpcv.consistency_ratio,
        rank_correlation=cpcv.rank_correlation,
        deflated_sharpe_source=cpcv.deflated_sharpe_source,
        adjusted_deflated_sharpe=cpcv.adjusted_deflated_sharpe,
        alerts=cpcv.alerts,
    )


def _verdict_to_response(verdict: ScreeningVerdict) -> ScreeningVerdictResponse:
    return ScreeningVerdictResponse(
        passed=verdict.passed,
        criteria=verdict.criteria,
        reasoning=verdict.reasoning,
    )


def screening_result_to_response(result: ScreeningResult) -> ScreeningResultResponse:
    """Convert a ScreeningResult dataclass to a Pydantic response model."""
    return ScreeningResultResponse(
        execution_id=result.execution_id,
        team_id=result.team_id,
        team_name=result.team_name,
        round_number=result.round_number,
        strategy_name=result.strategy_name,
        screened_at=result.screened_at,
        mode=result.mode,
        mixseek_score=result.mixseek_score,
        wfa=_wfa_to_response(result.wfa),
        cpcv=_cpcv_to_response(result.cpcv),
        verdict=_verdict_to_response(result.verdict),
        adapter_config=result.adapter_config,
        wfa_config=result.wfa_config,
        cpcv_config=result.cpcv_config,
    )


def batch_screening_to_response(batch: BatchScreeningResult) -> BatchScreeningResponse:
    """Convert a BatchScreeningResult dataclass to a Pydantic response model."""
    return BatchScreeningResponse(
        screened_at=batch.screened_at,
        n_candidates=batch.n_candidates,
        n_passed=batch.n_passed,
        n_failed=batch.n_failed,
        results=[screening_result_to_response(r) for r in batch.results],
        screening_criteria=batch.screening_criteria,
    )


def ensemble_result_to_response(result: EnsembleResult) -> EnsembleResultResponse:
    """Convert an EnsembleResult dataclass to a Pydantic response model."""
    return EnsembleResultResponse(
        weights=result.weights,
        expected_return=result.expected_return,
        volatility=result.volatility,
        sharpe_ratio=result.sharpe_ratio,
        max_drawdown=result.max_drawdown,
        diversification_ratio=result.diversification_ratio,
        component_strategies=result.component_strategies,
        risk_contributions=result.risk_contributions,
        optimization_method=result.optimization_method,
        n_strategies_input=result.n_strategies_input,
        correlation_matrix=result.correlation_matrix,
    )

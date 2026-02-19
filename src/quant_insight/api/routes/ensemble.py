"""Ensemble result API endpoints.

Serves ensemble portfolio optimization results from workspace JSON files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from quant_insight.api.schemas import (
    EnsembleOverviewResponse,
    EnsembleResultResponse,
)

router = APIRouter(prefix="/ensemble", tags=["ensemble"])


def _get_workspace(request: Request) -> Path:
    """Resolve workspace path from app state."""
    return request.app.state.workspace  # type: ignore[no-any-return]


def _load_json(path: Path) -> dict[str, Any] | None:
    """Load a JSON file, returning None if not found."""
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def _ensemble_from_dict(data: dict[str, Any]) -> EnsembleResultResponse:
    """Convert an ensemble dict (from JSON) to a response model."""
    return EnsembleResultResponse(
        weights=data["weights"],
        expected_return=data["expected_return"],
        volatility=data["volatility"],
        sharpe_ratio=data["sharpe_ratio"],
        max_drawdown=data["max_drawdown"],
        diversification_ratio=data["diversification_ratio"],
        component_strategies=data["component_strategies"],
        risk_contributions=data["risk_contributions"],
        optimization_method=data["optimization_method"],
        n_strategies_input=data["n_strategies_input"],
        correlation_matrix=data["correlation_matrix"],
    )


@router.get("/latest", response_model=EnsembleOverviewResponse)
def get_latest_ensemble(request: Request) -> EnsembleOverviewResponse:
    """Get the latest super-ensemble with regime analysis."""
    workspace = _get_workspace(request)
    reports = workspace / "reports"

    # Load super-ensemble
    ensemble_data = _load_json(reports / "super_ensemble_20260219.json")
    if ensemble_data is None:
        raise HTTPException(status_code=404, detail="No ensemble results found")

    best_key = ensemble_data.get("best", "option_a_5strat")
    key_map = {
        "5_strat_super": "option_a_5strat",
        "3_strat_indicator": "option_b_3strat_indicator",
        "3_group": "option_c_3group",
    }
    option_key = key_map.get(best_key, best_key)
    best_ensemble = ensemble_data.get(option_key)
    if best_ensemble is None:
        raise HTTPException(status_code=404, detail=f"Ensemble option '{option_key}' not found")

    # Load regime analysis (optional)
    regime_data = _load_json(reports / "regime_analysis_20260219.json")

    return EnsembleOverviewResponse(
        ensemble=_ensemble_from_dict(best_ensemble),
        individual_sharpe=ensemble_data.get("individual_sharpe", {}),
        regime_sharpe=regime_data.get("regime_sharpe") if regime_data else None,
        annual_sharpe=regime_data.get("annual_sharpe") if regime_data else None,
        quarterly_sharpe=regime_data.get("quarterly_sharpe") if regime_data else None,
        monthly_win_rate=regime_data.get("monthly_win_rate") if regime_data else None,
        max_drawdown_date=(
            regime_data["drawdown"]["max_dd_date"] if regime_data and "drawdown" in regime_data else None
        ),
        contribution=regime_data.get("contribution") if regime_data else None,
    )


@router.get("/by-execution/{execution_id}", response_model=EnsembleResultResponse | dict[str, str])
def get_ensemble_by_execution(request: Request, execution_id: str) -> EnsembleResultResponse | dict[str, str]:
    """Get ensemble result for a given execution_id.

    Currently returns the super-ensemble for the matching execution_id,
    or a not-found message.
    """
    workspace = _get_workspace(request)
    reports = workspace / "reports"

    # For now, we only have one ensemble (super-ensemble-v1)
    if execution_id == "super-ensemble-v1":
        ensemble_data = _load_json(reports / "super_ensemble_20260219.json")
        if ensemble_data and "option_a_5strat" in ensemble_data:
            return _ensemble_from_dict(ensemble_data["option_a_5strat"])

    return {"status": "not_found", "execution_id": execution_id}

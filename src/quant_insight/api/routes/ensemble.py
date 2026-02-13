"""Ensemble result API endpoints (stub for future implementation)."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="/ensemble", tags=["ensemble"])


@router.get("/by-execution/{execution_id}")
def get_ensemble_by_execution(execution_id: str) -> dict[str, str]:
    """Get ensemble result for a given execution_id (not yet implemented)."""
    return {"status": "not_implemented", "execution_id": execution_id}

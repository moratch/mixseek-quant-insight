"""Screening result API endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query, Request

from quant_insight.api.schemas import ScreeningResultResponse, screening_result_to_response
from quant_insight.storage.screening_store import ScreeningResultStore

router = APIRouter(prefix="/screening", tags=["screening"])


def _get_store(request: Request) -> ScreeningResultStore:
    """Resolve ScreeningResultStore from app state."""
    return request.app.state.screening_store  # type: ignore[no-any-return]


@router.get("/latest", response_model=list[ScreeningResultResponse])
def get_latest(
    request: Request,
    limit: Annotated[int, Query(ge=1, le=100)] = 10,
) -> list[ScreeningResultResponse]:
    """Get the most recent screening results."""
    store = _get_store(request)
    results = store.get_latest(limit=limit)
    return [screening_result_to_response(r) for r in results]


@router.get("/by-execution/{execution_id}", response_model=list[ScreeningResultResponse])
def get_by_execution(
    request: Request,
    execution_id: str,
) -> list[ScreeningResultResponse]:
    """Get all screening results for a given execution_id."""
    store = _get_store(request)
    results = store.get_by_execution(execution_id)
    return [screening_result_to_response(r) for r in results]


@router.get("/by-execution/{execution_id}/passed", response_model=list[ScreeningResultResponse])
def get_passed_by_execution(
    request: Request,
    execution_id: str,
) -> list[ScreeningResultResponse]:
    """Get only passed screening results for a given execution_id."""
    store = _get_store(request)
    results = store.get_passed(execution_id=execution_id)
    return [screening_result_to_response(r) for r in results]

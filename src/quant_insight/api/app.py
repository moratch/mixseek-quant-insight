"""FastAPI application factory for MixSeek Quant Insight API."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI

from quant_insight.api.routes.ensemble import router as ensemble_router
from quant_insight.api.routes.screening import router as screening_router
from quant_insight.storage.screening_store import ScreeningResultStore


def create_app(workspace: Path) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        workspace: Path to the MixSeek workspace directory.

    Returns:
        Configured FastAPI application instance.
    """
    app = FastAPI(
        title="MixSeek Quant Insight API",
        version="0.1.0",
        description="Statistical screening results for MixSeek signal functions",
    )

    # Initialize screening store and attach to app state
    store = ScreeningResultStore(workspace=workspace)
    app.state.screening_store = store

    # Register routers
    app.include_router(screening_router, prefix="/api/v1")
    app.include_router(ensemble_router, prefix="/api/v1")

    @app.get("/api/v1/health", tags=["health"])
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app

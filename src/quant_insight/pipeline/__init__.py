"""Pipeline module for screening MixSeek signal functions via quant-alpha-lab.

Result models are imported eagerly (lightweight, no heavy deps).
Screening classes are imported lazily to avoid pulling in polars/duckdb
when only result_models are needed (e.g. from ensemble.builder).
"""

from quant_insight.pipeline.result_models import (
    BatchScreeningResult,
    CPCVSummary,
    ScreeningResult,
    ScreeningVerdict,
    WFASummary,
)

__all__ = [
    "BatchScreeningResult",
    "CPCVSummary",
    "ScreeningConfig",
    "ScreeningMode",
    "ScreeningPipeline",
    "ScreeningResult",
    "ScreeningVerdict",
    "WFASummary",
]


def __getattr__(name: str) -> object:
    """Lazy-load screening classes to avoid heavy imports at package level."""
    if name in ("ScreeningConfig", "ScreeningMode", "ScreeningPipeline"):
        from quant_insight.pipeline.screening import (
            ScreeningConfig,
            ScreeningMode,
            ScreeningPipeline,
        )

        _lazy = {
            "ScreeningConfig": ScreeningConfig,
            "ScreeningMode": ScreeningMode,
            "ScreeningPipeline": ScreeningPipeline,
        }
        return _lazy[name]

    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)

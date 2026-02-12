"""Result models for the screening pipeline.

Defines dataclasses for WFA summaries, CPCV summaries, screening verdicts,
and batch screening results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime as dt
from typing import Any


@dataclass
class WFASummary:
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
    alerts: list[str] = field(default_factory=list)


@dataclass
class CPCVSummary:
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
    alerts: list[str] = field(default_factory=list)


@dataclass
class ScreeningVerdict:
    """Pass/fail verdict with per-criterion details."""

    passed: bool
    criteria: dict[str, bool]
    reasoning: str


@dataclass
class ScreeningResult:
    """Complete screening result for one signal function."""

    # Metadata
    execution_id: str
    team_id: str
    team_name: str
    round_number: int
    strategy_name: str
    screened_at: dt
    mode: str

    # MixSeek evaluation (reference)
    mixseek_score: float

    # quant-alpha-lab verification results
    wfa: WFASummary
    cpcv: CPCVSummary

    # Verdict
    verdict: ScreeningVerdict

    # Reproducibility parameters
    adapter_config: dict[str, Any]
    wfa_config: dict[str, Any]
    cpcv_config: dict[str, Any]


@dataclass
class BatchScreeningResult:
    """Batch screening result for multiple signal functions."""

    screened_at: dt
    n_candidates: int
    n_passed: int
    n_failed: int
    results: list[ScreeningResult]
    screening_criteria: dict[str, float]

"""Screening pipeline for MixSeek signal functions.

Extracts candidates from DuckDB, generates signals, runs WFA/CPCV
via quant-alpha-lab, and produces pass/fail verdicts.
"""

from __future__ import annotations

import logging
import sys
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime as dt
from enum import StrEnum
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import polars as pl

from quant_insight.adapter.signal_to_strategy import (
    AdapterConfig,
    SignalToStrategyAdapter,
    ThresholdMethod,
)
from quant_insight.evaluator.submission_parser import parse_submission_function
from quant_insight.pipeline.result_models import (
    BatchScreeningResult,
    CPCVSummary,
    ScreeningResult,
    ScreeningVerdict,
    WFASummary,
)
from quant_insight.utils.config import load_competition_config

logger = logging.getLogger(__name__)


class ScreeningMode(StrEnum):
    """Signal generation mode."""

    FAST = "fast"
    STRICT = "strict"

# SQL: extract best candidate per team from leader_board
_CANDIDATES_BY_EXECUTION_SQL = """
WITH ranked AS (
    SELECT
        execution_id, team_id, team_name, round_number,
        submission_content, score, final_submission,
        ROW_NUMBER() OVER (
            PARTITION BY execution_id, team_id
            ORDER BY final_submission DESC, round_number DESC
        ) AS rn
    FROM leader_board
    WHERE execution_id = ?
      AND score >= ?
)
SELECT execution_id, team_id, team_name, round_number,
       submission_content, score, final_submission
FROM ranked WHERE rn = 1
ORDER BY score DESC
"""

_CANDIDATES_LATEST_SQL = """
WITH latest AS (
    SELECT execution_id
    FROM leader_board
    WHERE final_submission = TRUE
    ORDER BY created_at DESC
    LIMIT 1
),
ranked AS (
    SELECT
        lb.execution_id, lb.team_id, lb.team_name, lb.round_number,
        lb.submission_content, lb.score, lb.final_submission,
        ROW_NUMBER() OVER (
            PARTITION BY lb.execution_id, lb.team_id
            ORDER BY lb.final_submission DESC, lb.round_number DESC
        ) AS rn
    FROM leader_board lb
    INNER JOIN latest l ON lb.execution_id = l.execution_id
    WHERE lb.score >= ?
)
SELECT execution_id, team_id, team_name, round_number,
       submission_content, score, final_submission
FROM ranked WHERE rn = 1
ORDER BY score DESC
"""

# Fallback SQL without final_submission column (old schema)
_CANDIDATES_BY_EXECUTION_FALLBACK_SQL = """
WITH ranked AS (
    SELECT
        execution_id, team_id, team_name, round_number,
        submission_content, score,
        ROW_NUMBER() OVER (
            PARTITION BY execution_id, team_id
            ORDER BY round_number DESC
        ) AS rn
    FROM leader_board
    WHERE execution_id = ?
      AND score >= ?
)
SELECT execution_id, team_id, team_name, round_number,
       submission_content, score, FALSE AS final_submission
FROM ranked WHERE rn = 1
ORDER BY score DESC
"""


@dataclass(frozen=True)
class ScreeningConfig:
    """Configuration for the screening pipeline."""

    # Paths
    workspace: Path
    qal_path: Path = Path("C:/Dev/quant-alpha-lab")

    # Adapter settings
    threshold_method: ThresholdMethod = ThresholdMethod.QUANTILE
    long_quantile: float = 0.9
    short_quantile: float = 0.1

    # WFA settings
    wfa_n_cycles: int = 6
    wfa_is_ratio: float = 0.7
    wfa_min_samples: int = 20

    # CPCV settings
    cpcv_n_splits: int = 6
    cpcv_n_test_groups: int = 2
    cpcv_purge_length: int = 5
    cpcv_embargo_pct: float = 0.01

    # Screening criteria
    min_oos_sharpe: float = 0.3
    min_wfe: float = 0.5
    min_consistency: float = 0.6
    max_pbo: float = 0.4
    min_dsr: float = 0.2
    min_degradation_pvalue: float = 0.05


class ScreeningPipeline:
    """MixSeek signal function verification pipeline via quant-alpha-lab.

    Usage:
        config = ScreeningConfig(workspace=Path("..."))
        pipeline = ScreeningPipeline(config)
        candidates = pipeline.extract_candidates(execution_id="...")
        result = pipeline.screen_batch(candidates, mode="fast")
    """

    def __init__(self, config: ScreeningConfig) -> None:
        self.config = config
        self._ensure_qal_path()
        self._ohlcv: pl.DataFrame | None = None
        self._returns: pl.DataFrame | None = None
        self._additional_data: dict[str, pl.DataFrame] | None = None

    def _ensure_qal_path(self) -> None:
        """Add quant-alpha-lab to sys.path if not already present."""
        qal_str = str(self.config.qal_path)
        if qal_str not in sys.path:
            sys.path.insert(0, qal_str)

    @staticmethod
    def _check_column_exists(
        conn: duckdb.DuckDBPyConnection,
        table: str,
        column: str,
    ) -> bool:
        """Check if a column exists in a DuckDB table."""
        try:
            cols = conn.execute(f"PRAGMA table_info('{table}')").fetchall()  # noqa: S608
            return any(row[1] == column for row in cols)
        except duckdb.CatalogException:
            return False

    @staticmethod
    def _latest_execution_id(conn: duckdb.DuckDBPyConnection) -> str:
        """Get the most recent execution_id from leader_board."""
        rows = conn.execute(
            "SELECT execution_id FROM leader_board ORDER BY created_at DESC LIMIT 1"
        ).fetchall()
        if not rows:
            msg = "No executions found in leader_board"
            raise ValueError(msg)
        return str(rows[0][0])

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_full_period_data(
        self,
    ) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, pl.DataFrame]]:
        """Load full-period OHLCV, returns, and additional data.

        Data is loaded from ``{workspace}/data/inputs/raw/`` and cached
        for subsequent calls within the same pipeline instance.

        Returns:
            Tuple of (ohlcv, returns, additional_data).

        Raises:
            FileNotFoundError: If required data files are missing.
            ValueError: If required columns are absent.
        """
        if self._ohlcv is not None and self._returns is not None and self._additional_data is not None:
            return self._ohlcv, self._returns, self._additional_data

        raw_dir = self.config.workspace / "data" / "inputs" / "raw"

        # OHLCV
        ohlcv_path = raw_dir / "ohlcv.parquet"
        if not ohlcv_path.exists():
            msg = f"OHLCV data not found: {ohlcv_path}"
            raise FileNotFoundError(msg)
        self._ohlcv = pl.read_parquet(ohlcv_path)

        # Returns
        returns_path = raw_dir / "returns.parquet"
        if not returns_path.exists():
            msg = f"Returns data not found: {returns_path}. Run 'quant-insight data build-returns' first."
            raise FileNotFoundError(msg)
        self._returns = pl.read_parquet(returns_path)

        for col in ("datetime", "symbol", "return_value"):
            if col not in self._returns.columns:
                msg = f"Required column '{col}' not found in returns data"
                raise ValueError(msg)

        # Additional data from competition.toml
        self._additional_data = {}
        config_path = self.config.workspace / "configs" / "competition.toml"
        competition_config = load_competition_config(config_path)
        if competition_config is not None:
            for data_config in competition_config.data:
                if data_config.name in ("ohlcv", "returns"):
                    continue
                path = raw_dir / f"{data_config.name}.parquet"
                if path.exists():
                    self._additional_data[data_config.name] = pl.read_parquet(path)
                    logger.info("Loaded additional data '%s': %s", data_config.name, path)
                elif data_config.required:
                    msg = f"Required additional data not found: {path}"
                    raise FileNotFoundError(msg)
                else:
                    logger.warning("Optional data '%s' not found, skipping", data_config.name)

        logger.info(
            "Data loaded: OHLCV=%d rows, Returns=%d rows, Additional=%s",
            len(self._ohlcv),
            len(self._returns),
            list(self._additional_data.keys()),
        )

        return self._ohlcv, self._returns, self._additional_data

    # ------------------------------------------------------------------
    # Candidate extraction
    # ------------------------------------------------------------------

    def extract_candidates(
        self,
        execution_id: str | None = None,
        min_score: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Extract screening candidates from DuckDB leader_board.

        Per (execution_id, team_id), selects one candidate preferring
        final_submission=true, then latest round_number.

        Args:
            execution_id: Specific execution to screen. None = latest.
            min_score: Minimum MixSeek score filter.

        Returns:
            List of candidate dicts with keys:
            execution_id, team_id, team_name, round_number,
            submission_content, score, final_submission.
        """
        db_path = self.config.workspace / "mixseek.db"
        if not db_path.exists():
            msg = f"DuckDB not found: {db_path}"
            raise FileNotFoundError(msg)

        conn = duckdb.connect(str(db_path), read_only=True)
        try:
            has_final_submission = self._check_column_exists(conn, "leader_board", "final_submission")
            if execution_id is not None:
                if has_final_submission:
                    rows = conn.execute(
                        _CANDIDATES_BY_EXECUTION_SQL, [execution_id, min_score]
                    ).fetchall()
                else:
                    logger.warning("leader_board missing 'final_submission' column; using fallback SQL")
                    rows = conn.execute(
                        _CANDIDATES_BY_EXECUTION_FALLBACK_SQL, [execution_id, min_score]
                    ).fetchall()
            else:
                if has_final_submission:
                    rows = conn.execute(_CANDIDATES_LATEST_SQL, [min_score]).fetchall()
                else:
                    logger.warning("leader_board missing 'final_submission' column; using fallback SQL")
                    rows = conn.execute(
                        _CANDIDATES_BY_EXECUTION_FALLBACK_SQL, [self._latest_execution_id(conn), min_score]
                    ).fetchall()
        finally:
            conn.close()

        candidates: list[dict[str, Any]] = []
        for row in rows:
            candidates.append(
                {
                    "execution_id": row[0],
                    "team_id": row[1],
                    "team_name": row[2],
                    "round_number": row[3],
                    "submission_content": row[4],
                    "score": row[5],
                    "final_submission": row[6],
                }
            )

        logger.info("Extracted %d candidates", len(candidates))
        return candidates

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def generate_full_signals(
        self,
        submission_code: str,
        ohlcv: pl.DataFrame,
        additional_data: dict[str, pl.DataFrame],
        mode: str = "fast",
    ) -> pl.DataFrame:
        """Execute a signal function and return signals.

        Args:
            submission_code: Markdown-wrapped Python code containing
                generate_signal().
            ohlcv: Full-period OHLCV data.
            additional_data: Additional datasets (e.g. indicators).
            mode: "fast" (one-shot) or "strict" (sequential per date).

        Returns:
            pl.DataFrame with columns: datetime, symbol, signal.

        Raises:
            ValueError: If mode is unknown or output schema is invalid.
        """
        signal_func = parse_submission_function(submission_code)

        if mode == "fast":
            result = signal_func(ohlcv, additional_data)
        elif mode == "strict":
            result = self._generate_strict(signal_func, ohlcv, additional_data)
        else:
            msg = f"Unknown mode: {mode}. Use 'fast' or 'strict'."
            raise ValueError(msg)

        if not isinstance(result, pl.DataFrame):
            result = pl.from_pandas(result)

        for col in ("datetime", "symbol", "signal"):
            if col not in result.columns:
                msg = f"Signal output missing required column: {col}"
                raise ValueError(msg)

        return result

    def _generate_strict(
        self,
        signal_func: Callable[..., Any],
        ohlcv: pl.DataFrame,
        additional_data: dict[str, pl.DataFrame],
    ) -> pl.DataFrame:
        """Generate signals sequentially (causal constraint).

        For each date, only data up to and including that date is passed
        to the signal function. This prevents future information leakage
        that might be missed in fast (one-shot) mode.
        """
        dates = ohlcv["datetime"].unique().sort()
        all_signals: list[pl.DataFrame] = []

        for i, date in enumerate(dates):
            mask = pl.col("datetime") <= date
            ohlcv_slice = ohlcv.filter(mask)
            additional_slice = {
                k: v.filter(mask) if "datetime" in v.columns else v for k, v in additional_data.items()
            }

            result = signal_func(ohlcv_slice, additional_slice)
            if not isinstance(result, pl.DataFrame):
                result = pl.from_pandas(result)

            day_signals = result.filter(pl.col("datetime") == date)
            all_signals.append(day_signals)

            if (i + 1) % 100 == 0:
                logger.info("Strict mode progress: %d/%d dates", i + 1, len(dates))

        if not all_signals:
            return pl.DataFrame(
                schema={
                    "datetime": pl.Datetime("us"),
                    "symbol": pl.Utf8,
                    "signal": pl.Float64,
                }
            )

        return pl.concat(all_signals)

    # ------------------------------------------------------------------
    # WFA / CPCV wrappers
    # ------------------------------------------------------------------

    def run_wfa(
        self,
        strategy_data: pd.DataFrame,
        strategy_func: Callable[[pd.DataFrame], pd.DataFrame],
        strategy_name: str = "Strategy",
    ) -> WFASummary:
        """Run Walk-Forward Analysis via quant-alpha-lab.

        Args:
            strategy_data: pd.DataFrame with CODE, DATE, FORWARD_RETURN, signal.
            strategy_func: Closure from adapter.make_strategy_func().
            strategy_name: Label for logging.

        Returns:
            WFASummary with cycle-level and aggregate metrics.
        """
        from core.walk_forward_analysis import WalkForwardAnalyzer

        analyzer = WalkForwardAnalyzer(
            is_ratio=self.config.wfa_is_ratio,
            oos_ratio=1.0 - self.config.wfa_is_ratio,
            n_cycles=self.config.wfa_n_cycles,
            min_samples_per_period=self.config.wfa_min_samples,
        )

        wfa_result = analyzer.run_wfa(
            data=strategy_data,
            strategy_func=strategy_func,
            strategy_name=strategy_name,
        )

        cycles: list[dict[str, float]] = []
        for c in wfa_result.cycles:
            cycles.append(
                {
                    "cycle_id": float(c.cycle_id),
                    "is_sharpe": c.is_sharpe,
                    "oos_sharpe": c.oos_sharpe,
                    "wfe": c.wfe,
                    "is_return": c.is_return,
                    "oos_return": c.oos_return,
                }
            )

        return WFASummary(
            n_cycles=wfa_result.n_cycles,
            mean_oos_sharpe=wfa_result.mean_oos_sharpe,
            std_oos_sharpe=wfa_result.std_oos_sharpe,
            mean_wfe=wfa_result.mean_wfe,
            std_wfe=wfa_result.std_wfe,
            consistency_score=wfa_result.consistency_score,
            degradation_rate=wfa_result.degradation_rate,
            degradation_pvalue=wfa_result.degradation_pvalue,
            trend_direction=wfa_result.trend_direction,
            cycles=cycles,
            alerts=list(wfa_result.alerts),
        )

    def run_cpcv(
        self,
        strategy_data: pd.DataFrame,
        strategy_func: Callable[[pd.DataFrame], pd.DataFrame],
        strategy_name: str = "Strategy",
    ) -> CPCVSummary:
        """Run CPCV analysis via quant-alpha-lab (includes PBO and DSR).

        Args:
            strategy_data: pd.DataFrame with CODE, DATE, FORWARD_RETURN, signal.
            strategy_func: Closure from adapter.make_strategy_func().
            strategy_name: Label for logging.

        Returns:
            CPCVSummary with PBO, DSR, and fold-level metrics.
        """
        from core.combinatorial_purged_cv import CPCVAnalyzer

        analyzer = CPCVAnalyzer(
            n_splits=self.config.cpcv_n_splits,
            n_test_groups=self.config.cpcv_n_test_groups,
            purge_length=self.config.cpcv_purge_length,
            embargo_pct=self.config.cpcv_embargo_pct,
        )

        cpcv_result = analyzer.run_cpcv(
            data=strategy_data,
            strategy_func=strategy_func,
            strategy_name=strategy_name,
        )

        return CPCVSummary(
            n_splits=cpcv_result.n_splits,
            purge_length=cpcv_result.purge_length,
            embargo_pct=self.config.cpcv_embargo_pct,
            mean_test_sharpe=cpcv_result.mean_test_sharpe,
            std_test_sharpe=cpcv_result.std_test_sharpe,
            pbo=cpcv_result.pbo,
            pbo_pvalue=cpcv_result.pbo_pvalue,
            deflated_sharpe=cpcv_result.deflated_sharpe,
            sharpe_haircut=cpcv_result.sharpe_haircut,
            consistency_ratio=cpcv_result.consistency_ratio,
            rank_correlation=cpcv_result.rank_correlation,
            alerts=list(cpcv_result.alerts),
        )

    # ------------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------------

    def evaluate_verdict(
        self,
        wfa: WFASummary,
        cpcv: CPCVSummary,
    ) -> ScreeningVerdict:
        """Evaluate pass/fail verdict against screening criteria.

        Args:
            wfa: WFA analysis summary.
            cpcv: CPCV analysis summary.

        Returns:
            ScreeningVerdict with per-criterion results and reasoning.
        """
        criteria: dict[str, bool] = {
            "min_oos_sharpe": wfa.mean_oos_sharpe >= self.config.min_oos_sharpe,
            "min_wfe": wfa.mean_wfe >= self.config.min_wfe,
            "min_consistency": wfa.consistency_score >= self.config.min_consistency,
            "max_pbo": cpcv.pbo <= self.config.max_pbo,
            "min_dsr": cpcv.deflated_sharpe >= self.config.min_dsr,
            "min_degradation_pvalue": wfa.degradation_pvalue >= self.config.min_degradation_pvalue,
        }

        passed = all(criteria.values())
        failed_criteria = [k for k, v in criteria.items() if not v]

        if passed:
            reasoning = "All screening criteria passed."
        else:
            reasoning = f"Failed criteria: {', '.join(failed_criteria)}"

        return ScreeningVerdict(passed=passed, criteria=criteria, reasoning=reasoning)

    # ------------------------------------------------------------------
    # Single & batch screening
    # ------------------------------------------------------------------

    def screen_single(
        self,
        submission_code: str,
        execution_id: str,
        team_id: str,
        team_name: str,
        round_number: int,
        mixseek_score: float,
        mode: str = "fast",
        strategy_name: str | None = None,
    ) -> ScreeningResult:
        """Run the full screening pipeline for one signal function.

        Flow:
            1. Load data (cached)
            2. Parse submission -> generate_signal()
            3. Generate signals (fast or strict)
            4. Adapter: discretize + make strategy_func closure
            5. WFA analysis
            6. CPCV analysis (PBO, DSR)
            7. Evaluate verdict
            8. Return ScreeningResult

        Args:
            submission_code: Markdown-wrapped Python code.
            execution_id: MixSeek execution ID.
            team_id: Team identifier.
            team_name: Team display name.
            round_number: Competition round.
            mixseek_score: MixSeek evaluation score (reference).
            mode: "fast" or "strict".
            strategy_name: Optional override for display name.

        Returns:
            ScreeningResult with WFA, CPCV, and verdict.
        """
        if strategy_name is None:
            strategy_name = f"{team_name} R{round_number}"

        logger.info("Screening '%s' (mode=%s)...", strategy_name, mode)

        # 1. Load data
        ohlcv, returns, additional_data = self.load_full_period_data()

        # 2-3. Generate signals
        signal_df = self.generate_full_signals(
            submission_code,
            ohlcv,
            additional_data,
            mode=mode,
        )

        # 3b. Signal dedup validation
        n_raw = len(signal_df)
        signal_dedup = signal_df.unique(subset=["datetime", "symbol"], keep="first", maintain_order=True)
        n_dedup = len(signal_dedup)
        if n_dedup < n_raw:
            logger.warning(
                "Signal has %d duplicate (datetime, symbol) rows (%d -> %d). Adapter will dedup.",
                n_raw - n_dedup,
                n_raw,
                n_dedup,
            )

        # 4. Adapter: convert + make_strategy_func
        adapter_config = AdapterConfig(
            threshold_method=self.config.threshold_method,
            long_quantile=self.config.long_quantile,
            short_quantile=self.config.short_quantile,
        )
        adapter = SignalToStrategyAdapter(adapter_config)

        strategy_data = adapter.convert(signal_df, returns, ohlcv)
        strategy_func = adapter.make_strategy_func(signal_df)

        # Drop signal/raw_signal from strategy_data to avoid column
        # collision in WFA/CPCV _calculate_metrics (which merges data
        # with strategy_func output, both having 'signal').
        analysis_data = strategy_data.drop(columns=["signal", "raw_signal"], errors="ignore")

        # 5. WFA
        wfa_summary = self.run_wfa(analysis_data, strategy_func, strategy_name)

        # 6. CPCV
        cpcv_summary = self.run_cpcv(
            analysis_data,
            strategy_func,
            strategy_name=strategy_name,
        )

        # 7. Verdict
        verdict = self.evaluate_verdict(wfa_summary, cpcv_summary)

        logger.info(
            "Screening '%s': %s (OOS Sharpe=%.3f, PBO=%.3f, DSR=%.3f)",
            strategy_name,
            "PASS" if verdict.passed else "FAIL",
            wfa_summary.mean_oos_sharpe,
            cpcv_summary.pbo,
            cpcv_summary.deflated_sharpe,
        )

        return ScreeningResult(
            execution_id=execution_id,
            team_id=team_id,
            team_name=team_name,
            round_number=round_number,
            strategy_name=strategy_name,
            screened_at=dt.now(),
            mode=mode,
            mixseek_score=mixseek_score,
            wfa=wfa_summary,
            cpcv=cpcv_summary,
            verdict=verdict,
            adapter_config={
                "threshold_method": str(self.config.threshold_method),
                "long_quantile": self.config.long_quantile,
                "short_quantile": self.config.short_quantile,
            },
            wfa_config={
                "n_cycles": self.config.wfa_n_cycles,
                "is_ratio": self.config.wfa_is_ratio,
                "min_samples": self.config.wfa_min_samples,
            },
            cpcv_config={
                "n_splits": self.config.cpcv_n_splits,
                "n_test_groups": self.config.cpcv_n_test_groups,
                "purge_length": self.config.cpcv_purge_length,
                "embargo_pct": self.config.cpcv_embargo_pct,
            },
        )

    def screen_batch(
        self,
        candidates: list[dict[str, Any]],
        mode: ScreeningMode | str = ScreeningMode.FAST,
    ) -> BatchScreeningResult:
        """Run screening pipeline for multiple candidates.

        Each candidate is screened independently. If one candidate fails
        with an exception, its error is recorded and processing continues
        with remaining candidates.

        Args:
            candidates: List of candidate dicts from extract_candidates().
            mode: "fast" or "strict".

        Returns:
            BatchScreeningResult with per-candidate results and summary.
        """
        results: list[ScreeningResult] = []
        errors: list[str] = []

        for i, candidate in enumerate(candidates):
            team_name = candidate.get("team_name", "?")
            logger.info("Batch screening %d/%d: %s", i + 1, len(candidates), team_name)

            try:
                result = self.screen_single(
                    submission_code=candidate["submission_content"],
                    execution_id=candidate["execution_id"],
                    team_id=candidate["team_id"],
                    team_name=candidate["team_name"],
                    round_number=candidate["round_number"],
                    mixseek_score=candidate["score"],
                    mode=str(mode),
                )
                results.append(result)
            except Exception:
                tb = traceback.format_exc()
                error_msg = f"Candidate '{team_name}' failed: {tb}"
                logger.error(error_msg)
                errors.append(error_msg)

        if errors:
            logger.warning("%d candidate(s) failed with errors", len(errors))

        n_passed = sum(1 for r in results if r.verdict.passed)

        return BatchScreeningResult(
            screened_at=dt.now(),
            n_candidates=len(candidates),
            n_passed=n_passed,
            n_failed=len(candidates) - n_passed,
            results=results,
            screening_criteria={
                "min_oos_sharpe": self.config.min_oos_sharpe,
                "min_wfe": self.config.min_wfe,
                "min_consistency": self.config.min_consistency,
                "max_pbo": self.config.max_pbo,
                "min_dsr": self.config.min_dsr,
                "min_degradation_pvalue": self.config.min_degradation_pvalue,
            },
        )

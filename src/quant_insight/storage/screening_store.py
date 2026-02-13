"""DuckDB screening result store (P3).

Persists ScreeningResult objects to DuckDB for API retrieval.
Follows ImplementationStore pattern (thread-local connections, transaction management).
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime as dt
from pathlib import Path
from typing import Any, cast

import duckdb

from quant_insight.pipeline.result_models import (
    BatchScreeningResult,
    CPCVSummary,
    ScreeningResult,
    ScreeningVerdict,
    WFASummary,
)
from quant_insight.storage.schema import ALL_SCREENING_RESULT_DDL

logger = logging.getLogger(__name__)


def compute_config_hash(
    adapter_config: dict[str, Any],
    wfa_config: dict[str, Any],
    cpcv_config: dict[str, Any],
) -> str:
    """Compute a short SHA-256 hash of the combined config dicts.

    Args:
        adapter_config: Adapter configuration parameters.
        wfa_config: WFA configuration parameters.
        cpcv_config: CPCV configuration parameters.

    Returns:
        12-character hex hash string.
    """
    combined = {"adapter": adapter_config, "wfa": wfa_config, "cpcv": cpcv_config}
    raw = json.dumps(combined, sort_keys=True, default=str).encode()
    return hashlib.sha256(raw).hexdigest()[:12]


class ScreeningResultStore:
    """DuckDB store for screening results.

    Usage:
        store = ScreeningResultStore(workspace=Path("workspace"))
        store.initialize_schema()
        store.save(screening_result)
        results = store.get_by_execution("exec-123")
    """

    def __init__(self, workspace: Path, db_path: Path | None = None) -> None:
        """Initialize the store.

        Args:
            workspace: Workspace directory path.
            db_path: Database file path (default: {workspace}/mixseek.db).
        """
        self.db_path = db_path if db_path is not None else workspace / "mixseek.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()

    def _get_connection(self) -> duckdb.DuckDBPyConnection:
        """Get thread-local DuckDB connection."""
        if not hasattr(self._local, "conn"):
            self._local.conn = duckdb.connect(str(self.db_path))
        return cast(duckdb.DuckDBPyConnection, self._local.conn)

    @contextmanager
    def _transaction(self, conn: duckdb.DuckDBPyConnection) -> Iterator[duckdb.DuckDBPyConnection]:
        """Transaction context manager with rollback on failure."""
        try:
            conn.execute("BEGIN TRANSACTION")
            yield conn
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

    def _table_exists(self) -> bool:
        """Check if screening_result table exists."""
        conn = self._get_connection()
        result = conn.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'screening_result'"
        ).fetchone()
        return result is not None and result[0] > 0

    def initialize_schema(self) -> None:
        """Create screening_result table (idempotent)."""
        conn = self._get_connection()
        for ddl in ALL_SCREENING_RESULT_DDL:
            conn.execute(ddl)

    def save(self, result: ScreeningResult) -> None:
        """Save a single screening result to DuckDB (UPSERT).

        Args:
            result: ScreeningResult to persist.
        """
        conn = self._get_connection()
        config_hash = compute_config_hash(result.adapter_config, result.wfa_config, result.cpcv_config)

        # Determine failed criteria
        failed_criteria_list = [k for k, v in result.verdict.criteria.items() if not v]
        failed_criteria_str = ",".join(failed_criteria_list) if failed_criteria_list else None

        # Full result as JSON for detailed retrieval
        result_json = json.dumps(asdict(result), default=str, ensure_ascii=False)

        with self._transaction(conn):
            conn.execute(
                """
                INSERT INTO screening_result (
                    execution_id, team_id, team_name, round_number, strategy_name,
                    mode, config_hash, mixseek_score,
                    oos_sharpe, wfe, consistency,
                    pbo, deflated_sharpe,
                    passed, failed_criteria, result_json, screened_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (execution_id, team_id, round_number, strategy_name, mode, config_hash)
                DO UPDATE SET
                    team_name = EXCLUDED.team_name,
                    mixseek_score = EXCLUDED.mixseek_score,
                    oos_sharpe = EXCLUDED.oos_sharpe,
                    wfe = EXCLUDED.wfe,
                    consistency = EXCLUDED.consistency,
                    pbo = EXCLUDED.pbo,
                    deflated_sharpe = EXCLUDED.deflated_sharpe,
                    passed = EXCLUDED.passed,
                    failed_criteria = EXCLUDED.failed_criteria,
                    result_json = EXCLUDED.result_json,
                    screened_at = EXCLUDED.screened_at
                """,
                [
                    result.execution_id,
                    result.team_id,
                    result.team_name,
                    result.round_number,
                    result.strategy_name,
                    result.mode,
                    config_hash,
                    result.mixseek_score,
                    result.wfa.mean_oos_sharpe,
                    result.wfa.mean_wfe,
                    result.wfa.consistency_score,
                    result.cpcv.pbo,
                    result.cpcv.deflated_sharpe,
                    result.verdict.passed,
                    failed_criteria_str,
                    result_json,
                    result.screened_at,
                ],
            )

    def save_batch(self, batch: BatchScreeningResult) -> None:
        """Save all results in a batch.

        Args:
            batch: BatchScreeningResult containing multiple results.
        """
        for result in batch.results:
            self.save(result)

    def _row_to_screening_result(self, result_json: str) -> ScreeningResult:
        """Deserialize a JSON string back to ScreeningResult.

        Args:
            result_json: JSON string from result_json column.

        Returns:
            Reconstructed ScreeningResult.
        """
        data: dict[str, Any] = json.loads(result_json)

        return ScreeningResult(
            execution_id=data["execution_id"],
            team_id=data["team_id"],
            team_name=data["team_name"],
            round_number=data["round_number"],
            strategy_name=data["strategy_name"],
            screened_at=dt.fromisoformat(data["screened_at"]),
            mode=data["mode"],
            mixseek_score=data["mixseek_score"],
            wfa=WFASummary(**data["wfa"]),
            cpcv=CPCVSummary(**data["cpcv"]),
            verdict=ScreeningVerdict(**data["verdict"]),
            adapter_config=data["adapter_config"],
            wfa_config=data["wfa_config"],
            cpcv_config=data["cpcv_config"],
        )

    def get_by_execution(self, execution_id: str) -> list[ScreeningResult]:
        """Get all screening results for a given execution_id.

        Returns empty list if table does not exist or no results found.

        Args:
            execution_id: The execution identifier.

        Returns:
            List of ScreeningResult objects.
        """
        if not self._table_exists():
            return []

        conn = self._get_connection()
        rows = conn.execute(
            "SELECT result_json FROM screening_result WHERE execution_id = ? ORDER BY id",
            [execution_id],
        ).fetchall()

        return [self._row_to_screening_result(str(row[0])) for row in rows]

    def get_latest(self, limit: int = 10) -> list[ScreeningResult]:
        """Get the most recent screening results.

        Returns empty list if table does not exist.

        Args:
            limit: Maximum number of results to return.

        Returns:
            List of ScreeningResult objects, newest first.
        """
        if not self._table_exists():
            return []

        conn = self._get_connection()
        rows = conn.execute(
            "SELECT result_json FROM screening_result ORDER BY id DESC LIMIT ?",
            [limit],
        ).fetchall()

        return [self._row_to_screening_result(str(row[0])) for row in rows]

    def get_passed(self, execution_id: str | None = None) -> list[ScreeningResult]:
        """Get screening results that passed all criteria.

        Returns empty list if table does not exist.

        Args:
            execution_id: Optional filter by execution_id.

        Returns:
            List of passed ScreeningResult objects.
        """
        if not self._table_exists():
            return []

        conn = self._get_connection()
        if execution_id is not None:
            rows = conn.execute(
                "SELECT result_json FROM screening_result WHERE passed = true AND execution_id = ? ORDER BY id",
                [execution_id],
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT result_json FROM screening_result WHERE passed = true ORDER BY id",
            ).fetchall()

        return [self._row_to_screening_result(str(row[0])) for row in rows]

"""DuckDB並列書き込み対応ストア（エージェント実装保存用）

このモジュールはLocalCodeExecutorが生成したスクリプトをDuckDBに永続化します。
mixseek-coreのAggregationStoreパターン（スレッドローカル接続、トランザクション管理）に準拠。

Technical Strategy:
    DuckDB Python APIは同期のみのため、asyncio.to_threadで
    スレッドプールに退避して非同期実行を実現。
    スレッドローカルコネクションにより、各エージェントが独立した
    コネクションを使用してMVCC並列書き込みを実現。
"""

import asyncio
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import cast

import duckdb
from mixseek.utils.env import get_workspace_path

from quant_insight.storage import schema


class DatabaseWriteError(Exception):
    """データベース書き込み失敗（3回リトライ後）"""


class DatabaseReadError(Exception):
    """データベース読み込み失敗"""


class ImplementationStore:
    """DuckDB並列書き込み対応ストア（エージェント実装保存用）

    DuckDB Python APIは同期のみのため、asyncio.to_threadで
    スレッドプールに退避して非同期実行を実現。
    スレッドローカルコネクションにより、各エージェントが独立した
    コネクションを使用してMVCC並列書き込みを実現。
    """

    def __init__(self, workspace: Path | None = None, db_path: Path | None = None) -> None:
        """初期化

        Args:
            workspace: ワークスペースディレクトリパス
                      Noneの場合はConfigurationManager経由で取得
            db_path: データベースファイルパス
                    Noneの場合は{workspace}/mixseek.dbを使用

        Raises:
            PermissionError: DBファイル作成権限なし
        """
        self.db_path = self._get_db_path(workspace, db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # スレッドローカル変数（各スレッドが独立したコネクション保持）
        self._local = threading.local()

    def _get_db_path(self, workspace: Path | None, db_path: Path | None) -> Path:
        """データベースファイルパス取得

        Args:
            workspace: ワークスペースディレクトリパス
            db_path: 指定されたパス（Noneの場合は{workspace}/mixseek.db）

        Returns:
            データベースファイルパス
        """
        if db_path is not None:
            return db_path

        # Article 9準拠: ConfigurationManager経由でworkspaceを取得
        if workspace is None:
            workspace = get_workspace_path(cli_arg=None)

        return workspace / "mixseek.db"

    def _get_connection(self) -> duckdb.DuckDBPyConnection:
        """スレッドローカルコネクション取得

        各スレッドが独立したDuckDBコネクションを使用することで、
        MVCC並列書き込みを実現。

        Returns:
            DuckDBコネクション
        """
        if not hasattr(self._local, "conn"):
            self._local.conn = duckdb.connect(str(self.db_path))
        return cast(duckdb.DuckDBPyConnection, self._local.conn)

    @contextmanager
    def _transaction(self, conn: duckdb.DuckDBPyConnection) -> Iterator[duckdb.DuckDBPyConnection]:
        """同期トランザクション管理

        Args:
            conn: DuckDBコネクション

        Yields:
            トランザクション内のコネクション

        Raises:
            Exception: トランザクション失敗時（ROLLBACK実行）
        """
        try:
            conn.execute("BEGIN TRANSACTION")
            yield conn
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

    def initialize_schema(self) -> None:
        """agent_implementationテーブルを初期化

        このメソッドはべき等であり、複数回呼び出し可能。

        Raises:
            Exception: データベース初期化失敗
        """
        conn = self._get_connection()

        # 全DDL文を実行
        for ddl in schema.ALL_AGENT_IMPLEMENTATION_DDL:
            conn.execute(ddl)

    def table_exists(self) -> bool:
        """agent_implementationテーブルが存在するか確認

        Returns:
            テーブルが存在すればTrue
        """
        conn = self._get_connection()
        result = conn.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'agent_implementation'"
        ).fetchone()
        return result is not None and result[0] > 0

    def save_script_sync(
        self,
        execution_id: str,
        team_id: str,
        round_number: int,
        member_agent_name: str,
        file_name: str,
        code: str,
    ) -> None:
        """スクリプトを保存（同期版、UPSERT）

        Args:
            execution_id: 実行識別子(UUID)
            team_id: チームID
            round_number: ラウンド番号
            member_agent_name: メンバーエージェント名
            file_name: ファイル名
            code: Pythonコード

        Raises:
            Exception: データベース書き込み失敗
        """
        conn = self._get_connection()

        with self._transaction(conn):
            conn.execute(
                """
                INSERT INTO agent_implementation
                (execution_id, team_id, round_number, member_agent_name, file_name, code)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT (execution_id, team_id, round_number, member_agent_name, file_name) DO UPDATE SET
                    code = EXCLUDED.code
            """,
                [
                    execution_id,
                    team_id,
                    round_number,
                    member_agent_name,
                    file_name,
                    code,
                ],
            )

    async def save_script(
        self,
        execution_id: str,
        team_id: str,
        round_number: int,
        member_agent_name: str,
        file_name: str,
        code: str,
    ) -> None:
        """スクリプトを保存（非同期版、UPSERT）

        複数エージェントから同時呼び出しされても安全（ロックフリー）。
        DuckDB同期APIをasyncio.to_threadでスレッドプールに退避し、
        真の非同期並列実行を実現。

        Args:
            execution_id: 実行識別子(UUID)
            team_id: チームID
            round_number: ラウンド番号
            member_agent_name: メンバーエージェント名
            file_name: ファイル名
            code: Pythonコード

        Raises:
            DatabaseWriteError: 書き込み失敗（3回リトライ後）
        """
        # エクスポネンシャルバックオフリトライ
        delays = [1, 2, 4]

        for attempt, delay in enumerate(delays, 1):
            try:
                await asyncio.to_thread(
                    self.save_script_sync,
                    execution_id,
                    team_id,
                    round_number,
                    member_agent_name,
                    file_name,
                    code,
                )
                return
            except Exception as e:
                if attempt == len(delays):
                    raise DatabaseWriteError(f"Failed to save script after {attempt} retries: {e}") from e
                await asyncio.sleep(delay)

    def read_script_sync(
        self,
        execution_id: str,
        team_id: str,
        round_number: int,
        file_name: str,
    ) -> str | None:
        """スクリプトを読み込み（同期版）

        Args:
            execution_id: 実行識別子(UUID)
            team_id: チームID
            round_number: ラウンド番号
            file_name: ファイル名

        Returns:
            コード文字列、またはNone（未存在時）
        """
        conn = self._get_connection()

        result = conn.execute(
            """
            SELECT code
            FROM agent_implementation
            WHERE execution_id = ?
              AND team_id = ?
              AND round_number = ?
              AND file_name = ?
        """,
            [execution_id, team_id, round_number, file_name],
        ).fetchone()

        if not result:
            return None

        return str(result[0])

    async def read_script(
        self,
        execution_id: str,
        team_id: str,
        round_number: int,
        file_name: str,
    ) -> str | None:
        """スクリプトを読み込み（非同期版）

        Args:
            execution_id: 実行識別子(UUID)
            team_id: チームID
            round_number: ラウンド番号
            file_name: ファイル名

        Returns:
            コード文字列、またはNone（未存在時）

        Raises:
            DatabaseReadError: 読み込み失敗
        """
        try:
            return await asyncio.to_thread(
                self.read_script_sync,
                execution_id,
                team_id,
                round_number,
                file_name,
            )
        except Exception as e:
            raise DatabaseReadError(f"Failed to read script: {e}") from e

    def list_scripts_sync(
        self,
        execution_id: str,
        team_id: str,
        round_number: int,
    ) -> list[dict[str, str]]:
        """スクリプト一覧を取得（同期版）

        Args:
            execution_id: 実行識別子(UUID)
            team_id: チームID
            round_number: ラウンド番号

        Returns:
            ファイル情報のリスト [{"file_name": ..., "created_at": ...}, ...]
        """
        conn = self._get_connection()

        results = conn.execute(
            """
            SELECT file_name, created_at
            FROM agent_implementation
            WHERE execution_id = ?
              AND team_id = ?
              AND round_number = ?
            ORDER BY file_name
        """,
            [execution_id, team_id, round_number],
        ).fetchall()

        return [{"file_name": str(row[0]), "created_at": str(row[1])} for row in results]

    async def list_scripts(
        self,
        execution_id: str,
        team_id: str,
        round_number: int,
    ) -> list[dict[str, str]]:
        """スクリプト一覧を取得（非同期版）

        Args:
            execution_id: 実行識別子(UUID)
            team_id: チームID
            round_number: ラウンド番号

        Returns:
            ファイル情報のリスト [{"file_name": ..., "created_at": ...}, ...]

        Raises:
            DatabaseReadError: 読み込み失敗
        """
        try:
            return await asyncio.to_thread(
                self.list_scripts_sync,
                execution_id,
                team_id,
                round_number,
            )
        except Exception as e:
            raise DatabaseReadError(f"Failed to list scripts: {e}") from e


_instance: ImplementationStore | None = None


def get_implementation_store() -> ImplementationStore:
    """シングルトンインスタンスを取得

    Returns:
        ImplementationStoreインスタンス（共有）

    Note:
        複数エージェントが同時に使用しても、threading.local()により
        各スレッドが独立したコネクションを使用するため安全。
    """
    global _instance
    if _instance is None:
        _instance = ImplementationStore()
    return _instance


__all__ = [
    "ImplementationStore",
    "DatabaseWriteError",
    "DatabaseReadError",
    "get_implementation_store",
]

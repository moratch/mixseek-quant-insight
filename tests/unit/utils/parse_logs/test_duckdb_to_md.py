"""DuckDBエクスポーターのユニットテスト."""

import json
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import duckdb
import pytest

from quant_insight.utils.parse_logs.duckdb.exporter import (
    LeaderBoardExporter,
    RoundHistoryExporter,
)


@pytest.fixture
def temp_db() -> Iterator[Path]:
    """テスト用一時DuckDBファイルを作成."""
    with tempfile.TemporaryDirectory() as d:
        db_path = Path(d) / "test.db"

        # スキーマ作成
        conn = duckdb.connect(str(db_path))

        # シーケンス作成
        conn.execute("CREATE SEQUENCE IF NOT EXISTS round_history_id_seq")
        conn.execute("CREATE SEQUENCE IF NOT EXISTS leader_board_id_seq")

        # round_historyテーブル
        conn.execute("""
            CREATE TABLE IF NOT EXISTS round_history (
                id INTEGER PRIMARY KEY DEFAULT nextval('round_history_id_seq'),
                execution_id TEXT NOT NULL,
                team_id TEXT NOT NULL,
                team_name TEXT NOT NULL,
                round_number INTEGER NOT NULL,
                message_history JSON,
                member_submissions_record JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(execution_id, team_id, round_number)
            )
        """)

        # leader_boardテーブル
        conn.execute("""
            CREATE TABLE IF NOT EXISTS leader_board (
                id INTEGER PRIMARY KEY DEFAULT nextval('leader_board_id_seq'),
                execution_id VARCHAR NOT NULL,
                team_id VARCHAR NOT NULL,
                team_name VARCHAR NOT NULL,
                round_number INTEGER NOT NULL,
                submission_content TEXT NOT NULL,
                submission_format VARCHAR NOT NULL DEFAULT 'md',
                score FLOAT NOT NULL,
                score_details JSON NOT NULL,
                final_submission BOOLEAN NOT NULL DEFAULT FALSE,
                exit_reason VARCHAR NULL,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (execution_id, team_id, round_number)
            )
        """)

        conn.close()

        yield db_path


@pytest.fixture
def temp_output_dir() -> Iterator[Path]:
    """テスト用一時出力ディレクトリ."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def sample_team_configs() -> dict[str, dict[str, Any]]:
    """テスト用チーム設定."""
    return {
        "team-a": {
            "team_id": "team-a",
            "team_name": "Team Alpha",
            "leader_model": "openai:gpt-4o",
            "leader_system_instruction": "Be a helpful leader",
            "members": [
                {
                    "agent_name": "analyzer",
                    "agent_type": "code_execution",
                    "model": "openai:gpt-4o-mini",
                    "system_instruction": "Analyze code",
                }
            ],
        },
    }


class TestRoundHistoryExporter:
    """RoundHistoryExporterのテスト."""

    def test_export_team_log_no_data(
        self,
        temp_db: Path,
        temp_output_dir: Path,
    ) -> None:
        """データがない場合."""
        exporter = RoundHistoryExporter(temp_db)
        output_path = exporter.export_team_log("exec-123", "team-a", temp_output_dir)

        # ファイルパスは返されるがデータがないので警告のみ
        assert output_path.name == "exec-123_team-a_logs.md"

    def test_export_team_log_with_data(
        self,
        temp_db: Path,
        temp_output_dir: Path,
        sample_team_configs: dict[str, dict[str, Any]],
    ) -> None:
        """データがある場合."""
        # テストデータ挿入
        conn = duckdb.connect(str(temp_db))
        conn.execute(
            """
            INSERT INTO round_history
            (execution_id, team_id, team_name, round_number, message_history, member_submissions_record)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            [
                "exec-123",
                "team-a",
                "Team Alpha",
                1,
                json.dumps([{"kind": "request", "parts": [{"part_kind": "user-prompt", "content": "Hello"}]}]),
                json.dumps(
                    {
                        "all_submissions": [{"agent_name": "analyzer", "status": "success", "content": "Result"}],
                        "success_count": 1,
                        "total_count": 1,
                    }
                ),
            ],
        )
        conn.close()

        exporter = RoundHistoryExporter(temp_db, sample_team_configs)
        output_path = exporter.export_team_log("exec-123", "team-a", temp_output_dir)

        assert output_path.exists()
        content = output_path.read_text()

        assert "Team Alpha" in content
        assert "Round 1" in content
        assert "Hello" in content

    def test_export_all_teams(
        self,
        temp_db: Path,
        temp_output_dir: Path,
    ) -> None:
        """全チームエクスポート."""
        # テストデータ挿入
        conn = duckdb.connect(str(temp_db))
        for team_id, team_name in [("team-a", "Team Alpha"), ("team-b", "Team Beta")]:
            conn.execute(
                """
                INSERT INTO round_history
                (execution_id, team_id, team_name, round_number, message_history)
                VALUES (?, ?, ?, ?, ?)
            """,
                [
                    "exec-123",
                    team_id,
                    team_name,
                    1,
                    json.dumps([]),
                ],
            )
        conn.close()

        exporter = RoundHistoryExporter(temp_db)
        output_paths = exporter.export_all_teams("exec-123", temp_output_dir)

        assert len(output_paths) == 2
        team_ids = {p.name.split("_")[1] for p in output_paths}
        assert "team-a" in team_ids
        assert "team-b" in team_ids


class TestLeaderBoardExporter:
    """LeaderBoardExporterのテスト."""

    def test_export_team_submissions_no_data(
        self,
        temp_db: Path,
        temp_output_dir: Path,
    ) -> None:
        """データがない場合."""
        exporter = LeaderBoardExporter(temp_db)
        output_path = exporter.export_team_submissions("exec-123", "team-a", temp_output_dir)

        assert output_path.name == "exec-123_team-a_submissions.md"

    def test_export_team_submissions_with_data(
        self,
        temp_db: Path,
        temp_output_dir: Path,
    ) -> None:
        """データがある場合."""
        # テストデータ挿入
        conn = duckdb.connect(str(temp_db))
        conn.execute(
            """
            INSERT INTO leader_board
            (execution_id, team_id, team_name, round_number, submission_content,
             score, score_details, final_submission)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            [
                "exec-123",
                "team-a",
                "Team Alpha",
                1,
                "# My Submission\nHello world",
                85.5,
                json.dumps({"overall_score": 85.5, "metrics": [{"metric_name": "accuracy", "score": 90}]}),
                False,
            ],
        )
        conn.close()

        exporter = LeaderBoardExporter(temp_db)
        output_path = exporter.export_team_submissions("exec-123", "team-a", temp_output_dir)

        assert output_path.exists()
        content = output_path.read_text()

        assert "Team Alpha" in content
        assert "Round 1" in content
        assert "85.5" in content
        assert "My Submission" in content

    def test_export_all_teams(
        self,
        temp_db: Path,
        temp_output_dir: Path,
    ) -> None:
        """全チームエクスポート."""
        # テストデータ挿入
        conn = duckdb.connect(str(temp_db))
        for team_id, team_name in [("team-a", "Team Alpha"), ("team-b", "Team Beta")]:
            conn.execute(
                """
                INSERT INTO leader_board
                (execution_id, team_id, team_name, round_number, submission_content, score, score_details)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                [
                    "exec-123",
                    team_id,
                    team_name,
                    1,
                    "Submission",
                    80.0,
                    json.dumps({}),
                ],
            )
        conn.close()

        exporter = LeaderBoardExporter(temp_db)
        output_paths = exporter.export_all_teams("exec-123", temp_output_dir)

        assert len(output_paths) == 2

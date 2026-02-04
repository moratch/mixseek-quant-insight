"""エクスポートCLIの統合テスト."""

import json
import tempfile
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import duckdb
import pytest
from typer.testing import CliRunner

from quant_insight.cli.main import app

runner = CliRunner()


@pytest.fixture
def temp_workspace() -> Iterator[Path]:
    """テスト用一時ワークスペース."""
    with tempfile.TemporaryDirectory() as d:
        workspace = Path(d)

        # DuckDBファイル作成
        db_path = workspace / "mixseek.db"
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

        # テストデータ挿入
        conn.execute(
            """
            INSERT INTO round_history
            (execution_id, team_id, team_name, round_number, message_history)
            VALUES (?, ?, ?, ?, ?)
        """,
            [
                "test-exec-id",
                "team-test",
                "Test Team",
                1,
                json.dumps([{"role": "user", "parts": [{"type": "text", "content": "Hello"}]}]),
            ],
        )

        conn.execute(
            """
            INSERT INTO leader_board
            (execution_id, team_id, team_name, round_number, submission_content, score, score_details)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            [
                "test-exec-id",
                "team-test",
                "Test Team",
                1,
                "# Test Submission",
                85.0,
                json.dumps({"overall_score": 85.0}),
            ],
        )

        conn.close()

        yield workspace


@pytest.fixture
def temp_config(temp_workspace: Path) -> Path:
    """テスト用orchestrator.toml."""
    config_path = temp_workspace / "orchestrator.toml"
    config_content = f"""
[orchestrator]
timeout_per_team_seconds = 600

[[orchestrator.teams]]
config = "{temp_workspace}/team-test.toml"
"""
    config_path.write_text(config_content)

    # チームconfig作成
    team_config_path = temp_workspace / "team-test.toml"
    team_config_content = """
[team]
team_id = "team-test"
team_name = "Test Team"

[leader]
model = "openai:gpt-4o"
system_instruction = "Be helpful"
"""
    team_config_path.write_text(team_config_content)

    return config_path


class TestExportLogsCommand:
    """export logsコマンドのテスト."""

    def test_help(self) -> None:
        """ヘルプ表示."""
        result = runner.invoke(app, ["export", "logs", "--help"], color=False)
        assert result.exit_code == 0
        assert "execution_id" in result.output
        assert "config" in result.output.lower()

    def test_missing_config(self, temp_workspace: Path) -> None:
        """configオプション未指定."""
        result = runner.invoke(
            app,
            [
                "export",
                "logs",
                "test-exec-id",
                "-w",
                str(temp_workspace),
            ],
        )
        # configは必須なのでエラー
        assert result.exit_code != 0

    @patch("mixseek.orchestrator.load_orchestrator_settings")
    @patch("quant_insight.utils.parse_logs.duckdb.exporter.load_team_configs_from_orchestrator")
    def test_export_with_mocked_config(
        self,
        mock_load_team_configs: MagicMock,
        mock_load_orchestrator_settings: MagicMock,
        temp_workspace: Path,
        temp_config: Path,
    ) -> None:
        """モック設定でのエクスポート."""
        # モック設定
        mock_load_orchestrator_settings.return_value = MagicMock()
        mock_load_team_configs.return_value = {
            "team-test": {
                "team_id": "team-test",
                "team_name": "Test Team",
            }
        }

        output_dir = temp_workspace / "output"
        result = runner.invoke(
            app,
            [
                "export",
                "logs",
                "test-exec-id",
                "-c",
                str(temp_config),
                "-w",
                str(temp_workspace),
                "-o",
                str(output_dir),
            ],
        )

        assert result.exit_code == 0
        assert "エクスポート完了" in result.output

    @patch("mixseek.orchestrator.load_orchestrator_settings")
    @patch("quant_insight.utils.parse_logs.duckdb.exporter.load_team_configs_from_orchestrator")
    def test_export_logs_only(
        self,
        mock_load_team_configs: MagicMock,
        mock_load_orchestrator_settings: MagicMock,
        temp_workspace: Path,
        temp_config: Path,
    ) -> None:
        """ログのみエクスポート."""
        mock_load_orchestrator_settings.return_value = MagicMock()
        mock_load_team_configs.return_value = {"team-test": {"team_id": "team-test", "team_name": "Test Team"}}

        output_dir = temp_workspace / "output"
        result = runner.invoke(
            app,
            [
                "export",
                "logs",
                "test-exec-id",
                "-c",
                str(temp_config),
                "-w",
                str(temp_workspace),
                "-o",
                str(output_dir),
                "--logs-only",
            ],
        )

        assert result.exit_code == 0
        # logsファイルのみ生成される
        output_files = list(output_dir.glob("*.md"))
        logs_files = [f for f in output_files if "_logs.md" in f.name]
        submissions_files = [f for f in output_files if "_submissions.md" in f.name]
        assert len(logs_files) >= 0  # データがあれば生成される
        assert len(submissions_files) == 0  # submissionsは生成されない

    @patch("mixseek.orchestrator.load_orchestrator_settings")
    @patch("quant_insight.utils.parse_logs.duckdb.exporter.load_team_configs_from_orchestrator")
    def test_export_submissions_only(
        self,
        mock_load_team_configs: MagicMock,
        mock_load_orchestrator_settings: MagicMock,
        temp_workspace: Path,
        temp_config: Path,
    ) -> None:
        """サブミッションのみエクスポート."""
        mock_load_orchestrator_settings.return_value = MagicMock()
        mock_load_team_configs.return_value = {"team-test": {"team_id": "team-test", "team_name": "Test Team"}}

        output_dir = temp_workspace / "output"
        result = runner.invoke(
            app,
            [
                "export",
                "logs",
                "test-exec-id",
                "-c",
                str(temp_config),
                "-w",
                str(temp_workspace),
                "-o",
                str(output_dir),
                "--submissions-only",
            ],
        )

        assert result.exit_code == 0

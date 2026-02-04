"""DuckDBからMarkdownへのエクスポーター

round_history/leader_boardテーブルからMarkdownファイルを生成。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import duckdb

from quant_insight.utils.parse_logs.common.fence import get_fence, parse_json_safe
from quant_insight.utils.parse_logs.duckdb.formatters import (
    format_messages,
    format_metadata_section,
    format_score_details,
)

if TYPE_CHECKING:
    from mixseek.config import OrchestratorSettings

from mixseek.agents.leader.config import load_team_config

logger = logging.getLogger(__name__)


def load_team_configs_from_orchestrator(
    orchestrator_settings: OrchestratorSettings,
    workspace: Path,
) -> dict[str, dict[str, Any]]:
    """OrchestratorSettingsからチーム設定を読み込む

    Args:
        orchestrator_settings: オーケストレータ設定
        workspace: ワークスペースパス

    Returns:
        team_idをキーとしたチーム設定辞書
    """
    team_configs: dict[str, dict[str, Any]] = {}

    for team_entry in orchestrator_settings.teams:
        config_path = Path(team_entry["config"])
        team_config = load_team_config(config_path, workspace)

        # メンバー情報を収集
        members: list[dict[str, Any]] = []
        if team_config.members:
            for member in team_config.members:
                members.append(
                    {
                        "agent_name": member.agent_name,
                        "agent_type": str(member.agent_type),
                        "model": member.model,
                        "system_instruction": member.system_instruction,
                    }
                )

        team_configs[team_config.team_id] = {
            "team_id": team_config.team_id,
            "team_name": team_config.team_name,
            "leader_model": team_config.leader.model if team_config.leader else None,
            "leader_system_instruction": team_config.leader.system_instruction if team_config.leader else None,
            "members": members,
        }

    return team_configs


class RoundHistoryExporter:
    """round_historyテーブルからログMarkdownを生成"""

    def __init__(
        self,
        db_path: Path,
        team_configs: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """初期化

        Args:
            db_path: DuckDBファイルパス
            team_configs: チーム設定情報（メタデータ出力用）
        """
        self.db_path = db_path
        self.team_configs = team_configs or {}

    def _get_connection(self) -> duckdb.DuckDBPyConnection:
        """DuckDB接続取得"""
        return duckdb.connect(str(self.db_path), read_only=True)

    def _get_round_history(
        self,
        execution_id: str,
        team_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """round_historyテーブルからデータ取得

        Args:
            execution_id: 実行ID
            team_id: チームID（Noneの場合は全チーム）

        Returns:
            ラウンド履歴リスト
        """
        conn = self._get_connection()

        if team_id:
            query = """
                SELECT
                    team_id,
                    team_name,
                    round_number,
                    message_history,
                    member_submissions_record,
                    created_at
                FROM round_history
                WHERE execution_id = ? AND team_id = ?
                ORDER BY round_number
            """
            result = conn.execute(query, [execution_id, team_id]).fetchall()
        else:
            query = """
                SELECT
                    team_id,
                    team_name,
                    round_number,
                    message_history,
                    member_submissions_record,
                    created_at
                FROM round_history
                WHERE execution_id = ?
                ORDER BY team_id, round_number
            """
            result = conn.execute(query, [execution_id]).fetchall()

        conn.close()

        return [
            {
                "team_id": row[0],
                "team_name": row[1],
                "round_number": row[2],
                "message_history": row[3],
                "member_submissions_record": row[4],
                "created_at": row[5],
            }
            for row in result
        ]

    def export_team_log(
        self,
        execution_id: str,
        team_id: str,
        output_dir: Path,
    ) -> Path:
        """特定チームのログMarkdownを生成

        Args:
            execution_id: 実行ID
            team_id: チームID
            output_dir: 出力ディレクトリ

        Returns:
            生成されたMarkdownファイルパス
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        rounds = self._get_round_history(execution_id, team_id)
        if not rounds:
            logger.warning(f"No round history found for execution_id={execution_id}, team_id={team_id}")
            return output_dir / f"{execution_id}_{team_id}_logs.md"

        team_name = rounds[0]["team_name"]
        team_config = self.team_configs.get(team_id)

        md_lines: list[str] = []

        # ヘッダー
        md_lines.append(f"# Agent Log: {team_name}")
        md_lines.append("")

        # メタ情報
        md_lines.append(
            format_metadata_section(
                execution_id=execution_id,
                team_id=team_id,
                team_name=team_name,
                total_rounds=len(rounds),
                leader_model=team_config.get("leader_model") if team_config else None,
                leader_system_instruction=team_config.get("leader_system_instruction") if team_config else None,
                members=team_config.get("members") if team_config else None,
            )
        )
        md_lines.append("---")
        md_lines.append("")

        # 各ラウンド
        for round_data in rounds:
            round_number = round_data["round_number"]
            created_at = round_data["created_at"]

            md_lines.append(f"## Round {round_number}")
            md_lines.append(f"- **created_at:** `{created_at}`")
            md_lines.append("")

            # Member Agent応答を先にパース（メッセージ履歴に埋め込むため）
            member_submissions_raw = parse_json_safe(round_data["member_submissions_record"])
            member_submissions: list[dict[str, Any]] = []
            if member_submissions_raw and isinstance(member_submissions_raw, dict):
                member_submissions = member_submissions_raw.get("submissions", [])

            # メッセージ履歴（Member Agentの思考ログをtool-returnに埋め込む）
            message_history_raw = parse_json_safe(round_data["message_history"])
            if message_history_raw and isinstance(message_history_raw, list):
                md_lines.append("### Messages")
                md_lines.append("")
                # Pydantic AI形式のメッセージをフォーマット（member_submissionsを渡す）
                formatted_messages = self._format_pydantic_messages(
                    message_history_raw,
                    member_submissions=member_submissions,
                )
                md_lines.append(formatted_messages)
                md_lines.append("")

            md_lines.append("---")
            md_lines.append("")

        # ファイル書き出し
        output_path = output_dir / f"{execution_id}_{team_id}_logs.md"
        output_path.write_text("\n".join(md_lines), encoding="utf-8")

        logger.info(f"Generated: {output_path}")
        return output_path

    def _format_pydantic_messages(
        self,
        messages: list[dict[str, Any]],
        member_submissions: list[dict[str, Any]] | None = None,
    ) -> str:
        """Pydantic AI形式のメッセージをMarkdown形式にフォーマット

        Args:
            messages: Pydantic AI形式のメッセージリスト
            member_submissions: Member Agent応答リスト（tool-returnにall_messagesを展開するため）

        Returns:
            フォーマットされたMarkdown文字列
        """
        return format_messages(messages, member_submissions=member_submissions)

    def export_all_teams(
        self,
        execution_id: str,
        output_dir: Path,
    ) -> list[Path]:
        """全チームのログMarkdownを生成

        Args:
            execution_id: 実行ID
            output_dir: 出力ディレクトリ

        Returns:
            生成されたMarkdownファイルパスのリスト
        """
        # 全チームのteam_idを取得
        conn = self._get_connection()
        result = conn.execute(
            "SELECT DISTINCT team_id FROM round_history WHERE execution_id = ?",
            [execution_id],
        ).fetchall()
        conn.close()

        team_ids = [row[0] for row in result]

        output_paths: list[Path] = []
        for team_id in team_ids:
            output_path = self.export_team_log(execution_id, team_id, output_dir)
            output_paths.append(output_path)

        return output_paths


class LeaderBoardExporter:
    """leader_boardテーブルからサブミッションMarkdownを生成"""

    def __init__(
        self,
        db_path: Path,
        team_configs: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """初期化

        Args:
            db_path: DuckDBファイルパス
            team_configs: チーム設定情報（メタデータ出力用）
        """
        self.db_path = db_path
        self.team_configs = team_configs or {}

    def _get_connection(self) -> duckdb.DuckDBPyConnection:
        """DuckDB接続取得"""
        return duckdb.connect(str(self.db_path), read_only=True)

    def _get_leader_board(
        self,
        execution_id: str,
        team_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """leader_boardテーブルからデータ取得

        Args:
            execution_id: 実行ID
            team_id: チームID（Noneの場合は全チーム）

        Returns:
            リーダーボードエントリリスト
        """
        conn = self._get_connection()

        if team_id:
            query = """
                SELECT
                    team_id,
                    team_name,
                    round_number,
                    submission_content,
                    submission_format,
                    score,
                    score_details,
                    final_submission,
                    exit_reason,
                    created_at
                FROM leader_board
                WHERE execution_id = ? AND team_id = ?
                ORDER BY round_number
            """
            result = conn.execute(query, [execution_id, team_id]).fetchall()
        else:
            query = """
                SELECT
                    team_id,
                    team_name,
                    round_number,
                    submission_content,
                    submission_format,
                    score,
                    score_details,
                    final_submission,
                    exit_reason,
                    created_at
                FROM leader_board
                WHERE execution_id = ?
                ORDER BY team_id, round_number
            """
            result = conn.execute(query, [execution_id]).fetchall()

        conn.close()

        return [
            {
                "team_id": row[0],
                "team_name": row[1],
                "round_number": row[2],
                "submission_content": row[3],
                "submission_format": row[4],
                "score": row[5],
                "score_details": row[6],
                "final_submission": row[7],
                "exit_reason": row[8],
                "created_at": row[9],
            }
            for row in result
        ]

    def export_team_submissions(
        self,
        execution_id: str,
        team_id: str,
        output_dir: Path,
    ) -> Path:
        """特定チームのサブミッションMarkdownを生成

        Args:
            execution_id: 実行ID
            team_id: チームID
            output_dir: 出力ディレクトリ

        Returns:
            生成されたMarkdownファイルパス
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        entries = self._get_leader_board(execution_id, team_id)
        if not entries:
            logger.warning(f"No leader board entries found for execution_id={execution_id}, team_id={team_id}")
            return output_dir / f"{execution_id}_{team_id}_submissions.md"

        team_name = entries[0]["team_name"]
        final_round = max(entry["round_number"] for entry in entries)

        md_lines: list[str] = []

        # ヘッダー
        md_lines.append(f"# {team_name} - Submissions")
        md_lines.append("")

        # メタ情報テーブル
        md_lines.append("| 項目 | 値 |")
        md_lines.append("|------|-----|")
        md_lines.append(f"| execution_id | `{execution_id}` |")
        md_lines.append(f"| team_id | `{team_id}` |")
        md_lines.append(f"| 最終ラウンド | {final_round} |")
        md_lines.append("")
        md_lines.append("---")
        md_lines.append("")

        # 各ラウンド
        for entry in entries:
            round_number = entry["round_number"]
            created_at = entry["created_at"]
            score = entry["score"]
            final_submission = entry["final_submission"]
            exit_reason = entry["exit_reason"]
            submission_content = entry["submission_content"]
            score_details_raw = parse_json_safe(entry["score_details"])

            md_lines.append(f"## Round {round_number}")
            md_lines.append(f"- **created_at:** `{created_at}`")
            md_lines.append(f"- **score:** `{score}`")
            md_lines.append(f"- **final_submission:** `{final_submission}`")
            if exit_reason:
                md_lines.append(f"- **exit_reason:** {exit_reason}")
            md_lines.append("")

            # スコア詳細
            if score_details_raw and isinstance(score_details_raw, dict):
                md_lines.append("### スコア詳細")
                md_lines.append(format_score_details(score_details_raw))
                md_lines.append("")

            # 提出内容
            md_lines.append("### 提出内容")
            open_fence, close_fence = get_fence(submission_content)
            md_lines.append(open_fence)
            md_lines.append(submission_content)
            md_lines.append(close_fence)
            md_lines.append("")
            md_lines.append("---")
            md_lines.append("")

        # ファイル書き出し
        output_path = output_dir / f"{execution_id}_{team_id}_submissions.md"
        output_path.write_text("\n".join(md_lines), encoding="utf-8")

        logger.info(f"Generated: {output_path}")
        return output_path

    def export_all_teams(
        self,
        execution_id: str,
        output_dir: Path,
    ) -> list[Path]:
        """全チームのサブミッションMarkdownを生成

        Args:
            execution_id: 実行ID
            output_dir: 出力ディレクトリ

        Returns:
            生成されたMarkdownファイルパスのリスト
        """
        # 全チームのteam_idを取得
        conn = self._get_connection()
        result = conn.execute(
            "SELECT DISTINCT team_id FROM leader_board WHERE execution_id = ?",
            [execution_id],
        ).fetchall()
        conn.close()

        team_ids = [row[0] for row in result]

        output_paths: list[Path] = []
        for team_id in team_ids:
            output_path = self.export_team_submissions(execution_id, team_id, output_dir)
            output_paths.append(output_path)

        return output_paths

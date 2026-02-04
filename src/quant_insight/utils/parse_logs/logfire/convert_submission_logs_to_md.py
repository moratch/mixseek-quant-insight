#!/usr/bin/env python3
"""evaluator_logs CSVをteam_idごとのMarkdownファイルに変換するスクリプト."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

import typer


def extract_submission_history(content: str) -> str:
    """contentから # 提出履歴 セクションを抽出する."""
    # # 提出履歴 から # リーダーボード の手前までを抽出
    match = re.search(r"(# 提出履歴.*?)(?=# リーダーボード|$)", content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def format_score_details(content: str) -> str:
    """スコア詳細のJSONをMarkdownテーブルに変換する."""

    def replace_json_block(match: re.Match[str] | None) -> str:
        if match is None:
            return ""
        json_str = match.group(1)
        try:
            data = json.loads(json_str)
            lines: list[str] = []
            lines.append(f"- **総合スコア**: {data.get('overall_score', 'N/A')}")
            lines.append("")
            metrics = data.get("metrics", [])
            if metrics:
                lines.append("| メトリクス | スコア | コメント |")
                lines.append("|-----------|--------|----------|")
                for m in metrics:
                    name = m.get("metric_name", "N/A")
                    score = m.get("score", "N/A")
                    comment = m.get("evaluator_comment", "N/A")
                    lines.append(f"| {name} | {score} | {comment} |")
            return "\n".join(lines)
        except json.JSONDecodeError:
            # パースできなければそのまま返す
            return str(match.group(0))

    # ### スコア詳細: の後のJSONブロックを置換
    result = re.sub(
        r"### スコア詳細:\s*\n\{(.*?)\}",
        lambda m: "### スコア詳細:\n"
        + replace_json_block(re.match(r".*?(\{.*\})", "{" + m.group(1) + "}", re.DOTALL)),
        content,
        flags=re.DOTALL,
    )
    return result


def format_score_details_v2(content: str) -> str:
    """スコア詳細のJSONをMarkdownに変換する（改良版）."""

    def json_to_md(json_str: str) -> str:
        try:
            data = json.loads(json_str)
            lines: list[str] = []
            lines.append(f"- **総合スコア**: {data.get('overall_score', 'N/A')}")
            lines.append("")
            metrics = data.get("metrics", [])
            if metrics:
                lines.append("| メトリクス | スコア | コメント |")
                lines.append("|-----------|--------|----------|")
                for m in metrics:
                    name = m.get("metric_name", "N/A")
                    score = m.get("score", "N/A")
                    comment = m.get("evaluator_comment", "N/A")
                    lines.append(f"| {name} | {score} | {comment} |")
            return "\n".join(lines)
        except json.JSONDecodeError:
            return json_str

    # ### スコア詳細: の後の { ... } を探して置換
    pattern = r"(### スコア詳細:\s*\n)(\{[\s\S]*?\n\})"

    def replacer(m: re.Match[str]) -> str:
        header = m.group(1)
        json_block = m.group(2)
        return header + json_to_md(json_block)

    return re.sub(pattern, replacer, content)


app = typer.Typer()


@app.command()
def main(
    evaluator_csv: Path = typer.Argument(..., help="evaluator_logs CSVファイルのパス"),
    output_dir: Path = typer.Argument(..., help="出力ディレクトリ"),
) -> None:
    """evaluator_logs CSVをteam_idごとのMarkdownファイルに変換する."""
    csv_path = evaluator_csv

    # CSVを読み込む
    with csv_path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # team_idごとにグループ化（最新のround=最大roundのものを使用）
    team_data: dict[str, dict[str, Any]] = {}
    for row in rows:
        team_id = row["team_id"]
        round_num = int(row["round"])
        if team_id not in team_data or round_num > team_data[team_id]["round"]:
            team_data[team_id] = {
                "trace_id": row["trace_id"],
                "team_id": team_id,
                "round": round_num,
                "start_timestamp": row["start_timestamp"],
                "content": row["content"],
            }

    # team_idごとにMarkdownファイルを生成
    for team_id, data in team_data.items():
        submission_history = extract_submission_history(data["content"])
        # スコア詳細をMarkdownに変換
        submission_history = format_score_details_v2(submission_history)

        # Markdownの内容を構築（フロントマターではなく通常のMarkdown）
        md_content = f"""# {data["team_id"]}

| 項目 | 値 |
|------|-----|
| trace_id | `{data["trace_id"]}` |
| team_id | `{data["team_id"]}` |
| round | {data["round"]} |
| timestamp | {data["start_timestamp"]} |

{submission_history}
"""
        filename = f"{team_id}_submissions.md"
        output_path = output_dir / filename

        output_path.write_text(md_content, encoding="utf-8")
        print(f"Generated: {output_path}")


if __name__ == "__main__":
    app()

#!/usr/bin/env python3
"""CSVログをmodel_name別のMarkdownファイルに変換するスクリプト"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

import typer

# member_logsをtool_call_idでインデックス化するグローバル変数
member_logs_by_tool_call_id: dict[str, list[dict[str, Any]]] = {}


def get_fence(content: str, lang: str = "") -> tuple[str, str]:
    """コンテンツに応じて適切なフェンスを返す

    コンテンツ内にバッククォートが含まれる場合はチルダフェンスを使用し、
    チルダも含まれる場合は十分な長さのフェンスを使用する。

    Returns:
        tuple: (開始フェンス, 終了フェンス)
    """
    has_backticks = "```" in content
    has_tildes = "~~~" in content

    if not has_backticks:
        # バッククォートがなければ通常の3つで十分
        return f"```{lang}", "```"
    elif not has_tildes:
        # バッククォートがあるがチルダがなければチルダを使用
        return f"~~~{lang}", "~~~"
    else:
        # 両方ある場合は、十分な長さのチルダを使用
        max_tildes = 3
        matches = re.findall(r"~+", content)
        if matches:
            max_tildes = max(len(m) for m in matches)
        fence = "~" * (max_tildes + 1)
        return f"{fence}{lang}", fence


def parse_json_safe(json_str: str) -> list[Any] | dict[str, Any] | None:
    """JSONを安全にパースする"""
    if not json_str:
        return None
    try:
        result = json.loads(json_str)
        if isinstance(result, list | dict):
            return result
        return None
    except json.JSONDecodeError:
        return None


def load_member_logs(csv_path: Path) -> None:
    """member_logsをtool_call_idでインデックス化して読み込む"""
    global member_logs_by_tool_call_id
    member_logs_by_tool_call_id = {}

    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tool_call_id = row.get("tool_call_id", "")
            if tool_call_id:
                if tool_call_id not in member_logs_by_tool_call_id:
                    member_logs_by_tool_call_id[tool_call_id] = []
                member_logs_by_tool_call_id[tool_call_id].append(row)

    # 各tool_call_id内でstart_timestampでソート
    for tool_call_id in member_logs_by_tool_call_id:
        member_logs_by_tool_call_id[tool_call_id].sort(key=lambda x: x.get("start_timestamp", ""))


def format_member_log(member_row: dict[str, Any], parent_msg_index: int) -> str:
    """member_logを読みやすい形式にフォーマット

    Args:
        member_row: member_logの1行
        parent_msg_index: 親メッセージのインデックス（例: 4）
    """
    output: list[str] = []
    tool_name = member_row.get("tool_name", "unknown")
    model_name = member_row.get("model_name", "unknown")
    start_timestamp = member_row.get("start_timestamp", "N/A")
    message = member_row.get("message", "")
    all_messages_raw = member_row.get("all_messages", "")

    output.append(f"#### Tool Message: `{tool_name}` → `{model_name}`")
    output.append(f"- **start_timestamp:** `{start_timestamp}`")
    output.append(f"- **message:** {message}")
    output.append("")

    all_messages = parse_json_safe(all_messages_raw)
    if all_messages and isinstance(all_messages, list):
        output.append(format_messages(all_messages, indent_level=1, parent_index=parent_msg_index))
    else:
        output.append("*パースエラーまたは空*")
    output.append("")

    return "\n".join(output)


def format_messages(messages: list[Any], indent_level: int = 0, parent_index: int | None = None) -> str:
    """all_messagesリストを読みやすい形式にフォーマット

    Args:
        messages: メッセージリスト
        indent_level: インデントレベル（0=####, 1=#####, ...）
        parent_index: 親メッセージのインデックス（例: 4 → Message 4.1, 4.2...）
    """
    if not messages:
        return "*メッセージなし*"

    # ヘッダーレベルを決定
    header_prefix = "#" * (4 + indent_level)

    output = []
    for i, msg in enumerate(messages, 1):
        role = msg.get("role", "unknown")
        parts = msg.get("parts", [])
        finish_reason = msg.get("finish_reason", "")

        # メッセージ番号を決定
        if parent_index is not None:
            msg_number = f"{parent_index}.{i}"
        else:
            msg_number = str(i)

        output.append(f"{header_prefix} Message {msg_number}: `{role}`")
        if finish_reason:
            output.append(f"*finish_reason: {finish_reason}*\n")

        tool_call_counter = 0
        for part in parts:
            part_type = part.get("type", "unknown")

            if part_type == "text":
                content = part.get("content", "")
                open_fence, close_fence = get_fence(content)
                output.append(open_fence)
                output.append(content)
                output.append(f"{close_fence}\n")

            elif part_type == "tool_call":
                tool_call_counter += 1
                tool_name = part.get("name", "unknown")
                tool_id = part.get("id", "")
                arguments = part.get("arguments", {})
                args_json = json.dumps(arguments, ensure_ascii=False, indent=2)
                open_fence, close_fence = get_fence(args_json, "json")
                output.append(f"**Tool Call:** `{tool_name}`")
                output.append(f"- ID: `{tool_id}`")
                output.append("- Arguments:")
                output.append(open_fence)
                output.append(args_json)
                output.append(f"{close_fence}\n")

                # member_logsから該当するtool_call_idのログを展開
                if tool_id and tool_id in member_logs_by_tool_call_id:
                    member_rows = member_logs_by_tool_call_id[tool_id]
                    for member_row in member_rows:
                        output.append(format_member_log(member_row, i))

            elif part_type == "tool_call_response":
                tool_name = part.get("name", "unknown")
                tool_id = part.get("id", "")
                result = part.get("result", "")
                open_fence, close_fence = get_fence(result)
                output.append(f"**Tool Response:** `{tool_name}`")
                output.append(f"- ID: `{tool_id}`")
                output.append("- Result:")
                output.append(open_fence)
                output.append(result)
                output.append(f"{close_fence}\n")

    return "\n".join(output)


def format_system_instructions(instructions: list[Any]) -> str:
    """system_instructionsを読みやすい形式にフォーマット"""
    if not instructions:
        return "*システム指示なし*"

    output: list[str] = []
    for i, instr in enumerate(instructions, 1):
        instr_type = instr.get("type", "unknown")
        content = instr.get("content", "")
        open_fence, close_fence = get_fence(content)

        if len(instructions) > 1:
            output.append(f"#### Instruction {i} (`{instr_type}`)")
        output.append(open_fence)
        output.append(content)
        output.append(close_fence)

    return "\n".join(output)


app = typer.Typer()


@app.command()
def main(
    leader_csv: Path = typer.Argument(..., help="leader_logs CSVファイルのパス"),
    output_dir: Path = typer.Argument(..., help="出力ディレクトリ"),
    member_csv: Path | None = typer.Option(
        None, "--member-csv", "-m", help="member_logs CSVファイルのパス（オプション）"
    ),
) -> None:
    """leader_logs CSVをmodel_name別のMarkdownファイルに変換する."""
    csv_path = leader_csv

    # member_logsを読み込んでインデックス化
    if member_csv and member_csv.exists():
        load_member_logs(member_csv)
        print(f"member_logs読み込み: {len(member_logs_by_tool_call_id)} tool_call_ids")

    # CSVを読み込み（BOM付きUTF-8対応）
    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # model_nameでグループ化
    models: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        model_name = row.get("model_name", "unknown")
        if model_name not in models:
            models[model_name] = []
        models[model_name].append(row)

    # 各モデルごとにMarkdownファイルを生成
    for model_name, model_rows in models.items():
        # roundでソート
        model_rows.sort(key=lambda x: int(x.get("round", 0)))

        # trace_idを取得（最初の行から）
        trace_id = model_rows[0].get("trace_id", "unknown")

        # Markdownを生成
        md_lines = []
        md_lines.append(f"# Agent Log: {model_name}")
        md_lines.append("")
        md_lines.append("## メタ情報")
        md_lines.append("")
        md_lines.append(f"- **trace_id:** `{trace_id}`")
        md_lines.append(f"- **model_name:** `{model_name}`")
        md_lines.append(f"- **総ラウンド数:** {len(model_rows)}")
        md_lines.append("")
        md_lines.append("---")
        md_lines.append("")

        for row in model_rows:
            round_num = row.get("round", "?")
            start_timestamp = row.get("start_timestamp", "N/A")
            message = row.get("message", "")
            system_instructions_raw = row.get("system_instructions", "")
            all_messages_raw = row.get("all_messages", "")

            md_lines.append(f"## Round {round_num}")
            md_lines.append("")
            md_lines.append(f"- **start_timestamp:** `{start_timestamp}`")
            md_lines.append(f"- **message:** {message}")
            md_lines.append("")

            # System Instructions
            md_lines.append("### System Instructions")
            md_lines.append("")
            system_instructions = parse_json_safe(system_instructions_raw)
            if system_instructions and isinstance(system_instructions, list):
                md_lines.append(format_system_instructions(system_instructions))
            else:
                md_lines.append("*パースエラーまたは空*")
            md_lines.append("")

            # All Messages
            md_lines.append("### Messages")
            md_lines.append("")
            all_messages = parse_json_safe(all_messages_raw)
            if all_messages and isinstance(all_messages, list):
                md_lines.append(format_messages(all_messages))
            else:
                md_lines.append("*パースエラーまたは空*")
            md_lines.append("")
            md_lines.append("---")
            md_lines.append("")

        # ファイルに書き出し
        safe_name = model_name.replace("/", "_").replace(" ", "_")
        output_path = output_dir / f"{safe_name}_logs.md"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))

        print(f"生成: {output_path}")


if __name__ == "__main__":
    app()

"""DuckDB用Markdownフォーマッター

round_history/leader_boardテーブルからMarkdownへの変換フォーマッター。
"""

from __future__ import annotations

import json
from typing import Any

from quant_insight.utils.parse_logs.common.fence import get_fence


def format_arguments_expanded(args: dict[str, Any], tool_name: str) -> str:
    """tool callのargumentsを展開してフォーマット

    1階層のJSONを分解し、各keyに対してvalueをコードブロックで表示する。
    execute_python_codeの場合は"code"キーをPythonコードブロックで表示。

    Args:
        args: arguments辞書
        tool_name: ツール名

    Returns:
        フォーマットされたMarkdown文字列
    """
    output: list[str] = []

    for key, value in args.items():
        # execute_python_codeの"code"キーは特別扱い
        if tool_name == "execute_python_code" and key == "code":
            output.append(f"- **{key}:**")
            open_fence, close_fence = get_fence(str(value), "python")
            output.append(open_fence)
            output.append(str(value))
            output.append(close_fence)
        elif isinstance(value, str):
            # 文字列値はコードブロックで表示
            output.append(f"- **{key}:**")
            open_fence, close_fence = get_fence(value)
            output.append(open_fence)
            output.append(value)
            output.append(close_fence)
        elif isinstance(value, dict | list):
            # 複雑な値はJSON形式で表示
            output.append(f"- **{key}:**")
            json_str = json.dumps(value, ensure_ascii=False, indent=2)
            open_fence, close_fence = get_fence(json_str, "json")
            output.append(open_fence)
            output.append(json_str)
            output.append(close_fence)
        else:
            # その他の値（数値、boolean等）はインライン表示
            output.append(f"- **{key}:** `{value}`")

    return "\n".join(output)


def format_nested_json(data: Any, indent_level: int = 0) -> str:
    """ネストしたJSONを階層的にフォーマット

    複数階層のJSONを分解して、それぞれを読みやすいように表示する。
    最下層のvalueはコードブロックで表示する。

    Args:
        data: JSONデータ（dict, list, または基本型）
        indent_level: インデントレベル

    Returns:
        フォーマットされたMarkdown文字列
    """
    output: list[str] = []
    indent = "  " * indent_level

    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, dict):
                output.append(f"{indent}- **{key}:**")
                output.append(format_nested_json(value, indent_level + 1))
            elif isinstance(value, list):
                output.append(f"{indent}- **{key}:** (リスト: {len(value)}件)")
                for i, item in enumerate(value):
                    if isinstance(item, dict | list):
                        output.append(f"{indent}  - [{i}]:")
                        output.append(format_nested_json(item, indent_level + 2))
                    else:
                        # 単純な値のリストアイテム
                        item_str = str(item)
                        if "\n" in item_str or len(item_str) > 80:
                            open_fence, close_fence = get_fence(item_str)
                            output.append(f"{indent}  - [{i}]:")
                            output.append(f"{indent}    {open_fence}")
                            for line in item_str.split("\n"):
                                output.append(f"{indent}    {line}")
                            output.append(f"{indent}    {close_fence}")
                        else:
                            output.append(f"{indent}  - [{i}]: `{item_str}`")
            elif isinstance(value, str) and ("\n" in value or len(value) > 80):
                # 長い文字列またはマルチラインはコードブロックで表示
                output.append(f"{indent}- **{key}:**")
                open_fence, close_fence = get_fence(value)
                output.append(f"{indent}  {open_fence}")
                for line in value.split("\n"):
                    output.append(f"{indent}  {line}")
                output.append(f"{indent}  {close_fence}")
            else:
                # 短い値はインライン表示
                output.append(f"{indent}- **{key}:** `{value}`")
    elif isinstance(data, list):
        for i, item in enumerate(data):
            if isinstance(item, dict | list):
                output.append(f"{indent}- [{i}]:")
                output.append(format_nested_json(item, indent_level + 1))
            else:
                output.append(f"{indent}- [{i}]: `{item}`")
    else:
        # 基本型
        data_str = str(data)
        if "\n" in data_str or len(data_str) > 80:
            open_fence, close_fence = get_fence(data_str)
            output.append(f"{indent}{open_fence}")
            for line in data_str.split("\n"):
                output.append(f"{indent}{line}")
            output.append(f"{indent}{close_fence}")
        else:
            output.append(f"{indent}`{data_str}`")

    return "\n".join(output)


def format_messages(
    messages: list[dict[str, Any]],
    indent_level: int = 0,
    parent_index: int | None = None,
    member_submissions: list[dict[str, Any]] | None = None,
) -> str:
    """メッセージリストをMarkdown形式にフォーマット

    Pydantic AI形式のメッセージを処理する。
    - メッセージには "kind" キー（"request" or "response"）がある
    - パーツには "part_kind" キー（"user-prompt", "text", "tool-call", "tool-return"）がある

    Args:
        messages: Pydantic AI形式のメッセージリスト
        indent_level: インデントレベル（0=####, 1=#####, ...）
        parent_index: 親メッセージのインデックス（例: 4 → Message 4.1, 4.2...）
        member_submissions: Member Agent応答リスト（tool-returnにall_messagesを展開するため）

    Returns:
        フォーマットされたMarkdown文字列
    """
    if not messages:
        return "*メッセージなし*"

    # Member submissionsをagent_nameでインデックス化
    member_by_agent: dict[str, dict[str, Any]] = {}
    if member_submissions:
        for sub in member_submissions:
            agent_name = sub.get("agent_name", "")
            if agent_name:
                member_by_agent[agent_name] = sub

    header_prefix = "#" * (4 + indent_level)
    output: list[str] = []

    for i, msg in enumerate(messages, 1):
        # Pydantic AI形式: "kind" キーを使用（"request" or "response"）
        kind = msg.get("kind", "unknown")
        parts = msg.get("parts", [])
        finish_reason = msg.get("finish_reason", "")

        # メッセージ番号を決定
        msg_number = f"{parent_index}.{i}" if parent_index is not None else str(i)

        output.append(f"{header_prefix} Message {msg_number}: `{kind}`")
        if finish_reason:
            output.append(f"*finish_reason: {finish_reason}*\n")

        for part in parts:
            # Pydantic AI形式: "part_kind" キーを使用
            part_kind = part.get("part_kind", "unknown")

            if part_kind == "user-prompt":
                # ユーザープロンプト
                content = part.get("content", "")
                open_fence, close_fence = get_fence(content)
                output.append("**User Prompt:**")
                output.append(open_fence)
                output.append(content)
                output.append(f"{close_fence}\n")

            elif part_kind == "text":
                # テキスト応答
                content = part.get("content", "")
                open_fence, close_fence = get_fence(content)
                output.append(open_fence)
                output.append(content)
                output.append(f"{close_fence}\n")

            elif part_kind == "tool-call":
                # ツール呼び出し
                tool_name = part.get("tool_name", "unknown")
                tool_call_id = part.get("tool_call_id", "")
                args_raw = part.get("args", "{}")

                # argsはJSON文字列の場合があるのでパースを試みる
                if isinstance(args_raw, str):
                    try:
                        arguments = json.loads(args_raw)
                    except json.JSONDecodeError:
                        arguments = {"_raw": args_raw}
                else:
                    arguments = args_raw if isinstance(args_raw, dict) else {"_raw": args_raw}

                output.append(f"**Tool Call:** `{tool_name}`")
                output.append(f"- ID: `{tool_call_id}`")

                # argumentsを展開表示
                if arguments:
                    output.append("- Arguments:")
                    output.append(format_arguments_expanded(arguments, tool_name))
                output.append("")

            elif part_kind == "tool-return":
                # ツール応答
                tool_name = part.get("tool_name", "unknown")
                tool_call_id = part.get("tool_call_id", "")
                content = part.get("content", "")

                # delegate_to_XXX からagent_nameを抽出してMember Agentの思考ログを取得
                agent_name = tool_name.replace("delegate_to_", "") if tool_name.startswith("delegate_to_") else None
                member_sub = member_by_agent.get(agent_name, {}) if agent_name else {}
                all_messages = member_sub.get("all_messages")

                output.append(f"**Tool Response:** `{tool_name}`")
                output.append(f"- ID: `{tool_call_id}`")

                # Member Agentの思考ログがあれば展開
                if all_messages:
                    output.append("")
                    msg_count = len(all_messages)
                    summary = f"Member Agent `{agent_name}` の思考ログ ({msg_count} messages)"
                    output.append(f"<details><summary>{summary}</summary>\n")
                    # Member Agentのメッセージを再帰的にフォーマット
                    member_messages_formatted = format_messages(
                        all_messages,
                        indent_level=indent_level + 1,
                        parent_index=int(msg_number.split(".")[0]) if "." not in msg_number else None,
                    )
                    output.append(member_messages_formatted)
                    output.append("\n</details>\n")
                    output.append("- Final Result:")
                elif agent_name:
                    # delegate_to_XXXだがall_messagesがない場合
                    output.append(f"- *Member Agent `{agent_name}` の思考ログはありません*")
                    output.append("- Result:")
                else:
                    # 通常のツール応答
                    output.append("- Result:")

                # contentがJSON形式かチェックして適切にフォーマット
                content_str = str(content)
                parsed_content = None
                if content_str.strip().startswith(("{", "[")):
                    try:
                        parsed_content = json.loads(content_str)
                    except json.JSONDecodeError:
                        pass

                if parsed_content is not None and isinstance(parsed_content, dict | list):
                    # JSONを階層的に展開表示
                    output.append(format_nested_json(parsed_content, indent_level=1))
                    output.append("")
                else:
                    # 通常のテキストとして表示
                    open_fence, close_fence = get_fence(content_str)
                    output.append(open_fence)
                    output.append(content_str)
                    output.append(f"{close_fence}\n")

    return "\n".join(output)


def format_system_instructions(instructions: list[dict[str, Any]]) -> str:
    """system_instructionをMarkdown形式にフォーマット

    Args:
        instructions: system_instruction リスト

    Returns:
        フォーマットされたMarkdown文字列
    """
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


def format_score_details(score_details: dict[str, Any]) -> str:
    """スコア詳細をMarkdownテーブルにフォーマット

    Args:
        score_details: スコア詳細辞書

    Returns:
        フォーマットされたMarkdown文字列
    """
    output: list[str] = []

    overall_score = score_details.get("overall_score")
    if overall_score is not None:
        output.append(f"- **総合スコア:** {overall_score}")
        output.append("")

    metrics = score_details.get("metrics", [])
    if metrics:
        output.append("| メトリクス | スコア | コメント |")
        output.append("|-----------|--------|----------|")
        for m in metrics:
            name = m.get("metric_name", "N/A")
            score = m.get("score", "N/A")
            comment = m.get("evaluator_comment", "N/A")
            output.append(f"| {name} | {score} | {comment} |")

    return "\n".join(output) if output else "*スコア詳細なし*"


def format_metadata_section(
    execution_id: str,
    team_id: str,
    team_name: str,
    total_rounds: int,
    leader_model: str | None = None,
    leader_system_instruction: str | None = None,
    members: list[dict[str, Any]] | None = None,
) -> str:
    """メタ情報セクションをMarkdown形式にフォーマット

    Args:
        execution_id: 実行ID
        team_id: チームID
        team_name: チーム名
        total_rounds: 総ラウンド数
        leader_model: Leaderのモデル名（オプション）
        leader_system_instruction: Leaderのシステム指示（オプション）
        members: メンバー情報リスト（オプション）

    Returns:
        フォーマットされたMarkdown文字列
    """
    output: list[str] = []

    output.append("## メタ情報")
    output.append(f"- **execution_id:** `{execution_id}`")
    output.append(f"- **team_id:** `{team_id}`")
    output.append(f"- **team_name:** {team_name}")
    output.append(f"- **総ラウンド数:** {total_rounds}")
    output.append("")

    if leader_model:
        output.append("### Leader Agent")
        output.append(f"- **model:** `{leader_model}`")
        if leader_system_instruction:
            output.append("- **system_instruction:**")
            open_fence, close_fence = get_fence(leader_system_instruction)
            output.append(open_fence)
            output.append(leader_system_instruction)
            output.append(close_fence)
        output.append("")

    if members:
        output.append("### Member Agents")
        output.append("")
        output.append("| Agent名 | 種別 | モデル |")
        output.append("|---------|------|--------|")
        for member in members:
            name = member.get("agent_name", "unknown")
            agent_type = member.get("agent_type", "unknown")
            model = member.get("model", "unknown")
            output.append(f"| {name} | {agent_type} | {model} |")
        output.append("")

        # 各メンバーのsystem_instruction
        for member in members:
            name = member.get("agent_name", "unknown")
            sys_instr = member.get("system_instruction")
            if sys_instr:
                output.append(f"#### {name} の system_instruction")
                open_fence, close_fence = get_fence(sys_instr)
                output.append(open_fence)
                output.append(sys_instr)
                output.append(close_fence)
                output.append("")

    return "\n".join(output)

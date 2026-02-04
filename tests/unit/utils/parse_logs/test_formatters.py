"""フォーマッターのユニットテスト."""

from quant_insight.utils.parse_logs.common.fence import get_fence, parse_json_safe
from quant_insight.utils.parse_logs.duckdb.formatters import (
    format_messages,
    format_metadata_section,
    format_score_details,
    format_system_instructions,
)


class TestGetFence:
    """get_fence関数のテスト."""

    def test_no_special_chars(self) -> None:
        """特殊文字がない場合はバッククォート."""
        content = "Hello, world!"
        open_fence, close_fence = get_fence(content)
        assert open_fence == "```"
        assert close_fence == "```"

    def test_with_backticks(self) -> None:
        """バッククォートがある場合はチルダ."""
        content = "Here is some ```code```"
        open_fence, close_fence = get_fence(content)
        assert open_fence == "~~~"
        assert close_fence == "~~~"

    def test_with_both_backticks_and_tildes(self) -> None:
        """両方ある場合は長いチルダ."""
        content = "Here is ```code``` and ~~~more~~~"
        open_fence, close_fence = get_fence(content)
        # ~~~が3つなので4つ以上のチルダが必要
        assert open_fence.startswith("~~~~")
        assert close_fence.startswith("~~~~")

    def test_with_lang(self) -> None:
        """言語指定付き."""
        content = "Hello, world!"
        open_fence, close_fence = get_fence(content, "python")
        assert open_fence == "```python"
        assert close_fence == "```"


class TestParseJsonSafe:
    """parse_json_safe関数のテスト."""

    def test_valid_json_dict(self) -> None:
        """有効なJSON辞書."""
        result = parse_json_safe('{"key": "value"}')
        assert result == {"key": "value"}

    def test_valid_json_list(self) -> None:
        """有効なJSONリスト."""
        result = parse_json_safe("[1, 2, 3]")
        assert result == [1, 2, 3]

    def test_invalid_json(self) -> None:
        """無効なJSON."""
        result = parse_json_safe("not valid json")
        assert result is None

    def test_none_input(self) -> None:
        """None入力."""
        result = parse_json_safe(None)
        assert result is None

    def test_empty_string(self) -> None:
        """空文字列."""
        result = parse_json_safe("")
        assert result is None


class TestFormatMessages:
    """format_messages関数のテスト."""

    def test_empty_messages(self) -> None:
        """空のメッセージリスト."""
        result = format_messages([])
        assert result == "*メッセージなし*"

    def test_text_message(self) -> None:
        """テキストメッセージ (Pydantic AI形式)."""
        messages = [
            {
                "kind": "response",
                "parts": [{"part_kind": "text", "content": "Hello"}],
            }
        ]
        result = format_messages(messages)
        assert "Message 1: `response`" in result
        assert "Hello" in result

    def test_tool_call_message(self) -> None:
        """ツールコールメッセージ (Pydantic AI形式)."""
        messages = [
            {
                "kind": "response",
                "parts": [
                    {
                        "part_kind": "tool-call",
                        "tool_name": "test_tool",
                        "tool_call_id": "call_123",
                        "args": {"arg1": "value1"},
                    }
                ],
            }
        ]
        result = format_messages(messages)
        assert "Tool Call:" in result
        assert "`test_tool`" in result
        assert "`call_123`" in result
        # 新形式: argumentsが展開表示されているか確認
        assert "**arg1:**" in result
        assert "value1" in result

    def test_tool_call_execute_python_code(self) -> None:
        """execute_python_codeツールコールメッセージ."""
        messages = [
            {
                "kind": "response",
                "parts": [
                    {
                        "part_kind": "tool-call",
                        "tool_name": "execute_python_code",
                        "tool_call_id": "call_456",
                        "args": {"code": "print('hello')"},
                    }
                ],
            }
        ]
        result = format_messages(messages)
        assert "Tool Call:" in result
        assert "`execute_python_code`" in result
        assert "**code:**" in result
        # Pythonコードブロックとして表示されているか
        assert "```python" in result
        assert "print('hello')" in result

    def test_tool_return_with_json(self) -> None:
        """JSONを含むtool-returnメッセージ."""
        messages = [
            {
                "kind": "request",
                "parts": [
                    {
                        "part_kind": "tool-return",
                        "tool_name": "test_tool",
                        "tool_call_id": "call_123",
                        "content": '{"status": "success", "data": {"value": 42}}',
                    }
                ],
            }
        ]
        result = format_messages(messages)
        assert "Tool Response:" in result
        # 階層的に展開されているか
        assert "**status:**" in result
        assert "**data:**" in result

    def test_with_parent_index(self) -> None:
        """親インデックス付き (Pydantic AI形式)."""
        messages = [
            {
                "kind": "request",
                "parts": [{"part_kind": "user-prompt", "content": "Hello"}],
            }
        ]
        result = format_messages(messages, parent_index=2)
        assert "Message 2.1:" in result


class TestFormatSystemInstructions:
    """format_system_instructions関数のテスト."""

    def test_empty_instructions(self) -> None:
        """空の指示リスト."""
        result = format_system_instructions([])
        assert result == "*システム指示なし*"

    def test_single_instruction(self) -> None:
        """単一の指示."""
        instructions = [{"type": "text", "content": "Be helpful"}]
        result = format_system_instructions(instructions)
        assert "Be helpful" in result

    def test_multiple_instructions(self) -> None:
        """複数の指示."""
        instructions = [
            {"type": "text", "content": "Be helpful"},
            {"type": "text", "content": "Be concise"},
        ]
        result = format_system_instructions(instructions)
        assert "Instruction 1" in result
        assert "Instruction 2" in result


class TestFormatScoreDetails:
    """format_score_details関数のテスト."""

    def test_empty_details(self) -> None:
        """空の詳細."""
        result = format_score_details({})
        assert result == "*スコア詳細なし*"

    def test_with_overall_score(self) -> None:
        """総合スコアあり."""
        details = {"overall_score": 85.5}
        result = format_score_details(details)
        assert "総合スコア" in result
        assert "85.5" in result

    def test_with_metrics(self) -> None:
        """メトリクスあり."""
        details = {
            "overall_score": 85.5,
            "metrics": [
                {"metric_name": "accuracy", "score": 90, "evaluator_comment": "Good"},
                {"metric_name": "speed", "score": 80, "evaluator_comment": "OK"},
            ],
        }
        result = format_score_details(details)
        assert "accuracy" in result
        assert "90" in result
        assert "speed" in result


class TestFormatMetadataSection:
    """format_metadata_section関数のテスト."""

    def test_basic_metadata(self) -> None:
        """基本メタデータ."""
        result = format_metadata_section(
            execution_id="exec-123",
            team_id="team-a",
            team_name="Team Alpha",
            total_rounds=3,
        )
        assert "exec-123" in result
        assert "team-a" in result
        assert "Team Alpha" in result
        assert "3" in result

    def test_with_leader_info(self) -> None:
        """リーダー情報付き."""
        result = format_metadata_section(
            execution_id="exec-123",
            team_id="team-a",
            team_name="Team Alpha",
            total_rounds=3,
            leader_model="openai:gpt-4o",
            leader_system_instruction="Be helpful",
        )
        assert "Leader Agent" in result
        assert "openai:gpt-4o" in result
        assert "Be helpful" in result

    def test_with_members(self) -> None:
        """メンバー情報付き."""
        result = format_metadata_section(
            execution_id="exec-123",
            team_id="team-a",
            team_name="Team Alpha",
            total_rounds=3,
            members=[
                {
                    "agent_name": "analyzer",
                    "agent_type": "code_execution",
                    "model": "openai:gpt-4o-mini",
                    "system_instruction": "Analyze code",
                }
            ],
        )
        assert "Member Agents" in result
        assert "analyzer" in result
        assert "code_execution" in result

"""Unit tests for LocalCodeExecutorAgent."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mixseek.models.member_agent import MemberAgentConfig, ResultStatus

from quant_insight.agents.local_code_executor.agent import LocalCodeExecutorAgent
from quant_insight.agents.local_code_executor.models import LocalCodeExecutorConfig


@pytest.mark.unit
class TestLocalCodeExecutorAgentInit:
    """LocalCodeExecutorAgent初期化のテスト。"""

    @patch("quant_insight.agents.local_code_executor.agent.Agent")
    def test_init_with_valid_config(self, mock_agent_class, valid_member_config, tmp_path, monkeypatch):
        """有効な設定で初期化成功。"""
        monkeypatch.setenv("MIXSEEK_WORKSPACE", str(tmp_path))
        mock_agent_instance = MagicMock()
        mock_agent_class.return_value = mock_agent_instance

        agent = LocalCodeExecutorAgent(valid_member_config)

        assert agent.config == valid_member_config
        assert isinstance(agent.executor_config, LocalCodeExecutorConfig)
        assert agent.agent is not None

    def test_raises_when_tool_settings_missing(self, member_config_without_tool_settings):
        """tool_settings未設定時にValueErrorを発生。"""
        with pytest.raises(ValueError, match="TOMLに.*local_code_executor.*設定がありません"):
            LocalCodeExecutorAgent(member_config_without_tool_settings)

    def test_raises_when_local_code_executor_missing(self, member_config_without_executor):
        """local_code_executor設定未設定時にValueErrorを発生。"""
        with pytest.raises(ValueError, match="TOMLに.*local_code_executor.*設定がありません"):
            LocalCodeExecutorAgent(member_config_without_executor)

    @patch("quant_insight.agents.local_code_executor.agent.Agent")
    def test_creates_executor_config(self, mock_agent_class, valid_member_config, tmp_path, monkeypatch):
        """executor_configを正しく作成。"""
        monkeypatch.setenv("MIXSEEK_WORKSPACE", str(tmp_path))
        mock_agent_instance = MagicMock()
        mock_agent_class.return_value = mock_agent_instance

        agent = LocalCodeExecutorAgent(valid_member_config)

        assert agent.executor_config.available_data_paths == ["data/input"]
        assert agent.executor_config.timeout_seconds == 60

    @patch("quant_insight.agents.local_code_executor.agent.Agent")
    def test_creates_pydantic_ai_agent(self, mock_agent_class, valid_member_config, tmp_path, monkeypatch):
        """Pydantic AI Agentを作成。"""
        monkeypatch.setenv("MIXSEEK_WORKSPACE", str(tmp_path))
        mock_agent_instance = MagicMock()
        mock_agent_class.return_value = mock_agent_instance

        agent = LocalCodeExecutorAgent(valid_member_config)

        # Agent[LocalCodeExecutorConfig, str]型であることを確認
        assert hasattr(agent.agent, "run")


@pytest.mark.unit
class TestLocalCodeExecutorAgentExecute:
    """LocalCodeExecutorAgent.executeのテスト。"""

    @pytest.mark.asyncio
    @patch("quant_insight.agents.local_code_executor.agent.Agent")
    async def test_execute_returns_success_result(self, mock_agent_class, valid_member_config, tmp_path, monkeypatch):
        """成功時にSUCCESS結果を返す。"""
        monkeypatch.setenv("MIXSEEK_WORKSPACE", str(tmp_path))
        mock_agent_instance = MagicMock()
        mock_agent_class.return_value = mock_agent_instance

        # agent.agentをモックに置き換え
        mock_result = MagicMock()
        mock_result.output = "テスト出力"
        mock_agent_instance.run = AsyncMock(return_value=mock_result)

        agent = LocalCodeExecutorAgent(valid_member_config)
        result = await agent.execute("テストタスク")

        assert result.status == ResultStatus.SUCCESS
        assert result.content == "テスト出力"
        assert result.agent_name == "test-executor"
        assert result.agent_type == "custom"

    @pytest.mark.asyncio
    @patch("quant_insight.agents.local_code_executor.agent.Agent")
    async def test_execute_returns_error_result_on_exception(
        self, mock_agent_class, valid_member_config, tmp_path, monkeypatch
    ):
        """例外発生時にERROR結果を返す。"""
        monkeypatch.setenv("MIXSEEK_WORKSPACE", str(tmp_path))
        mock_agent_instance = MagicMock()
        mock_agent_class.return_value = mock_agent_instance

        # agent.agentがエラーを発生させるようにモック
        mock_agent_instance.run = AsyncMock(side_effect=RuntimeError("テストエラー"))

        agent = LocalCodeExecutorAgent(valid_member_config)
        result = await agent.execute("テストタスク")

        assert result.status == ResultStatus.ERROR
        assert "タスク実行エラー" in result.content
        assert result.error_message is not None
        assert "テストエラー" in result.error_message

    @pytest.mark.asyncio
    @patch("quant_insight.agents.local_code_executor.agent.Agent")
    async def test_execute_passes_task_to_agent(self, mock_agent_class, valid_member_config, tmp_path, monkeypatch):
        """タスクをPydantic AI Agentに渡す。"""
        monkeypatch.setenv("MIXSEEK_WORKSPACE", str(tmp_path))
        mock_agent_instance = MagicMock()
        mock_agent_class.return_value = mock_agent_instance

        mock_result = MagicMock()
        mock_result.output = "出力"
        mock_agent_instance.run = AsyncMock(return_value=mock_result)

        agent = LocalCodeExecutorAgent(valid_member_config)
        await agent.execute("タスク内容")

        # runが正しい引数で呼ばれたことを確認
        mock_agent_instance.run.assert_called_once()
        call_args = mock_agent_instance.run.call_args
        assert call_args[0][0] == "タスク内容"  # 第1引数がタスク

    @pytest.mark.asyncio
    @patch("quant_insight.agents.local_code_executor.agent.Agent")
    async def test_execute_passes_deps_to_agent(self, mock_agent_class, valid_member_config, tmp_path, monkeypatch):
        """依存関係をPydantic AI Agentに渡す。"""
        monkeypatch.setenv("MIXSEEK_WORKSPACE", str(tmp_path))
        mock_agent_instance = MagicMock()
        mock_agent_class.return_value = mock_agent_instance

        mock_result = MagicMock()
        mock_result.output = "出力"
        mock_agent_instance.run = AsyncMock(return_value=mock_result)

        agent = LocalCodeExecutorAgent(valid_member_config)
        await agent.execute("タスク")

        # runのdeps引数がexecutor_configであることを確認
        call_args = mock_agent_instance.run.call_args
        assert call_args[1]["deps"] == agent.executor_config


@pytest.mark.unit
class TestLocalCodeExecutorAgentOutputType:
    """LocalCodeExecutorAgent._resolve_output_typeのテスト。"""

    @patch("quant_insight.agents.local_code_executor.agent.Agent")
    def test_returns_str_when_no_output_model_config(
        self, mock_agent_class, valid_member_config, tmp_path, monkeypatch
    ):
        """output_model設定がない場合はstr型を返す。"""
        monkeypatch.setenv("MIXSEEK_WORKSPACE", str(tmp_path))
        mock_agent_class.return_value = MagicMock()

        agent = LocalCodeExecutorAgent(valid_member_config)

        # output_modelが未設定なのでstr型
        output_type = agent._resolve_output_type()
        assert output_type is str

    @patch("quant_insight.agents.local_code_executor.agent.Agent")
    def test_loads_output_model_when_configured(self, mock_agent_class, tmp_path, monkeypatch):
        """output_model設定がある場合はモデルをロード。"""
        monkeypatch.setenv("MIXSEEK_WORKSPACE", str(tmp_path))
        mock_agent_class.return_value = MagicMock()

        # output_model設定を含むconfig
        config = MemberAgentConfig(
            name="test-executor",
            type="custom",
            model="anthropic:claude-sonnet-4-5-20250929",
            metadata={
                "tool_settings": {
                    "local_code_executor": {
                        "available_data_paths": ["data/input"],
                        "timeout_seconds": 60,
                        "output_model": {
                            "module_path": "quant_insight.agents.local_code_executor.output_models",
                            "class_name": "AnalyzerOutput",
                        },
                    }
                }
            },
        )

        agent = LocalCodeExecutorAgent(config)
        output_type = agent._resolve_output_type()

        # AnalyzerOutputモデルがロードされる
        from quant_insight.agents.local_code_executor.output_models import AnalyzerOutput

        assert output_type is AnalyzerOutput


@pytest.mark.unit
class TestLocalCodeExecutorAgentLoadOutputModel:
    """LocalCodeExecutorAgent._load_output_modelのテスト。"""

    @patch("quant_insight.agents.local_code_executor.agent.Agent")
    def test_loads_valid_model_class(self, mock_agent_class, valid_member_config, tmp_path, monkeypatch):
        """有効なモデルクラスをロード。"""
        monkeypatch.setenv("MIXSEEK_WORKSPACE", str(tmp_path))
        mock_agent_class.return_value = MagicMock()

        agent = LocalCodeExecutorAgent(valid_member_config)

        model_class = agent._load_output_model(
            "quant_insight.agents.local_code_executor.output_models", "AnalyzerOutput"
        )

        from quant_insight.agents.local_code_executor.output_models import AnalyzerOutput

        assert model_class is AnalyzerOutput

    @patch("quant_insight.agents.local_code_executor.agent.Agent")
    def test_raises_on_invalid_module_path(self, mock_agent_class, valid_member_config, tmp_path, monkeypatch):
        """無効なモジュールパスでImportErrorを発生。"""
        monkeypatch.setenv("MIXSEEK_WORKSPACE", str(tmp_path))
        mock_agent_class.return_value = MagicMock()

        agent = LocalCodeExecutorAgent(valid_member_config)

        with pytest.raises(ImportError, match="モジュール .* のインポートに失敗しました"):
            agent._load_output_model("invalid.module.path", "SomeClass")

    @patch("quant_insight.agents.local_code_executor.agent.Agent")
    def test_raises_on_invalid_class_name(self, mock_agent_class, valid_member_config, tmp_path, monkeypatch):
        """無効なクラス名でAttributeErrorを発生。"""
        monkeypatch.setenv("MIXSEEK_WORKSPACE", str(tmp_path))
        mock_agent_class.return_value = MagicMock()

        agent = LocalCodeExecutorAgent(valid_member_config)

        with pytest.raises(AttributeError, match="モジュール .* にクラス .* が見つかりません"):
            agent._load_output_model("quant_insight.agents.local_code_executor.output_models", "NonExistentClass")

    @patch("quant_insight.agents.local_code_executor.agent.Agent")
    def test_raises_when_class_not_basemodel_subclass(
        self, mock_agent_class, valid_member_config, tmp_path, monkeypatch
    ):
        """BaseModelのサブクラスでないクラスでTypeErrorを発生。"""
        monkeypatch.setenv("MIXSEEK_WORKSPACE", str(tmp_path))
        mock_agent_class.return_value = MagicMock()

        agent = LocalCodeExecutorAgent(valid_member_config)

        # LocalCodeExecutorConfigはBaseModelだが、意図的に失敗させるためにstr型を使用
        with pytest.raises(TypeError, match="クラス .* はBaseModelのサブクラスではありません"):
            # pathlib.Pathクラスを試す（BaseModelでない）
            agent._load_output_model("pathlib", "Path")


@pytest.mark.unit
class TestLocalCodeExecutorAgentStructuredOutput:
    """LocalCodeExecutorAgent.executeの構造化出力テスト。"""

    @pytest.mark.asyncio
    @patch("quant_insight.agents.local_code_executor.agent.Agent")
    async def test_serializes_basemodel_output_to_json(self, mock_agent_class, tmp_path, monkeypatch):
        """BaseModel出力をJSON文字列にシリアライズ。"""
        monkeypatch.setenv("MIXSEEK_WORKSPACE", str(tmp_path))
        mock_agent_instance = MagicMock()
        mock_agent_class.return_value = mock_agent_instance

        # 構造化出力を返すモック
        from quant_insight.agents.local_code_executor.output_models import AnalyzerOutput, ScriptEntry

        mock_result = MagicMock()
        mock_result.output = AnalyzerOutput(
            scripts=[ScriptEntry(file_name="script.py", code="print('hello')")],
            report="テストレポート",
        )
        mock_agent_instance.run = AsyncMock(return_value=mock_result)

        config = MemberAgentConfig(
            name="test-executor",
            type="custom",
            model="anthropic:claude-sonnet-4-5-20250929",
            metadata={
                "tool_settings": {
                    "local_code_executor": {
                        "available_data_paths": ["data/input"],
                        "timeout_seconds": 60,
                    }
                }
            },
        )

        agent = LocalCodeExecutorAgent(config)
        result = await agent.execute("テストタスク")

        # JSON文字列にシリアライズされている
        assert result.status == ResultStatus.SUCCESS
        import json

        parsed = json.loads(result.content)
        assert parsed["scripts"] == [{"file_name": "script.py", "code": "print('hello')"}]
        assert parsed["report"] == "テストレポート"

    @pytest.mark.asyncio
    @patch("quant_insight.agents.local_code_executor.agent.Agent")
    async def test_uses_str_output_directly(self, mock_agent_class, valid_member_config, tmp_path, monkeypatch):
        """str出力はそのまま使用。"""
        monkeypatch.setenv("MIXSEEK_WORKSPACE", str(tmp_path))
        mock_agent_instance = MagicMock()
        mock_agent_class.return_value = mock_agent_instance

        # str出力を返すモック
        mock_result = MagicMock()
        mock_result.output = "テスト出力文字列"
        mock_agent_instance.run = AsyncMock(return_value=mock_result)

        agent = LocalCodeExecutorAgent(valid_member_config)
        result = await agent.execute("テストタスク")

        # そのまま使用される
        assert result.status == ResultStatus.SUCCESS
        assert result.content == "テスト出力文字列"

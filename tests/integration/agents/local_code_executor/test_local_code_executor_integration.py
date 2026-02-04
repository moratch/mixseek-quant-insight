"""Integration tests for LocalCodeExecutorAgent."""

import tomllib
from unittest.mock import MagicMock

import pytest
from mixseek.models.member_agent import MemberAgentConfig

from quant_insight.agents.local_code_executor.agent import LocalCodeExecutorAgent
from quant_insight.agents.local_code_executor.tools import (
    execute_python_code,
    list_available_data,
    list_preloaded_modules,
)


@pytest.mark.integration
class TestLocalCodeExecutorIntegration:
    """LocalCodeExecutorAgent統合テスト。"""

    def test_agent_with_real_subprocess_execution(self, tmp_path, monkeypatch):
        """実際のサブプロセスでコード実行。"""
        from unittest.mock import patch

        monkeypatch.setenv("MIXSEEK_WORKSPACE", str(tmp_path))

        # テスト用データファイル作成
        data_dir = tmp_path / "data" / "input"
        data_dir.mkdir(parents=True)
        (data_dir / "test.txt").write_text("test data")

        config = MemberAgentConfig(
            name="integration-test-executor",
            type="custom",
            model="anthropic:claude-sonnet-4-5-20250929",
            metadata={
                "tool_settings": {
                    "local_code_executor": {
                        "available_data_paths": ["data/input"],
                        "timeout_seconds": 10,
                    }
                }
            },
        )

        # Agentをモックして初期化
        with patch("quant_insight.agents.local_code_executor.agent.Agent"):
            agent = LocalCodeExecutorAgent(config)

        # RunContextを手動で作成してexecute_python_codeを直接テスト
        ctx = MagicMock()
        ctx.deps = agent.executor_config

        # Pythonコードを実行
        code = """
from pathlib import Path
print("Hello from subprocess")
print("Test completed successfully")
"""
        result = execute_python_code(ctx, code)

        assert "Hello from subprocess" in result
        assert "Test completed successfully" in result

    def test_tool_functions_with_real_filesystem(self, tmp_path, monkeypatch):
        """実際のファイルシステムでツール関数動作確認。"""
        monkeypatch.setenv("MIXSEEK_WORKSPACE", str(tmp_path))

        # テストデータ作成
        data_dir = tmp_path / "data" / "input"
        data_dir.mkdir(parents=True)
        (data_dir / "file1.parquet").write_bytes(b"data1" * 100)
        (data_dir / "file2.parquet").write_bytes(b"data2" * 200)

        from quant_insight.agents.local_code_executor.models import LocalCodeExecutorConfig

        ctx = MagicMock()
        ctx.deps = LocalCodeExecutorConfig(available_data_paths=["data/input"])

        # list_available_dataテスト
        result_data = list_available_data(ctx)
        assert "file1.parquet" in result_data
        assert "file2.parquet" in result_data
        assert "bytes" in result_data

        # list_preloaded_modulesテスト（DEFAULT_PRELOADED_MODULESが表示される）
        result_modules = list_preloaded_modules(ctx)
        assert "polars" in result_modules
        assert "pl" in result_modules
        assert "pandas" in result_modules
        assert "pd" in result_modules

    def test_config_loading_from_toml(self, tmp_path, monkeypatch):
        """TOMLファイルからの設定読み込み統合テスト。"""
        from unittest.mock import patch

        monkeypatch.setenv("MIXSEEK_WORKSPACE", str(tmp_path))

        # TOMLファイルを作成
        toml_content = """
[agent]
type = "custom"
name = "test-code-executor"
model = "anthropic:claude-sonnet-4-5-20250929"
description = "Test code execution agent"
temperature = 0.5

[agent.system_instruction]
text = "Test system instruction"

[agent.plugin]
path = "src/quant_insight/agents/local_code_executor/agent.py"
agent_class = "LocalCodeExecutorAgent"

[agent.metadata.tool_settings.local_code_executor]
available_data_paths = ["data/inputs/ohlcv/train.parquet", "data/inputs/returns/train.parquet"]
timeout_seconds = 120
"""
        toml_path = tmp_path / "test_agent.toml"
        toml_path.write_text(toml_content)

        # TOMLファイルを読み込み
        with open(toml_path, "rb") as f:
            toml_data = tomllib.load(f)

        # MemberAgentConfigを作成
        config = MemberAgentConfig.model_validate(toml_data["agent"])

        assert config.name == "test-code-executor"
        assert config.type == "custom"
        assert config.model == "anthropic:claude-sonnet-4-5-20250929"

        # LocalCodeExecutorAgentを初期化（Agentをモック）
        with patch("quant_insight.agents.local_code_executor.agent.Agent"):
            agent = LocalCodeExecutorAgent(config)

        assert agent.executor_config.available_data_paths == [
            "data/inputs/ohlcv/train.parquet",
            "data/inputs/returns/train.parquet",
        ]
        assert agent.executor_config.timeout_seconds == 120

    def test_config_loading_with_output_model(self, tmp_path, monkeypatch):
        """構造化出力モデルを含むTOML設定の読み込みテスト。"""
        from unittest.mock import patch

        monkeypatch.setenv("MIXSEEK_WORKSPACE", str(tmp_path))

        # output_model設定を含むTOMLファイルを作成
        toml_content = """
[agent]
type = "custom"
name = "test-analyzer"
model = "anthropic:claude-sonnet-4-5-20250929"
description = "Test analyzer agent with structured output"
temperature = 0.5

[agent.system_instruction]
text = "Test system instruction"

[agent.plugin]
path = "src/quant_insight/agents/local_code_executor/agent.py"
agent_class = "LocalCodeExecutorAgent"

[agent.metadata.tool_settings.local_code_executor]
available_data_paths = ["data/inputs/ohlcv/train.parquet"]
timeout_seconds = 120

[agent.metadata.tool_settings.local_code_executor.output_model]
module_path = "quant_insight.agents.local_code_executor.output_models"
class_name = "AnalyzerOutput"
"""
        toml_path = tmp_path / "test_analyzer_agent.toml"
        toml_path.write_text(toml_content)

        # TOMLファイルを読み込み
        with open(toml_path, "rb") as f:
            toml_data = tomllib.load(f)

        # MemberAgentConfigを作成
        config = MemberAgentConfig.model_validate(toml_data["agent"])

        assert config.name == "test-analyzer"
        assert config.metadata is not None
        assert "tool_settings" in config.metadata
        assert "local_code_executor" in config.metadata["tool_settings"]

        # LocalCodeExecutorAgentを初期化（Agentをモック）
        with patch("quant_insight.agents.local_code_executor.agent.Agent"):
            agent = LocalCodeExecutorAgent(config)

        # output_model設定が正しく読み込まれていることを確認
        assert agent.executor_config.output_model is not None
        assert (
            agent.executor_config.output_model.module_path == "quant_insight.agents.local_code_executor.output_models"
        )
        assert agent.executor_config.output_model.class_name == "AnalyzerOutput"

    async def test_structured_output_agent_execution(self, tmp_path, monkeypatch):
        """構造化出力を含むエージェント実行テスト。"""
        from unittest.mock import AsyncMock, patch

        monkeypatch.setenv("MIXSEEK_WORKSPACE", str(tmp_path))

        # テスト用データファイル作成
        data_dir = tmp_path / "data" / "inputs" / "ohlcv"
        data_dir.mkdir(parents=True)
        (data_dir / "train.parquet").write_text("test data")

        config = MemberAgentConfig(
            name="analyzer-test",
            type="custom",
            model="anthropic:claude-sonnet-4-5-20250929",
            metadata={
                "tool_settings": {
                    "local_code_executor": {
                        "available_data_paths": ["data/inputs/ohlcv/train.parquet"],
                        "timeout_seconds": 10,
                        "output_model": {
                            "module_path": "quant_insight.agents.local_code_executor.output_models",
                            "class_name": "AnalyzerOutput",
                        },
                    }
                }
            },
        )

        # output_modelsをインポートして実際のモデルを使用
        from quant_insight.agents.local_code_executor.output_models import AnalyzerOutput, ScriptEntry

        # Agentをモックして初期化
        with patch("quant_insight.agents.local_code_executor.agent.Agent") as mock_agent_class:
            # runの戻り値をモック
            mock_result = MagicMock()
            mock_result.output = AnalyzerOutput(
                scripts=[ScriptEntry(file_name="script1.py", code="print('hello')")],
                report="# 分析結果\nテスト分析レポート",
            )

            mock_agent_instance = AsyncMock()
            mock_agent_instance.run.return_value = mock_result
            mock_agent_class.return_value = mock_agent_instance

            agent = LocalCodeExecutorAgent(config)

            # executeを実行
            result = await agent.execute("テストデータを分析してください")

        # 構造化出力が正しく処理されていることを確認
        assert result.content is not None
        assert "scripts" in result.content
        assert "report" in result.content
        assert result.status.value == "success"

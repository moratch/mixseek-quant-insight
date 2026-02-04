"""Local Code Executor テスト用フィクスチャ。"""

from collections.abc import Generator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from mixseek.models.member_agent import MemberAgentConfig

from quant_insight.agents.local_code_executor.models import LocalCodeExecutorConfig


@pytest.fixture
def mock_ctx():
    """RunContext[LocalCodeExecutorConfig]のモック。"""
    ctx = MagicMock()
    ctx.deps = LocalCodeExecutorConfig(
        available_data_paths=["data/input"],
        timeout_seconds=60,
        max_output_chars=1000,
    )
    return ctx


@pytest.fixture
def valid_member_config():
    """有効なMemberAgentConfig。"""
    return MemberAgentConfig(
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


@pytest.fixture
def member_config_without_tool_settings():
    """tool_settings未設定のMemberAgentConfig。"""
    return MemberAgentConfig(
        name="test-executor",
        type="custom",
        model="anthropic:claude-sonnet-4-5-20250929",
        metadata={},
    )


@pytest.fixture
def member_config_without_executor():
    """local_code_executor設定未設定のMemberAgentConfig。"""
    return MemberAgentConfig(
        name="test-executor",
        type="custom",
        model="anthropic:claude-sonnet-4-5-20250929",
        metadata={"tool_settings": {}},
    )


@pytest.fixture
def tmp_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """一時ワークスペース環境（DuckDBスキーマ初期化済み）。"""
    from quant_insight.storage import ImplementationStore

    monkeypatch.setenv("MIXSEEK_WORKSPACE", str(tmp_path))
    # テスト用ディレクトリ/ファイル作成
    data_input = tmp_path / "data" / "input"
    data_input.mkdir(parents=True)
    (data_input / "test.parquet").write_bytes(b"dummy")

    # DuckDBスキーマを初期化
    store = ImplementationStore(workspace=tmp_path)
    store.initialize_schema()

    return tmp_path


@pytest.fixture
def mock_pydantic_agent() -> MagicMock:
    """Pydantic AI Agentのモック。"""
    mock_agent = MagicMock()
    mock_result = MagicMock()
    mock_result.output = "モックされた出力"
    mock_agent.run = AsyncMock(return_value=mock_result)
    return mock_agent


@pytest.fixture(autouse=True)
def init_duckdb_schema_for_agent_tests(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[Path]:
    """agent.pyテスト用にDuckDBスキーマを自動初期化。

    各テストの tmp_path でDuckDBスキーマを初期化し、
    MIXSEEK_WORKSPACE環境変数を設定する。
    これにより _verify_database_schema() が成功するようになる。
    """
    import quant_insight.storage.implementation_store as impl_store_module
    from quant_insight.storage import ImplementationStore

    # 環境変数を設定（tmp_pathを使用）
    monkeypatch.setenv("MIXSEEK_WORKSPACE", str(tmp_path))

    # グローバルシングルトンをリセット
    impl_store_module._instance = None

    # tmp_pathでスキーマを初期化（テストごとに独立したDB）
    store = ImplementationStore(workspace=tmp_path)
    store.initialize_schema()

    # シングルトンにこのストアを設定（get_implementation_store()が使用）
    impl_store_module._instance = store

    yield tmp_path

    # テスト後にシングルトンをクリーンアップ
    impl_store_module._instance = None

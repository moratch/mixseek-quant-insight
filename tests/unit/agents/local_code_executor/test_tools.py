"""Unit tests for local_code_executor tools."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from quant_insight.agents.local_code_executor.models import ImplementationContext, LocalCodeExecutorConfig
from quant_insight.agents.local_code_executor.tools import (
    _build_wrapper_script,
    _get_workspace_path,
    _resolve_data_path,
    execute_python_code,
    list_available_data,
    list_preloaded_modules,
    list_scripts,
    read_script,
)
from quant_insight.models.preloaded_module import PreloadedModule
from quant_insight.storage import ImplementationStore


@pytest.mark.unit
class TestGetWorkspacePath:
    """_get_workspace_path関数のテスト。"""

    def test_returns_path_when_env_set(self, monkeypatch):
        """環境変数設定時にPathを返す。"""
        monkeypatch.setenv("MIXSEEK_WORKSPACE", "/test/workspace")
        result = _get_workspace_path()
        assert result == Path("/test/workspace")

    def test_raises_when_env_not_set(self, monkeypatch):
        """環境変数未設定時にValueErrorを発生。"""
        monkeypatch.delenv("MIXSEEK_WORKSPACE", raising=False)
        with pytest.raises(ValueError, match="環境変数MIXSEEK_WORKSPACEが設定されていません"):
            _get_workspace_path()

    def test_raises_when_env_empty(self, monkeypatch):
        """環境変数が空文字列の場合にValueErrorを発生。"""
        monkeypatch.setenv("MIXSEEK_WORKSPACE", "")
        with pytest.raises(ValueError, match="環境変数MIXSEEK_WORKSPACEが設定されていません"):
            _get_workspace_path()


@pytest.mark.unit
class TestResolveDataPath:
    """_resolve_data_path関数のテスト。"""

    def test_resolves_relative_path(self, monkeypatch):
        """相対パスをワークスペース配下の絶対パスに変換。"""
        monkeypatch.setenv("MIXSEEK_WORKSPACE", "/workspace")
        result = _resolve_data_path("data/input")
        assert result == Path("/workspace/data/input")

    def test_handles_nested_path(self, monkeypatch):
        """ネストしたパスを正しく解決。"""
        monkeypatch.setenv("MIXSEEK_WORKSPACE", "/workspace")
        result = _resolve_data_path("data/input/train/2024/01")
        assert result == Path("/workspace/data/input/train/2024/01")


@pytest.mark.unit
class TestListAvailableData:
    """list_available_data関数のテスト。"""

    def test_returns_message_when_no_paths_configured(self):
        """パス未設定時にメッセージを返す。"""
        ctx = MagicMock()
        ctx.deps = LocalCodeExecutorConfig(available_data_paths=[])

        result = list_available_data(ctx)

        assert "データパスが設定されていません" in result

    def test_lists_existing_file(self, tmp_path, monkeypatch):
        """存在するファイルの情報を一覧表示。"""
        monkeypatch.setenv("MIXSEEK_WORKSPACE", str(tmp_path))
        test_file = tmp_path / "test.parquet"
        test_file.write_bytes(b"dummy" * 100)

        ctx = MagicMock()
        ctx.deps = LocalCodeExecutorConfig(available_data_paths=["test.parquet"])

        result = list_available_data(ctx)

        assert "test.parquet" in result
        assert "bytes" in result

    def test_lists_existing_directory(self, tmp_path, monkeypatch):
        """存在するディレクトリの内容を一覧表示。"""
        monkeypatch.setenv("MIXSEEK_WORKSPACE", str(tmp_path))
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "file1.parquet").write_bytes(b"data1")
        (data_dir / "file2.parquet").write_bytes(b"data2")

        ctx = MagicMock()
        ctx.deps = LocalCodeExecutorConfig(available_data_paths=["data"])

        result = list_available_data(ctx)

        assert "file1.parquet" in result
        assert "file2.parquet" in result

    def test_handles_non_existing_path(self, tmp_path, monkeypatch):
        """存在しないパスにメッセージを表示。"""
        monkeypatch.setenv("MIXSEEK_WORKSPACE", str(tmp_path))

        ctx = MagicMock()
        ctx.deps = LocalCodeExecutorConfig(available_data_paths=["nonexistent"])

        result = list_available_data(ctx)

        assert "パスが存在しません" in result

    def test_handles_empty_directory(self, tmp_path, monkeypatch):
        """空ディレクトリにメッセージを表示。"""
        monkeypatch.setenv("MIXSEEK_WORKSPACE", str(tmp_path))
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        ctx = MagicMock()
        ctx.deps = LocalCodeExecutorConfig(available_data_paths=["empty"])

        result = list_available_data(ctx)

        assert "空のディレクトリ" in result

    def test_limits_entries_to_50(self, tmp_path, monkeypatch):
        """エントリ数を50件に制限。"""
        monkeypatch.setenv("MIXSEEK_WORKSPACE", str(tmp_path))
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        # 60個のファイルを作成
        for i in range(60):
            (data_dir / f"file{i:03d}.txt").write_bytes(b"data")

        ctx = MagicMock()
        ctx.deps = LocalCodeExecutorConfig(available_data_paths=["data"])

        result = list_available_data(ctx)

        assert "他10件" in result

    def test_handles_permission_error(self, tmp_path, monkeypatch):
        """アクセス権限エラーを適切に処理。"""
        monkeypatch.setenv("MIXSEEK_WORKSPACE", str(tmp_path))
        restricted_dir = tmp_path / "restricted"
        restricted_dir.mkdir()

        ctx = MagicMock()
        ctx.deps = LocalCodeExecutorConfig(available_data_paths=["restricted"])

        # iterdir()がPermissionErrorを発生させるようにモック
        with patch("pathlib.Path.iterdir", side_effect=PermissionError("Access denied")):
            result = list_available_data(ctx)

        assert "アクセス権限がありません" in result


@pytest.mark.unit
class TestListPreloadedModules:
    """list_preloaded_modules関数のテスト。"""

    def test_lists_default_modules(self):
        """デフォルトモジュールを一覧表示。"""
        ctx = MagicMock()
        ctx.deps = LocalCodeExecutorConfig()

        result = list_preloaded_modules(ctx)

        # DEFAULT_PRELOADED_MODULESの主要モジュールが含まれることを確認
        assert "polars" in result
        assert "pl" in result
        assert "pandas" in result
        assert "pd" in result
        assert "numpy" in result
        assert "np" in result


@pytest.mark.unit
class TestBuildWrapperScript:
    """_build_wrapper_script関数のテスト。"""

    def test_builds_script_with_user_code(self):
        """ユーザーコードを含むスクリプトを生成。"""
        user_code = "print('Hello, World!')"
        script = _build_wrapper_script(user_code, [], [])

        assert "print('Hello, World!')" in script
        assert "exec(" in script

    def test_escapes_triple_quotes_in_user_code(self):
        """ユーザーコード内の三重引用符をエスケープ。"""
        user_code = 'print("""triple quotes""")'
        script = _build_wrapper_script(user_code, [], [])

        assert r"\"\"\"" in script

    def test_escapes_backslashes_in_user_code(self):
        """ユーザーコード内のバックスラッシュをエスケープ。"""
        user_code = r"print('path\\to\\file')"
        script = _build_wrapper_script(user_code, [], [])

        assert r"\\\\" in script

    def test_includes_preloaded_modules(self):
        """プレimportモジュールを含む。"""
        modules = [
            PreloadedModule(name="numpy", alias="np"),
            PreloadedModule(name="pandas", alias="pd"),
        ]
        script = _build_wrapper_script("pass", modules, [])

        assert "import numpy as np" in script
        assert "import pandas as pd" in script

    def test_includes_available_paths(self):
        """利用可能パスを含む（現在は未使用）。"""
        # Note: available_pathsパラメータは現在使用されていないが、
        # シグネチャの後方互換性のために保持されている
        paths = ["/workspace/data/input", "/workspace/data/output"]
        script = _build_wrapper_script("pass", [], paths)

        # スクリプトが正常に生成されることを確認
        assert "exec(" in script

    def test_handles_dotted_module_names_with_alias(self):
        """ドット付きモジュール名をエイリアス付きで正しく処理。"""
        modules = [PreloadedModule(name="sklearn.preprocessing", alias="preprocessing")]
        script = _build_wrapper_script("pass", modules, [])

        assert "import sklearn.preprocessing as preprocessing" in script

    def test_handles_module_without_alias(self):
        """エイリアスなしのモジュールをモジュール名そのままで処理。"""
        modules = [PreloadedModule(name="json")]
        script = _build_wrapper_script("pass", modules, [])

        assert "import json as json" in script


@pytest.mark.unit
class TestExecutePythonCode:
    """execute_python_code関数のテスト。"""

    def test_executes_simple_code(self, tmp_path, monkeypatch):
        """単純なコードを実行してstdoutを返す。"""
        monkeypatch.setenv("MIXSEEK_WORKSPACE", str(tmp_path))

        ctx = MagicMock()
        ctx.deps = LocalCodeExecutorConfig(
            available_data_paths=[],
            timeout_seconds=5,
        )

        result = execute_python_code(ctx, "print('Hello')")

        assert "Hello" in result

    def test_captures_stderr(self, tmp_path, monkeypatch):
        """stderrも出力に含める。"""
        monkeypatch.setenv("MIXSEEK_WORKSPACE", str(tmp_path))

        ctx = MagicMock()
        ctx.deps = LocalCodeExecutorConfig(
            available_data_paths=[],
            timeout_seconds=5,
        )

        result = execute_python_code(ctx, "import sys; print('error', file=sys.stderr)")

        assert "error" in result

    def test_returns_no_output_message(self, tmp_path, monkeypatch):
        """出力がない場合にメッセージを返す。"""
        monkeypatch.setenv("MIXSEEK_WORKSPACE", str(tmp_path))

        ctx = MagicMock()
        ctx.deps = LocalCodeExecutorConfig(
            available_data_paths=[],
            timeout_seconds=5,
        )

        result = execute_python_code(ctx, "x = 1 + 1")

        assert "出力なし" in result

    def test_handles_timeout(self, tmp_path, monkeypatch):
        """タイムアウト時にエラーメッセージを返す。"""
        monkeypatch.setenv("MIXSEEK_WORKSPACE", str(tmp_path))

        ctx = MagicMock()
        ctx.deps = LocalCodeExecutorConfig(
            available_data_paths=[],
            timeout_seconds=1,
        )

        result = execute_python_code(ctx, "import time; time.sleep(10)")

        assert "タイムアウト" in result

    def test_truncates_output_when_max_chars_set(self, tmp_path, monkeypatch):
        """max_output_chars設定時に出力を切り詰め。"""
        monkeypatch.setenv("MIXSEEK_WORKSPACE", str(tmp_path))

        ctx = MagicMock()
        ctx.deps = LocalCodeExecutorConfig(
            available_data_paths=[],
            timeout_seconds=5,
            max_output_chars=10,
        )

        result = execute_python_code(ctx, "print('A' * 100)")

        assert len(result) > 10  # メッセージ含む
        assert "切り詰め" in result

    def test_cleans_up_temp_file(self, tmp_path, monkeypatch):
        """一時ファイルをクリーンアップ。"""
        monkeypatch.setenv("MIXSEEK_WORKSPACE", str(tmp_path))

        ctx = MagicMock()
        ctx.deps = LocalCodeExecutorConfig(
            available_data_paths=[],
            timeout_seconds=5,
        )

        # 実行前の一時ファイル数を記録
        temp_dir = Path(os.environ.get("TMPDIR", "/tmp"))
        before_count = len([f for f in temp_dir.glob("tmp*.py") if f.is_file()])

        execute_python_code(ctx, "print('test')")

        # 実行後も一時ファイル数が増えていないことを確認
        after_count = len([f for f in temp_dir.glob("tmp*.py") if f.is_file()])
        assert after_count == before_count

    def test_does_not_inherit_arbitrary_environment_variables(self, tmp_path, monkeypatch):
        """セキュリティ対策: 任意の環境変数を継承しない（APIキーなどの漏洩防止）。"""
        monkeypatch.setenv("MIXSEEK_WORKSPACE", str(tmp_path))
        monkeypatch.setenv("TEST_VAR", "test_value")
        monkeypatch.setenv("MINKABU_API_KEY", "secret_key")

        ctx = MagicMock()
        ctx.deps = LocalCodeExecutorConfig(
            available_data_paths=[],
            timeout_seconds=5,
        )

        # 親プロセスの任意の環境変数はサブプロセスに渡されない
        result = execute_python_code(ctx, "import os; print(os.environ.get('TEST_VAR', 'NOT_FOUND'))")
        assert "NOT_FOUND" in result
        assert "test_value" not in result

        # 機密情報（APIキー）もサブプロセスに渡されない
        result2 = execute_python_code(ctx, "import os; print(os.environ.get('MINKABU_API_KEY', 'NOT_FOUND'))")
        assert "NOT_FOUND" in result2
        assert "secret_key" not in result2

    def test_sets_working_directory_to_workspace(self, tmp_path, monkeypatch):
        """作業ディレクトリをワークスペースに設定。"""
        monkeypatch.setenv("MIXSEEK_WORKSPACE", str(tmp_path))

        ctx = MagicMock()
        ctx.deps = LocalCodeExecutorConfig(
            available_data_paths=[],
            timeout_seconds=5,
        )

        result = execute_python_code(ctx, "import os; print(os.getcwd())")

        assert str(tmp_path) in result

    def test_handles_code_execution_error(self, tmp_path, monkeypatch):
        """コード実行エラーをキャプチャ。"""
        monkeypatch.setenv("MIXSEEK_WORKSPACE", str(tmp_path))

        ctx = MagicMock()
        ctx.deps = LocalCodeExecutorConfig(
            available_data_paths=[],
            timeout_seconds=5,
        )

        result = execute_python_code(ctx, "raise ValueError('Test error')")

        assert "ValueError" in result
        assert "Test error" in result


@pytest.fixture
def duckdb_ctx(tmp_path, monkeypatch):
    """DuckDB対応のテストコンテキストを作成。"""
    monkeypatch.setenv("MIXSEEK_WORKSPACE", str(tmp_path))

    # デフォルトのDBパス（{workspace}/mixseek.db）を使用してスキーマを初期化
    store = ImplementationStore(workspace=tmp_path)
    store.initialize_schema()

    # シングルトンをテスト用のストアに置き換え
    import quant_insight.storage.implementation_store as store_module

    monkeypatch.setattr(store_module, "_instance", store)

    # implementation_context付きのLocalCodeExecutorConfigを作成
    impl_ctx = ImplementationContext(
        execution_id="test-exec-123",
        team_id="test-team-1",
        round_number=1,
        member_agent_name="test-analyzer",
    )
    config = LocalCodeExecutorConfig(implementation_context=impl_ctx)

    ctx = MagicMock()
    ctx.deps = config
    return ctx


@pytest.mark.unit
class TestReadScript:
    """read_script関数のテスト（DuckDB対応）。"""

    async def test_reads_script_content(self, duckdb_ctx, tmp_path, monkeypatch):
        """スクリプトの内容を読み込む。"""
        from quant_insight.storage import get_implementation_store

        code = "print('Hello, World!')"

        # 先にstore経由で保存
        store = get_implementation_store()
        store.save_script_sync(
            execution_id="test-exec-123",
            team_id="test-team-1",
            round_number=1,
            member_agent_name="test-analyzer",
            file_name="test_script",
            code=code,
        )

        # 読み込み
        result = await read_script(duckdb_ctx, "test_script")

        assert result == code

    async def test_reads_multiline_script(self, duckdb_ctx, tmp_path, monkeypatch):
        """複数行のスクリプトを読み込む。"""
        from quant_insight.storage import get_implementation_store

        code = """import numpy as np
import pandas as pd

def main():
    print("Hello")

if __name__ == "__main__":
    main()
"""
        store = get_implementation_store()
        store.save_script_sync(
            execution_id="test-exec-123",
            team_id="test-team-1",
            round_number=1,
            member_agent_name="test-analyzer",
            file_name="multiline",
            code=code,
        )

        result = await read_script(duckdb_ctx, "multiline")

        assert result == code

    async def test_raises_when_script_not_found(self, duckdb_ctx):
        """スクリプトが存在しない場合にFileNotFoundErrorを発生。"""
        with pytest.raises(FileNotFoundError, match="スクリプトが見つかりません"):
            await read_script(duckdb_ctx, "non_existent.py")

    async def test_raises_when_implementation_context_not_set(self):
        """implementation_contextが設定されていない場合にValueErrorを発生。"""
        ctx = MagicMock()
        ctx.deps = LocalCodeExecutorConfig(implementation_context=None)

        with pytest.raises(ValueError, match="implementation_contextが設定されていません"):
            await read_script(ctx, "test.py")

    async def test_handles_utf8_encoding(self, duckdb_ctx, tmp_path, monkeypatch):
        """UTF-8エンコーディングを正しく処理。"""
        from quant_insight.storage import get_implementation_store

        code = "# 日本語のコメント\nprint('こんにちは')"
        store = get_implementation_store()
        store.save_script_sync(
            execution_id="test-exec-123",
            team_id="test-team-1",
            round_number=1,
            member_agent_name="test-analyzer",
            file_name="utf8_test",
            code=code,
        )

        result = await read_script(duckdb_ctx, "utf8_test")

        assert result == code
        assert "日本語" in result
        assert "こんにちは" in result

    async def test_reads_with_py_extension(self, duckdb_ctx, tmp_path, monkeypatch):
        """ファイル名に.py拡張子を含めて読み込み。"""
        from quant_insight.storage import get_implementation_store

        code = "pass"
        store = get_implementation_store()
        store.save_script_sync(
            execution_id="test-exec-123",
            team_id="test-team-1",
            round_number=1,
            member_agent_name="test-analyzer",
            file_name="extension_test.py",
            code=code,
        )

        # 拡張子付きで読み込み
        result = await read_script(duckdb_ctx, "extension_test.py")

        assert result == code


@pytest.mark.unit
class TestListScripts:
    """list_scripts関数のテスト。"""

    async def test_returns_message_when_no_scripts(self, duckdb_ctx):
        """スクリプトがない場合にメッセージを返す。"""
        result = await list_scripts(duckdb_ctx)
        assert "保存されているスクリプトはありません" in result

    async def test_lists_saved_scripts(self, duckdb_ctx, tmp_path, monkeypatch):
        """保存したスクリプトを一覧表示。"""
        from quant_insight.storage import get_implementation_store

        store = get_implementation_store()
        store.save_script_sync(
            execution_id="test-exec-123",
            team_id="test-team-1",
            round_number=1,
            member_agent_name="test-analyzer",
            file_name="script_a.py",
            code="pass",
        )
        store.save_script_sync(
            execution_id="test-exec-123",
            team_id="test-team-1",
            round_number=1,
            member_agent_name="test-analyzer",
            file_name="script_b.py",
            code="pass",
        )

        result = await list_scripts(duckdb_ctx)

        assert "script_a.py" in result
        assert "script_b.py" in result
        assert "# 保存済みスクリプト" in result

    async def test_raises_when_implementation_context_not_set(self):
        """implementation_contextが設定されていない場合にValueErrorを発生。"""
        ctx = MagicMock()
        ctx.deps = LocalCodeExecutorConfig(implementation_context=None)

        with pytest.raises(ValueError, match="implementation_contextが設定されていません"):
            await list_scripts(ctx)

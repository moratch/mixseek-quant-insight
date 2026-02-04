"""Integration test フィクスチャ for LocalCodeExecutorAgent."""

from collections.abc import Generator
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def init_duckdb_schema_for_integration_tests(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[Path]:
    """Integration test用にDuckDBスキーマを自動初期化。

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

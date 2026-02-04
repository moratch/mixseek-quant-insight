"""Unit tests for ImplementationStore."""

import pytest

from quant_insight.storage import (
    ImplementationStore,
)


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    """一時DuckDBを使用するストアを作成。"""
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("MIXSEEK_WORKSPACE", str(tmp_path))
    store = ImplementationStore(db_path=db_path)
    store.initialize_schema()
    return store


@pytest.mark.unit
class TestImplementationStoreInit:
    """ImplementationStoreの初期化テスト。"""

    def test_creates_db_file(self, tmp_path, monkeypatch):
        """DBファイルが作成される。"""
        db_path = tmp_path / "test.db"
        monkeypatch.setenv("MIXSEEK_WORKSPACE", str(tmp_path))
        store = ImplementationStore(db_path=db_path)
        store.initialize_schema()

        assert db_path.exists()

    def test_uses_default_db_path(self, tmp_path, monkeypatch):
        """デフォルトでmixseek.dbを使用。"""
        monkeypatch.setenv("MIXSEEK_WORKSPACE", str(tmp_path))
        store = ImplementationStore(workspace=tmp_path)

        assert store.db_path == tmp_path / "mixseek.db"

    def test_creates_parent_directory(self, tmp_path, monkeypatch):
        """親ディレクトリを作成。"""
        nested_db = tmp_path / "nested" / "dir" / "test.db"
        monkeypatch.setenv("MIXSEEK_WORKSPACE", str(tmp_path))
        store = ImplementationStore(db_path=nested_db)
        store.initialize_schema()

        assert nested_db.exists()


@pytest.mark.unit
class TestImplementationStoreTableExists:
    """table_existsメソッドのテスト。"""

    def test_returns_false_before_init(self, tmp_path, monkeypatch):
        """初期化前はFalseを返す。"""
        db_path = tmp_path / "test.db"
        monkeypatch.setenv("MIXSEEK_WORKSPACE", str(tmp_path))
        store = ImplementationStore(db_path=db_path)

        assert store.table_exists() is False

    def test_returns_true_after_init(self, tmp_db):
        """初期化後はTrueを返す。"""
        assert tmp_db.table_exists() is True


@pytest.mark.unit
class TestImplementationStoreSaveScriptSync:
    """save_script_syncメソッドのテスト。"""

    def test_saves_script(self, tmp_db):
        """スクリプトを保存。"""
        tmp_db.save_script_sync(
            execution_id="exec-123",
            team_id="team-1",
            round_number=1,
            member_agent_name="analyzer",
            file_name="test.py",
            code="print('hello')",
        )

        # 保存されたことを確認
        code = tmp_db.read_script_sync(
            execution_id="exec-123",
            team_id="team-1",
            round_number=1,
            file_name="test.py",
        )
        assert code == "print('hello')"

    def test_upsert_on_conflict(self, tmp_db):
        """同一キーの場合はUPSERT。"""
        # 最初の保存
        tmp_db.save_script_sync(
            execution_id="exec-123",
            team_id="team-1",
            round_number=1,
            member_agent_name="analyzer",
            file_name="test.py",
            code="old code",
        )

        # 同一キーで上書き
        tmp_db.save_script_sync(
            execution_id="exec-123",
            team_id="team-1",
            round_number=1,
            member_agent_name="analyzer",
            file_name="test.py",
            code="new code",
        )

        code = tmp_db.read_script_sync(
            execution_id="exec-123",
            team_id="team-1",
            round_number=1,
            file_name="test.py",
        )
        assert code == "new code"


@pytest.mark.unit
class TestImplementationStoreReadScriptSync:
    """read_script_syncメソッドのテスト。"""

    def test_returns_none_when_not_found(self, tmp_db):
        """存在しない場合はNoneを返す。"""
        code = tmp_db.read_script_sync(
            execution_id="nonexistent",
            team_id="team-1",
            round_number=1,
            file_name="test.py",
        )
        assert code is None

    def test_reads_saved_script(self, tmp_db):
        """保存したスクリプトを読み込む。"""
        tmp_db.save_script_sync(
            execution_id="exec-123",
            team_id="team-1",
            round_number=1,
            member_agent_name="analyzer",
            file_name="main.py",
            code="import numpy as np\nprint(np.__version__)",
        )

        code = tmp_db.read_script_sync(
            execution_id="exec-123",
            team_id="team-1",
            round_number=1,
            file_name="main.py",
        )

        assert "import numpy" in code
        assert "print" in code


@pytest.mark.unit
class TestImplementationStoreListScriptsSync:
    """list_scripts_syncメソッドのテスト。"""

    def test_returns_empty_list_when_no_scripts(self, tmp_db):
        """スクリプトがない場合は空リストを返す。"""
        scripts = tmp_db.list_scripts_sync(
            execution_id="exec-123",
            team_id="team-1",
            round_number=1,
        )
        assert scripts == []

    def test_lists_saved_scripts(self, tmp_db):
        """保存したスクリプトを一覧表示。"""
        tmp_db.save_script_sync(
            execution_id="exec-123",
            team_id="team-1",
            round_number=1,
            member_agent_name="analyzer",
            file_name="a_first.py",
            code="pass",
        )
        tmp_db.save_script_sync(
            execution_id="exec-123",
            team_id="team-1",
            round_number=1,
            member_agent_name="analyzer",
            file_name="b_second.py",
            code="pass",
        )

        scripts = tmp_db.list_scripts_sync(
            execution_id="exec-123",
            team_id="team-1",
            round_number=1,
        )

        assert len(scripts) == 2
        file_names = [s["file_name"] for s in scripts]
        assert "a_first.py" in file_names
        assert "b_second.py" in file_names

    def test_filters_by_context(self, tmp_db):
        """コンテキストでフィルタ。"""
        # 異なるコンテキストで保存
        tmp_db.save_script_sync(
            execution_id="exec-123",
            team_id="team-1",
            round_number=1,
            member_agent_name="analyzer",
            file_name="team1_round1.py",
            code="pass",
        )
        tmp_db.save_script_sync(
            execution_id="exec-123",
            team_id="team-2",
            round_number=1,
            member_agent_name="analyzer",
            file_name="team2_round1.py",
            code="pass",
        )

        # team-1のみ取得
        scripts = tmp_db.list_scripts_sync(
            execution_id="exec-123",
            team_id="team-1",
            round_number=1,
        )

        assert len(scripts) == 1
        assert scripts[0]["file_name"] == "team1_round1.py"


@pytest.mark.unit
class TestImplementationStoreAsync:
    """非同期メソッドのテスト。"""

    @pytest.mark.asyncio
    async def test_save_script_async(self, tmp_db):
        """非同期でスクリプトを保存。"""
        await tmp_db.save_script(
            execution_id="exec-async",
            team_id="team-1",
            round_number=1,
            member_agent_name="analyzer",
            file_name="async_test.py",
            code="print('async')",
        )

        code = tmp_db.read_script_sync(
            execution_id="exec-async",
            team_id="team-1",
            round_number=1,
            file_name="async_test.py",
        )
        assert code == "print('async')"

    @pytest.mark.asyncio
    async def test_read_script_async(self, tmp_db):
        """非同期でスクリプトを読み込み。"""
        tmp_db.save_script_sync(
            execution_id="exec-async",
            team_id="team-1",
            round_number=1,
            member_agent_name="analyzer",
            file_name="async_read.py",
            code="print('read')",
        )

        code = await tmp_db.read_script(
            execution_id="exec-async",
            team_id="team-1",
            round_number=1,
            file_name="async_read.py",
        )
        assert code == "print('read')"

    @pytest.mark.asyncio
    async def test_list_scripts_async(self, tmp_db):
        """非同期でスクリプト一覧を取得。"""
        tmp_db.save_script_sync(
            execution_id="exec-async",
            team_id="team-1",
            round_number=1,
            member_agent_name="analyzer",
            file_name="async_list.py",
            code="pass",
        )

        scripts = await tmp_db.list_scripts(
            execution_id="exec-async",
            team_id="team-1",
            round_number=1,
        )
        assert len(scripts) == 1
        assert scripts[0]["file_name"] == "async_list.py"

"""config CLIコマンドのユニットテスト."""

import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest
from typer.testing import CliRunner

from quant_insight.cli.main import app

runner = CliRunner()


@pytest.fixture
def temp_workspace() -> Iterator[Path]:
    """テスト用一時ワークスペース."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


class TestConfigInitCommand:
    """config initコマンドのテスト."""

    def test_help(self) -> None:
        """ヘルプ表示."""
        result = runner.invoke(app, ["config", "init", "--help"], color=False)
        assert result.exit_code == 0
        assert "workspace" in result.output.lower()
        assert "force" in result.output.lower()

    def test_copies_all_files_to_workspace(self, temp_workspace: Path) -> None:
        """ワークスペースに全ファイルがコピーされる."""
        result = runner.invoke(
            app,
            ["config", "init", "-w", str(temp_workspace)],
        )

        assert result.exit_code == 0

        # configs/ディレクトリが作成される
        configs_dir = temp_workspace / "configs"
        assert configs_dir.exists()

        # ルートのTOMLファイルがコピーされる
        assert (configs_dir / "competition.toml").exists()
        assert (configs_dir / "evaluator.toml").exists()
        assert (configs_dir / "orchestrator.toml").exists()

        # agents/配下もコピーされる
        assert (configs_dir / "agents" / "teams").is_dir()
        assert (configs_dir / "agents" / "members").is_dir()

        # teams配下のファイル
        teams_files = list((configs_dir / "agents" / "teams").glob("*.toml"))
        assert len(teams_files) >= 2

        # members配下のファイル
        members_files = list((configs_dir / "agents" / "members").glob("*.toml"))
        assert len(members_files) >= 4

    def test_prompts_when_configs_exists(self, temp_workspace: Path) -> None:
        """configs/が既存の場合は確認プロンプトを表示."""
        # 既存のconfigsディレクトリを作成
        configs_dir = temp_workspace / "configs"
        configs_dir.mkdir()
        (configs_dir / "existing.toml").write_text("existing = true")

        # 'n'で拒否
        result = runner.invoke(
            app,
            ["config", "init", "-w", str(temp_workspace)],
            input="n\n",
        )

        assert result.exit_code == 1
        # 既存ファイルは残っている
        assert (configs_dir / "existing.toml").exists()

    def test_force_deletes_existing_configs(self, temp_workspace: Path) -> None:
        """--forceで既存configs/を削除してからコピー."""
        # 既存のconfigsディレクトリを作成
        configs_dir = temp_workspace / "configs"
        configs_dir.mkdir()
        (configs_dir / "existing.toml").write_text("existing = true")

        result = runner.invoke(
            app,
            ["config", "init", "-w", str(temp_workspace), "--force"],
        )

        assert result.exit_code == 0

        # 既存ファイルは削除されている
        assert not (configs_dir / "existing.toml").exists()

        # 新しいファイルがコピーされている
        assert (configs_dir / "competition.toml").exists()

    def test_error_without_workspace(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """MIXSEEK_WORKSPACE未設定でエラー."""
        monkeypatch.delenv("MIXSEEK_WORKSPACE", raising=False)

        result = runner.invoke(
            app,
            ["config", "init"],
        )

        assert result.exit_code != 0
        assert "エラー" in result.output or "error" in result.output.lower()

    def test_uses_env_workspace(self, temp_workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """MIXSEEK_WORKSPACE環境変数を使用."""
        monkeypatch.setenv("MIXSEEK_WORKSPACE", str(temp_workspace))

        result = runner.invoke(
            app,
            ["config", "init"],
        )

        assert result.exit_code == 0
        assert (temp_workspace / "configs" / "competition.toml").exists()

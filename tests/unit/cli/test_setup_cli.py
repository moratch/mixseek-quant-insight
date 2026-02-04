"""setup CLIコマンドのユニットテスト."""

import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest
from typer.testing import CliRunner

from quant_insight.cli.commands.config import install_sample_configs
from quant_insight.cli.main import app

runner = CliRunner()


@pytest.fixture
def temp_workspace() -> Iterator[Path]:
    """テスト用一時ワークスペース."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def new_workspace() -> Iterator[Path]:
    """新規ワークスペース用（存在しないパス）."""
    with tempfile.TemporaryDirectory() as d:
        # サブディレクトリとして新規パスを返す（まだ存在しない）
        workspace = Path(d) / "workspace"
        yield workspace


class TestInstallSampleConfigs:
    """install_sample_configs関数のユニットテスト."""

    def test_copies_files_to_workspace(self, temp_workspace: Path) -> None:
        """ワークスペースにファイルをコピーする."""
        copied_files = install_sample_configs(temp_workspace, force=True)

        assert len(copied_files) >= 9

        configs_dir = temp_workspace / "configs"
        assert configs_dir.exists()
        assert (configs_dir / "competition.toml").exists()
        assert (configs_dir / "evaluator.toml").exists()
        assert (configs_dir / "orchestrator.toml").exists()

    def test_raises_when_configs_exists_without_force(self, temp_workspace: Path) -> None:
        """force=Falseでconfigs/が既存の場合はFileExistsError."""
        configs_dir = temp_workspace / "configs"
        configs_dir.mkdir()

        with pytest.raises(FileExistsError):
            install_sample_configs(temp_workspace, force=False)

    def test_force_overwrites_existing(self, temp_workspace: Path) -> None:
        """force=Trueで既存configs/を上書き."""
        configs_dir = temp_workspace / "configs"
        configs_dir.mkdir()
        (configs_dir / "old_file.toml").write_text("old")

        install_sample_configs(temp_workspace, force=True)

        # 古いファイルは削除されている
        assert not (configs_dir / "old_file.toml").exists()
        # 新しいファイルがコピーされている
        assert (configs_dir / "competition.toml").exists()


class TestSetupCommand:
    """setupコマンドのテスト."""

    def test_help(self) -> None:
        """ヘルプ表示."""
        result = runner.invoke(app, ["setup", "--help"], color=False)
        assert result.exit_code == 0
        assert "workspace" in result.output.lower()
        assert "mixseek init" in result.output.lower() or "一括" in result.output

    def test_setup_creates_workspace_structure(self, new_workspace: Path) -> None:
        """セットアップでワークスペース構造が作成される."""
        result = runner.invoke(
            app,
            ["setup", "-w", str(new_workspace)],
        )

        assert result.exit_code == 0
        assert "セットアップ完了" in result.output

        # mixseek initで作成されるディレクトリ
        assert (new_workspace / "logs").is_dir()
        assert (new_workspace / "templates").is_dir()

        # config initでコピーされる設定
        configs_dir = new_workspace / "configs"
        assert configs_dir.is_dir()
        assert (configs_dir / "competition.toml").exists()
        assert (configs_dir / "evaluator.toml").exists()
        assert (configs_dir / "orchestrator.toml").exists()

        # db initで作成されるデータベース
        assert (new_workspace / "mixseek.db").exists()

    def test_uses_env_workspace(self, new_workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """MIXSEEK_WORKSPACE環境変数を使用."""
        monkeypatch.setenv("MIXSEEK_WORKSPACE", str(new_workspace))

        result = runner.invoke(app, ["setup"])

        assert result.exit_code == 0
        assert (new_workspace / "configs" / "competition.toml").exists()

    def test_error_without_workspace(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """MIXSEEK_WORKSPACE未設定でエラー."""
        monkeypatch.delenv("MIXSEEK_WORKSPACE", raising=False)

        result = runner.invoke(app, ["setup"])

        assert result.exit_code != 0

    def test_progress_messages(self, new_workspace: Path) -> None:
        """進捗メッセージが表示される."""
        result = runner.invoke(
            app,
            ["setup", "-w", str(new_workspace)],
        )

        assert "Step 1/3" in result.output
        assert "Step 2/3" in result.output
        assert "Step 3/3" in result.output

    def test_existing_workspace_prompts_confirmation(self, new_workspace: Path) -> None:
        """既存ワークスペースでは確認プロンプトが表示される."""
        # 最初のセットアップ
        result = runner.invoke(app, ["setup", "-w", str(new_workspace)])
        assert result.exit_code == 0

        # 再実行時は確認プロンプトで'n'を入力
        result = runner.invoke(
            app,
            ["setup", "-w", str(new_workspace)],
            input="n\n",
        )

        # mixseek initの確認で中断される
        assert result.exit_code != 0

    def test_config_always_overwritten(self, new_workspace: Path) -> None:
        """config initは常に強制上書きされる（再実行時）."""
        # 最初のセットアップ
        result = runner.invoke(app, ["setup", "-w", str(new_workspace)])
        assert result.exit_code == 0

        # configsに追加ファイルを作成
        configs_dir = new_workspace / "configs"
        (configs_dir / "old_file.toml").write_text("old")

        # 再実行時は'y'で確認を通過
        result = runner.invoke(
            app,
            ["setup", "-w", str(new_workspace)],
            input="y\n",
        )

        assert result.exit_code == 0
        # 古いファイルは削除されている
        assert not (configs_dir / "old_file.toml").exists()
        # 新しいファイルがコピーされている
        assert (configs_dir / "competition.toml").exists()

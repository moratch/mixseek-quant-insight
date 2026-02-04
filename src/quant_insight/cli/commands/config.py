"""設定管理CLIコマンド.

サンプル設定ファイルのインストール機能を提供。
"""

import shutil
from pathlib import Path

import typer

from quant_insight.utils.env import get_workspace

config_app = typer.Typer(
    name="config",
    help="設定管理コマンド",
)


def _get_examples_dir() -> Path:
    """パッケージ外のexamplesディレクトリを取得.

    Returns:
        examplesディレクトリの絶対パス

    Raises:
        FileNotFoundError: examplesディレクトリが見つからない場合
    """
    # プロジェクトルートからの相対パスで取得
    # cli/commands/config.py -> src/quant_insight/cli/commands/
    # -> src/quant_insight/cli/ -> src/quant_insight/ -> src/ -> project_root/
    project_root = Path(__file__).parent.parent.parent.parent.parent
    examples_dir = project_root / "examples"

    if not examples_dir.exists():
        msg = f"examplesディレクトリが見つかりません: {examples_dir}"
        raise FileNotFoundError(msg)

    return examples_dir


def _copy_examples(examples_dir: Path, dest_dir: Path) -> list[Path]:
    """examplesディレクトリの内容をコピー.

    Args:
        examples_dir: コピー元のexamplesディレクトリ
        dest_dir: コピー先のconfigsディレクトリ

    Returns:
        コピーされたファイルのリスト
    """
    copied_files: list[Path] = []

    for src_file in examples_dir.rglob("*.toml"):
        # 相対パスを維持してコピー
        rel_path = src_file.relative_to(examples_dir)
        dest_file = dest_dir / rel_path

        # 親ディレクトリを作成
        dest_file.parent.mkdir(parents=True, exist_ok=True)

        # ファイルをコピー
        shutil.copy2(src_file, dest_file)
        copied_files.append(rel_path)

    return copied_files


def install_sample_configs(workspace: Path, *, force: bool = False) -> list[Path]:
    """サンプル設定をワークスペースにインストール.

    Args:
        workspace: ワークスペースパス
        force: True の場合、既存ディレクトリを確認なしで削除

    Returns:
        コピーされたファイルの相対パスリスト

    Raises:
        FileExistsError: force=False で configs/ が既に存在する場合
        FileNotFoundError: examples ディレクトリが見つからない場合
    """
    configs_dir = workspace / "configs"

    if configs_dir.exists() and not force:
        raise FileExistsError(f"{configs_dir} は既に存在します")

    if configs_dir.exists():
        shutil.rmtree(configs_dir)

    examples_dir = _get_examples_dir()
    configs_dir.mkdir(parents=True, exist_ok=True)
    return _copy_examples(examples_dir, configs_dir)


@config_app.command("init")
def init(
    workspace: Path | None = typer.Option(
        None,
        "--workspace",
        "-w",
        help="ワークスペースパス（未指定時は$MIXSEEK_WORKSPACE）",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="既存のconfigs/ディレクトリを削除してからコピー",
    ),
) -> None:
    """サンプル設定ファイルをワークスペースにコピー.

    examples/配下の設定ファイル（competition.toml, orchestrator.toml,
    evaluator.toml, agents/）を$MIXSEEK_WORKSPACE/configs/にコピーします。

    Examples:
        quant-insight config init
        quant-insight config init --workspace /path/to/workspace
        quant-insight config init --force
    """

    try:
        # ワークスペース解決
        ws = workspace if workspace else get_workspace()

        # インストール実行
        try:
            copied_files = install_sample_configs(ws, force=force)
        except FileExistsError:
            if not typer.confirm(
                f"{ws / 'configs'} は既に存在します。削除して再作成しますか？",
                default=False,
            ):
                typer.echo("キャンセルしました。", err=True)
                raise typer.Exit(code=1)
            copied_files = install_sample_configs(ws, force=True)

        # 結果表示
        typer.echo(f"サンプル設定をコピーしました: {ws / 'configs'}")
        for file_path in sorted(copied_files):
            typer.echo(f"  - {file_path}")

    except (ValueError, FileNotFoundError) as e:
        typer.echo(f"エラー: {e}", err=True)
        raise typer.Exit(code=1) from e

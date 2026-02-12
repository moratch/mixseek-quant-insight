"""CLI entry point for quant-insight."""

from pathlib import Path

import typer
from mixseek.cli.commands.init import init as mixseek_init

from quant_insight import __version__
from quant_insight.cli.commands import config_app, data_app, db_app, export_app, screening_app
from quant_insight.cli.commands.config import install_sample_configs
from quant_insight.storage import ImplementationStore
from quant_insight.utils.env import get_workspace

app = typer.Typer(
    name="quant-insight",
    help="Agentic trading strategies using mixseek-core framework",
)

# サブコマンドを登録
app.add_typer(config_app, name="config")
app.add_typer(data_app, name="data")
app.add_typer(db_app, name="db")
app.add_typer(export_app, name="export")
app.add_typer(screening_app, name="screening")


@app.command()
def version() -> None:
    """Show version information."""
    typer.echo(f"quant-insight version {__version__}")


@app.command()
def setup(
    workspace: Path | None = typer.Option(
        None,
        "--workspace",
        "-w",
        help="ワークスペースパス（未指定時は$MIXSEEK_WORKSPACE）",
    ),
) -> None:
    """環境を一括セットアップ（mixseek init → config init → db init）"""

    # ワークスペース解決
    ws = workspace if workspace else get_workspace()

    # Step 1: mixseek init（本家を厳密に呼び出し）
    typer.echo("Step 1/3: ワークスペース構造を作成...")
    try:
        mixseek_init(workspace=ws)
    except SystemExit as e:
        if e.code != 0:
            raise typer.Exit(code=e.code if isinstance(e.code, int) else 1)

    # Step 2: config init（常に強制上書き）
    typer.echo("Step 2/3: サンプル設定をコピー...")
    copied_files = install_sample_configs(ws, force=True)
    typer.echo(f"  {len(copied_files)} ファイルをコピーしました")

    # Step 3: db init
    typer.echo("Step 3/3: データベースを初期化...")
    store = ImplementationStore(workspace=ws)
    store.initialize_schema()
    typer.echo(f"  {store.db_path}")

    typer.echo("セットアップ完了")


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version_flag: bool = typer.Option(False, "--version", "-v", help="Show version information"),
) -> None:
    """Agentic trading strategies using mixseek-core framework."""
    if version_flag:
        typer.echo(f"quant-insight version {__version__}")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


if __name__ == "__main__":
    app()

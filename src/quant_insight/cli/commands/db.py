"""データベース管理CLIコマンド

エージェント実装保存用のDuckDBスキーマを初期化するコマンドを提供。
"""

from pathlib import Path

import typer

db_app = typer.Typer(
    name="db",
    help="データベース管理コマンド",
)


@db_app.command("init")
def init_db(
    workspace: Path | None = typer.Option(
        None,
        "--workspace",
        "-w",
        help="ワークスペースパス（未指定時は$MIXSEEK_WORKSPACE）",
    ),
) -> None:
    """DuckDBスキーマを初期化

    agent_implementationテーブルを作成します。
    既存テーブルがある場合は何もしません（べき等）。
    """
    from quant_insight.storage import ImplementationStore

    try:
        store = ImplementationStore(workspace=workspace)
        store.initialize_schema()

        typer.echo(f"データベースを初期化しました: {store.db_path}")
    except Exception as e:
        typer.echo(f"エラー: {e}", err=True)
        raise typer.Exit(code=1) from e

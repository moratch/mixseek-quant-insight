"""エクスポートCLIコマンド

DuckDBからMarkdownへのログエクスポート機能を提供。
"""

from pathlib import Path

import typer

from quant_insight.utils.parse_logs.duckdb.exporter import (
    LeaderBoardExporter,
    RoundHistoryExporter,
    load_team_configs_from_orchestrator,
)

export_app = typer.Typer(
    name="export",
    help="ログエクスポートコマンド",
)


@export_app.command("logs")
def export_logs(
    execution_id: str = typer.Argument(..., help="エクスポート対象のexecution_id"),
    config: Path = typer.Option(
        ...,
        "--config",
        "-c",
        help="orchestrator.tomlパス（必須）",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="出力ディレクトリ（デフォルト: $MIXSEEK_WORKSPACE/data/outputs/export）",
    ),
    workspace: Path | None = typer.Option(
        None,
        "--workspace",
        "-w",
        help="ワークスペースパス（未指定時は$MIXSEEK_WORKSPACE）",
    ),
    team: str | None = typer.Option(
        None,
        "--team",
        "-t",
        help="特定チームのみエクスポート",
    ),
    logs_only: bool = typer.Option(
        False,
        "--logs-only",
        help="ログMDのみエクスポート",
    ),
    submissions_only: bool = typer.Option(
        False,
        "--submissions-only",
        help="サブミッションMDのみエクスポート",
    ),
) -> None:
    """DuckDBからMarkdownログをエクスポート

    execution_idで指定された実行のログをMarkdown形式でエクスポートします。
    orchestrator.tomlからチーム設定を読み込み、メタデータを含むログを生成します。
    """
    from mixseek.orchestrator import load_orchestrator_settings
    from mixseek.utils.env import get_workspace_path

    try:
        # ワークスペース解決
        if workspace is None:
            workspace = get_workspace_path(cli_arg=None)

        # 出力ディレクトリ解決
        if output is None:
            output = workspace / "data" / "outputs" / "export"

        typer.echo(f"Workspace: {workspace}")
        typer.echo(f"Execution ID: {execution_id}")
        typer.echo(f"Config: {config}")

        # orchestrator.toml読み込み
        orchestrator_settings = load_orchestrator_settings(config, workspace)

        # チーム設定読み込み
        team_configs = load_team_configs_from_orchestrator(orchestrator_settings, workspace)
        typer.echo(f"Loaded {len(team_configs)} team configs")

        # DuckDBパス
        db_path = workspace / "mixseek.db"
        if not db_path.exists():
            typer.echo(f"エラー: DuckDBファイルが見つかりません: {db_path}", err=True)
            raise typer.Exit(code=1)

        # 出力ディレクトリ作成
        output.mkdir(parents=True, exist_ok=True)

        exported_files: list[Path] = []

        # ログエクスポート
        if not submissions_only:
            typer.echo("Exporting logs...")
            logs_exporter = RoundHistoryExporter(db_path, team_configs)

            if team:
                if team not in team_configs:
                    typer.echo(f"エラー: チーム '{team}' が見つかりません", err=True)
                    raise typer.Exit(code=1)
                exported_files.append(logs_exporter.export_team_log(execution_id, team, output))
            else:
                exported_files.extend(logs_exporter.export_all_teams(execution_id, output))

        # サブミッションエクスポート
        if not logs_only:
            typer.echo("Exporting submissions...")
            submissions_exporter = LeaderBoardExporter(db_path, team_configs)

            if team:
                if team not in team_configs:
                    typer.echo(f"エラー: チーム '{team}' が見つかりません", err=True)
                    raise typer.Exit(code=1)
                exported_files.append(submissions_exporter.export_team_submissions(execution_id, team, output))
            else:
                exported_files.extend(submissions_exporter.export_all_teams(execution_id, output))

        # 結果出力
        typer.echo("")
        typer.echo(f"エクスポート完了: {len(exported_files)} ファイル")
        for file_path in exported_files:
            typer.echo(f"  - {file_path}")

    except Exception as e:
        typer.echo(f"エラー: {e}", err=True)
        raise typer.Exit(code=1) from e

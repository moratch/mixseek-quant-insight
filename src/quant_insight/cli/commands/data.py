"""Data build CLI commands."""

import asyncio
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Annotated

import polars as pl
import typer

from quant_insight.data_build.data_splitter import DataSplitter
from quant_insight.data_build.jquants import JQuantsAdapter, JQuantsPlan, JQuantsUniverse
from quant_insight.data_build.jquants.adapter import get_default_date_range
from quant_insight.data_build.return_builder import ReturnBuilder
from quant_insight.utils.config import load_competition_config
from quant_insight.utils.env import get_data_inputs_dir, get_raw_data_dir

data_app = typer.Typer(
    name="data",
    help="Data build commands for fetching, processing, and splitting data",
)


def _get_default_start_date() -> date:
    """デフォルト開始日（終了日から2年前）."""
    start, _ = get_default_date_range()
    return start


def _get_default_end_date() -> date:
    """デフォルト終了日（12週間前）."""
    _, end = get_default_date_range()
    return end


def _parse_date(date_str: str | None, default_getter: Callable[[], date]) -> date:
    """日付文字列をパース.

    Args:
        date_str: YYYY-MM-DD形式の文字列、またはNone
        default_getter: デフォルト日付を返す関数

    Returns:
        パースされた日付
    """
    if date_str is None:
        return default_getter()
    return date.fromisoformat(date_str)


@data_app.command("fetch-jquants")
def fetch_jquants(
    plan: Annotated[
        JQuantsPlan,
        typer.Option("--plan", "-p", help="J-Quants API plan (free/light/standard/premium)"),
    ] = JQuantsPlan.FREE,
    universe: Annotated[
        JQuantsUniverse,
        typer.Option("--universe", "-u", help="Target universe (prime/standard/growth/all)"),
    ] = JQuantsUniverse.PRIME,
    start_date_str: Annotated[
        str | None,
        typer.Option("--start-date", "-s", help="Start date (YYYY-MM-DD). Default: end_date - 2 years"),
    ] = None,
    end_date_str: Annotated[
        str | None,
        typer.Option("--end-date", "-e", help="End date (YYYY-MM-DD). Default: 12 weeks ago"),
    ] = None,
) -> None:
    """Fetch OHLCV and master data from J-Quants API.

    Requires JQUANTS_API_KEY and MIXSEEK_WORKSPACE environment variables.

    Example:
        quant-insight data fetch-jquants --plan free --universe prime
    """
    # 日付パース
    start_date = _parse_date(start_date_str, _get_default_start_date)
    end_date = _parse_date(end_date_str, _get_default_end_date)

    typer.echo("Fetching J-Quants data...")
    typer.echo(f"  Plan: {plan.value}")
    typer.echo(f"  Universe: {universe.value}")
    typer.echo(f"  Period: {start_date} to {end_date}")

    async def _fetch() -> None:
        adapter = JQuantsAdapter(plan=plan, universe=universe)

        try:
            await adapter.authenticate()
            typer.echo("  Authentication: OK")

            # 銘柄一覧を取得
            symbols = await adapter.get_universe()
            typer.echo(f"  Target symbols: {len(symbols)}")

            # データ取得
            data = await adapter.fetch_all_data(symbols, start_date, end_date)

            # 保存
            output_dir = get_raw_data_dir()
            adapter.save(data, output_dir)

            typer.echo(f"\nData saved to: {output_dir}")
            typer.echo(f"  ohlcv.parquet: {len(data['ohlcv'])} rows")
            typer.echo(f"  master.parquet: {len(data['master'])} rows")

        finally:
            await adapter.close()

    asyncio.run(_fetch())
    typer.echo("\nDone!")


@data_app.command("build-returns")
def build_returns(
    config_path: Annotated[
        Path,
        typer.Option("--config", "-c", help="Path to competition.toml"),
    ],
) -> None:
    """Calculate returns from OHLCV data.

    Uses window and method from [competition.return_definition] in config.

    Example:
        quant-insight data build-returns --config $MIXSEEK_WORKSPACE/configs/competition.toml
    """
    # 設定読み込み
    config = load_competition_config(config_path)
    if config is None:
        typer.echo(f"Error: Failed to load config from {config_path}", err=True)
        raise typer.Exit(1)

    typer.echo("Building returns...")
    typer.echo(f"  Window: {config.return_definition.window}")
    typer.echo(f"  Method: {config.return_definition.method}")

    # OHLCVデータ読み込み
    raw_dir = get_raw_data_dir()
    ohlcv_path = raw_dir / "ohlcv.parquet"

    if not ohlcv_path.exists():
        typer.echo(f"Error: OHLCV data not found at {ohlcv_path}", err=True)
        typer.echo("Run 'quant-insight data fetch-jquants' first.", err=True)
        raise typer.Exit(1)

    ohlcv = pl.read_parquet(ohlcv_path)
    typer.echo(f"  Loaded OHLCV: {len(ohlcv)} rows")

    # リターン計算
    builder = ReturnBuilder()
    returns = builder.calculate_returns(
        ohlcv,
        window=config.return_definition.window,
        method=config.return_definition.method,
    )

    # 保存
    output_path = raw_dir / "returns.parquet"
    returns.write_parquet(output_path)

    typer.echo(f"\nReturns saved to: {output_path}")
    typer.echo(f"  returns.parquet: {len(returns)} rows")
    typer.echo("\nDone!")


@data_app.command("split")
def split_data(
    config_path: Annotated[
        Path,
        typer.Option("--config", "-c", help="Path to competition.toml"),
    ],
) -> None:
    """Split data into train/valid/test sets.

    Uses data_split settings from competition.toml.

    Example:
        quant-insight data split --config $MIXSEEK_WORKSPACE/configs/competition.toml
    """
    # 設定読み込み
    config = load_competition_config(config_path)
    if config is None:
        typer.echo(f"Error: Failed to load config from {config_path}", err=True)
        raise typer.Exit(1)

    typer.echo("Splitting data...")
    typer.echo(f"  Train end: {config.data_split.train_end}")
    typer.echo(f"  Valid end: {config.data_split.valid_end}")
    typer.echo(f"  Purge rows: {config.data_split.purge_rows}")

    # データ読み込み
    raw_dir = get_raw_data_dir()
    data_inputs_dir = get_data_inputs_dir()

    datasets: dict[str, pl.DataFrame] = {}

    # tomlのデータセット設定から各データを読み込む
    for data_config in config.data:
        name = data_config.name

        # parquetまたはcsvを探す
        parquet_path = raw_dir / f"{name}.parquet"
        csv_path = raw_dir / f"{name}.csv"

        if parquet_path.exists():
            datasets[name] = pl.read_parquet(parquet_path)
        elif csv_path.exists():
            datasets[name] = pl.read_csv(csv_path)
        elif not data_config.required:
            typer.echo(f"  Skipping optional dataset '{name}' (not found)")
            continue
        else:
            typer.echo(f"Error: Dataset '{name}' not found at {parquet_path} or {csv_path}", err=True)
            if name == "returns":
                typer.echo("Run 'quant-insight data build-returns' first.", err=True)
            elif name == "ohlcv":
                typer.echo("Run 'quant-insight data fetch-jquants' first.", err=True)
            raise typer.Exit(1)

    typer.echo(f"  Loaded datasets: {list(datasets.keys())}")

    # データ分割（各データセットのdatetime_columnを使用）
    splitter = DataSplitter()
    config_map = {cfg.name: cfg for cfg in config.data}

    for name, df in datasets.items():
        cfg = config_map[name]
        train, valid, test = splitter.split_by_datetime(
            df,
            train_end=config.data_split.train_end,
            valid_end=config.data_split.valid_end,
            purge_rows=config.data_split.purge_rows,
            datetime_column=cfg.datetime_column,
        )

        # 出力ディレクトリ作成と保存
        output_dir = data_inputs_dir / name
        output_dir.mkdir(parents=True, exist_ok=True)

        train.write_parquet(output_dir / "train.parquet")
        valid.write_parquet(output_dir / "valid.parquet")
        test.write_parquet(output_dir / "test.parquet")

        typer.echo(f"  {name}: train={len(train)}, valid={len(valid)}, test={len(test)}")

    typer.echo(f"\nData split saved to: {data_inputs_dir}")
    typer.echo("\nDone!")

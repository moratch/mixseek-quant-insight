"""BOJ macro data adapter for Mixseek quant-insight."""

import asyncio
import logging
import os
import warnings
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl

from quant_insight.data_build.base_adapter import BaseDataSourceAdapter
from quant_insight.data_build.boj.models import BOJ_COLUMN_PREFIX

logger = logging.getLogger(__name__)

DEFAULT_BOJ_DATA_ROOT = "D:/Dev/Stock_data/Macro_Data/boj"
JST = ZoneInfo("Asia/Tokyo")


class BOJMacroAdapter(BaseDataSourceAdapter):
    """日銀マクロデータアダプター.

    boj_latest.parquet (long format) を読み込み、
    wide format (datetime + BOJ_*列) に変換して返す。
    """

    def __init__(self, boj_data_root: str | None = None) -> None:
        self._boj_root = Path(
            boj_data_root or os.environ.get("BOJ_DATA_ROOT", DEFAULT_BOJ_DATA_ROOT)
        )

    def _get_latest_path(self) -> Path:
        return self._boj_root / "latest" / "boj_latest.parquet"

    async def authenticate(self) -> bool:
        """BOJ API は認証不要."""
        return True

    async def get_universe(self) -> list[str]:
        """マクロデータは銘柄ユニバースを持たない."""
        return []

    async def fetch_ohlcv(
        self,
        symbols: list[str],
        start_date: date,
        end_date: date,
    ) -> pl.DataFrame:
        """OHLCVデータなし（マクロ指標のみ）."""
        return pl.DataFrame(
            schema={
                "datetime": pl.Datetime,
                "symbol": pl.Utf8,
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Float64,
            }
        )

    def _long_to_wide(self, df: pl.DataFrame) -> pl.DataFrame:
        """long format → wide format (datetime + BOJ_*列) に変換.

        カラム名生成、pivot、日次 forward-fill を行う共通メソッド。
        """
        # カラム名生成: BOJ_{db}_{series_code}
        df = df.with_columns(
            (pl.lit(BOJ_COLUMN_PREFIX) + pl.col("db") + pl.lit("_") + pl.col("series_code"))
            .alias("col_name")
        )

        # date を datetime に変換
        df = df.with_columns(
            pl.col("date").cast(pl.Date).cast(pl.Datetime("ms")).alias("datetime")
        )

        # Pivot: long → wide
        wide = df.pivot(
            on="col_name",
            index="datetime",
            values="value",
            aggregate_function="last",
        ).sort("datetime")

        # 日次 forward-fill (月次/四半期データの補間)
        dt_min_raw = wide["datetime"].min()
        dt_max_raw = wide["datetime"].max()
        assert isinstance(dt_min_raw, datetime) and isinstance(dt_max_raw, datetime)
        date_range = pl.datetime_range(
            dt_min_raw,
            dt_max_raw,
            interval="1d",
            eager=True,
        ).alias("datetime")

        full_dates = pl.DataFrame({"datetime": date_range})
        # Ensure matching datetime precision for join
        target_dtype = wide["datetime"].dtype
        full_dates = full_dates.with_columns(
            pl.col("datetime").cast(target_dtype)
        )
        wide = full_dates.join(wide, on="datetime", how="left").sort("datetime")

        # Forward-fill
        value_cols = [c for c in wide.columns if c.startswith(BOJ_COLUMN_PREFIX)]
        wide = wide.with_columns(
            [pl.col(c).forward_fill() for c in value_cols]
        )

        return wide

    async def fetch_all_data(
        self,
        symbols: list[str],
        start_date: date,
        end_date: date,
    ) -> dict[str, pl.DataFrame]:
        """boj_latest.parquet を読み込んで wide format に変換.

        Args:
            symbols: 未使用（マクロデータは銘柄非依存）
            start_date: 取得開始日（この日以降のデータのみ返す）
            end_date: 取得終了日（この日以前のデータのみ返す）
        """

        def _load() -> pl.DataFrame:
            path = self._get_latest_path()
            if not path.exists():
                raise FileNotFoundError(
                    f"BOJ data not found: {path}\n"
                    f"Run: python scripts/boj_fetcher.py fetch-phase1"
                )

            # Parquet 読み込み (long format)
            df = pl.read_parquet(path)
            logger.info(f"Loaded BOJ data: {df.shape[0]:,} rows from {path}")

            wide = self._long_to_wide(df)

            # 日付フィルタ
            start_dt = pl.lit(start_date).cast(pl.Datetime("ms"))
            end_dt = pl.lit(end_date).cast(pl.Datetime("ms"))
            wide = wide.filter(
                (pl.col("datetime") >= start_dt) & (pl.col("datetime") <= end_dt)
            )

            logger.info(f"BOJ wide format: {wide.shape[0]:,} rows x {wide.shape[1]} cols")
            return wide

        # sync I/O を別スレッドで実行
        wide = await asyncio.to_thread(_load)
        return {"boj_macro": wide}

    def load_history(self, as_of: datetime) -> pl.DataFrame:
        """Point-in-time バックテスト用: as_of 時点での最新観測値を返す.

        Args:
            as_of: 基準時刻。naive入力→JST付与、非JST tz-aware→JST変換。

        Returns:
            as_of 時点で観測可能だったデータの wide DataFrame。
            history/ 未存在時は空 DataFrame。
        """
        # as_of TZ 正規化
        if as_of.tzinfo is None:
            as_of = as_of.replace(tzinfo=JST)
        elif str(as_of.tzinfo) != "Asia/Tokyo":
            as_of = as_of.astimezone(JST)

        history_dir = self._boj_root / "history"
        if not history_dir.exists():
            warnings.warn(
                "history/ directory not found. Point-in-time data unavailable "
                "(Phase 4 not yet implemented).",
                stacklevel=2,
            )
            return pl.DataFrame()

        # history パーティションからファイル収集
        parquet_files = sorted(history_dir.rglob("*.parquet"))
        if not parquet_files:
            return pl.DataFrame()

        dfs = []
        for f in parquet_files:
            try:
                df = pl.read_parquet(f)
                dfs.append(df)
            except Exception as e:
                logger.warning(f"Skip {f}: {e}")

        if not dfs:
            return pl.DataFrame()

        combined = pl.concat(dfs)

        # fetch_ts <= as_of でフィルタ
        combined = combined.filter(pl.col("fetch_ts") <= as_of)

        if combined.is_empty():
            return pl.DataFrame()

        # 同一 (db, series_code, date) で fetch_ts 最大を採用
        combined = combined.sort("fetch_ts").group_by(
            ["db", "series_code", "date"]
        ).last()

        # long → wide 変換（fetch_all_data と同じ形式で返す）
        if combined.is_empty():
            return pl.DataFrame()

        return self._long_to_wide(combined)

    def save(
        self,
        data: dict[str, pl.DataFrame],
        output_dir: Path,
    ) -> None:
        """データを保存."""
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, df in data.items():
            df.write_parquet(output_dir / f"{name}.parquet")

    async def close(self) -> None:
        """リソースクリーンアップ（BOJアダプターは不要）."""
        pass

    async def __aenter__(self) -> "BOJMacroAdapter":
        await self.authenticate()
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()

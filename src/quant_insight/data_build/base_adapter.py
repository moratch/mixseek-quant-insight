"""Base data source adapter for fetching OHLCV and additional data."""

from abc import ABC, abstractmethod
from datetime import date
from pathlib import Path

import polars as pl


class BaseDataSourceAdapter(ABC):
    """データソースAPIアダプタの基底クラス.

    FR-015: 差し替え可能なアダプタパターンで実装
    """

    @abstractmethod
    async def authenticate(self) -> bool:
        """API認証を実行.

        Returns:
            bool: 認証成功ならTrue

        Raises:
            AuthenticationError: 認証失敗時
        """
        ...

    @abstractmethod
    async def get_universe(self) -> list[str]:
        """利用可能な銘柄一覧を取得.

        Returns:
            list[str]: 銘柄コードのリスト
        """
        ...

    @abstractmethod
    async def fetch_ohlcv(
        self,
        symbols: list[str],
        start_date: date,
        end_date: date,
    ) -> pl.DataFrame:
        """OHLCVデータを取得.

        Args:
            symbols: 銘柄コードのリスト
            start_date: 取得開始日
            end_date: 取得終了日

        Returns:
            pl.DataFrame: 以下のカラムを持つDataFrame
                - datetime: datetime型
                - symbol: str型
                - open: float型
                - high: float型
                - low: float型
                - close: float型
                - volume: int型
        """
        ...

    @abstractmethod
    async def fetch_all_data(
        self,
        symbols: list[str],
        start_date: date,
        end_date: date,
    ) -> dict[str, pl.DataFrame]:
        """全データを取得（OHLCV + 追加データ）.

        Args:
            symbols: 銘柄コードのリスト
            start_date: 取得開始日
            end_date: 取得終了日

        Returns:
            dict[str, pl.DataFrame]: {
                "ohlcv": pl.DataFrame,
                "追加データ名": pl.DataFrame,
                ...
            }
        """
        ...

    def save(
        self,
        data: dict[str, pl.DataFrame],
        output_dir: Path,
    ) -> None:
        """データを保存.

        Args:
            data: {"ohlcv": df, "returns": df, ...}
            output_dir: 出力ディレクトリ
                例: {MIXSEEK_WORKSPACE}/data/inputs/raw/

        Output Structure:
            {output_dir}/
            ├── ohlcv.parquet
            ├── returns.parquet
            └── {追加データ名}.parquet
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, df in data.items():
            df.write_parquet(output_dir / f"{name}.parquet")

"""Data splitter for splitting data into train/valid/test sets with purge support."""

from datetime import datetime

import polars as pl

from quant_insight.exceptions import DataSplitError
from quant_insight.models.data_config import DataConfig


class DataSplitter:
    """train/valid/test分割ロジック（API非依存）.

    FR-010: purge（パージ）を適用してデータ漏洩を防止
    """

    def _get_purge_dates(
        self,
        df: pl.DataFrame,
        purge_days: int,
        from_end: bool = True,
    ) -> set[datetime]:
        """パージ対象の日付を取得.

        Args:
            df: 対象DataFrame（"datetime"列を持つ）
            purge_days: パージする日数
            from_end: Trueなら末尾から、Falseなら先頭からパージ

        Returns:
            パージ対象の日付のset
        """
        if len(df) == 0 or purge_days <= 0:
            return set()

        unique_dates = df.select("datetime").unique().sort("datetime")
        dates_list = unique_dates["datetime"].to_list()

        if from_end:
            # 末尾からpurge_days日分
            purge_dates = dates_list[-purge_days:] if len(dates_list) >= purge_days else dates_list
        else:
            # 先頭からpurge_days日分
            purge_dates = dates_list[:purge_days] if len(dates_list) >= purge_days else dates_list

        return set(purge_dates)

    def split_by_datetime(
        self,
        df: pl.DataFrame,
        train_end: datetime,
        valid_end: datetime,
        purge_rows: int = 0,
        datetime_column: str = "datetime",
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """日時ベースでデータを分割（purge対応）.

        Args:
            df: 分割対象のDataFrame（datetime_column列を持つ）
            train_end: train期間の終了日時（この日時を含む）
            valid_end: valid期間の終了日時（この日時を含む）
            purge_rows: train/valid間、valid/test間でパージする日数
            datetime_column: 日時カラム名

        Returns:
            (train_df, valid_df, test_df)

        Raises:
            DataSplitError: purge適用後に空DataFrameになる場合

        Purge Logic:
            purge_rows > 0 の場合（日数単位）:
            - trainの末尾 purge_rows 日分を除外
            - validの先頭 purge_rows 日分を除外
            - validの末尾 purge_rows 日分を除外
            - testの先頭 purge_rows 日分を除外

        Note:
            purge_rowsが指定された場合、各境界からpurge_rows日分のデータを除外する。
            これによりリターン計算のwindow幅によるデータ漏洩を防止する。
            全銘柄の該当日データが一括で除外されるため、銘柄間のデータ漏洩も防止できる。
        """
        purge_days = purge_rows  # 日数として扱う

        # datetime_columnを"datetime"にリネーム（異なる場合のみ）
        if datetime_column != "datetime":
            df = df.rename({datetime_column: "datetime"})

        # datetime列でソート
        df_sorted = df.sort("datetime")

        # 基本分割
        train = df_sorted.filter(pl.col("datetime") <= train_end)
        valid = df_sorted.filter((pl.col("datetime") > train_end) & (pl.col("datetime") <= valid_end))
        test = df_sorted.filter(pl.col("datetime") > valid_end)

        # purge適用: 各境界からpurge_days日分を除外
        if purge_days > 0:
            # trainの末尾purge_days日分を除外
            train_purge_dates = self._get_purge_dates(train, purge_days, from_end=True)
            train = train.filter(~pl.col("datetime").is_in(list(train_purge_dates)))
            if len(train) == 0:
                raise DataSplitError(f"Purge resulted in empty DataFrame for train split. purge_days={purge_days}")

            # validの先頭と末尾からpurge_days日分を除外
            valid_head_purge = self._get_purge_dates(valid, purge_days, from_end=False)
            valid_tail_purge = self._get_purge_dates(valid, purge_days, from_end=True)
            all_valid_purge = valid_head_purge | valid_tail_purge
            valid = valid.filter(~pl.col("datetime").is_in(list(all_valid_purge)))
            if len(valid) == 0:
                raise DataSplitError(f"Purge resulted in empty DataFrame for valid split. purge_days={purge_days}")

            # testの先頭purge_days日分を除外
            test_purge_dates = self._get_purge_dates(test, purge_days, from_end=False)
            test = test.filter(~pl.col("datetime").is_in(list(test_purge_dates)))
            if len(test) == 0:
                raise DataSplitError(f"Purge resulted in empty DataFrame for test split. purge_days={purge_days}")

        return train, valid, test

    def split_all_datasets(
        self,
        datasets: dict[str, pl.DataFrame],
        data_configs: list[DataConfig],
        train_end: datetime,
        valid_end: datetime,
        purge_rows: int = 0,
    ) -> dict[str, tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]]:
        """全データセットを一括分割.

        Args:
            datasets: {"ohlcv": df, "returns": df, ...}
            data_configs: [[competition.data]]設定のリスト
            train_end: train期間の終了日時
            valid_end: valid期間の終了日時
            purge_rows: パージする行数

        Returns:
            {"ohlcv": (train, valid, test), "returns": (train, valid, test), ...}

        Note:
            各データセットのdatetime_columnは分割後に"datetime"にリネームされる。
            これによりバックテストループで統一的に"datetime"列を参照可能になる。
        """
        config_map = {cfg.name: cfg for cfg in data_configs}
        result = {}

        for name, df in datasets.items():
            cfg = config_map.get(name)
            if cfg is None:
                raise DataSplitError(f"DataConfig not found for dataset: {name}")
            datetime_col = cfg.datetime_column

            result[name] = self.split_by_datetime(df, train_end, valid_end, purge_rows, datetime_col)

        return result

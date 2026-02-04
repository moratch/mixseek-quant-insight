"""Unit tests for DataSplitter.

Test Cases:
1. Basic split without purge
2. Split with purge_rows > 0 (日数ベース)
3. Empty split handling
4. Datetime column renaming
5. split_all_datasets with multiple DataFrames
6. Edge case: purge_rows > data length
7. purge適用後に空DataFrameになる場合、DataSplitErrorを発生させること
8. 複数銘柄での日数ベースpurge動作確認
"""

from datetime import UTC, datetime

import polars as pl
import pytest

from quant_insight.data_build.data_splitter import DataSplitter
from quant_insight.exceptions import DataSplitError
from quant_insight.models.competition_config import DataConfig


class TestDataSplitter:
    """DataSplitterの単体テスト."""

    @pytest.fixture
    def sample_dataframe(self) -> pl.DataFrame:
        """サンプルDataFrame（10日分のデータ）."""
        return pl.DataFrame(
            {
                "datetime": [datetime(2023, 1, i + 1, tzinfo=UTC) for i in range(10)],
                "symbol": ["AAPL"] * 10,
                "value": list(range(10)),
            }
        )

    @pytest.fixture
    def custom_datetime_column_df(self) -> pl.DataFrame:
        """カスタム日時カラム名を持つDataFrame."""
        return pl.DataFrame(
            {
                "timestamp": [datetime(2023, 1, i + 1, tzinfo=UTC) for i in range(10)],
                "symbol": ["AAPL"] * 10,
                "value": list(range(10)),
            }
        )

    def test_basic_split_without_purge(self, sample_dataframe: pl.DataFrame) -> None:
        """Basic split without purge.

        train: 2023-01-01 ~ 2023-01-04 (4日間)
        valid: 2023-01-05 ~ 2023-01-07 (3日間)
        test:  2023-01-08 ~ 2023-01-10 (3日間)
        """
        splitter = DataSplitter()
        train_end = datetime(2023, 1, 4, tzinfo=UTC)
        valid_end = datetime(2023, 1, 7, tzinfo=UTC)

        train, valid, test = splitter.split_by_datetime(sample_dataframe, train_end, valid_end, purge_rows=0)

        assert len(train) == 4
        assert len(valid) == 3
        assert len(test) == 3

        # train: 2023-01-01 ~ 2023-01-04
        assert train["datetime"].min() == datetime(2023, 1, 1, tzinfo=UTC)
        assert train["datetime"].max() == datetime(2023, 1, 4, tzinfo=UTC)

        # valid: 2023-01-05 ~ 2023-01-07
        assert valid["datetime"].min() == datetime(2023, 1, 5, tzinfo=UTC)
        assert valid["datetime"].max() == datetime(2023, 1, 7, tzinfo=UTC)

        # test: 2023-01-08 ~ 2023-01-10
        assert test["datetime"].min() == datetime(2023, 1, 8, tzinfo=UTC)
        assert test["datetime"].max() == datetime(2023, 1, 10, tzinfo=UTC)

    def test_split_with_purge(self, sample_dataframe: pl.DataFrame) -> None:
        """Split with purge_rows > 0 (日数ベース).

        purge_rows=1の場合（1日分のデータを除外）:
        - trainの末尾1日分を除外
        - validの先頭1日分を除外
        - validの末尾1日分を除外
        - testの先頭1日分を除外

        train: 2023-01-01 ~ 2023-01-04 → 末尾1日除外 → 2023-01-01 ~ 2023-01-03 (3日)
        valid: 2023-01-05 ~ 2023-01-07 → 先頭1日除外、末尾1日除外 → 2023-01-06 (1日)
        test:  2023-01-08 ~ 2023-01-10 → 先頭1日除外 → 2023-01-09 ~ 2023-01-10 (2日)
        """
        splitter = DataSplitter()
        train_end = datetime(2023, 1, 4, tzinfo=UTC)
        valid_end = datetime(2023, 1, 7, tzinfo=UTC)

        train, valid, test = splitter.split_by_datetime(sample_dataframe, train_end, valid_end, purge_rows=1)

        # purge適用後の長さを検証
        assert len(train) == 3  # 4 - 1 (末尾)
        assert len(valid) == 1  # 3 - 1 (先頭) - 1 (末尾)
        assert len(test) == 2  # 3 - 1 (先頭)

        # train: 2023-01-01 ~ 2023-01-03 (末尾の2023-01-04が除外)
        assert train["datetime"].max() == datetime(2023, 1, 3, tzinfo=UTC)

        # valid: 2023-01-06 のみ (先頭の01-05と末尾の01-07が除外)
        assert len(valid) == 1
        assert valid["datetime"][0] == datetime(2023, 1, 6, tzinfo=UTC)

        # test: 2023-01-09 ~ 2023-01-10 (先頭の2023-01-08が除外)
        assert test["datetime"].min() == datetime(2023, 1, 9, tzinfo=UTC)

    def test_empty_split_handling(self, sample_dataframe: pl.DataFrame) -> None:
        """Empty split handling.

        valid_endをtrain_endより前に設定すると、validが空になる
        """
        splitter = DataSplitter()
        train_end = datetime(2023, 1, 10, tzinfo=UTC)
        valid_end = datetime(2023, 1, 10, tzinfo=UTC)

        train, valid, test = splitter.split_by_datetime(sample_dataframe, train_end, valid_end, purge_rows=0)

        assert len(train) == 10
        assert len(valid) == 0
        assert len(test) == 0

    def test_datetime_column_renaming(self, custom_datetime_column_df: pl.DataFrame) -> None:
        """Datetime column renaming.

        カスタム日時カラム名（timestamp）を"datetime"にリネーム
        """
        splitter = DataSplitter()
        train_end = datetime(2023, 1, 4, tzinfo=UTC)
        valid_end = datetime(2023, 1, 7, tzinfo=UTC)

        train, valid, test = splitter.split_by_datetime(
            custom_datetime_column_df,
            train_end,
            valid_end,
            purge_rows=0,
            datetime_column="timestamp",
        )

        # 分割後はすべて"datetime"列を持つ
        assert "datetime" in train.columns
        assert "datetime" in valid.columns
        assert "datetime" in test.columns

        # 元の"timestamp"列は存在しない
        assert "timestamp" not in train.columns
        assert "timestamp" not in valid.columns
        assert "timestamp" not in test.columns

    def test_split_all_datasets(self, sample_dataframe: pl.DataFrame) -> None:
        """split_all_datasets with multiple DataFrames.

        複数のDataFrameを一括分割
        """
        splitter = DataSplitter()

        # 複数データセット
        datasets = {
            "ohlcv": sample_dataframe,
            "returns": sample_dataframe.clone(),
        }

        data_configs = [
            DataConfig(name="ohlcv", datetime_column="datetime"),
            DataConfig(name="returns", datetime_column="datetime"),
        ]

        train_end = datetime(2023, 1, 4, tzinfo=UTC)
        valid_end = datetime(2023, 1, 7, tzinfo=UTC)

        result = splitter.split_all_datasets(datasets, data_configs, train_end, valid_end, purge_rows=0)

        # 結果の構造を検証
        assert "ohlcv" in result
        assert "returns" in result

        # 各データセットがtrain/valid/testに分割されている
        ohlcv_train, ohlcv_valid, ohlcv_test = result["ohlcv"]
        returns_train, returns_valid, returns_test = result["returns"]

        assert len(ohlcv_train) == 4
        assert len(ohlcv_valid) == 3
        assert len(ohlcv_test) == 3

        assert len(returns_train) == 4
        assert len(returns_valid) == 3
        assert len(returns_test) == 3

    def test_edge_case_purge_rows_exceeds_data_length(self, sample_dataframe: pl.DataFrame) -> None:
        """Edge case: purge_rows > data length.

        purge_rowsがデータ長より大きい場合、DataSplitErrorが発生する
        """
        splitter = DataSplitter()
        train_end = datetime(2023, 1, 4, tzinfo=UTC)
        valid_end = datetime(2023, 1, 7, tzinfo=UTC)

        # purge_rows=10は各分割のデータ長を超える（validが空になる）
        with pytest.raises(DataSplitError, match="Purge resulted in empty DataFrame"):
            splitter.split_by_datetime(sample_dataframe, train_end, valid_end, purge_rows=10)

    def test_purge_results_in_empty_dataframe_raises_error(self, sample_dataframe: pl.DataFrame) -> None:
        """purge適用後に空DataFrameになる場合、DataSplitErrorを発生させる.

        purge_rows=5でvalidが空になるケース
        """
        splitter = DataSplitter()
        train_end = datetime(2023, 1, 4, tzinfo=UTC)
        valid_end = datetime(2023, 1, 7, tzinfo=UTC)

        # valid期間は3日間だが、purge_rows=5なので先頭+末尾で全日除外
        # → validが空になる
        with pytest.raises(DataSplitError, match="Purge resulted in empty DataFrame"):
            splitter.split_by_datetime(sample_dataframe, train_end, valid_end, purge_rows=5)

    def test_purge_with_multiple_symbols(self) -> None:
        """複数銘柄での日数ベースpurge動作確認.

        日数ベースのpurgeでは、全銘柄の該当日データが一括で除外される。
        これにより銘柄間のデータ漏洩も防止できる。
        """
        # 3銘柄 × 10日間 = 30行
        dates = [datetime(2023, 1, i + 1, tzinfo=UTC) for i in range(10)]
        symbols = ["AAPL", "GOOG", "MSFT"]

        data = {
            "datetime": dates * 3,
            "symbol": [s for s in symbols for _ in range(10)],
            "value": list(range(30)),
        }
        df = pl.DataFrame(data)

        splitter = DataSplitter()
        train_end = datetime(2023, 1, 4, tzinfo=UTC)
        valid_end = datetime(2023, 1, 7, tzinfo=UTC)

        train, valid, test = splitter.split_by_datetime(df, train_end, valid_end, purge_rows=1)

        # purge_rows=1（日数）で各境界から1日分除外
        # train: 4日 × 3銘柄 = 12行 → 末尾1日除外 → 3日 × 3銘柄 = 9行
        # valid: 3日 × 3銘柄 = 9行 → 先頭1日+末尾1日除外 → 1日 × 3銘柄 = 3行
        # test:  3日 × 3銘柄 = 9行 → 先頭1日除外 → 2日 × 3銘柄 = 6行

        assert len(train) == 9, f"Expected 9 rows (3 days × 3 symbols), got {len(train)}"
        assert len(valid) == 3, f"Expected 3 rows (1 day × 3 symbols), got {len(valid)}"
        assert len(test) == 6, f"Expected 6 rows (2 days × 3 symbols), got {len(test)}"

        # 各セットに全銘柄が含まれることを確認
        assert set(train["symbol"].unique()) == {"AAPL", "GOOG", "MSFT"}
        assert set(valid["symbol"].unique()) == {"AAPL", "GOOG", "MSFT"}
        assert set(test["symbol"].unique()) == {"AAPL", "GOOG", "MSFT"}

        # trainの最終日が2023-01-03（01-04は除外）
        assert train["datetime"].max() == datetime(2023, 1, 3, tzinfo=UTC)

        # validの日付が2023-01-06のみ（01-05と01-07は除外）
        assert valid["datetime"].unique().to_list() == [datetime(2023, 1, 6, tzinfo=UTC)]

        # testの最初の日が2023-01-09（01-08は除外）
        assert test["datetime"].min() == datetime(2023, 1, 9, tzinfo=UTC)

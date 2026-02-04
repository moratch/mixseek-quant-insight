"""Contract tests for BaseDataSourceAdapter.

Verification:
1. Abstract methods are defined
2. save() method correctly outputs Parquet files
3. Concrete class implements all abstract methods
"""

from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl
import pytest

from quant_insight.data_build.base_adapter import BaseDataSourceAdapter


class TestBaseDataSourceAdapterContract:
    """BaseDataSourceAdapterの契約テスト."""

    @pytest.fixture
    def mock_adapter(self) -> type[BaseDataSourceAdapter]:
        """モックアダプタ（具象クラス）."""

        class MockAdapter(BaseDataSourceAdapter):
            """テスト用のモックアダプタ."""

            async def authenticate(self) -> bool:
                """認証成功をシミュレート."""
                return True

            async def get_universe(self) -> list[str]:
                """銘柄リストを返す."""
                return ["AAPL", "GOOGL"]

            async def fetch_ohlcv(
                self,
                symbols: list[str],
                start_date: date,
                end_date: date,
            ) -> pl.DataFrame:
                """OHLCVデータを返す."""
                return pl.DataFrame(
                    {
                        "datetime": [date(2023, 1, 1), date(2023, 1, 2)],
                        "symbol": ["AAPL", "AAPL"],
                        "open": [100.0, 102.0],
                        "high": [102.0, 104.0],
                        "low": [99.0, 101.0],
                        "close": [101.0, 103.0],
                        "volume": [1000, 1100],
                    }
                )

            async def fetch_all_data(
                self,
                symbols: list[str],
                start_date: date,
                end_date: date,
            ) -> dict[str, pl.DataFrame]:
                """全データを返す."""
                ohlcv = await self.fetch_ohlcv(symbols, start_date, end_date)
                return {"ohlcv": ohlcv}

        return MockAdapter

    def test_abstract_methods_defined(self) -> None:
        """Abstract methods are defined.

        BaseDataSourceAdapterは以下の抽象メソッドを持つ:
        - authenticate()
        - get_universe()
        - fetch_ohlcv()
        - fetch_all_data()
        """
        abstract_methods = BaseDataSourceAdapter.__abstractmethods__
        expected_methods = {
            "authenticate",
            "get_universe",
            "fetch_ohlcv",
            "fetch_all_data",
        }

        assert abstract_methods == expected_methods

    def test_cannot_instantiate_base_adapter(self) -> None:
        """BaseDataSourceAdapter cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseDataSourceAdapter()  # type: ignore[abstract]

    @pytest.mark.asyncio
    async def test_concrete_class_implements_all_methods(self, mock_adapter: type[BaseDataSourceAdapter]) -> None:
        """Concrete class implements all abstract methods."""
        adapter = mock_adapter()

        # すべての抽象メソッドが実装されている
        assert await adapter.authenticate()
        assert await adapter.get_universe() == ["AAPL", "GOOGL"]

        ohlcv = await adapter.fetch_ohlcv(["AAPL"], date(2023, 1, 1), date(2023, 1, 2))
        assert isinstance(ohlcv, pl.DataFrame)

        all_data = await adapter.fetch_all_data(["AAPL"], date(2023, 1, 1), date(2023, 1, 2))
        assert isinstance(all_data, dict)
        assert "ohlcv" in all_data

    @pytest.mark.asyncio
    async def test_save_method_outputs_parquet_files(self, mock_adapter: type[BaseDataSourceAdapter]) -> None:
        """save() method correctly outputs Parquet files.

        検証:
        1. 出力ディレクトリが作成される
        2. 各データセットが{name}.parquetとして保存される
        3. 保存されたファイルが読み込み可能
        """
        adapter = mock_adapter()

        # テスト用のデータセット
        data = {
            "ohlcv": pl.DataFrame(
                {
                    "datetime": [date(2023, 1, 1), date(2023, 1, 2)],
                    "symbol": ["AAPL", "AAPL"],
                    "close": [100.0, 101.0],
                }
            ),
            "returns": pl.DataFrame(
                {
                    "datetime": [date(2023, 1, 1), date(2023, 1, 2)],
                    "symbol": ["AAPL", "AAPL"],
                    "return_value": [0.01, 0.02],
                }
            ),
        }

        with TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"

            # save()を実行
            adapter.save(data, output_dir)

            # 出力ディレクトリが作成されている
            assert output_dir.exists()

            # 各ファイルが存在する
            ohlcv_path = output_dir / "ohlcv.parquet"
            returns_path = output_dir / "returns.parquet"

            assert ohlcv_path.exists()
            assert returns_path.exists()

            # 保存されたファイルが読み込み可能
            loaded_ohlcv = pl.read_parquet(ohlcv_path)
            loaded_returns = pl.read_parquet(returns_path)

            # データが正しく保存されている
            assert loaded_ohlcv.shape == data["ohlcv"].shape
            assert loaded_returns.shape == data["returns"].shape

            assert set(loaded_ohlcv.columns) == set(data["ohlcv"].columns)
            assert set(loaded_returns.columns) == set(data["returns"].columns)

    @pytest.mark.asyncio
    async def test_fetch_ohlcv_returns_correct_schema(self, mock_adapter: type[BaseDataSourceAdapter]) -> None:
        """fetch_ohlcv returns DataFrame with correct schema.

        必須カラム: datetime, symbol, open, high, low, close, volume
        """
        adapter = mock_adapter()
        ohlcv = await adapter.fetch_ohlcv(["AAPL"], date(2023, 1, 1), date(2023, 1, 2))

        required_columns = {"datetime", "symbol", "open", "high", "low", "close", "volume"}
        assert required_columns.issubset(set(ohlcv.columns))

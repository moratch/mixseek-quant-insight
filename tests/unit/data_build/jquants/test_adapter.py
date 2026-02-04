"""Unit tests for J-Quants adapter with mocks."""

from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from quant_insight.data_build.jquants.adapter import (
    BASE_URL,
    JQuantsAdapter,
    get_default_date_range,
)
from quant_insight.data_build.jquants.models import JQuantsPlan, JQuantsUniverse
from quant_insight.exceptions import AuthenticationError, DataFetchError


class TestBaseUrl:
    """BASE_URL定数のテスト."""

    def test_base_url_value(self) -> None:
        """ベースURLが正しいこと."""
        assert BASE_URL == "https://api.jquants.com/v2"


class TestJQuantsAdapterInit:
    """JQuantsAdapter初期化のテスト."""

    def test_default_init(self) -> None:
        """デフォルト値での初期化."""
        adapter = JQuantsAdapter()
        assert adapter.plan == JQuantsPlan.FREE
        assert adapter.universe == JQuantsUniverse.PRIME

    def test_init_with_custom_values(self) -> None:
        """カスタム値での初期化."""
        adapter = JQuantsAdapter(
            plan=JQuantsPlan.PREMIUM,
            universe=JQuantsUniverse.ALL,
        )
        assert adapter.plan == JQuantsPlan.PREMIUM
        assert adapter.universe == JQuantsUniverse.ALL

    def test_init_creates_rate_limiter(self) -> None:
        """Rate Limiterが作成されること."""
        adapter = JQuantsAdapter(plan=JQuantsPlan.FREE)
        assert adapter.rate_limiter.plan == JQuantsPlan.FREE


class TestJQuantsAdapterAuthenticate:
    """JQuantsAdapter.authenticateメソッドのテスト."""

    @pytest.mark.asyncio
    async def test_authenticate_missing_api_key(self) -> None:
        """APIキーが未設定の場合エラー."""
        adapter = JQuantsAdapter()

        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(AuthenticationError, match="JQUANTS_API_KEY.*not set"):
                await adapter.authenticate()

    @pytest.mark.asyncio
    async def test_authenticate_empty_api_key(self) -> None:
        """APIキーが空の場合エラー."""
        adapter = JQuantsAdapter()

        with patch.dict("os.environ", {"JQUANTS_API_KEY": "  "}, clear=True):
            with pytest.raises(AuthenticationError, match="JQUANTS_API_KEY.*not set"):
                await adapter.authenticate()

    @pytest.mark.asyncio
    async def test_authenticate_success(self) -> None:
        """認証成功."""
        adapter = JQuantsAdapter()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"data": []}

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        with (
            patch.dict("os.environ", {"JQUANTS_API_KEY": "test_key"}, clear=True),
            patch("httpx.AsyncClient", return_value=mock_client),
        ):
            result = await adapter.authenticate()
            assert result is True
            assert adapter._api_key == "test_key"

    @pytest.mark.asyncio
    async def test_authenticate_invalid_key(self) -> None:
        """無効なAPIキーでエラー."""
        adapter = JQuantsAdapter()

        mock_request = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Unauthorized", request=mock_request, response=mock_response
        )

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        with (
            patch.dict("os.environ", {"JQUANTS_API_KEY": "invalid_key"}, clear=True),
            patch("httpx.AsyncClient", return_value=mock_client),
        ):
            with pytest.raises(AuthenticationError, match="Invalid API key"):
                await adapter.authenticate()


class TestJQuantsAdapterGetUniverse:
    """JQuantsAdapter.get_universeメソッドのテスト."""

    @pytest.mark.asyncio
    async def test_get_universe_not_authenticated(self) -> None:
        """未認証時はエラー."""
        adapter = JQuantsAdapter()

        with pytest.raises(DataFetchError, match="Not authenticated"):
            await adapter.get_universe()

    @pytest.mark.asyncio
    async def test_get_universe_prime(self) -> None:
        """プライム市場の銘柄取得."""
        adapter = JQuantsAdapter(universe=JQuantsUniverse.PRIME)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {"Code": "7203", "Mkt": "0111"},  # プライム
                {"Code": "9984", "Mkt": "0111"},  # プライム
                {"Code": "6758", "Mkt": "0112"},  # スタンダード
            ]
        }

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        adapter._client = mock_client
        adapter._api_key = "test_key"

        symbols = await adapter.get_universe()
        assert symbols == ["7203", "9984"]

    @pytest.mark.asyncio
    async def test_get_universe_all(self) -> None:
        """全銘柄取得."""
        adapter = JQuantsAdapter(universe=JQuantsUniverse.ALL)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {"Code": "7203", "Mkt": "0111"},
                {"Code": "9984", "Mkt": "0112"},
                {"Code": "6758", "Mkt": "0113"},
            ]
        }

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        adapter._client = mock_client
        adapter._api_key = "test_key"

        symbols = await adapter.get_universe()
        assert len(symbols) == 3


class TestJQuantsAdapterFetchOhlcv:
    """JQuantsAdapter.fetch_ohlcvメソッドのテスト."""

    @pytest.mark.asyncio
    async def test_fetch_ohlcv_not_authenticated(self) -> None:
        """未認証時はエラー."""
        adapter = JQuantsAdapter()

        with pytest.raises(DataFetchError, match="Not authenticated"):
            await adapter.fetch_ohlcv(["7203"], date(2024, 1, 1), date(2024, 1, 1))

    @pytest.mark.asyncio
    async def test_fetch_ohlcv_empty_data(self) -> None:
        """空データの場合."""
        adapter = JQuantsAdapter()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"data": []}

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        adapter._client = mock_client
        adapter._api_key = "test_key"

        df = await adapter.fetch_ohlcv(["7203"], date(2024, 1, 1), date(2024, 1, 1))
        assert len(df) == 0
        assert "datetime" in df.columns
        assert "symbol" in df.columns

    @pytest.mark.asyncio
    async def test_fetch_ohlcv_with_data(self) -> None:
        """データ取得成功."""
        adapter = JQuantsAdapter()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {
                    "Date": "2024-01-04",
                    "Code": "7203",
                    "AdjO": 100.0,
                    "AdjH": 105.0,
                    "AdjL": 99.0,
                    "AdjC": 104.0,
                    "AdjVo": 1000000.0,
                    "O": 100.0,
                    "H": 105.0,
                    "L": 99.0,
                    "C": 104.0,
                    "Vo": 1000000.0,
                    "Va": 10400000000.0,
                },
            ]
        }

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        adapter._client = mock_client
        adapter._api_key = "test_key"

        df = await adapter.fetch_ohlcv(["7203"], date(2024, 1, 4), date(2024, 1, 4))

        assert len(df) == 1
        assert df["symbol"][0] == "7203"
        assert df["open"][0] == 100.0
        assert df["close"][0] == 104.0
        assert df["volume"][0] == 1000000.0

    @pytest.mark.asyncio
    async def test_fetch_ohlcv_filters_symbols(self) -> None:
        """指定銘柄のみフィルタすること."""
        adapter = JQuantsAdapter()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {
                    "Date": "2024-01-04",
                    "Code": "7203",
                    "AdjO": 100.0,
                    "AdjH": 105.0,
                    "AdjL": 99.0,
                    "AdjC": 104.0,
                    "AdjVo": 1000000.0,
                },
                {
                    "Date": "2024-01-04",
                    "Code": "9984",
                    "AdjO": 200.0,
                    "AdjH": 210.0,
                    "AdjL": 195.0,
                    "AdjC": 208.0,
                    "AdjVo": 2000000.0,
                },
                {
                    "Date": "2024-01-04",
                    "Code": "6758",
                    "AdjO": 300.0,
                    "AdjH": 310.0,
                    "AdjL": 290.0,
                    "AdjC": 305.0,
                    "AdjVo": 3000000.0,
                },
            ]
        }

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        adapter._client = mock_client
        adapter._api_key = "test_key"

        # 7203のみ取得
        df = await adapter.fetch_ohlcv(["7203"], date(2024, 1, 4), date(2024, 1, 4))

        assert len(df) == 1
        assert df["symbol"][0] == "7203"

    @pytest.mark.asyncio
    async def test_fetch_ohlcv_pagination(self) -> None:
        """ページネーション対応."""
        adapter = JQuantsAdapter()

        # 1ページ目
        mock_response1 = MagicMock()
        mock_response1.status_code = 200
        mock_response1.raise_for_status = MagicMock()
        mock_response1.json.return_value = {
            "data": [
                {
                    "Date": "2024-01-04",
                    "Code": "7203",
                    "AdjO": 100.0,
                    "AdjH": 105.0,
                    "AdjL": 99.0,
                    "AdjC": 104.0,
                    "AdjVo": 1000000.0,
                },
            ],
            "pagination_key": "page2",
        }

        # 2ページ目
        mock_response2 = MagicMock()
        mock_response2.status_code = 200
        mock_response2.raise_for_status = MagicMock()
        mock_response2.json.return_value = {
            "data": [
                {
                    "Date": "2024-01-04",
                    "Code": "9984",
                    "AdjO": 200.0,
                    "AdjH": 210.0,
                    "AdjL": 195.0,
                    "AdjC": 208.0,
                    "AdjVo": 2000000.0,
                },
            ],
        }

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=[mock_response1, mock_response2])

        adapter._client = mock_client
        adapter._api_key = "test_key"

        df = await adapter.fetch_ohlcv(["7203", "9984"], date(2024, 1, 4), date(2024, 1, 4))

        assert len(df) == 2


class TestJQuantsAdapterFetchMaster:
    """JQuantsAdapter.fetch_masterメソッドのテスト."""

    @pytest.mark.asyncio
    async def test_fetch_master_not_authenticated(self) -> None:
        """未認証時はエラー."""
        adapter = JQuantsAdapter()

        with pytest.raises(DataFetchError, match="Not authenticated"):
            await adapter.fetch_master(["7203"], date(2024, 1, 1), date(2024, 1, 1))

    @pytest.mark.asyncio
    async def test_fetch_master_with_data(self) -> None:
        """マスタデータ取得成功."""
        adapter = JQuantsAdapter()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {
                    "Date": "2024-01-04",
                    "Code": "7203",
                    "CoName": "トヨタ自動車",
                    "CoNameEn": "TOYOTA MOTOR CORPORATION",
                    "Mkt": "0111",
                    "MktNm": "プライム",
                    "S17": "0050",
                    "S17Nm": "輸送用機器",
                    "S33": "3700",
                    "S33Nm": "輸送用機器",
                },
            ]
        }

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        adapter._client = mock_client
        adapter._api_key = "test_key"

        df = await adapter.fetch_master(["7203"], date(2024, 1, 4), date(2024, 1, 4))

        assert len(df) == 1
        assert df["symbol"][0] == "7203"
        assert df["company_name"][0] == "トヨタ自動車"
        assert df["market_code"][0] == "0111"


class TestJQuantsAdapterFetchAllData:
    """JQuantsAdapter.fetch_all_dataメソッドのテスト."""

    @pytest.mark.asyncio
    async def test_fetch_all_data(self) -> None:
        """OHLCV + マスタ一括取得."""
        adapter = JQuantsAdapter()

        ohlcv_response = MagicMock()
        ohlcv_response.status_code = 200
        ohlcv_response.raise_for_status = MagicMock()
        ohlcv_response.json.return_value = {
            "data": [
                {
                    "Date": "2024-01-04",
                    "Code": "7203",
                    "AdjO": 100.0,
                    "AdjH": 105.0,
                    "AdjL": 99.0,
                    "AdjC": 104.0,
                    "AdjVo": 1000000.0,
                },
            ]
        }

        master_response = MagicMock()
        master_response.status_code = 200
        master_response.raise_for_status = MagicMock()
        master_response.json.return_value = {
            "data": [
                {"Date": "2024-01-04", "Code": "7203", "CoName": "トヨタ自動車", "Mkt": "0111"},
            ]
        }

        mock_client = AsyncMock()
        # fetch_ohlcv用、fetch_master用
        mock_client.get = AsyncMock(side_effect=[ohlcv_response, master_response])

        adapter._client = mock_client
        adapter._api_key = "test_key"

        data = await adapter.fetch_all_data(["7203"], date(2024, 1, 4), date(2024, 1, 4))

        assert "ohlcv" in data
        assert "master" in data
        assert len(data["ohlcv"]) == 1
        assert len(data["master"]) == 1


class TestJQuantsAdapterContextManager:
    """JQuantsAdapterコンテキストマネージャのテスト."""

    @pytest.mark.asyncio
    async def test_context_manager_calls_authenticate(self) -> None:
        """__aenter__でauthenticateが呼ばれること."""
        adapter = JQuantsAdapter()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"data": []}

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.aclose = AsyncMock()

        with (
            patch.dict("os.environ", {"JQUANTS_API_KEY": "test_key"}, clear=True),
            patch("httpx.AsyncClient", return_value=mock_client),
        ):
            async with adapter as a:
                assert a._api_key == "test_key"

    @pytest.mark.asyncio
    async def test_context_manager_closes_client(self) -> None:
        """__aexit__でcloseが呼ばれること."""
        adapter = JQuantsAdapter()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"data": []}

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.aclose = AsyncMock()

        with (
            patch.dict("os.environ", {"JQUANTS_API_KEY": "test_key"}, clear=True),
            patch("httpx.AsyncClient", return_value=mock_client),
        ):
            async with adapter:
                pass

            mock_client.aclose.assert_called_once()


class TestGetDefaultDateRange:
    """get_default_date_range関数のテスト."""

    def test_returns_tuple(self) -> None:
        """タプルを返すこと."""
        result = get_default_date_range()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_end_date_is_12_weeks_ago(self) -> None:
        """終了日が12週間前であること."""
        _, end = get_default_date_range()
        today = date.today()
        from datetime import timedelta

        expected = today - timedelta(weeks=12)
        assert end == expected

    def test_start_date_is_2_years_before_end(self) -> None:
        """開始日が終了日の2年前であること."""
        start, end = get_default_date_range()
        expected = date(end.year - 2, end.month, end.day)
        assert start == expected

    def test_start_before_end(self) -> None:
        """開始日が終了日より前であること."""
        start, end = get_default_date_range()
        assert start < end

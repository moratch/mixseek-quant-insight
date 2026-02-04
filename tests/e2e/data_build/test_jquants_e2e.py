"""E2E tests for J-Quants API adapter.

These tests make real API calls and require:
- JQUANTS_API_KEY environment variable to be set

Note: These tests are designed to be low-traffic to avoid rate limiting.
They only fetch 1-2 days of data for a small number of symbols.
"""

import os
from datetime import date, timedelta

import polars as pl
import pytest

from quant_insight.data_build.jquants import JQuantsAdapter, JQuantsPlan, JQuantsUniverse


def _get_api_key() -> str | None:
    """Get J-Quants API key from environment."""
    return os.environ.get("JQUANTS_API_KEY", "").strip() or None


# Skip all tests if API key is not set
pytestmark = pytest.mark.skipif(
    _get_api_key() is None,
    reason="JQUANTS_API_KEY not set",
)


@pytest.fixture
def api_key() -> str:
    """Get API key (only called when tests are not skipped)."""
    key = _get_api_key()
    assert key is not None
    return key


@pytest.fixture
def recent_trading_date() -> date:
    """Get a recent trading date (weekday, at least 3 days ago).

    Returns a date that is likely to have data available.

    Note:
        システム日付が未来の場合（CI環境など）を考慮し、
        環境変数 E2E_TEST_DATE または固定日付を使用する。
    """
    # 環境変数で日付を指定可能
    env_date = os.environ.get("E2E_TEST_DATE")
    if env_date:
        return date.fromisoformat(env_date)

    # システム日付が現実的かチェック（2025年以降は未来と判断）
    today = date.today()
    if today.year >= 2026:
        # 固定の取引日を使用（2025年1月の平日）
        return date(2025, 1, 17)

    # 通常は7日前の平日を使用
    target = today - timedelta(days=7)

    # Find a weekday (Monday=0, Sunday=6)
    while target.weekday() >= 5:  # Saturday or Sunday
        target -= timedelta(days=1)

    return target


@pytest.mark.e2e
class TestJQuantsAuthentication:
    """認証のE2Eテスト."""

    @pytest.mark.asyncio
    async def test_authenticate_with_valid_key(self, api_key: str) -> None:
        """有効なAPIキーで認証成功."""
        adapter = JQuantsAdapter(plan=JQuantsPlan.FREE)

        result = await adapter.authenticate()
        assert result is True

        await adapter.close()


@pytest.mark.e2e
class TestJQuantsGetUniverse:
    """銘柄ユニバース取得のE2Eテスト."""

    @pytest.mark.asyncio
    async def test_get_universe_prime(self, api_key: str) -> None:
        """プライム市場の銘柄取得."""
        adapter = JQuantsAdapter(
            plan=JQuantsPlan.FREE,
            universe=JQuantsUniverse.PRIME,
        )

        try:
            await adapter.authenticate()
            symbols = await adapter.get_universe()

            # プライム市場には一定数以上の銘柄があるはず
            assert len(symbols) > 100
            # 銘柄コードは文字列
            assert all(isinstance(s, str) for s in symbols)
            # 銘柄コードは4-5桁
            assert all(len(s) == 4 or len(s) == 5 for s in symbols)

        finally:
            await adapter.close()

    @pytest.mark.asyncio
    async def test_get_universe_all(self, api_key: str) -> None:
        """全銘柄取得."""
        adapter = JQuantsAdapter(
            plan=JQuantsPlan.FREE,
            universe=JQuantsUniverse.ALL,
        )

        try:
            await adapter.authenticate()
            symbols = await adapter.get_universe()

            # 全銘柄はプライム市場より多いはず
            assert len(symbols) > 1000

        finally:
            await adapter.close()


@pytest.mark.e2e
class TestJQuantsFetchOhlcv:
    """OHLCV取得のE2Eテスト."""

    @pytest.mark.asyncio
    async def test_fetch_ohlcv_single_day(
        self,
        api_key: str,
        recent_trading_date: date,
    ) -> None:
        """1日分のOHLCVデータ取得."""
        adapter = JQuantsAdapter(
            plan=JQuantsPlan.FREE,
            universe=JQuantsUniverse.PRIME,
        )

        try:
            await adapter.authenticate()

            # プライム市場の一部銘柄だけ取得（低負荷）
            symbols = (await adapter.get_universe())[:5]

            df = await adapter.fetch_ohlcv(
                symbols,
                start_date=recent_trading_date,
                end_date=recent_trading_date,
            )

            # 基本的なスキーマチェック
            assert "datetime" in df.columns
            assert "symbol" in df.columns
            assert "open" in df.columns
            assert "high" in df.columns
            assert "low" in df.columns
            assert "close" in df.columns
            assert "volume" in df.columns

            # データがあれば追加チェック
            if len(df) > 0:
                # datetime型チェック
                assert df["datetime"].dtype == pl.Datetime
                # 価格は正の値
                assert (df["open"] > 0).all()
                assert (df["close"] > 0).all()
                # high >= low
                assert (df["high"] >= df["low"]).all()

        finally:
            await adapter.close()

    @pytest.mark.asyncio
    async def test_fetch_ohlcv_column_mapping(
        self,
        api_key: str,
        recent_trading_date: date,
    ) -> None:
        """カラム名が正しくマッピングされること."""
        adapter = JQuantsAdapter(plan=JQuantsPlan.FREE)

        try:
            await adapter.authenticate()
            symbols = (await adapter.get_universe())[:3]

            df = await adapter.fetch_ohlcv(
                symbols,
                start_date=recent_trading_date,
                end_date=recent_trading_date,
            )

            # J-Quantsのオリジナルカラム名は含まれない
            jquants_columns = ["Date", "Code", "AdjO", "AdjH", "AdjL", "AdjC", "AdjVo"]
            for col in jquants_columns:
                assert col not in df.columns

        finally:
            await adapter.close()


@pytest.mark.e2e
class TestJQuantsFetchMaster:
    """マスタデータ取得のE2Eテスト."""

    @pytest.mark.asyncio
    async def test_fetch_master_single_day(
        self,
        api_key: str,
        recent_trading_date: date,
    ) -> None:
        """1日分のマスタデータ取得."""
        adapter = JQuantsAdapter(
            plan=JQuantsPlan.FREE,
            universe=JQuantsUniverse.PRIME,
        )

        try:
            await adapter.authenticate()

            # プライム市場の一部銘柄だけ取得（低負荷）
            symbols = (await adapter.get_universe())[:5]

            df = await adapter.fetch_master(
                symbols,
                start_date=recent_trading_date,
                end_date=recent_trading_date,
            )

            # 基本的なスキーマチェック
            assert "datetime" in df.columns
            assert "symbol" in df.columns
            assert "company_name" in df.columns
            assert "market_code" in df.columns

            # データがあれば追加チェック
            if len(df) > 0:
                # datetime型チェック
                assert df["datetime"].dtype == pl.Datetime
                # 会社名は空でない
                assert (df["company_name"].str.len_chars() > 0).all()

        finally:
            await adapter.close()


@pytest.mark.e2e
class TestJQuantsFetchAllData:
    """全データ一括取得のE2Eテスト."""

    @pytest.mark.asyncio
    async def test_fetch_all_data(
        self,
        api_key: str,
        recent_trading_date: date,
    ) -> None:
        """OHLCV + マスタの一括取得."""
        adapter = JQuantsAdapter(
            plan=JQuantsPlan.FREE,
            universe=JQuantsUniverse.PRIME,
        )

        try:
            await adapter.authenticate()

            # 最小限の銘柄で取得（低負荷）
            symbols = (await adapter.get_universe())[:3]

            data = await adapter.fetch_all_data(
                symbols,
                start_date=recent_trading_date,
                end_date=recent_trading_date,
            )

            # 両方のデータセットが含まれる
            assert "ohlcv" in data
            assert "master" in data
            assert isinstance(data["ohlcv"], pl.DataFrame)
            assert isinstance(data["master"], pl.DataFrame)

        finally:
            await adapter.close()


@pytest.mark.e2e
class TestJQuantsContextManager:
    """コンテキストマネージャのE2Eテスト."""

    @pytest.mark.asyncio
    async def test_context_manager(
        self,
        api_key: str,
        recent_trading_date: date,
    ) -> None:
        """async withでの使用."""
        async with JQuantsAdapter(plan=JQuantsPlan.FREE) as adapter:
            symbols = await adapter.get_universe()
            assert len(symbols) > 0


@pytest.mark.e2e
class TestJQuantsRateLimiting:
    """レート制限のE2Eテスト."""

    @pytest.mark.asyncio
    async def test_multiple_requests_with_rate_limiting(
        self,
        api_key: str,
        recent_trading_date: date,
    ) -> None:
        """複数リクエストでレート制限が機能すること.

        Note: FreeプランのRate Limitは5 req/min (12秒/req with buffer)
        このテストは実際のスリープを行うため、時間がかかる場合がある。
        """
        adapter = JQuantsAdapter(plan=JQuantsPlan.FREE)

        try:
            await adapter.authenticate()

            # 2日分取得（複数リクエスト発生）
            start = recent_trading_date - timedelta(days=1)
            end = recent_trading_date
            symbols = (await adapter.get_universe())[:2]

            # Rate Limitエラーが発生しないこと
            df = await adapter.fetch_ohlcv(symbols, start_date=start, end_date=end)

            # データが取得できていること（祝日等で0件の可能性あり）
            assert isinstance(df, pl.DataFrame)

        finally:
            await adapter.close()

"""J-Quants API adapter for fetching OHLCV and master data."""

import os
from datetime import date, timedelta
from typing import Any

import httpx
import polars as pl
from tqdm import tqdm

from quant_insight.data_build.base_adapter import BaseDataSourceAdapter
from quant_insight.data_build.jquants.models import (
    MARGIN_COLUMN_MAPPING,
    MASTER_COLUMN_MAPPING,
    OHLCV_COLUMN_MAPPING,
    SHORT_RATIO_COLUMN_MAPPING,
    UNIVERSE_TO_MARKET_CODE,
    JQuantsPlan,
    JQuantsUniverse,
)
from quant_insight.data_build.jquants.rate_limiter import RateLimiter
from quant_insight.exceptions import AuthenticationError, DataFetchError

# J-Quants API Base URL
BASE_URL = "https://api.jquants.com/v2"


class JQuantsAdapter(BaseDataSourceAdapter):
    """J-Quants APIアダプタ.

    FR-015: 差し替え可能なアダプタパターンで実装

    J-Quants APIからOHLCVデータとマスタデータを取得し、
    標準スキーマに変換して保存する。

    データ取得方式:
    - 日付単位で全銘柄のデータを取得（日付でループ）
    - ページネーション対応（pagination_key）
    - Rate Limit対応（sleepを挿入）
    """

    def __init__(
        self,
        plan: JQuantsPlan = JQuantsPlan.FREE,
        universe: JQuantsUniverse = JQuantsUniverse.PRIME,
    ) -> None:
        """JQuantsAdapterを初期化.

        Args:
            plan: J-Quants APIプラン（Rate Limit制御に使用）
            universe: 対象銘柄ユニバース
        """
        self.plan = plan
        self.universe = universe
        self.rate_limiter = RateLimiter(plan)
        self._api_key: str | None = None
        self._client: httpx.AsyncClient | None = None

    async def _request_with_retry(
        self,
        path: str,
        params: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """リトライ付きでAPIリクエストを実行.

        Rate Limit待機 + 指数的バックオフリトライを行う。

        Args:
            path: APIエンドポイントパス（例: "/equities/master"）
            params: クエリパラメータ

        Returns:
            dict[str, Any]: APIレスポンスのJSONデータ

        Raises:
            DataFetchError: 認証前に呼び出された場合
            httpx.HTTPError: リトライ後もリクエストが失敗した場合
        """
        if self._client is None:
            raise DataFetchError("Not authenticated. Call authenticate() first.")

        await self.rate_limiter.wait()

        async def do_request() -> dict[str, Any]:
            assert self._client is not None
            response = await self._client.get(path, params=params)
            response.raise_for_status()
            return response.json()  # type: ignore[no-any-return]

        return await self.rate_limiter.execute_with_retry(do_request)

    async def authenticate(self) -> bool:
        """API認証を実行.

        環境変数JQUANTS_API_KEYからAPIキーを取得し、認証を確認する。

        Returns:
            bool: 認証成功ならTrue

        Raises:
            AuthenticationError: APIキーが未設定または認証失敗時
        """
        api_key = os.environ.get("JQUANTS_API_KEY", "").strip()
        if not api_key:
            raise AuthenticationError("JQUANTS_API_KEY environment variable is not set")

        self._api_key = api_key
        self._client = httpx.AsyncClient(
            base_url=BASE_URL,
            headers={"X-API-KEY": api_key},
            timeout=30.0,
        )

        # 認証確認のためマスタエンドポイントにアクセス
        try:
            await self._request_with_retry("/equities/master")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise AuthenticationError("Invalid API key") from e
            raise AuthenticationError(f"Authentication failed: {e}") from e
        except httpx.RequestError as e:
            raise AuthenticationError(f"Connection error: {e}") from e

        return True

    async def get_universe(self, market: JQuantsUniverse | None = None) -> list[str]:
        """利用可能な銘柄一覧を取得.

        Args:
            market: 対象市場（Noneの場合はself.universeを使用）

        Returns:
            list[str]: 銘柄コードのリスト

        Raises:
            DataFetchError: データ取得失敗時
        """
        target_market = market if market is not None else self.universe
        market_code = UNIVERSE_TO_MARKET_CODE[target_market]

        try:
            data = await self._request_with_retry("/equities/master")
        except httpx.HTTPError as e:
            raise DataFetchError(f"Failed to fetch universe: {e}") from e

        info_list = data.get("data", [])
        if market_code is None:
            return [item["Code"] for item in info_list]
        return [item["Code"] for item in info_list if item.get("Mkt") == market_code]

    async def _fetch_paginated_data_by_date(
        self,
        endpoint: str,
        symbols: list[str],
        start_date: date,
        end_date: date,
        description: str,
    ) -> list[dict[str, Any]]:
        """日付単位でページネーション対応のデータ取得を行う共通ヘルパー.

        Args:
            endpoint: APIエンドポイントパス（例: "/equities/bars/daily"）
            symbols: 銘柄コードのリスト（フィルタ用）
            start_date: 取得開始日
            end_date: 取得終了日
            description: プログレスバーの説明文

        Returns:
            list[dict[str, Any]]: 取得した生データのリスト

        Raises:
            DataFetchError: データ取得失敗時
        """
        all_data: list[dict[str, Any]] = []
        symbol_set = set(symbols)
        current_date = start_date
        total_days = (end_date - start_date).days + 1

        with tqdm(total=total_days, desc=description) as pbar:
            while current_date <= end_date:
                date_str = current_date.isoformat()
                pagination_key: str | None = None

                while True:
                    params: dict[str, str] = {"date": date_str}
                    if pagination_key:
                        params["pagination_key"] = pagination_key

                    try:
                        data = await self._request_with_retry(endpoint, params)
                    except httpx.HTTPError as e:
                        raise DataFetchError(f"Failed to fetch {description} for {date_str}: {e}") from e

                    for item in data.get("data", []):
                        if item.get("Code") in symbol_set:
                            all_data.append(item)

                    pagination_key = data.get("pagination_key")
                    if not pagination_key:
                        break

                current_date += timedelta(days=1)
                pbar.update(1)

        return all_data

    async def fetch_ohlcv(
        self,
        symbols: list[str],
        start_date: date,
        end_date: date,
    ) -> pl.DataFrame:
        """OHLCVデータを取得（日付単位）.

        Args:
            symbols: 銘柄コードのリスト（universeフィルタ用、APIは全銘柄返す）
            start_date: 取得開始日
            end_date: 取得終了日

        Returns:
            pl.DataFrame: 標準スキーマに変換されたOHLCVデータ
        """
        all_data = await self._fetch_paginated_data_by_date(
            endpoint="/equities/bars/daily",
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            description="Fetching OHLCV",
        )

        if not all_data:
            # 空のDataFrameを返す
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

        # DataFrameに変換
        df = pl.DataFrame(all_data)

        # マッピングに含まれないカラムがあればエラー
        unmapped_columns = set(df.columns) - set(OHLCV_COLUMN_MAPPING.keys())
        if unmapped_columns:
            raise DataFetchError(f"Unmapped columns in OHLCV response: {unmapped_columns}")

        # カラム名マッピング（存在するカラムのみ）
        rename_dict = {k: v for k, v in OHLCV_COLUMN_MAPPING.items() if k in df.columns}
        df = df.rename(rename_dict)

        # datetime変換（Date文字列 → datetime）
        if "datetime" in df.columns:
            df = df.with_columns(pl.col("datetime").str.to_datetime("%Y-%m-%d").alias("datetime"))

        # ソート
        df = df.sort(["symbol", "datetime"])

        return df

    async def fetch_master(
        self,
        symbols: list[str],
        start_date: date,
        end_date: date,
    ) -> pl.DataFrame:
        """マスタデータを取得（日付単位）.

        Args:
            symbols: 銘柄コードのリスト（フィルタ用）
            start_date: 取得開始日
            end_date: 取得終了日

        Returns:
            pl.DataFrame: 変換されたマスタデータ
        """
        all_data = await self._fetch_paginated_data_by_date(
            endpoint="/equities/master",
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            description="Fetching Master",
        )

        if not all_data:
            # 空のDataFrameを返す
            return pl.DataFrame(
                schema={
                    "datetime": pl.Datetime,
                    "symbol": pl.Utf8,
                    "company_name": pl.Utf8,
                    "market_code": pl.Utf8,
                }
            )

        # DataFrameに変換
        df = pl.DataFrame(all_data)

        # マッピングに含まれないカラムがあればエラー
        unmapped_columns = set(df.columns) - set(MASTER_COLUMN_MAPPING.keys())
        if unmapped_columns:
            raise DataFetchError(f"Unmapped columns in Master response: {unmapped_columns}")

        # カラム名マッピング（存在するカラムのみ）
        rename_dict = {k: v for k, v in MASTER_COLUMN_MAPPING.items() if k in df.columns}
        df = df.rename(rename_dict)

        # datetime変換（Date文字列 → datetime）
        if "datetime" in df.columns:
            df = df.with_columns(pl.col("datetime").str.to_datetime("%Y-%m-%d").alias("datetime"))

        # 重複削除（土日祝にAPIを呼び出すと前営業日のデータが重複するため）
        df = df.unique(subset=["datetime", "symbol"])

        # ソート
        df = df.sort(["symbol", "datetime"])

        return df

    async def fetch_margin_interest(
        self,
        symbols: list[str],
        start_date: date,
        end_date: date,
    ) -> pl.DataFrame:
        """信用取引週末残高を取得（週次、日付単位で全銘柄）.

        Standard プラン以上で利用可能。週次データのため金曜日のみデータが存在する。

        Args:
            symbols: 銘柄コードのリスト（フィルタ用）
            start_date: 取得開始日
            end_date: 取得終了日

        Returns:
            pl.DataFrame: 信用取引残高データ
        """
        all_data = await self._fetch_paginated_data_by_date(
            endpoint="/markets/margin-interest",
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            description="Fetching Margin Interest",
        )

        if not all_data:
            return pl.DataFrame(
                schema={
                    "datetime": pl.Datetime,
                    "symbol": pl.Utf8,
                    "margin_short_vol": pl.Float64,
                    "margin_long_vol": pl.Float64,
                }
            )

        df = pl.DataFrame(all_data)
        rename_dict = {k: v for k, v in MARGIN_COLUMN_MAPPING.items() if k in df.columns}
        df = df.rename(rename_dict)

        if "datetime" in df.columns:
            df = df.with_columns(pl.col("datetime").str.to_datetime("%Y-%m-%d").alias("datetime"))

        df = df.sort(["symbol", "datetime"])
        return df

    async def fetch_short_ratio(
        self,
        start_date: date,
        end_date: date,
    ) -> pl.DataFrame:
        """業種別空売り比率を取得（日次、全33業種）.

        Standard プラン以上で利用可能。

        Args:
            start_date: 取得開始日
            end_date: 取得終了日

        Returns:
            pl.DataFrame: 業種別空売り比率データ
        """
        all_data: list[dict[str, object]] = []
        current_date = start_date
        total_days = (end_date - start_date).days + 1

        with tqdm(total=total_days, desc="Fetching Short Ratio") as pbar:
            while current_date <= end_date:
                date_str = current_date.isoformat()

                try:
                    data = await self._request_with_retry(
                        "/markets/short-ratio", {"date": date_str}
                    )
                except httpx.HTTPError as e:
                    raise DataFetchError(f"Failed to fetch short ratio for {date_str}: {e}") from e

                all_data.extend(data.get("data", []))
                current_date += timedelta(days=1)
                pbar.update(1)

        if not all_data:
            return pl.DataFrame(
                schema={
                    "datetime": pl.Datetime,
                    "sector33_code": pl.Utf8,
                    "sell_ex_short_value": pl.Float64,
                }
            )

        df = pl.DataFrame(all_data)
        rename_dict = {k: v for k, v in SHORT_RATIO_COLUMN_MAPPING.items() if k in df.columns}
        df = df.rename(rename_dict)

        if "datetime" in df.columns:
            df = df.with_columns(pl.col("datetime").str.to_datetime("%Y-%m-%d").alias("datetime"))

        df = df.sort(["datetime", "sector33_code"])
        return df

    async def fetch_all_data(
        self,
        symbols: list[str],
        start_date: date,
        end_date: date,
    ) -> dict[str, pl.DataFrame]:
        """全データを取得（OHLCV + マスタ + マージン）.

        Standard プラン以上の場合は信用取引データも取得する。

        Args:
            symbols: 銘柄コードのリスト
            start_date: 取得開始日
            end_date: 取得終了日

        Returns:
            dict[str, pl.DataFrame]: {
                "ohlcv": pl.DataFrame,
                "master": pl.DataFrame,
                "margin": pl.DataFrame,       # Standard+
                "short_ratio": pl.DataFrame,   # Standard+
            }
        """
        ohlcv = await self.fetch_ohlcv(symbols, start_date, end_date)
        master = await self.fetch_master(symbols, start_date, end_date)

        result: dict[str, pl.DataFrame] = {
            "ohlcv": ohlcv,
            "master": master,
        }

        # Standard プラン以上で信用取引データを取得
        if self.plan in (JQuantsPlan.STANDARD, JQuantsPlan.PREMIUM):
            margin = await self.fetch_margin_interest(symbols, start_date, end_date)
            short_ratio = await self.fetch_short_ratio(start_date, end_date)
            result["margin"] = margin
            result["short_ratio"] = short_ratio

        return result

    async def close(self) -> None:
        """HTTPクライアントを閉じる."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "JQuantsAdapter":
        """Async context manager entry."""
        await self.authenticate()
        return self

    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: object,
    ) -> None:
        """Async context manager exit."""
        await self.close()


def get_default_date_range() -> tuple[date, date]:
    """Freeプランのデフォルト日付範囲を取得.

    J-Quants Freeプランは「直近12週間を除く2年分」のデータが取得可能。
    - 終了日: today - 12週間
    - 開始日: 終了日 - 2年

    Returns:
        (start_date, end_date): 取得可能な日付範囲
    """
    today = date.today()
    end_date = today - timedelta(weeks=12)
    start_date = date(end_date.year - 2, end_date.month, end_date.day)
    return start_date, end_date

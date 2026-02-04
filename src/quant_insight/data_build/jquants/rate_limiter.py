"""Rate limiter for J-Quants API requests."""

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

import httpx

from quant_insight.data_build.jquants.models import JQuantsPlan

# プラン別のRate Limit（1分あたりのリクエスト数）
RATE_LIMITS_PER_MINUTE: dict[JQuantsPlan, int] = {
    JQuantsPlan.FREE: 5,
    JQuantsPlan.LIGHT: 60,
    JQuantsPlan.STANDARD: 120,
    JQuantsPlan.PREMIUM: 500,
}

# バッファ係数（安全マージン）
RATE_LIMIT_BUFFER = 0.8

# 指数的バックオフの最大遅延（秒）
MAX_BACKOFF_DELAY = 70.0

# リトライ対象のHTTPステータスコード
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

T = TypeVar("T")


class RetryableError(Exception):
    """リトライ可能なエラー.

    このエラーが発生した場合、execute_with_retryはリトライを試みる。
    """

    def __init__(self, message: str, cause: Exception | None = None) -> None:
        """RetryableErrorを初期化.

        Args:
            message: エラーメッセージ
            cause: 原因となった例外
        """
        super().__init__(message)
        if cause is not None:
            self.__cause__ = cause


class RateLimiter:
    """プラン別のRate Limit管理.

    APIリクエスト間に適切なスリープを挿入し、
    Rate Limitを超えないようにする。
    指数的バックオフによるリトライ機能も提供する。
    """

    def __init__(self, plan: JQuantsPlan) -> None:
        """RateLimiterを初期化.

        Args:
            plan: J-Quants APIプラン
        """
        self.plan = plan
        self._requests_per_minute = int(RATE_LIMITS_PER_MINUTE[plan] * RATE_LIMIT_BUFFER)
        self._sleep_interval = 60.0 / self._requests_per_minute
        self._last_request_time: float | None = None

    @property
    def sleep_interval(self) -> float:
        """リクエスト間のスリープ間隔（秒）."""
        return self._sleep_interval

    @property
    def base_delay(self) -> float:
        """指数的バックオフのベース遅延（秒）.

        プランのsleep_intervalを基準とする。
        """
        return self._sleep_interval

    @property
    def requests_per_minute(self) -> int:
        """バッファ適用後の1分あたりのリクエスト数."""
        return self._requests_per_minute

    def calculate_backoff_delay(self, retry_count: int) -> float:
        """指数的バックオフの遅延時間を計算.

        Args:
            retry_count: 現在のリトライ回数（0始まり）

        Returns:
            float: 遅延時間（秒）。base_delay * 2^retry_count を返すが、
                   MAX_BACKOFF_DELAY を超えない。
        """
        delay = self.base_delay * (2**retry_count)
        return float(min(delay, MAX_BACKOFF_DELAY))

    def _is_retryable_error(self, error: Exception) -> bool:
        """エラーがリトライ可能か判定.

        Args:
            error: 発生した例外

        Returns:
            bool: リトライ可能ならTrue
        """
        if isinstance(error, RetryableError):
            return True

        if isinstance(error, httpx.HTTPStatusError):
            return error.response.status_code in RETRYABLE_STATUS_CODES

        if isinstance(error, (httpx.ConnectError, httpx.TimeoutException)):
            return True

        return False

    async def execute_with_retry(
        self,
        func: Callable[[], Awaitable[T]],
        max_retries: int = 3,
    ) -> T:
        """リトライ付きで関数を実行.

        指数的バックオフを使用してリトライを行う。

        Args:
            func: 実行する非同期関数
            max_retries: 最大リトライ回数（デフォルト: 3）

        Returns:
            関数の戻り値

        Raises:
            Exception: 最大リトライ回数を超えた場合、
                       または非リトライ対象のエラーが発生した場合
        """
        last_error: Exception | None = None

        for attempt in range(max_retries + 1):
            try:
                return await func()
            except Exception as e:
                if not self._is_retryable_error(e):
                    raise

                last_error = e

                if attempt < max_retries:
                    delay = self.calculate_backoff_delay(attempt)
                    await asyncio.sleep(delay)

        # max_retriesを超えた場合、最後のエラーを再送出
        if last_error is not None:
            raise last_error

        # ここには到達しないはずだが、型チェッカー用
        raise RuntimeError("Unexpected state in execute_with_retry")  # pragma: no cover

    async def wait(self) -> None:
        """次のリクエストまで待機.

        前回のリクエストからの経過時間を計算し、
        Rate Limitを超えないようにスリープする。
        """
        if self._last_request_time is not None:
            elapsed = time.monotonic() - self._last_request_time
            if elapsed < self._sleep_interval:
                await asyncio.sleep(self._sleep_interval - elapsed)

        self._last_request_time = time.monotonic()

    def reset(self) -> None:
        """Rate Limiter状態をリセット."""
        self._last_request_time = None

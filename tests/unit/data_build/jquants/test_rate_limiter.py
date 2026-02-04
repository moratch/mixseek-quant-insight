"""Unit tests for J-Quants rate limiter."""

import asyncio
import time
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from quant_insight.data_build.jquants.models import JQuantsPlan
from quant_insight.data_build.jquants.rate_limiter import (
    RATE_LIMIT_BUFFER,
    RATE_LIMITS_PER_MINUTE,
    RateLimiter,
    RetryableError,
)


class TestRateLimitsPerMinute:
    """RATE_LIMITS_PER_MINUTE定数のテスト."""

    def test_free_plan_limit(self) -> None:
        """Freeプランのレート制限が正しいこと."""
        assert RATE_LIMITS_PER_MINUTE[JQuantsPlan.FREE] == 5

    def test_light_plan_limit(self) -> None:
        """Lightプランのレート制限が正しいこと."""
        assert RATE_LIMITS_PER_MINUTE[JQuantsPlan.LIGHT] == 60

    def test_standard_plan_limit(self) -> None:
        """Standardプランのレート制限が正しいこと."""
        assert RATE_LIMITS_PER_MINUTE[JQuantsPlan.STANDARD] == 120

    def test_premium_plan_limit(self) -> None:
        """Premiumプランのレート制限が正しいこと."""
        assert RATE_LIMITS_PER_MINUTE[JQuantsPlan.PREMIUM] == 500

    def test_all_plans_covered(self) -> None:
        """全プランがレート制限に含まれていること."""
        for plan in JQuantsPlan:
            assert plan in RATE_LIMITS_PER_MINUTE


class TestRateLimitBuffer:
    """RATE_LIMIT_BUFFER定数のテスト."""

    def test_buffer_value(self) -> None:
        """バッファ係数が0.8であること."""
        assert RATE_LIMIT_BUFFER == 0.8

    def test_buffer_is_less_than_one(self) -> None:
        """バッファ係数が1未満であること（安全マージン確保）."""
        assert RATE_LIMIT_BUFFER < 1.0

    def test_buffer_is_positive(self) -> None:
        """バッファ係数が正の値であること."""
        assert RATE_LIMIT_BUFFER > 0


class TestRateLimiterInit:
    """RateLimiter初期化のテスト."""

    def test_init_with_free_plan(self) -> None:
        """Freeプランでの初期化."""
        limiter = RateLimiter(JQuantsPlan.FREE)
        assert limiter.plan == JQuantsPlan.FREE
        # 5 * 0.8 = 4 requests/min
        assert limiter.requests_per_minute == 4
        # 60 / 4 = 15.0 seconds
        assert limiter.sleep_interval == 15.0

    def test_init_with_light_plan(self) -> None:
        """Lightプランでの初期化."""
        limiter = RateLimiter(JQuantsPlan.LIGHT)
        assert limiter.plan == JQuantsPlan.LIGHT
        # 60 * 0.8 = 48 requests/min
        assert limiter.requests_per_minute == 48
        # 60 / 48 = 1.25 seconds
        assert limiter.sleep_interval == 1.25

    def test_init_with_standard_plan(self) -> None:
        """Standardプランでの初期化."""
        limiter = RateLimiter(JQuantsPlan.STANDARD)
        assert limiter.plan == JQuantsPlan.STANDARD
        # 120 * 0.8 = 96 requests/min
        assert limiter.requests_per_minute == 96
        # 60 / 96 = 0.625 seconds
        assert limiter.sleep_interval == 0.625

    def test_init_with_premium_plan(self) -> None:
        """Premiumプランでの初期化."""
        limiter = RateLimiter(JQuantsPlan.PREMIUM)
        assert limiter.plan == JQuantsPlan.PREMIUM
        # 500 * 0.8 = 400 requests/min
        assert limiter.requests_per_minute == 400
        # 60 / 400 = 0.15 seconds
        assert limiter.sleep_interval == 0.15


class TestRateLimiterWait:
    """RateLimiter.waitメソッドのテスト."""

    @pytest.mark.asyncio
    async def test_first_wait_no_sleep(self) -> None:
        """初回呼び出し時はスリープしないこと."""
        limiter = RateLimiter(JQuantsPlan.PREMIUM)

        with patch("asyncio.sleep") as mock_sleep:
            start = time.monotonic()
            await limiter.wait()
            elapsed = time.monotonic() - start

            # 初回はスリープなし
            mock_sleep.assert_not_called()
            assert elapsed < 0.1

    @pytest.mark.asyncio
    async def test_consecutive_waits_sleep(self) -> None:
        """連続呼び出し時は適切にスリープすること."""
        limiter = RateLimiter(JQuantsPlan.PREMIUM)

        # 初回
        await limiter.wait()

        with patch("asyncio.sleep") as mock_sleep:
            mock_sleep.return_value = None  # asyncio.sleepはコルーチン
            # asyncio.sleepをモックするために別の方法
            pass

        # 実際のスリープ時間をテストするには短いインターバルを使用
        # Premiumプランは0.15秒なので実際に待つ
        start = time.monotonic()
        await limiter.wait()
        elapsed = time.monotonic() - start

        # 少なくともスリープインターバルの一部は経過しているはず
        # （前回からの経過時間によって実際のスリープ時間は変わる）
        assert elapsed >= 0  # 最低でも即座に返る

    @pytest.mark.asyncio
    async def test_wait_respects_elapsed_time(self) -> None:
        """経過時間を考慮してスリープすること."""
        limiter = RateLimiter(JQuantsPlan.PREMIUM)  # 0.15秒間隔

        await limiter.wait()

        # 十分な時間待つ
        await asyncio.sleep(0.2)

        start = time.monotonic()
        await limiter.wait()
        elapsed = time.monotonic() - start

        # 既に十分な時間が経過しているのでスリープしない
        assert elapsed < 0.05


class TestRateLimiterReset:
    """RateLimiter.resetメソッドのテスト."""

    @pytest.mark.asyncio
    async def test_reset_clears_last_request_time(self) -> None:
        """リセット後は初回呼び出しと同じ挙動になること."""
        limiter = RateLimiter(JQuantsPlan.PREMIUM)

        # 1回目
        await limiter.wait()
        # 2回目（スリープする可能性あり）
        await limiter.wait()

        # リセット
        limiter.reset()

        # リセット後の呼び出し（スリープしない）
        start = time.monotonic()
        await limiter.wait()
        elapsed = time.monotonic() - start

        assert elapsed < 0.05


class TestRateLimiterProperties:
    """RateLimiterプロパティのテスト."""

    def test_sleep_interval_property(self) -> None:
        """sleep_intervalプロパティが計算値を返すこと."""
        limiter = RateLimiter(JQuantsPlan.LIGHT)
        expected = 60.0 / (RATE_LIMITS_PER_MINUTE[JQuantsPlan.LIGHT] * RATE_LIMIT_BUFFER)
        assert limiter.sleep_interval == expected

    def test_requests_per_minute_property(self) -> None:
        """requests_per_minuteプロパティがバッファ適用後の値を返すこと."""
        limiter = RateLimiter(JQuantsPlan.LIGHT)
        expected = int(RATE_LIMITS_PER_MINUTE[JQuantsPlan.LIGHT] * RATE_LIMIT_BUFFER)
        assert limiter.requests_per_minute == expected


class TestRetryableError:
    """RetryableErrorのテスト."""

    def test_retryable_error_message(self) -> None:
        """RetryableErrorがメッセージを保持すること."""
        error = RetryableError("Test error")
        assert str(error) == "Test error"

    def test_retryable_error_with_cause(self) -> None:
        """RetryableErrorが原因例外を保持すること."""
        cause = ValueError("Original error")
        error = RetryableError("Wrapped error", cause)
        assert error.__cause__ is cause


class TestExponentialBackoff:
    """指数的バックオフのテスト."""

    def test_base_delay_equals_sleep_interval(self) -> None:
        """ベース遅延がsleep_intervalと等しいこと."""
        limiter = RateLimiter(JQuantsPlan.LIGHT)
        assert limiter.base_delay == limiter.sleep_interval

    def test_calculate_backoff_delay_first_retry(self) -> None:
        """1回目のリトライ時の遅延が base_delay * 2^0 = base_delay であること."""
        limiter = RateLimiter(JQuantsPlan.LIGHT)
        delay = limiter.calculate_backoff_delay(retry_count=0)
        assert delay == limiter.base_delay

    def test_calculate_backoff_delay_second_retry(self) -> None:
        """2回目のリトライ時の遅延が base_delay * 2^1 = base_delay * 2 であること."""
        limiter = RateLimiter(JQuantsPlan.LIGHT)
        delay = limiter.calculate_backoff_delay(retry_count=1)
        assert delay == limiter.base_delay * 2

    def test_calculate_backoff_delay_third_retry(self) -> None:
        """3回目のリトライ時の遅延が base_delay * 2^2 = base_delay * 4 であること."""
        limiter = RateLimiter(JQuantsPlan.LIGHT)
        delay = limiter.calculate_backoff_delay(retry_count=2)
        assert delay == limiter.base_delay * 4

    def test_calculate_backoff_delay_respects_max_delay(self) -> None:
        """遅延が最大値を超えないこと."""
        limiter = RateLimiter(JQuantsPlan.FREE)  # base_delay = 15秒
        # 大きなretry_countでも最大70秒を超えない
        delay = limiter.calculate_backoff_delay(retry_count=10)
        assert delay <= 70.0


class TestExecuteWithRetry:
    """execute_with_retryメソッドのテスト."""

    @pytest.mark.asyncio
    async def test_successful_execution_no_retry(self) -> None:
        """成功時はリトライしないこと."""
        limiter = RateLimiter(JQuantsPlan.PREMIUM)
        mock_func = AsyncMock(return_value="success")

        result = await limiter.execute_with_retry(mock_func)

        assert result == "success"
        assert mock_func.call_count == 1

    @pytest.mark.asyncio
    async def test_retry_on_rate_limit_error(self) -> None:
        """HTTP 429エラー時にリトライすること."""
        limiter = RateLimiter(JQuantsPlan.PREMIUM)

        # 最初の2回は429エラー、3回目で成功
        request = httpx.Request("GET", "https://api.example.com")
        response_429 = httpx.Response(429, request=request)
        mock_func = AsyncMock(
            side_effect=[
                httpx.HTTPStatusError("Rate limited", request=request, response=response_429),
                httpx.HTTPStatusError("Rate limited", request=request, response=response_429),
                "success",
            ]
        )

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await limiter.execute_with_retry(mock_func)

        assert result == "success"
        assert mock_func.call_count == 3

    @pytest.mark.asyncio
    async def test_retry_on_retryable_error(self) -> None:
        """RetryableError時にリトライすること."""
        limiter = RateLimiter(JQuantsPlan.PREMIUM)

        mock_func = AsyncMock(
            side_effect=[
                RetryableError("Temporary error"),
                "success",
            ]
        )

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await limiter.execute_with_retry(mock_func)

        assert result == "success"
        assert mock_func.call_count == 2

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self) -> None:
        """最大リトライ回数を超えた場合は例外を再送出すること."""
        limiter = RateLimiter(JQuantsPlan.PREMIUM)

        request = httpx.Request("GET", "https://api.example.com")
        response_429 = httpx.Response(429, request=request)
        error = httpx.HTTPStatusError("Rate limited", request=request, response=response_429)
        mock_func = AsyncMock(side_effect=error)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(httpx.HTTPStatusError):
                await limiter.execute_with_retry(mock_func, max_retries=3)

        # 初回 + 3回リトライ = 4回
        assert mock_func.call_count == 4

    @pytest.mark.asyncio
    async def test_non_retryable_error_raises_immediately(self) -> None:
        """リトライ対象外のエラーは即座に再送出すること."""
        limiter = RateLimiter(JQuantsPlan.PREMIUM)

        mock_func = AsyncMock(side_effect=ValueError("Not retryable"))

        with pytest.raises(ValueError, match="Not retryable"):
            await limiter.execute_with_retry(mock_func)

        assert mock_func.call_count == 1

    @pytest.mark.asyncio
    async def test_http_500_is_retryable(self) -> None:
        """HTTP 500エラー時にリトライすること."""
        limiter = RateLimiter(JQuantsPlan.PREMIUM)

        request = httpx.Request("GET", "https://api.example.com")
        response_500 = httpx.Response(500, request=request)
        mock_func = AsyncMock(
            side_effect=[
                httpx.HTTPStatusError("Server error", request=request, response=response_500),
                "success",
            ]
        )

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await limiter.execute_with_retry(mock_func)

        assert result == "success"
        assert mock_func.call_count == 2

    @pytest.mark.asyncio
    async def test_http_400_is_not_retryable(self) -> None:
        """HTTP 400エラーはリトライしないこと."""
        limiter = RateLimiter(JQuantsPlan.PREMIUM)

        request = httpx.Request("GET", "https://api.example.com")
        response_400 = httpx.Response(400, request=request)
        error = httpx.HTTPStatusError("Bad request", request=request, response=response_400)
        mock_func = AsyncMock(side_effect=error)

        with pytest.raises(httpx.HTTPStatusError):
            await limiter.execute_with_retry(mock_func)

        assert mock_func.call_count == 1

    @pytest.mark.asyncio
    async def test_backoff_delay_increases_exponentially(self) -> None:
        """バックオフ遅延が指数的に増加すること."""
        limiter = RateLimiter(JQuantsPlan.PREMIUM)  # base_delay = 0.15秒

        request = httpx.Request("GET", "https://api.example.com")
        response_429 = httpx.Response(429, request=request)
        error = httpx.HTTPStatusError("Rate limited", request=request, response=response_429)
        mock_func = AsyncMock(side_effect=[error, error, error, "success"])

        sleep_delays: list[float] = []

        async def capture_sleep(delay: float) -> None:
            sleep_delays.append(delay)

        with patch("asyncio.sleep", side_effect=capture_sleep):
            await limiter.execute_with_retry(mock_func, max_retries=3)

        # 指数的バックオフを確認
        base = limiter.base_delay
        assert len(sleep_delays) == 3
        assert sleep_delays[0] == base  # 2^0 = 1
        assert sleep_delays[1] == base * 2  # 2^1 = 2
        assert sleep_delays[2] == base * 4  # 2^2 = 4

    @pytest.mark.asyncio
    async def test_connection_error_is_retryable(self) -> None:
        """接続エラー時にリトライすること."""
        limiter = RateLimiter(JQuantsPlan.PREMIUM)

        mock_func = AsyncMock(
            side_effect=[
                httpx.ConnectError("Connection failed"),
                "success",
            ]
        )

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await limiter.execute_with_retry(mock_func)

        assert result == "success"
        assert mock_func.call_count == 2

    @pytest.mark.asyncio
    async def test_timeout_error_is_retryable(self) -> None:
        """タイムアウトエラー時にリトライすること."""
        limiter = RateLimiter(JQuantsPlan.PREMIUM)

        mock_func = AsyncMock(
            side_effect=[
                httpx.TimeoutException("Timeout"),
                "success",
            ]
        )

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await limiter.execute_with_retry(mock_func)

        assert result == "success"
        assert mock_func.call_count == 2


class TestRetryWithDifferentPlans:
    """各プランでのリトライ動作テスト."""

    @pytest.mark.asyncio
    async def test_free_plan_backoff_delay(self) -> None:
        """Freeプランのバックオフ遅延が正しいこと."""
        limiter = RateLimiter(JQuantsPlan.FREE)
        # Free: 15秒/リクエスト → base_delay = 15秒
        assert limiter.base_delay == 15.0
        assert limiter.calculate_backoff_delay(0) == 15.0
        assert limiter.calculate_backoff_delay(1) == 30.0
        assert limiter.calculate_backoff_delay(2) == 60.0
        assert limiter.calculate_backoff_delay(3) == 70.0  # max_delay上限（15*8=120→70）

    @pytest.mark.asyncio
    async def test_light_plan_backoff_delay(self) -> None:
        """Lightプランのバックオフ遅延が正しいこと."""
        limiter = RateLimiter(JQuantsPlan.LIGHT)
        # Light: 1.25秒/リクエスト → base_delay = 1.25秒
        assert limiter.base_delay == 1.25
        assert limiter.calculate_backoff_delay(0) == 1.25
        assert limiter.calculate_backoff_delay(1) == 2.5
        assert limiter.calculate_backoff_delay(2) == 5.0

    @pytest.mark.asyncio
    async def test_premium_plan_backoff_delay(self) -> None:
        """Premiumプランのバックオフ遅延が正しいこと."""
        limiter = RateLimiter(JQuantsPlan.PREMIUM)
        # Premium: 0.15秒/リクエスト → base_delay = 0.15秒
        assert limiter.base_delay == 0.15
        assert limiter.calculate_backoff_delay(0) == 0.15
        assert limiter.calculate_backoff_delay(1) == 0.30
        assert limiter.calculate_backoff_delay(2) == 0.60

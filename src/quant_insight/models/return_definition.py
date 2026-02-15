"""Return definition model for quant-insight."""

from typing import Self

from pydantic import BaseModel, Field, model_validator


class ReturnDefinition(BaseModel):
    """Return calculation definition (FR-007, FR-008).

    Defines how returns are calculated from OHLCV data.

    Supported methods:
        - close2close: close[t] → close[t+window] (default)
        - open2close: open[t+1] → close[t+window] (realistic entry)
        - daytrade_market: open[t+1] → close[t+1] (intraday return, window=1 only)

    Note:
        daytrade_market is numerically equivalent to open2close(window=1).
        It is separated to make the intent explicit:
        - open2close: swing-style (buy at open, sell at close after window days)
        - daytrade_market: day-trade evaluation (next-day intraday return)

        Limit order types (daytrade_open_limit, daytrade_intraday_limit) are NOT
        included here. They live in ExecutionAnalyzer, which is independent of
        the MixSeek evaluation pipeline.
    """

    window: int = Field(
        default=1,
        ge=1,
        description="Return calculation window width (in days)",
    )
    method: str = Field(
        default="close2close",
        pattern=r"^(close2close|open2close|daytrade_market)$",
        description="Return calculation method: close2close, open2close, or daytrade_market",
    )

    @model_validator(mode="after")
    def validate_daytrade_window(self) -> Self:
        """daytrade_market requires window==1."""
        if self.method == "daytrade_market" and self.window != 1:
            msg = f"daytrade_market requires window=1, got {self.window}"
            raise ValueError(msg)
        return self

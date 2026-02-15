"""Execution analyzer for limit order fill analysis.

WARNING: This module is COMPLETELY INDEPENDENT of the MixSeek evaluation pipeline.
Do NOT feed ExecutionResult.data into the evaluator / backtest_loop.
Non-fill = null introduces selection bias.
Use this only for qal-compatible script verification.
"""

from dataclasses import dataclass

import polars as pl


@dataclass
class ExecutionResult:
    """Result of execution analysis.

    Attributes:
        data: DataFrame with columns:
            datetime, symbol, is_executed, entry_price, limit_return
            NOTE: Non-executed rows have null for limit_return and entry_price.
        execution_rate: Fraction of rows where the order was filled.
        method: The limit order method used.
        position_side: "long" or "short".
        limit_offset_pct: The offset percentage used for limit price.
    """

    data: pl.DataFrame
    execution_rate: float
    method: str
    position_side: str
    limit_offset_pct: float


class ExecutionAnalyzer:
    """Limit order fill analysis. Independent of MixSeek evaluation pipeline.

    WARNING: Do NOT pass this class's output to MixSeek evaluator / backtest_loop.
    Non-fill = null introduces selection bias.
    Use this only for qal-compatible script verification.

    Supported methods:
        - daytrade_open_limit: Limit order at open, fill if open meets limit condition.
        - daytrade_intraday_limit: Limit order during the day, fill if intraday
          price touches the limit.
    """

    VALID_METHODS: frozenset[str] = frozenset({"daytrade_open_limit", "daytrade_intraday_limit"})
    VALID_SIDES: frozenset[str] = frozenset({"long", "short"})

    def analyze(
        self,
        ohlcv: pl.DataFrame,
        method: str = "daytrade_open_limit",
        position_side: str = "long",
        limit_offset_pct: float = 1.0,
    ) -> ExecutionResult:
        """Analyze limit order execution conditions.

        Args:
            ohlcv: OHLCV data with columns: datetime, symbol, open, high, low, close, volume
            method: Limit order method.
                - "daytrade_open_limit": Fill if next-day open meets limit condition.
                - "daytrade_intraday_limit": Fill if next-day intraday price touches limit.
            position_side: "long" or "short".
            limit_offset_pct: Offset percentage for limit price (e.g., 1.0 = 1%).

        Returns:
            ExecutionResult with fill analysis.

        Raises:
            ValueError: If method or position_side is invalid, or limit_offset_pct < 0.

        Note:
            Non-executed rows have null for limit_return and entry_price.
            This is intentional for qal-compatible verification.
            Do NOT use this output for MixSeek evaluation.
        """
        if method not in self.VALID_METHODS:
            msg = f"Invalid method: {method}. Must be one of {sorted(self.VALID_METHODS)}"
            raise ValueError(msg)
        if position_side not in self.VALID_SIDES:
            msg = f"Invalid position_side: {position_side}. Must be one of {sorted(self.VALID_SIDES)}"
            raise ValueError(msg)
        if limit_offset_pct < 0:
            msg = f"limit_offset_pct must be >= 0, got {limit_offset_pct}"
            raise ValueError(msg)

        ohlcv_sorted = ohlcv.sort(["symbol", "datetime"])
        offset_ratio = limit_offset_pct / 100.0

        # Compute limit price: close[t] * (1 +/- offset)
        # Next-day OHLCV for fill checks
        enriched = ohlcv_sorted.with_columns(
            next_open=pl.col("open").shift(-1).over("symbol"),
            next_high=pl.col("high").shift(-1).over("symbol"),
            next_low=pl.col("low").shift(-1).over("symbol"),
            next_close=pl.col("close").shift(-1).over("symbol"),
        )

        if position_side == "long":
            # Long: buy limit at close[t] * (1 - offset)
            limit_price_expr = pl.col("close") * (1 - offset_ratio)
            enriched = enriched.with_columns(limit_price=limit_price_expr)

            if method == "daytrade_open_limit":
                # Fill if next_open <= limit_price
                is_exec = pl.col("next_open") <= pl.col("limit_price")
                entry = pl.col("next_open")
            else:  # daytrade_intraday_limit
                # Fill if next_low <= limit_price
                is_exec = pl.col("next_low") <= pl.col("limit_price")
                # Entry price: min(next_open, limit_price) — if gap down below limit, fill at open
                entry = pl.min_horizontal("next_open", "limit_price")
        else:
            # Short: sell limit at close[t] * (1 + offset)
            limit_price_expr = pl.col("close") * (1 + offset_ratio)
            enriched = enriched.with_columns(limit_price=limit_price_expr)

            if method == "daytrade_open_limit":
                # Fill if next_open >= limit_price
                is_exec = pl.col("next_open") >= pl.col("limit_price")
                entry = pl.col("next_open")
            else:  # daytrade_intraday_limit
                # Fill if next_high >= limit_price
                is_exec = pl.col("next_high") >= pl.col("limit_price")
                # Entry price: max(next_open, limit_price) — if gap up above limit, fill at open
                entry = pl.max_horizontal("next_open", "limit_price")

        # Set is_executed, then conditionally set entry_price only for executed rows
        enriched = enriched.with_columns(is_executed=is_exec)
        enriched = enriched.with_columns(
            entry_price=pl.when(pl.col("is_executed")).then(entry).otherwise(pl.lit(None, dtype=pl.Float64)),
        )

        # Boundary rows (last row per symbol): next_* is null → not executed
        enriched = enriched.with_columns(
            is_executed=pl.when(pl.col("next_open").is_null()).then(pl.lit(False)).otherwise(pl.col("is_executed")),
        )

        # Calculate limit_return for executed rows only
        # limit_return for long: (next_close - entry_price) / entry_price
        # limit_return for short: (entry_price - next_close) / entry_price
        if position_side == "long":
            return_expr = (pl.col("next_close") - pl.col("entry_price")) / pl.col("entry_price")
        else:
            return_expr = (pl.col("entry_price") - pl.col("next_close")) / pl.col("entry_price")

        enriched = enriched.with_columns(
            limit_return=pl.when(pl.col("is_executed")).then(return_expr).otherwise(pl.lit(None, dtype=pl.Float64)),
        )

        result_df = enriched.select(["datetime", "symbol", "is_executed", "entry_price", "limit_return"])

        # Calculate execution rate (excluding boundary rows)
        non_boundary = enriched.filter(pl.col("next_open").is_not_null())
        total = len(non_boundary)
        executed = non_boundary.filter(pl.col("is_executed")).height
        execution_rate = executed / total if total > 0 else 0.0

        return ExecutionResult(
            data=result_df,
            execution_rate=execution_rate,
            method=method,
            position_side=position_side,
            limit_offset_pct=limit_offset_pct,
        )

"""Signal-to-Strategy Adapter.

Converts MixSeek continuous signals to quant-alpha-lab discrete strategy format.

MixSeek outputs continuous float signals (e.g. -57.6 to +11.5).
quant-alpha-lab expects discrete signals {-1, 0, 1} with specific column naming:
  - CODE (str), DATE (datetime), FORWARD_RETURN (float), signal (int)

quant-alpha-lab internally computes: strategy_return = signal * FORWARD_RETURN
  - signal=1:  long position (profit when FORWARD_RETURN > 0)
  - signal=-1: short position (profit when FORWARD_RETURN < 0)
  - signal=0:  no position
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

import pandas as pd
import polars as pl

logger = logging.getLogger(__name__)


class ThresholdMethod(StrEnum):
    """Signal discretization method."""

    QUANTILE = "quantile"
    FIXED = "fixed"
    ZERO = "zero"


@dataclass(frozen=True)
class AdapterConfig:
    """Configuration for Signal-to-Strategy conversion.

    Attributes:
        threshold_method: Discretization method. QUANTILE is recommended
            as it handles skewed signal distributions (e.g. 95% negative).
        long_quantile: For QUANTILE method, signals above this percentile
            within each day become long (1). Default 0.9 = top 10%.
        short_quantile: For QUANTILE method, signals below this percentile
            within each day become short (-1). Default 0.1 = bottom 10%.
        fixed_long_threshold: For FIXED method, signals >= this value become long.
        fixed_short_threshold: For FIXED method, signals <= this value become short.
    """

    threshold_method: ThresholdMethod = ThresholdMethod.QUANTILE
    long_quantile: float = 0.9
    short_quantile: float = 0.1
    fixed_long_threshold: float = 0.0
    fixed_short_threshold: float = 0.0


class SignalToStrategyAdapter:
    """Converts MixSeek continuous signals to quant-alpha-lab discrete strategy format.

    The adapter performs two key transformations:
    1. Column renaming: datetime→DATE, symbol→CODE, return_value→FORWARD_RETURN
    2. Signal discretization: continuous float → {-1, 0, 1}

    Example:
        >>> adapter = SignalToStrategyAdapter()
        >>> strategy_df = adapter.convert(signal_df, returns_df, ohlcv_df)
        >>> strategy_func = adapter.make_strategy_func(signal_df)
        >>> # strategy_func can be passed to WFA/CPCV
    """

    def __init__(self, config: AdapterConfig | None = None) -> None:
        self.config = config or AdapterConfig()

    def convert(
        self,
        signal_df: pl.DataFrame,
        returns_df: pl.DataFrame,
        ohlcv_df: pl.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Convert MixSeek output to quant-alpha-lab input format.

        Args:
            signal_df: MixSeek signal output (datetime, symbol, signal).
            returns_df: Return data (datetime, symbol, return_value).
            ohlcv_df: OHLCV data for CLOSE column (optional).

        Returns:
            pd.DataFrame with columns:
                CODE (str), DATE (datetime64[ns]), FORWARD_RETURN (float64),
                signal (int in {-1, 0, 1}), raw_signal (float64).
                If ohlcv_df is provided, CLOSE (float64) is also included.
        """
        # Deduplicate input signals on (datetime, symbol) before discretization
        n_before = len(signal_df)
        signal_df_dedup = signal_df.unique(subset=["datetime", "symbol"], keep="first", maintain_order=True)
        if len(signal_df_dedup) < n_before:
            logger.warning(
                "signal_df had %d duplicate (datetime, symbol) rows (reduced %d → %d). Kept first occurrence.",
                n_before - len(signal_df_dedup),
                n_before,
                len(signal_df_dedup),
            )

        # Discretize signals
        discretized = self._discretize(signal_df_dedup)

        # Deduplicate join targets on (datetime, symbol) to prevent row multiplication
        returns_dedup = returns_df.select(["datetime", "symbol", "return_value"]).unique(
            subset=["datetime", "symbol"], keep="first", maintain_order=True
        )

        # Join with returns
        joined = discretized.join(returns_dedup, on=["datetime", "symbol"], how="inner")

        # Build output columns
        output_cols = [
            pl.col("symbol").alias("CODE"),
            pl.col("datetime").alias("DATE"),
            pl.col("return_value").alias("FORWARD_RETURN"),
            pl.col("discrete_signal").alias("signal"),
            pl.col("signal").alias("raw_signal"),
        ]

        # Add CLOSE if ohlcv provided
        if ohlcv_df is not None:
            ohlcv_dedup = ohlcv_df.select(["datetime", "symbol", "close"]).unique(
                subset=["datetime", "symbol"], keep="first"
            )
            joined = joined.join(ohlcv_dedup, on=["datetime", "symbol"], how="left")
            output_cols.insert(2, pl.col("close").alias("CLOSE"))

        result_pl = joined.sort(["datetime", "symbol"]).select(output_cols)

        # Convert to pandas
        result_pd = result_pl.to_pandas()
        result_pd["DATE"] = pd.to_datetime(result_pd["DATE"])
        result_pd["signal"] = result_pd["signal"].astype(int)

        return result_pd.reset_index(drop=True)

    def make_strategy_func(
        self,
        signal_df: pl.DataFrame,
    ) -> Callable[[pd.DataFrame], pd.DataFrame]:
        """Generate a strategy_func closure for WFA/CPCV.

        quant-alpha-lab's WFA and CPCV expect:
            strategy_func(data: pd.DataFrame) -> pd.DataFrame
            Input:  CODE, DATE, FORWARD_RETURN, ...
            Output: CODE, DATE, signal (int in {-1, 0, 1})

        This method pre-computes discrete signals for all dates,
        then returns a closure that looks up signals by (CODE, DATE).

        Args:
            signal_df: Full-period signal data (datetime, symbol, signal).

        Returns:
            A callable matching the WFA/CPCV strategy_func contract.
        """
        # Pre-compute all discrete signals
        discretized = self._discretize(signal_df)
        signal_lookup = discretized.select(
            [
                pl.col("datetime").alias("DATE"),
                pl.col("symbol").alias("CODE"),
                pl.col("discrete_signal"),
            ]
        ).to_pandas()
        signal_lookup["DATE"] = pd.to_datetime(signal_lookup["DATE"])

        # Deduplicate (CODE, DATE) pairs — warn and keep first occurrence
        dupes = signal_lookup.duplicated(subset=["CODE", "DATE"], keep="first")
        if dupes.any():
            n_dupes = int(dupes.sum())
            n_before = len(signal_lookup)
            signal_lookup = signal_lookup[~dupes].reset_index(drop=True)
            logger.warning(
                "signal_lookup had %d duplicate (CODE, DATE) rows (reduced %d → %d). Kept first occurrence.",
                n_dupes,
                n_before,
                len(signal_lookup),
            )

        def strategy_func(data: pd.DataFrame) -> pd.DataFrame:
            merged = data[["CODE", "DATE"]].merge(
                signal_lookup,
                on=["CODE", "DATE"],
                how="left",
            )
            merged["signal"] = merged["discrete_signal"].fillna(0).astype(int)
            return merged[["CODE", "DATE", "signal"]]

        return strategy_func

    def _discretize(self, signal_df: pl.DataFrame) -> pl.DataFrame:
        """Dispatch to the appropriate discretization method.

        Args:
            signal_df: DataFrame with datetime, symbol, signal columns.

        Returns:
            DataFrame with added discrete_signal column (Int32).
        """
        method = self.config.threshold_method
        if method == ThresholdMethod.QUANTILE:
            return self._discretize_quantile(signal_df)
        elif method == ThresholdMethod.FIXED:
            return self._discretize_fixed(signal_df)
        elif method == ThresholdMethod.ZERO:
            return self._discretize_zero(signal_df)
        else:
            msg = f"Unknown threshold method: {method}"
            raise ValueError(msg)

    def _discretize_quantile(self, signal_df: pl.DataFrame) -> pl.DataFrame:
        """Discretize using daily quantile thresholds.

        For each date, compute the rank percentile of each signal.
        Signals above long_quantile become 1 (long).
        Signals below short_quantile become -1 (short).
        Others become 0 (no position).

        This is robust to skewed distributions (e.g. Focused strategy
        where 95% of signals are negative).
        """
        long_q = self.config.long_quantile
        short_q = self.config.short_quantile

        return (
            signal_df.with_columns(
                (
                    pl.col("signal")
                    .rank(method="ordinal")
                    .over("datetime")
                    .truediv(pl.col("signal").count().over("datetime"))
                ).alias("_pctrank"),
            )
            .with_columns(
                pl.when(pl.col("_pctrank") > long_q)
                .then(pl.lit(1))
                .when(pl.col("_pctrank") <= short_q)
                .then(pl.lit(-1))
                .otherwise(pl.lit(0))
                .cast(pl.Int32)
                .alias("discrete_signal"),
            )
            .drop("_pctrank")
        )

    def _discretize_fixed(self, signal_df: pl.DataFrame) -> pl.DataFrame:
        """Discretize using fixed threshold values."""
        long_t = self.config.fixed_long_threshold
        short_t = self.config.fixed_short_threshold

        return signal_df.with_columns(
            pl.when(pl.col("signal") >= long_t)
            .then(pl.lit(1))
            .when(pl.col("signal") <= short_t)
            .then(pl.lit(-1))
            .otherwise(pl.lit(0))
            .cast(pl.Int32)
            .alias("discrete_signal"),
        )

    def _discretize_zero(self, signal_df: pl.DataFrame) -> pl.DataFrame:
        """Discretize using zero threshold. signal >= 0 → 1, signal < 0 → -1."""
        return signal_df.with_columns(
            pl.when(pl.col("signal") >= 0)
            .then(pl.lit(1))
            .when(pl.col("signal") < 0)
            .then(pl.lit(-1))
            .otherwise(pl.lit(0))
            .cast(pl.Int32)
            .alias("discrete_signal"),
        )

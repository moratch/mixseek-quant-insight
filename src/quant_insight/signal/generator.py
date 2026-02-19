"""Signal generator for the super-ensemble portfolio.

Generates daily position signals by running all validated strategies,
applying quantile thresholds, and weighting by ensemble composition.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl

from quant_insight.evaluator.submission_parser import parse_submission_function

logger = logging.getLogger(__name__)

# Default super-ensemble weights (from super_ensemble_20260219.json option_a_5strat)
DEFAULT_WEIGHTS: dict[str, float] = {
    "Creative": 0.239089285070701,
    "RETURN_2D": 0.07624872818839572,
    "STOCH_K_5": 0.05,
    "I5_cp_gap3d": 0.23466198674089753,
    "C2C_cp_vol": 0.4,
}


@dataclass
class StrategySpec:
    """Specification for a single strategy in the ensemble."""

    name: str
    code: str
    weight: float
    long_quantile: float = 0.9
    short_quantile: float = 0.1


@dataclass
class SignalOutput:
    """Output of signal generation for one date."""

    date: str
    n_long: int
    n_short: int
    n_neutral: int
    n_total: int
    positions: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Strategy code definitions
# ---------------------------------------------------------------------------

CREATIVE_CODE = """```python
import polars as pl
import pandas as pd

def generate_signal(ohlcv, additional_data):
    ind = additional_data["indicators"].to_pandas()
    s = ind[["datetime","symbol","INTRADAY_VOL"]].copy()
    s = s.dropna(subset=["INTRADAY_VOL"])
    s["signal"] = -s.groupby("datetime")["INTRADAY_VOL"].rank(pct=True)
    return s[["datetime","symbol","signal"]]
```"""

RETURN_2D_CODE = """```python
import polars as pl
import pandas as pd

def generate_signal(ohlcv, additional_data):
    ind = additional_data["indicators"].to_pandas()
    s = ind[["datetime","symbol","RETURN_2D"]].copy()
    s = s.dropna(subset=["RETURN_2D"])
    s["signal"] = -s.groupby("datetime")["RETURN_2D"].rank(pct=True)
    return s[["datetime","symbol","signal"]]
```"""

STOCH_K_5_CODE = """```python
import polars as pl
import pandas as pd

def generate_signal(ohlcv, additional_data):
    ind = additional_data["indicators"].to_pandas()
    s = ind[["datetime","symbol","STOCH_K_5"]].copy()
    s = s.dropna(subset=["STOCH_K_5"])
    s["signal"] = -s.groupby("datetime")["STOCH_K_5"].rank(pct=True)
    return s[["datetime","symbol","signal"]]
```"""

I5_CODE = """```python
import polars as pl
def generate_signal(ohlcv, additional_data):
    df = ohlcv.sort(["symbol", "datetime"])
    cp = pl.when(pl.col("high") == pl.col("low")).then(0.5).otherwise(
        (pl.col("close") - pl.col("low")) / (pl.col("high") - pl.col("low"))
    )
    df = df.with_columns((-cp).fill_null(0.0).alias("raw_cp"))
    prev_close = pl.col("close").shift(1).over("symbol")
    overnight = (pl.col("open") - prev_close) / (prev_close + 1e-10)
    df = df.with_columns(overnight.fill_null(0.0).alias("gap"))
    df = df.with_columns(
        (-pl.col("gap").rolling_mean(3).over("symbol")).fill_null(0.0).alias("raw_gap3")
    )
    df = df.with_columns(
        ((pl.col("raw_cp").rank().over("datetime") - 1) /
         (pl.count().over("datetime") - 1)).fill_null(0.5).alias("n_cp"),
        ((pl.col("raw_gap3").rank().over("datetime") - 1) /
         (pl.count().over("datetime") - 1)).fill_null(0.5).alias("n_gap3"),
    )
    df = df.with_columns((0.36 * pl.col("n_cp") + 0.64 * pl.col("n_gap3")).alias("signal"))
    return df.select(["datetime", "symbol", "signal"])
```"""

C2C_CODE = """```python
import polars as pl
def generate_signal(ohlcv, additional_data):
    df = ohlcv.sort(["symbol", "datetime"])
    cp = pl.when(pl.col("high") == pl.col("low")).then(0.5).otherwise(
        (pl.col("close") - pl.col("low")) / (pl.col("high") - pl.col("low"))
    )
    df = df.with_columns((-cp).fill_null(0.0).alias("raw_cp"))
    vol_ma5 = pl.col("volume").rolling_mean(5).over("symbol")
    vol_ratio = pl.col("volume") / (vol_ma5 + 1e-10)
    df = df.with_columns((-vol_ratio).fill_null(0.0).alias("raw_vol"))
    df = df.with_columns(
        ((pl.col("raw_cp").rank().over("datetime") - 1) /
         (pl.count().over("datetime") - 1)).fill_null(0.5).alias("n_cp"),
        ((pl.col("raw_vol").rank().over("datetime") - 1) /
         (pl.count().over("datetime") - 1)).fill_null(0.5).alias("n_vol"),
    )
    df = df.with_columns((0.65 * pl.col("n_cp") + 0.35 * pl.col("n_vol")).alias("signal"))
    return df.select(["datetime", "symbol", "signal"])
```"""

DEFAULT_STRATEGIES: list[StrategySpec] = [
    StrategySpec("Creative", CREATIVE_CODE, DEFAULT_WEIGHTS["Creative"], 0.9, 0.1),
    StrategySpec("RETURN_2D", RETURN_2D_CODE, DEFAULT_WEIGHTS["RETURN_2D"], 0.9, 0.1),
    StrategySpec("STOCH_K_5", STOCH_K_5_CODE, DEFAULT_WEIGHTS["STOCH_K_5"], 0.9, 0.1),
    StrategySpec("I5_cp_gap3d", I5_CODE, DEFAULT_WEIGHTS["I5_cp_gap3d"], 0.85, 0.15),
    StrategySpec("C2C_cp_vol", C2C_CODE, DEFAULT_WEIGHTS["C2C_cp_vol"], 0.85, 0.15),
]


class EnsembleSignalGenerator:
    """Generates ensemble portfolio signals from validated strategies.

    Usage:
        gen = EnsembleSignalGenerator(workspace=Path("workspace"))
        gen.load_data()
        output = gen.generate(date="2025-12-30")
    """

    def __init__(
        self,
        workspace: Path,
        strategies: list[StrategySpec] | None = None,
    ) -> None:
        self.workspace = workspace
        self.strategies = strategies or DEFAULT_STRATEGIES
        self._ohlcv: pl.DataFrame | None = None
        self._additional_data: dict[str, pl.DataFrame] | None = None

    def load_data(self) -> None:
        """Load OHLCV and additional data from workspace."""
        raw_dir = self.workspace / "data" / "inputs" / "raw"

        logger.info("Loading OHLCV data...")
        self._ohlcv = pl.read_parquet(raw_dir / "ohlcv.parquet")

        self._additional_data = {}
        master_path = raw_dir / "master.parquet"
        if master_path.exists():
            self._additional_data["master"] = pl.read_parquet(master_path)
        indicators_path = raw_dir / "indicators.parquet"
        if indicators_path.exists():
            self._additional_data["indicators"] = pl.read_parquet(indicators_path)

        logger.info(
            "Data loaded: OHLCV=%d rows, indicators=%s",
            len(self._ohlcv),
            "loaded" if "indicators" in self._additional_data else "not found",
        )

    def _generate_strategy_signals(self, spec: StrategySpec) -> pl.DataFrame:
        """Generate signals for one strategy across all dates.

        Returns:
            DataFrame with columns: datetime, symbol, signal, position
        """
        if self._ohlcv is None or self._additional_data is None:
            msg = "Data not loaded. Call load_data() first."
            raise RuntimeError(msg)

        func = parse_submission_function(spec.code)
        raw_result = func(self._ohlcv, self._additional_data)
        sig_df: pl.DataFrame = raw_result if isinstance(raw_result, pl.DataFrame) else pl.from_pandas(raw_result)

        sig_df = sig_df.unique(subset=["datetime", "symbol"], keep="first")

        # Apply quantile threshold to get positions per date
        sig_df = sig_df.with_columns(
            pl.col("signal")
            .rank(descending=False)
            .over("datetime")
            .truediv(pl.col("signal").count().over("datetime"))
            .alias("pct_rank")
        )

        sig_df = sig_df.with_columns(
            pl.when(pl.col("pct_rank") > spec.long_quantile)
            .then(1)
            .when(pl.col("pct_rank") < spec.short_quantile)
            .then(-1)
            .otherwise(0)
            .alias("position")
        )

        return sig_df.select(["datetime", "symbol", "signal", "position"])

    def generate(
        self,
        date: str | None = None,
        top_n: int = 50,
    ) -> list[SignalOutput]:
        """Generate ensemble signals.

        Args:
            date: Specific date (YYYY-MM-DD) or None for all dates.
            top_n: Number of top positions to include in output.

        Returns:
            List of SignalOutput per date.
        """
        if self._ohlcv is None:
            self.load_data()

        # Generate signals for all strategies
        all_positions: dict[str, pl.DataFrame] = {}
        for spec in self.strategies:
            logger.info("Generating signals for %s (weight=%.1f%%)...", spec.name, spec.weight * 100)
            all_positions[spec.name] = self._generate_strategy_signals(spec)

        # Combine: weighted sum of positions
        # Start with first strategy
        first_name = self.strategies[0].name
        combined = (
            all_positions[first_name]
            .select(["datetime", "symbol"])
            .with_columns(
                (pl.col("symbol").cast(pl.Utf8)).alias("symbol")  # ensure string type
            )
        )

        # Re-select to get unique datetime/symbol pairs across all strategies
        all_pairs: list[pl.DataFrame] = []
        for name, df in all_positions.items():
            all_pairs.append(df.select(["datetime", "symbol"]))
        combined = pl.concat(all_pairs).unique(subset=["datetime", "symbol"])

        # Join each strategy's position and compute weighted sum
        combined = combined.with_columns(pl.lit(0.0).alias("ensemble_signal"))
        for spec in self.strategies:
            strat_df = all_positions[spec.name].select(
                ["datetime", "symbol", pl.col("position").alias(f"pos_{spec.name}")]
            )
            combined = combined.join(strat_df, on=["datetime", "symbol"], how="left")
            combined = combined.with_columns(
                (pl.col("ensemble_signal") + pl.col(f"pos_{spec.name}").fill_null(0) * spec.weight).alias(
                    "ensemble_signal"
                )
            )
            combined = combined.drop(f"pos_{spec.name}")

        # Filter by date if specified
        if date is not None:
            combined = combined.filter(pl.col("datetime").cast(pl.Utf8).str.starts_with(date))

        if len(combined) == 0:
            logger.warning("No data found for date=%s", date)
            return []

        # Group by date and produce output
        dates = combined.select("datetime").unique().sort("datetime")
        outputs: list[SignalOutput] = []

        for row in dates.iter_rows():
            dt_val = row[0]
            day_df = combined.filter(pl.col("datetime") == dt_val).sort("ensemble_signal", descending=True)

            n_long = day_df.filter(pl.col("ensemble_signal") > 0).height
            n_short = day_df.filter(pl.col("ensemble_signal") < 0).height
            n_neutral = day_df.filter(pl.col("ensemble_signal") == 0).height

            # Top N long + bottom N short
            top_long = day_df.head(top_n)
            top_short = day_df.tail(top_n)
            positions_df = pl.concat([top_long, top_short]).unique(subset=["symbol"])

            positions = [
                {
                    "symbol": r[1],
                    "ensemble_signal": round(float(r[2]), 4),
                }
                for r in positions_df.iter_rows()
                if r[2] != 0  # skip neutral
            ]

            outputs.append(
                SignalOutput(
                    date=str(dt_val),
                    n_long=n_long,
                    n_short=n_short,
                    n_neutral=n_neutral,
                    n_total=day_df.height,
                    positions=positions,
                )
            )

        return outputs


def load_weights_from_json(json_path: Path) -> dict[str, float]:
    """Load ensemble weights from super-ensemble JSON.

    Args:
        json_path: Path to super_ensemble JSON file.

    Returns:
        Strategy name to weight mapping.
    """
    data = json.loads(json_path.read_text(encoding="utf-8"))
    best_key = data.get("best", "option_a_5strat")
    key_map = {
        "5_strat_super": "option_a_5strat",
        "3_strat_indicator": "option_b_3strat_indicator",
        "3_group": "option_c_3group",
    }
    option_key = key_map.get(best_key, best_key)
    return data[option_key]["weights"]  # type: ignore[no-any-return]

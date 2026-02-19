"""Production signal pipeline for validated strategies.

Generates actionable trading signals from statistically validated strategies.
Currently supports I5_cp_gap3d (the only strategy passing WFA/CPCV at all horizons).

Usage:
    from quant_insight.pipeline.production import ProductionPipeline, ProductionConfig

    config = ProductionConfig(workspace=Path("workspace"))
    pipeline = ProductionPipeline(config)
    result = pipeline.run()  # latest date
    result = pipeline.run(date="2025-12-26")  # specific date
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime as dt
from pathlib import Path
from typing import Any

import polars as pl

from quant_insight.evaluator.submission_parser import parse_submission_function
from quant_insight.signal.generator import I5_CODE

logger = logging.getLogger(__name__)

# I5_HL5 composite: I5 (80%) + HL5_CLOSE_TO_LOW_DEV (20%)
# Validated: WFE=0.508, PBO=0.389, DSR=3.046
I5_HL5_CODE = """```python
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
    i5_signal = 0.36 * pl.col("n_cp") + 0.64 * pl.col("n_gap3")
    df = df.with_columns(i5_signal.alias("i5_raw"))

    ind = additional_data["indicators"]
    hl5 = ind.select(["datetime", "symbol", "HL5_CLOSE_TO_LOW_DEV"]).drop_nulls()
    hl5 = hl5.with_columns(
        ((-pl.col("HL5_CLOSE_TO_LOW_DEV")).rank().over("datetime") /
         pl.count().over("datetime")).fill_null(0.5).alias("n_hl5")
    )

    merged = df.select(["datetime", "symbol", "i5_raw"]).join(
        hl5.select(["datetime", "symbol", "n_hl5"]),
        on=["datetime", "symbol"],
        how="inner"
    )
    merged = merged.with_columns(
        (0.80 * pl.col("i5_raw") + 0.20 * pl.col("n_hl5")).alias("signal")
    )
    return merged.select(["datetime", "symbol", "signal"])
```"""

# Strategy registry: validated strategies with their screening results
VALIDATED_STRATEGIES: dict[str, str] = {
    "I5_cp_gap3d": I5_CODE,
    "I5_HL5_w80": I5_HL5_CODE,
}


@dataclass
class StrategyConfig:
    """Configuration for a single strategy."""

    name: str
    code: str
    long_quantile: float = 0.85
    short_quantile: float = 0.15
    weight: float = 1.0


@dataclass
class Position:
    """A single stock position."""

    symbol: str
    direction: str  # "long" | "short"
    signal: float
    pct_rank: float


@dataclass
class DailySignal:
    """Signal output for one trading day."""

    date: str
    strategy: str
    n_long: int
    n_short: int
    n_universe: int
    long_positions: list[Position]
    short_positions: list[Position]


@dataclass
class ProductionResult:
    """Result of a production pipeline run."""

    signals: list[DailySignal]
    config: dict[str, Any]
    generated_at: str
    data_range: str


@dataclass
class LiquidityStats:
    """Liquidity filter statistics."""

    total_symbols: int
    filtered_symbols: int
    excluded_symbols: int
    min_turnover_yen: float
    lookback_days: int


@dataclass
class ProductionConfig:
    """Configuration for the production pipeline."""

    workspace: Path
    strategy_name: str = "I5_cp_gap3d"
    long_quantile: float = 0.85
    short_quantile: float = 0.15
    top_n: int = 50
    holding_period_days: int = 10
    min_avg_turnover_yen: float = 5e7  # 5,000万円 minimum average daily turnover
    liquidity_lookback_days: int = 60  # lookback period for avg turnover calculation
    additional_data_names: tuple[str, ...] = ("master", "indicators", "fundamental", "margin")


class ProductionPipeline:
    """Production signal generation pipeline.

    Generates actionable long/short positions from validated strategies.
    Designed for daily production use with the 10-day swing approach.
    """

    def __init__(self, config: ProductionConfig) -> None:
        self.config = config
        self._ohlcv: pl.DataFrame | None = None
        self._additional_data: dict[str, pl.DataFrame] = {}
        self._liquidity_stats: LiquidityStats | None = None

    def _load_data(self) -> None:
        """Load OHLCV and additional data from workspace."""
        raw_dir = self.config.workspace / "data" / "inputs" / "raw"

        logger.info("Loading OHLCV data...")
        self._ohlcv = pl.read_parquet(raw_dir / "ohlcv.parquet")

        self._additional_data = {}
        for name in self.config.additional_data_names:
            path = raw_dir / f"{name}.parquet"
            if path.exists():
                self._additional_data[name] = pl.read_parquet(path)
                logger.info("Loaded %s: %d rows", name, self._additional_data[name].height)
            else:
                logger.warning("Optional data '%s' not found at %s", name, path)

        logger.info(
            "Data loaded: OHLCV=%d rows, additional=%s",
            self._ohlcv.height,
            list(self._additional_data.keys()),
        )

        # Apply liquidity filter
        if self.config.min_avg_turnover_yen > 0:
            self._apply_liquidity_filter()

    def _apply_liquidity_filter(self) -> None:
        """Filter out low-liquidity symbols based on average daily turnover."""
        assert self._ohlcv is not None

        # Use recent data for turnover calculation
        max_date = self._ohlcv["datetime"].max()
        lookback_start = max_date - pl.duration(days=self.config.liquidity_lookback_days * 2)
        recent = self._ohlcv.filter(pl.col("datetime") >= lookback_start)

        # Calculate average daily turnover (volume × close price) per symbol
        turnover = (
            recent.with_columns((pl.col("volume") * pl.col("close")).alias("turnover_yen"))
            .group_by("symbol")
            .agg(pl.col("turnover_yen").mean().alias("avg_turnover"))
        )

        total_symbols = turnover.height
        liquid_symbols = turnover.filter(pl.col("avg_turnover") >= self.config.min_avg_turnover_yen)["symbol"]
        liquid_set = set(liquid_symbols.to_list())

        # Filter OHLCV
        before = self._ohlcv.height
        self._ohlcv = self._ohlcv.filter(pl.col("symbol").is_in(liquid_symbols))

        # Filter additional data
        for name, df in self._additional_data.items():
            if "symbol" in df.columns:
                self._additional_data[name] = df.filter(pl.col("symbol").is_in(liquid_symbols))

        self._liquidity_stats = LiquidityStats(
            total_symbols=total_symbols,
            filtered_symbols=len(liquid_set),
            excluded_symbols=total_symbols - len(liquid_set),
            min_turnover_yen=self.config.min_avg_turnover_yen,
            lookback_days=self.config.liquidity_lookback_days,
        )

        logger.info(
            "Liquidity filter: %d/%d symbols pass (min avg turnover %.0f JPY, OHLCV %d -> %d rows)",
            len(liquid_set),
            total_symbols,
            self.config.min_avg_turnover_yen,
            before,
            self._ohlcv.height,
        )

    def _get_strategy_code(self) -> str:
        """Get strategy code from registry or config."""
        code = VALIDATED_STRATEGIES.get(self.config.strategy_name)
        if code is None:
            msg = (
                f"Strategy '{self.config.strategy_name}' not found in validated registry. "
                f"Available: {list(VALIDATED_STRATEGIES.keys())}"
            )
            raise ValueError(msg)
        return code

    def _generate_signals(self, code: str) -> pl.DataFrame:
        """Generate raw signals from strategy code.

        Returns:
            DataFrame with columns: datetime, symbol, signal, pct_rank, position
        """
        assert self._ohlcv is not None

        func = parse_submission_function(code)
        raw_result = func(self._ohlcv, self._additional_data)
        sig_df: pl.DataFrame = raw_result if isinstance(raw_result, pl.DataFrame) else pl.from_pandas(raw_result)
        sig_df = sig_df.unique(subset=["datetime", "symbol"], keep="first")

        # Compute percentile rank per date
        sig_df = sig_df.with_columns(
            pl.col("signal")
            .rank(descending=False)
            .over("datetime")
            .truediv(pl.col("signal").count().over("datetime"))
            .alias("pct_rank")
        )

        # Apply quantile thresholds
        sig_df = sig_df.with_columns(
            pl.when(pl.col("pct_rank") > self.config.long_quantile)
            .then(pl.lit("long"))
            .when(pl.col("pct_rank") < self.config.short_quantile)
            .then(pl.lit("short"))
            .otherwise(pl.lit("neutral"))
            .alias("direction")
        )

        return sig_df

    def run(self, date: str | None = None) -> ProductionResult:
        """Run the production pipeline.

        Args:
            date: Target date (YYYY-MM-DD) or None for latest available.

        Returns:
            ProductionResult with actionable positions.
        """
        if self._ohlcv is None:
            self._load_data()

        code = self._get_strategy_code()
        logger.info("Generating %s signals...", self.config.strategy_name)

        sig_df = self._generate_signals(code)

        # Filter to target date
        if date is not None:
            sig_df = sig_df.filter(pl.col("datetime").cast(pl.Utf8).str.starts_with(date))
        else:
            # Latest available date
            latest = sig_df.select("datetime").max().item()
            sig_df = sig_df.filter(pl.col("datetime") == latest)

        if sig_df.height == 0:
            logger.warning("No data found for date=%s", date)
            return ProductionResult(
                signals=[],
                config={"strategy": self.config.strategy_name, "date": date},
                generated_at=dt.now().isoformat(),
                data_range="N/A",
            )

        # Build DailySignal for each date
        dates = sig_df.select("datetime").unique().sort("datetime")
        signals: list[DailySignal] = []

        for row in dates.iter_rows():
            dt_val = row[0]
            day_df = sig_df.filter(pl.col("datetime") == dt_val)

            long_df = day_df.filter(pl.col("direction") == "long").sort("signal", descending=True)
            short_df = day_df.filter(pl.col("direction") == "short").sort("signal", descending=False)

            long_positions = [
                Position(
                    symbol=str(r[1]),
                    direction="long",
                    signal=round(float(r[2]), 6),
                    pct_rank=round(float(r[3]), 4),
                )
                for r in long_df.head(self.config.top_n)
                .select(["datetime", "symbol", "signal", "pct_rank"])
                .iter_rows()
            ]
            short_positions = [
                Position(
                    symbol=str(r[1]),
                    direction="short",
                    signal=round(float(r[2]), 6),
                    pct_rank=round(float(r[3]), 4),
                )
                for r in short_df.head(self.config.top_n)
                .select(["datetime", "symbol", "signal", "pct_rank"])
                .iter_rows()
            ]

            signals.append(
                DailySignal(
                    date=str(dt_val),
                    strategy=self.config.strategy_name,
                    n_long=long_df.height,
                    n_short=short_df.height,
                    n_universe=day_df.height,
                    long_positions=long_positions,
                    short_positions=short_positions,
                )
            )

        # Data range info
        assert self._ohlcv is not None
        data_min = self._ohlcv["datetime"].min()
        data_max = self._ohlcv["datetime"].max()

        config_dict: dict[str, Any] = {
            "strategy": self.config.strategy_name,
            "long_quantile": self.config.long_quantile,
            "short_quantile": self.config.short_quantile,
            "top_n": self.config.top_n,
            "holding_period_days": self.config.holding_period_days,
            "min_avg_turnover_yen": self.config.min_avg_turnover_yen,
        }
        if self._liquidity_stats is not None:
            config_dict["liquidity"] = asdict(self._liquidity_stats)

        return ProductionResult(
            signals=signals,
            config=config_dict,
            generated_at=dt.now().isoformat(),
            data_range=f"{data_min!s} to {data_max!s}",
        )

    def save_result(self, result: ProductionResult, output_dir: Path | None = None) -> Path:
        """Save production result to JSON.

        Args:
            result: Production result to save.
            output_dir: Output directory (default: workspace/reports/signals/).

        Returns:
            Path to saved JSON file.
        """
        if output_dir is None:
            output_dir = self.config.workspace / "reports" / "signals"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Filename: strategy_date.json
        if result.signals:
            date_str = result.signals[0].date[:10].replace("-", "")
        else:
            date_str = dt.now().strftime("%Y%m%d")

        filename = f"{self.config.strategy_name}_{date_str}.json"
        out_path = output_dir / filename

        # Convert to serializable dict
        data: dict[str, Any] = {
            "generated_at": result.generated_at,
            "data_range": result.data_range,
            "config": result.config,
            "signals": [_daily_signal_to_dict(s) for s in result.signals],
        }

        out_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        return out_path


def _daily_signal_to_dict(signal: DailySignal) -> dict[str, Any]:
    """Convert DailySignal to serializable dict."""
    return {
        "date": signal.date,
        "strategy": signal.strategy,
        "n_long": signal.n_long,
        "n_short": signal.n_short,
        "n_universe": signal.n_universe,
        "long_positions": [asdict(p) for p in signal.long_positions],
        "short_positions": [asdict(p) for p in signal.short_positions],
    }

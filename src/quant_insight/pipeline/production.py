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

# Strategy registry: validated strategies with their screening results
VALIDATED_STRATEGIES: dict[str, str] = {
    "I5_cp_gap3d": I5_CODE,
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
class ProductionConfig:
    """Configuration for the production pipeline."""

    workspace: Path
    strategy_name: str = "I5_cp_gap3d"
    long_quantile: float = 0.85
    short_quantile: float = 0.15
    top_n: int = 50
    holding_period_days: int = 10
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

        return ProductionResult(
            signals=signals,
            config={
                "strategy": self.config.strategy_name,
                "long_quantile": self.config.long_quantile,
                "short_quantile": self.config.short_quantile,
                "top_n": self.config.top_n,
                "holding_period_days": self.config.holding_period_days,
            },
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

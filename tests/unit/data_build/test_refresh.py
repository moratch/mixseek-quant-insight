"""Tests for incremental data refresh logic."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import polars as pl
import pytest


@pytest.fixture()
def raw_dir(tmp_path: Path) -> Path:
    """Create a raw data directory with existing OHLCV data."""
    d = tmp_path / "data" / "inputs" / "raw"
    d.mkdir(parents=True)

    dates = ["2025-01-01", "2025-01-02", "2025-01-03"]
    symbols = ["SYM0001", "SYM0002", "SYM0003"]
    rows = []
    for dt_str in dates:
        for s in symbols:
            rows.append({
                "datetime": dt_str,
                "symbol": s,
                "open": 100.0,
                "high": 105.0,
                "low": 95.0,
                "close": 102.0,
                "volume": 1000.0,
            })

    ohlcv = pl.DataFrame(rows).with_columns(
        pl.col("datetime").str.strptime(pl.Datetime, "%Y-%m-%d")
    )
    ohlcv.write_parquet(d / "ohlcv.parquet")

    # Master data
    master_rows = []
    for dt_str in dates:
        for s in symbols:
            master_rows.append({
                "datetime": dt_str,
                "symbol": s,
                "market_code": "0111",
            })
    master = pl.DataFrame(master_rows).with_columns(
        pl.col("datetime").str.strptime(pl.Datetime, "%Y-%m-%d")
    )
    master.write_parquet(d / "master.parquet")
    return d


class TestDetectMaxDate:
    """Test max date detection from existing parquet."""

    def test_detect_max_date(self, raw_dir: Path) -> None:
        ohlcv = pl.read_parquet(raw_dir / "ohlcv.parquet")
        max_dt = ohlcv["datetime"].max()
        assert max_dt is not None
        assert isinstance(max_dt, datetime)
        assert max_dt.date() == date(2025, 1, 3)

    def test_empty_ohlcv(self, raw_dir: Path) -> None:
        # Write empty parquet
        empty = pl.DataFrame({
            "datetime": pl.Series([], dtype=pl.Datetime),
            "symbol": pl.Series([], dtype=pl.Utf8),
            "open": pl.Series([], dtype=pl.Float64),
            "high": pl.Series([], dtype=pl.Float64),
            "low": pl.Series([], dtype=pl.Float64),
            "close": pl.Series([], dtype=pl.Float64),
            "volume": pl.Series([], dtype=pl.Float64),
        })
        empty.write_parquet(raw_dir / "ohlcv.parquet")
        ohlcv = pl.read_parquet(raw_dir / "ohlcv.parquet")
        assert ohlcv["datetime"].max() is None


class TestMergeDedup:
    """Test merge and deduplication logic for incremental refresh."""

    def test_append_new_dates(self, raw_dir: Path) -> None:
        existing = pl.read_parquet(raw_dir / "ohlcv.parquet")
        assert existing.height == 9  # 3 dates × 3 symbols

        # Simulate new data (2 new dates)
        new_rows = []
        for dt_str in ["2025-01-04", "2025-01-06"]:
            for s in ["SYM0001", "SYM0002", "SYM0003"]:
                new_rows.append({
                    "datetime": dt_str,
                    "symbol": s,
                    "open": 101.0,
                    "high": 106.0,
                    "low": 96.0,
                    "close": 103.0,
                    "volume": 1100.0,
                })
        new_df = pl.DataFrame(new_rows).with_columns(
            pl.col("datetime").str.strptime(pl.Datetime, "%Y-%m-%d")
        )

        # Merge
        merged = pl.concat([existing, new_df], how="diagonal_relaxed")
        merged = merged.unique(subset=["datetime", "symbol"], keep="last").sort(["datetime", "symbol"])
        assert merged.height == 15  # 5 dates × 3 symbols

    def test_dedup_overlapping_dates(self, raw_dir: Path) -> None:
        existing = pl.read_parquet(raw_dir / "ohlcv.parquet")

        # Simulate overlapping data (Jan 3 repeated + Jan 4 new)
        new_rows = []
        for dt_str in ["2025-01-03", "2025-01-04"]:
            for s in ["SYM0001", "SYM0002", "SYM0003"]:
                new_rows.append({
                    "datetime": dt_str,
                    "symbol": s,
                    "open": 200.0,  # Different values
                    "high": 210.0,
                    "low": 190.0,
                    "close": 205.0,
                    "volume": 2000.0,
                })
        new_df = pl.DataFrame(new_rows).with_columns(
            pl.col("datetime").str.strptime(pl.Datetime, "%Y-%m-%d")
        )

        merged = pl.concat([existing, new_df], how="diagonal_relaxed")
        merged = merged.unique(subset=["datetime", "symbol"], keep="last").sort(["datetime", "symbol"])

        # Should have 4 dates × 3 symbols = 12 (Jan 3 deduped, Jan 4 added)
        assert merged.height == 12

        # Jan 3 data should be from new_df (keep="last")
        jan3 = merged.filter(
            pl.col("datetime") == pl.lit("2025-01-03").str.strptime(pl.Datetime, "%Y-%m-%d")
        )
        assert jan3["close"][0] == 205.0  # new value

    def test_merge_without_symbol_column(self, raw_dir: Path) -> None:
        """Test merge for data without symbol column (e.g., short_ratio)."""
        existing = pl.DataFrame({
            "datetime": ["2025-01-01", "2025-01-02"],
            "sector33_code": ["A", "A"],
            "value": [1.0, 2.0],
        }).with_columns(pl.col("datetime").str.strptime(pl.Datetime, "%Y-%m-%d"))

        new_df = pl.DataFrame({
            "datetime": ["2025-01-03"],
            "sector33_code": ["A"],
            "value": [3.0],
        }).with_columns(pl.col("datetime").str.strptime(pl.Datetime, "%Y-%m-%d"))

        # When symbol not in columns, use only datetime as key
        key_cols = ["datetime", "symbol"] if "symbol" in new_df.columns else ["datetime"]
        available_keys = [c for c in key_cols if c in existing.columns and c in new_df.columns]
        merged = pl.concat([existing, new_df], how="diagonal_relaxed")
        merged = merged.unique(subset=available_keys, keep="last").sort(available_keys)
        assert merged.height == 3

"""Data row reference models for quant-insight.

These models serve as reference schemas for Polars DataFrames.
They are primarily used for type hints and documentation purposes.
"""

from datetime import datetime as dt
from decimal import Decimal

from pydantic import BaseModel, Field


class OHLCVRow(BaseModel):
    """OHLCV data row schema.

    Reference model for OHLCV data structure.
    """

    datetime: dt = Field(
        ...,
        description="Date and time",
    )
    symbol: str = Field(
        ...,
        description="Stock symbol code",
    )
    open: Decimal = Field(
        ...,
        description="Opening price",
    )
    high: Decimal = Field(
        ...,
        description="High price",
    )
    low: Decimal = Field(
        ...,
        description="Low price",
    )
    close: Decimal = Field(
        ...,
        description="Closing price",
    )
    volume: int = Field(
        ...,
        ge=0,
        description="Trading volume",
    )


class ReturnRow(BaseModel):
    """Return data row schema.

    Reference model for return data structure.
    """

    datetime: dt = Field(
        ...,
        description="Date and time",
    )
    symbol: str = Field(
        ...,
        description="Stock symbol code",
    )
    return_value: float = Field(
        ...,
        description="Return value (can be negative, zero, or positive)",
    )


class SignalRow(BaseModel):
    """Signal data row schema.

    Reference model for signal data structure.
    """

    datetime: dt = Field(
        ...,
        description="Date and time",
    )
    symbol: str = Field(
        ...,
        description="Stock symbol code",
    )
    signal: float = Field(
        ...,
        description="Signal value (can be negative, zero, or positive)",
    )

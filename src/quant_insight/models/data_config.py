"""Data configuration model for quant-insight."""

from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    """Dataset configuration ([[competition.data]] section).

    Defines a dataset available in the competition.
    """

    name: str = Field(
        ...,
        description="Dataset name (e.g., ohlcv, returns, fundamentals)",
    )
    datetime_column: str = Field(
        default="datetime",
        description="Datetime column name in the dataset",
    )

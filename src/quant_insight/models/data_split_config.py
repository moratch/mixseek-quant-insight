"""Data split configuration model for quant-insight."""

from datetime import datetime

from pydantic import BaseModel, Field, model_validator


class DataSplitConfig(BaseModel):
    """Data split configuration (FR-010).

    Defines how to split data into train/valid/test sets.
    """

    train_end: datetime = Field(
        ...,
        description="Training period end datetime (inclusive)",
    )
    valid_end: datetime = Field(
        ...,
        description="Validation period end datetime (inclusive, test starts after this)",
    )
    purge_rows: int = Field(
        default=0,
        ge=0,
        description=(
            "Number of rows to purge between train/valid and valid/test. "
            "Set according to return window to prevent data leakage"
        ),
    )

    @model_validator(mode="after")
    def validate_date_order(self) -> "DataSplitConfig":
        """Validate train_end is before valid_end."""
        if self.train_end >= self.valid_end:
            raise ValueError("train_end must be before valid_end")
        return self

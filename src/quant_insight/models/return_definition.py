"""Return definition model for quant-insight."""

from pydantic import BaseModel, Field


class ReturnDefinition(BaseModel):
    """Return calculation definition (FR-007, FR-008).

    Defines how returns are calculated from OHLCV data.
    """

    window: int = Field(
        default=1,
        ge=1,
        description="Return calculation window width (in days)",
    )
    method: str = Field(
        default="close2close",
        pattern=r"^(close2close|open2close)$",
        description="Return calculation method: close2close or open2close",
    )

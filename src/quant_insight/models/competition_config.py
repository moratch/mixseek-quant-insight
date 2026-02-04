"""Competition configuration model for quant-insight."""

from pydantic import BaseModel, Field, model_validator

from quant_insight.models.data_config import DataConfig
from quant_insight.models.data_split_config import DataSplitConfig
from quant_insight.models.return_definition import ReturnDefinition


class CompetitionConfig(BaseModel):
    """Competition configuration (FR-010).

    Defines the entire competition setup including datasets, data splits, and return calculation.
    """

    name: str = Field(
        ...,
        description="Competition name",
    )
    description: str = Field(
        default="",
        description="Competition description",
    )
    data: list[DataConfig] = Field(
        ...,
        min_length=2,
        description="Dataset configuration list. ohlcv and returns are required",
    )
    data_split: DataSplitConfig = Field(
        ...,
        description="Data split configuration",
    )
    return_definition: ReturnDefinition = Field(
        default_factory=ReturnDefinition,
        description="Return calculation definition",
    )

    @model_validator(mode="after")
    def validate_required_datasets(self) -> "CompetitionConfig":
        """Validate that ohlcv and returns datasets are present."""
        names = {d.name for d in self.data}
        if "ohlcv" not in names:
            raise ValueError("ohlcv dataset is required in [[competition.data]]")
        if "returns" not in names:
            raise ValueError("returns dataset is required in [[competition.data]]")
        return self

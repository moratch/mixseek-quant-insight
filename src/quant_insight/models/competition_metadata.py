"""Competition metadata models for quant-insight."""

from pydantic import BaseModel, Field


class DatasetSchema(BaseModel):
    """Dataset schema definition (FR-019).

    Describes the structure and columns of a dataset.
    """

    name: str = Field(
        ...,
        description="Dataset name",
    )
    description: str = Field(
        default="",
        description="Dataset description",
    )
    columns: list[dict[str, str]] = Field(
        ...,
        description="Column definitions with name, type, and description",
    )


class SubmissionFormat(BaseModel):
    """Submission format definition (FR-019).

    Defines the expected format of submission functions.
    """

    function_signature: str = Field(
        default="def generate_signal(ohlcv: pl.DataFrame, additional_data: Dict[str, pl.DataFrame]) -> pl.DataFrame",
        description="Function signature for submissions",
    )
    ohlcv_schema: DatasetSchema = Field(
        ...,
        description="OHLCV input data schema",
    )
    additional_data_schemas: list[DatasetSchema] = Field(
        default_factory=list,
        description="Additional input data schemas",
    )
    output_schema: DatasetSchema = Field(
        ...,
        description="Output data schema (signal)",
    )
    example: str = Field(
        ...,
        description="Example function implementation",
    )


class CompetitionMetadata(BaseModel):
    """Competition metadata (FR-019).

    Provides information about the competition to agents and participants.
    """

    competition_name: str = Field(
        ...,
        description="Competition name",
    )
    competition_description: str = Field(
        default="",
        description="Competition description",
    )
    available_datasets: list[DatasetSchema] = Field(
        ...,
        description="List of available datasets",
    )
    submission_format: SubmissionFormat = Field(
        ...,
        description="Submission format specification",
    )
    evaluation_metric: str = Field(
        default="sharpe_ratio",
        description="Primary evaluation metric",
    )
    evaluation_description: str = Field(
        default="順位相関(Spearman)系列のシャープレシオ",
        description="Evaluation metric description",
    )

"""Unit tests for competition metadata models."""

from quant_insight.models.competition_metadata import (
    CompetitionMetadata,
    DatasetSchema,
    SubmissionFormat,
)


class TestDatasetSchema:
    """Test DatasetSchema model."""

    def test_valid_dataset_schema(self) -> None:
        """Test valid DatasetSchema."""
        schema = DatasetSchema(
            name="ohlcv",
            description="OHLCV data",
            columns=[
                {"name": "datetime", "type": "datetime", "description": "Date and time"},
                {"name": "symbol", "type": "str", "description": "Stock symbol"},
                {"name": "close", "type": "float", "description": "Closing price"},
            ],
        )
        assert schema.name == "ohlcv"
        assert schema.description == "OHLCV data"
        assert len(schema.columns) == 3
        assert schema.columns[0]["name"] == "datetime"

    def test_dataset_schema_minimal(self) -> None:
        """Test DatasetSchema with minimal fields."""
        schema = DatasetSchema(
            name="test_data",
            columns=[],
        )
        assert schema.name == "test_data"
        assert schema.description == ""
        assert schema.columns == []


class TestSubmissionFormat:
    """Test SubmissionFormat model."""

    def test_valid_submission_format(self) -> None:
        """Test valid SubmissionFormat."""
        ohlcv_schema = DatasetSchema(
            name="ohlcv",
            columns=[{"name": "close", "type": "float", "description": "Close"}],
        )
        output_schema = DatasetSchema(
            name="signal",
            columns=[{"name": "signal", "type": "float", "description": "Signal"}],
        )

        submission_format = SubmissionFormat(
            ohlcv_schema=ohlcv_schema,
            output_schema=output_schema,
            example="def generate_signal(ohlcv, additional_data): return ohlcv",
        )
        assert submission_format.ohlcv_schema.name == "ohlcv"
        assert submission_format.output_schema.name == "signal"
        assert "generate_signal" in submission_format.example

    def test_submission_format_default_signature(self) -> None:
        """Test SubmissionFormat uses default function signature."""
        ohlcv_schema = DatasetSchema(name="ohlcv", columns=[])
        output_schema = DatasetSchema(name="output", columns=[])

        submission_format = SubmissionFormat(
            ohlcv_schema=ohlcv_schema,
            output_schema=output_schema,
            example="example code",
        )
        assert "generate_signal" in submission_format.function_signature
        assert "pl.DataFrame" in submission_format.function_signature

    def test_submission_format_with_additional_data(self) -> None:
        """Test SubmissionFormat with additional data schemas."""
        ohlcv_schema = DatasetSchema(name="ohlcv", columns=[])
        output_schema = DatasetSchema(name="output", columns=[])
        sentiment_schema = DatasetSchema(name="sentiment", columns=[])

        submission_format = SubmissionFormat(
            ohlcv_schema=ohlcv_schema,
            additional_data_schemas=[sentiment_schema],
            output_schema=output_schema,
            example="example",
        )
        assert len(submission_format.additional_data_schemas) == 1
        assert submission_format.additional_data_schemas[0].name == "sentiment"


class TestCompetitionMetadata:
    """Test CompetitionMetadata model."""

    def test_valid_competition_metadata(self) -> None:
        """Test valid CompetitionMetadata."""
        ohlcv_schema = DatasetSchema(name="ohlcv", columns=[])
        returns_schema = DatasetSchema(name="returns", columns=[])
        output_schema = DatasetSchema(name="signal", columns=[])

        submission_format = SubmissionFormat(
            ohlcv_schema=ohlcv_schema,
            output_schema=output_schema,
            example="example",
        )

        metadata = CompetitionMetadata(
            competition_name="Test Competition",
            competition_description="Test description",
            available_datasets=[ohlcv_schema, returns_schema],
            submission_format=submission_format,
        )
        assert metadata.competition_name == "Test Competition"
        assert len(metadata.available_datasets) == 2
        assert metadata.evaluation_metric == "sharpe_ratio"

    def test_competition_metadata_defaults(self) -> None:
        """Test CompetitionMetadata with default values."""
        ohlcv_schema = DatasetSchema(name="ohlcv", columns=[])
        output_schema = DatasetSchema(name="signal", columns=[])

        submission_format = SubmissionFormat(
            ohlcv_schema=ohlcv_schema,
            output_schema=output_schema,
            example="example",
        )

        metadata = CompetitionMetadata(
            competition_name="Test",
            available_datasets=[ohlcv_schema],
            submission_format=submission_format,
        )
        assert metadata.competition_description == ""
        assert metadata.evaluation_metric == "sharpe_ratio"
        assert "Spearman" in metadata.evaluation_description


class TestCompetitionMetadataSerialization:
    """Test CompetitionMetadata serialization."""

    def test_model_dump(self) -> None:
        """Test model_dump returns correct structure."""
        ohlcv_schema = DatasetSchema(
            name="ohlcv",
            description="OHLCV data",
            columns=[{"name": "close", "type": "float", "description": "Close"}],
        )
        output_schema = DatasetSchema(name="signal", columns=[])

        submission_format = SubmissionFormat(
            ohlcv_schema=ohlcv_schema,
            output_schema=output_schema,
            example="example code",
        )

        metadata = CompetitionMetadata(
            competition_name="Test Competition",
            available_datasets=[ohlcv_schema],
            submission_format=submission_format,
            evaluation_metric="custom_metric",
        )

        data = metadata.model_dump()
        assert data["competition_name"] == "Test Competition"
        assert len(data["available_datasets"]) == 1
        assert data["submission_format"]["ohlcv_schema"]["name"] == "ohlcv"
        assert data["evaluation_metric"] == "custom_metric"

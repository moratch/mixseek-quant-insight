"""Models package for quant-insight."""

from quant_insight.models.backtest_result import BacktestResult, IterationResult
from quant_insight.models.competition_config import CompetitionConfig
from quant_insight.models.competition_metadata import (
    CompetitionMetadata,
    DatasetSchema,
    SubmissionFormat,
)
from quant_insight.models.data_config import DataConfig
from quant_insight.models.data_rows import OHLCVRow, ReturnRow, SignalRow
from quant_insight.models.data_split_config import DataSplitConfig
from quant_insight.models.preloaded_module import PreloadedModule
from quant_insight.models.return_definition import ReturnDefinition

__all__ = [
    # Configuration models
    "CompetitionConfig",
    "DataConfig",
    "DataSplitConfig",
    "ReturnDefinition",
    # Backtest result models
    "BacktestResult",
    "IterationResult",
    # Metadata models
    "CompetitionMetadata",
    "DatasetSchema",
    "SubmissionFormat",
    # Data row models (reference schemas)
    "OHLCVRow",
    "ReturnRow",
    "SignalRow",
    # Preloaded module model
    "PreloadedModule",
]

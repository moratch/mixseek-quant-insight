"""Data build package for quant-insight.

This package provides:
- BaseDataSourceAdapter: Abstract base class for data source adapters
- ReturnBuilder: Calculate return series from OHLCV data
- DataSplitter: Split data into train/valid/test sets with purge support
- ExecutionAnalyzer: Limit order fill analysis (independent of MixSeek evaluation)
- ExecutionResult: Result dataclass for ExecutionAnalyzer
"""

from quant_insight.data_build.base_adapter import BaseDataSourceAdapter
from quant_insight.data_build.data_splitter import DataSplitter
from quant_insight.data_build.execution_analyzer import ExecutionAnalyzer, ExecutionResult
from quant_insight.data_build.return_builder import ReturnBuilder

__all__ = [
    "BaseDataSourceAdapter",
    "DataSplitter",
    "ExecutionAnalyzer",
    "ExecutionResult",
    "ReturnBuilder",
]

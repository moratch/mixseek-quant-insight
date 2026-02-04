"""Contract tests for CorrelationSharpeRatio metric integration with mixseek-core.

These tests verify that CorrelationSharpeRatio correctly implements the BaseMetric
interface and can be loaded via TOML custom_metrics configuration.

Task: T026 - custom_metrics登録でTOML設定例が動作することを確認
"""

import importlib
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest
from mixseek.evaluator.metrics.base import BaseMetric
from mixseek.models.evaluation_config import (
    EvaluationConfig,
    LLMDefaultConfig,
    MetricConfig,
)

from quant_insight.evaluator.correlation_sharpe_ratio import CorrelationSharpeRatio


@pytest.mark.contract
def test_correlation_sharpe_ratio_inherits_base_metric():
    """Verify CorrelationSharpeRatio inherits from BaseMetric."""
    assert issubclass(CorrelationSharpeRatio, BaseMetric)


@pytest.mark.contract
def test_correlation_sharpe_ratio_has_evaluate_method():
    """Verify CorrelationSharpeRatio has async evaluate method."""
    metric = CorrelationSharpeRatio()
    assert hasattr(metric, "evaluate")
    assert callable(metric.evaluate)


@pytest.mark.contract
async def test_evaluate_returns_metric_score(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify evaluate() returns MetricScore with correct structure."""
    # Set up workspace environment
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    data_dir = workspace / "data" / "inputs"
    data_dir.mkdir(parents=True)

    # Create subdirectories for ohlcv and returns
    ohlcv_dir = data_dir / "ohlcv"
    returns_dir = data_dir / "returns"
    ohlcv_dir.mkdir()
    returns_dir.mkdir()

    monkeypatch.setenv("MIXSEEK_WORKSPACE", str(workspace))

    # Create test data
    ohlcv = pl.DataFrame(
        {
            "datetime": [datetime(2023, 1, 1), datetime(2023, 1, 1)],
            "symbol": ["AAPL", "GOOGL"],
            "open": [100.0, 200.0],
            "high": [102.0, 205.0],
            "low": [99.0, 198.0],
            "close": [101.0, 203.0],
            "volume": [1000, 2000],
        }
    )
    returns = pl.DataFrame(
        {
            "datetime": [datetime(2023, 1, 1), datetime(2023, 1, 1)],
            "symbol": ["AAPL", "GOOGL"],
            "return_value": [0.01, 0.02],
        }
    )

    ohlcv_path = ohlcv_dir / "test.parquet"
    returns_path = returns_dir / "test.parquet"
    ohlcv.write_parquet(ohlcv_path)
    returns.write_parquet(returns_path)

    submission = """
import polars as pl

def generate_signal(ohlcv: pl.DataFrame, additional_data: dict) -> pl.DataFrame:
    return ohlcv.select(["datetime", "symbol"]).with_columns(
        pl.when(pl.col("symbol") == "AAPL").then(1.0).otherwise(2.0).alias("signal")
    )
"""

    metric = CorrelationSharpeRatio()
    result = await metric.evaluate(
        user_query="test query",
        submission=submission,
    )

    # Verify MetricScore structure
    assert result.metric_name == "CorrelationSharpeRatio"
    assert isinstance(result.score, float)
    assert isinstance(result.evaluator_comment, str)


@pytest.mark.contract
def test_custom_metric_dynamic_loading():
    """Verify CorrelationSharpeRatio can be loaded via custom_metrics configuration pattern.

    This test verifies that the module can be dynamically loaded using the same
    mechanism that mixseek-core's Evaluator uses for custom_metrics:
    ```toml
    [custom_metrics]
    CorrelationSharpeRatio = { module = "quant_insight.evaluator.correlation_sharpe_ratio",
                               class = "CorrelationSharpeRatio" }
    ```
    """
    # Simulate the dynamic loading mechanism used by mixseek-core Evaluator
    module_path = "quant_insight.evaluator.correlation_sharpe_ratio"
    class_name = "CorrelationSharpeRatio"

    # Dynamic import (same as Evaluator._load_custom_metrics_from_config)
    module = importlib.import_module(module_path)
    metric_class = getattr(module, class_name)

    # Verify it's a subclass of BaseMetric
    assert issubclass(metric_class, BaseMetric)

    # Verify instantiation works
    metric_instance = metric_class()
    assert isinstance(metric_instance, CorrelationSharpeRatio)
    assert isinstance(metric_instance, BaseMetric)

    # Verify it has required methods
    assert hasattr(metric_instance, "evaluate")
    assert callable(metric_instance.evaluate)


@pytest.mark.contract
def test_toml_config_parsing_with_custom_metrics(tmp_path: Path) -> None:
    """Verify the TOML configuration example from quickstart.md parses correctly."""
    # This is the TOML config example pattern for custom_metrics
    toml_content = """
[llm_default]
model = "google-gla:gemini-2.5-flash"

[[metrics]]
name = "CorrelationSharpeRatio"
weight = 1.0

[custom_metrics.CorrelationSharpeRatio]
module = "quant_insight.evaluator.correlation_sharpe_ratio"
class = "CorrelationSharpeRatio"
"""

    # Write TOML to temp file
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    configs_dir = workspace / "configs"
    configs_dir.mkdir()
    toml_path = configs_dir / "evaluator.toml"
    toml_path.write_text(toml_content)

    # Load and validate config via EvaluationConfig (legacy API)
    config = EvaluationConfig.from_toml_file(workspace)

    # Verify config loaded correctly
    assert config.llm_default.model == "google-gla:gemini-2.5-flash"
    assert len(config.metrics) == 1
    assert config.metrics[0].name == "CorrelationSharpeRatio"
    assert config.metrics[0].weight == 1.0
    assert config.custom_metrics is not None
    assert "CorrelationSharpeRatio" in config.custom_metrics

    # Verify custom_metrics config structure
    custom_config = config.custom_metrics["CorrelationSharpeRatio"]
    assert custom_config["module"] == "quant_insight.evaluator.correlation_sharpe_ratio"
    assert custom_config["class"] == "CorrelationSharpeRatio"


@pytest.mark.contract
def test_evaluation_config_with_custom_metrics():
    """Verify EvaluationConfig can be created with custom_metrics programmatically."""
    config = EvaluationConfig(
        llm_default=LLMDefaultConfig(model="google-gla:gemini-2.5-flash"),
        metrics=[
            MetricConfig(name="CorrelationSharpeRatio", weight=1.0),
        ],
        custom_metrics={
            "CorrelationSharpeRatio": {
                "module": "quant_insight.evaluator.correlation_sharpe_ratio",
                "class": "CorrelationSharpeRatio",
            }
        },
    )

    # Verify config structure
    assert config.llm_default.model == "google-gla:gemini-2.5-flash"
    assert len(config.metrics) == 1
    assert config.metrics[0].name == "CorrelationSharpeRatio"
    assert config.custom_metrics is not None
    assert "CorrelationSharpeRatio" in config.custom_metrics

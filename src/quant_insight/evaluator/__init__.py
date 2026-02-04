"""バックテスト評価用のEvaluatorモジュール。

このモジュールはmixseek-coreフレームワーク用のカスタムevaluatorを提供し、
特にTime Series API形式を使用したシグナル生成関数のバックテストを実装する。
"""

from quant_insight.evaluator.backtest_loop import BacktestLoop
from quant_insight.evaluator.correlation_sharpe_ratio import CorrelationSharpeRatio
from quant_insight.evaluator.submission_parser import (
    extract_code_from_submission,
    parse_submission_function,
)

__all__ = [
    "BacktestLoop",
    "CorrelationSharpeRatio",
    "extract_code_from_submission",
    "parse_submission_function",
]

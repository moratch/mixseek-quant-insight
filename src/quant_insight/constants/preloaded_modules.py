"""共通プレロードモジュール定数定義。

Article 8（データ精度）およびArticle 9（DRY原則）に準拠。
EvaluatorとLocalCodeExecutorで同一のモジュール環境を提供する。
"""

from quant_insight.models.preloaded_module import PreloadedModule

# デフォルトでプレロードされるモジュール
# EvaluatorとLocalCodeExecutorの両方で使用される
DEFAULT_PRELOADED_MODULES: list[PreloadedModule] = [
    PreloadedModule(name="polars", alias="pl"),
    PreloadedModule(name="pandas", alias="pd"),
    PreloadedModule(name="numpy", alias="np"),
    PreloadedModule(name="sklearn"),
    PreloadedModule(name="scipy"),
    PreloadedModule(name="statsmodels"),
    PreloadedModule(name="sys"),
    PreloadedModule(name="pathlib"),
    PreloadedModule(name="os"),
    PreloadedModule(name="json"),
]

__all__ = ["DEFAULT_PRELOADED_MODULES"]

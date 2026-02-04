"""共通ユーティリティ

ログパースに使用する共通関数を提供。
"""

from quant_insight.utils.parse_logs.common.fence import get_fence, parse_json_safe

__all__ = [
    "get_fence",
    "parse_json_safe",
]

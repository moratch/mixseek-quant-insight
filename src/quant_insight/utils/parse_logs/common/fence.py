"""共通フェンス・JSONユーティリティ

Markdown変換に使用する共通関数を提供。
"""

from __future__ import annotations

import json
import re
from typing import Any


def get_fence(content: str, lang: str = "") -> tuple[str, str]:
    """コンテンツに応じて適切なフェンスを返す

    コンテンツ内にバッククォートが含まれる場合はチルダフェンスを使用し、
    チルダも含まれる場合は十分な長さのフェンスを使用する。

    Args:
        content: フェンスで囲むコンテンツ
        lang: 言語指定（オプション）

    Returns:
        (開始フェンス, 終了フェンス)のタプル
    """
    has_backticks = "```" in content
    has_tildes = "~~~" in content

    if not has_backticks:
        # バッククォートがなければ通常の3つで十分
        return f"```{lang}", "```"
    elif not has_tildes:
        # バッククォートがあるがチルダがなければチルダを使用
        return f"~~~{lang}", "~~~"
    else:
        # 両方ある場合は、十分な長さのチルダを使用
        max_tildes = 3
        matches = re.findall(r"~+", content)
        if matches:
            max_tildes = max(len(m) for m in matches)
        fence = "~" * (max_tildes + 1)
        return f"{fence}{lang}", fence


def parse_json_safe(json_str: str | None) -> list[Any] | dict[str, Any] | None:
    """JSONを安全にパースする

    Args:
        json_str: JSONまたはNone

    Returns:
        パース結果（リストまたは辞書）、パース失敗またはNoneの場合はNone
    """
    if not json_str:
        return None
    try:
        return json.loads(json_str)  # type: ignore[no-any-return]
    except json.JSONDecodeError:
        return None

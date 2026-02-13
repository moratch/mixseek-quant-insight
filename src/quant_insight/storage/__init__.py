"""ストレージモジュール

エージェント実装およびスクリーニング結果をDuckDBに永続化するためのストレージ層を提供。

スキーマ定義（文字列DDL）は即時importされるが、ストアクラスは遅延importにより
mixseek-core未導入環境でもScreeningResultStore単体利用が可能。
"""

from quant_insight.storage.schema import (
    AGENT_IMPLEMENTATION_INDEX_DDL,
    AGENT_IMPLEMENTATION_SEQUENCE_DDL,
    AGENT_IMPLEMENTATION_TABLE_DDL,
    ALL_AGENT_IMPLEMENTATION_DDL,
    ALL_SCREENING_RESULT_DDL,
    SCREENING_RESULT_INDEX_DDL,
    SCREENING_RESULT_SEQUENCE_DDL,
    SCREENING_RESULT_TABLE_DDL,
)

__all__ = [
    # ストアクラス (lazy)
    "ImplementationStore",
    "get_implementation_store",
    "ScreeningResultStore",
    # 例外 (lazy)
    "DatabaseWriteError",
    "DatabaseReadError",
    # スキーマ定義 (eager)
    "AGENT_IMPLEMENTATION_SEQUENCE_DDL",
    "AGENT_IMPLEMENTATION_TABLE_DDL",
    "AGENT_IMPLEMENTATION_INDEX_DDL",
    "ALL_AGENT_IMPLEMENTATION_DDL",
    "SCREENING_RESULT_SEQUENCE_DDL",
    "SCREENING_RESULT_TABLE_DDL",
    "SCREENING_RESULT_INDEX_DDL",
    "ALL_SCREENING_RESULT_DDL",
]


def __getattr__(name: str) -> object:
    """Lazy-load store classes to avoid pulling in mixseek-core at package level.

    ImplementationStore depends on mixseek.utils.env, so importing it eagerly
    forces a mixseek-core dependency even when only ScreeningResultStore is needed.
    """
    if name in ("ImplementationStore", "get_implementation_store", "DatabaseWriteError", "DatabaseReadError"):
        from quant_insight.storage.implementation_store import (
            DatabaseReadError,
            DatabaseWriteError,
            ImplementationStore,
            get_implementation_store,
        )

        _lazy = {
            "ImplementationStore": ImplementationStore,
            "get_implementation_store": get_implementation_store,
            "DatabaseWriteError": DatabaseWriteError,
            "DatabaseReadError": DatabaseReadError,
        }
        return _lazy[name]

    if name == "ScreeningResultStore":
        from quant_insight.storage.screening_store import ScreeningResultStore

        return ScreeningResultStore

    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)

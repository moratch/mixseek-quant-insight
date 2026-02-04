"""ストレージモジュール

エージェント実装をDuckDBに永続化するためのストレージ層を提供。
"""

from quant_insight.storage.implementation_store import (
    DatabaseReadError,
    DatabaseWriteError,
    ImplementationStore,
    get_implementation_store,
)
from quant_insight.storage.schema import (
    AGENT_IMPLEMENTATION_INDEX_DDL,
    AGENT_IMPLEMENTATION_SEQUENCE_DDL,
    AGENT_IMPLEMENTATION_TABLE_DDL,
    ALL_AGENT_IMPLEMENTATION_DDL,
)

__all__ = [
    # ストアクラス
    "ImplementationStore",
    "get_implementation_store",
    # 例外
    "DatabaseWriteError",
    "DatabaseReadError",
    # スキーマ定義
    "AGENT_IMPLEMENTATION_SEQUENCE_DDL",
    "AGENT_IMPLEMENTATION_TABLE_DDL",
    "AGENT_IMPLEMENTATION_INDEX_DDL",
    "ALL_AGENT_IMPLEMENTATION_DDL",
]

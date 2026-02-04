"""DuckDBスキーマ定義（エージェント実装保存用）

このモジュールはagent_implementationテーブルのDDL定義を提供します。
LocalCodeExecutorが生成したスクリプトをDuckDBに永続化するために使用。
"""

# シーケンス定義
AGENT_IMPLEMENTATION_SEQUENCE_DDL = """
CREATE SEQUENCE IF NOT EXISTS agent_implementation_id_seq
"""

# agent_implementationテーブルDDL
AGENT_IMPLEMENTATION_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS agent_implementation (
    -- 主キー
    id INTEGER PRIMARY KEY DEFAULT nextval('agent_implementation_id_seq'),

    -- 識別子
    execution_id TEXT NOT NULL,
    team_id TEXT NOT NULL,
    round_number INTEGER NOT NULL,
    member_agent_name TEXT NOT NULL,
    file_name TEXT NOT NULL,

    -- コンテンツ
    code TEXT NOT NULL,

    -- メタデータ
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- 一意性制約
    UNIQUE(execution_id, team_id, round_number, member_agent_name, file_name)
)
"""

# インデックス定義
AGENT_IMPLEMENTATION_INDEX_DDL = """
CREATE INDEX IF NOT EXISTS idx_agent_implementation_context
ON agent_implementation (execution_id, team_id, round_number, member_agent_name)
"""

# 全DDL文（実行順）
ALL_AGENT_IMPLEMENTATION_DDL: list[str] = [
    AGENT_IMPLEMENTATION_SEQUENCE_DDL,
    AGENT_IMPLEMENTATION_TABLE_DDL,
    AGENT_IMPLEMENTATION_INDEX_DDL,
]

__all__ = [
    "AGENT_IMPLEMENTATION_SEQUENCE_DDL",
    "AGENT_IMPLEMENTATION_TABLE_DDL",
    "AGENT_IMPLEMENTATION_INDEX_DDL",
    "ALL_AGENT_IMPLEMENTATION_DDL",
]

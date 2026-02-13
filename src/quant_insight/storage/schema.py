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

# ---------------------------------------------------------------------------
# screening_result テーブル (P3)
# ---------------------------------------------------------------------------

SCREENING_RESULT_SEQUENCE_DDL = """
CREATE SEQUENCE IF NOT EXISTS screening_result_id_seq
"""

SCREENING_RESULT_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS screening_result (
    -- 主キー
    id INTEGER PRIMARY KEY DEFAULT nextval('screening_result_id_seq'),

    -- 識別子
    execution_id TEXT NOT NULL,
    team_id TEXT NOT NULL,
    team_name TEXT NOT NULL,
    round_number INTEGER NOT NULL,
    strategy_name TEXT NOT NULL,
    mode TEXT NOT NULL,
    config_hash TEXT NOT NULL,

    -- MixSeek スコア
    mixseek_score DOUBLE,

    -- WFA 主要指標
    oos_sharpe DOUBLE,
    wfe DOUBLE,
    consistency DOUBLE,

    -- CPCV 主要指標
    pbo DOUBLE,
    deflated_sharpe DOUBLE,

    -- 判定
    passed BOOLEAN NOT NULL,
    failed_criteria TEXT,

    -- 全結果JSON (詳細取得用)
    result_json TEXT NOT NULL,

    -- タイムスタンプ
    screened_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- 一意性制約
    UNIQUE(execution_id, team_id, round_number, strategy_name, mode, config_hash)
)
"""

SCREENING_RESULT_INDEX_DDL = """
CREATE INDEX IF NOT EXISTS idx_screening_result_execution
ON screening_result (execution_id)
"""

ALL_SCREENING_RESULT_DDL: list[str] = [
    SCREENING_RESULT_SEQUENCE_DDL,
    SCREENING_RESULT_TABLE_DDL,
    SCREENING_RESULT_INDEX_DDL,
]


__all__ = [
    "AGENT_IMPLEMENTATION_SEQUENCE_DDL",
    "AGENT_IMPLEMENTATION_TABLE_DDL",
    "AGENT_IMPLEMENTATION_INDEX_DDL",
    "ALL_AGENT_IMPLEMENTATION_DDL",
    "SCREENING_RESULT_SEQUENCE_DDL",
    "SCREENING_RESULT_TABLE_DDL",
    "SCREENING_RESULT_INDEX_DDL",
    "ALL_SCREENING_RESULT_DDL",
]

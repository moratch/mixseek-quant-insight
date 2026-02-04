"""Local Code Executor Agent用Pydanticモデル。

Article 16（型安全性）に準拠。
"""

from pydantic import BaseModel, ConfigDict, Field


class ImplementationContext(BaseModel):
    """エージェント実装を特定するためのコンテキスト情報。

    DuckDBへのスクリプト保存時に使用する識別情報。
    """

    execution_id: str = Field(
        ...,
        description="実行識別子(UUID)",
    )
    team_id: str = Field(
        ...,
        description="チームID",
    )
    round_number: int = Field(
        ...,
        ge=0,
        description="ラウンド番号",
    )
    member_agent_name: str = Field(
        ...,
        description="メンバーエージェント名",
    )


class OutputModelConfig(BaseModel):
    """構造化出力モデル設定。"""

    module_path: str = Field(
        ...,
        description="モジュールパス（例: quant_insight.agents.local_code_executor.output_models）",
    )
    class_name: str = Field(
        ...,
        description="クラス名（例: AnalyzerOutput）",
    )


class LocalCodeExecutorConfig(BaseModel):
    """Local Code Executor設定（Article 9準拠）。"""

    available_data_paths: list[str] = Field(
        default_factory=list,
        description="$MIXSEEK_WORKSPACEからの相対パスリスト",
    )
    timeout_seconds: int = Field(
        default=120,
        gt=0,
        description="実行タイムアウト（秒）",
    )
    max_output_chars: int | None = Field(
        default=None,
        description="最大出力文字数（None=無制限）",
    )
    output_model: OutputModelConfig | None = Field(
        default=None,
        description="構造化出力モデル設定",
    )
    implementation_context: ImplementationContext | None = Field(
        default=None,
        description="実装コンテキスト（DuckDB保存時に使用）",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "available_data_paths": ["data/input", "data/output"],
                    "timeout_seconds": 120,
                    "max_output_chars": None,
                }
            ]
        }
    )


__all__ = ["ImplementationContext", "OutputModelConfig", "LocalCodeExecutorConfig"]

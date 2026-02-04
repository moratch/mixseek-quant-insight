"""Local Code Executor Agent用構造化出力モデル。

Article 16（型安全性）に準拠。
"""

from pydantic import BaseModel, Field


class ScriptEntry(BaseModel):
    """保存するスクリプトのエントリ。"""

    file_name: str = Field(..., description="ファイル名（.py拡張子）")
    code: str = Field(..., description="Pythonコード文字列")


class AnalyzerOutput(BaseModel):
    """分析エージェントの構造化出力。"""

    scripts: list[ScriptEntry] = Field(
        ...,
        description="分析で作成したスクリプトのリスト",
    )
    report: str = Field(
        ...,
        description="Markdown形式の分析結果レポート",
    )


class SubmitterOutput(BaseModel):
    """Submission作成エージェントの構造化出力。"""

    submission: str = Field(
        ...,
        description="Submission形式に整合するシグナル生成関数を含むSubmissionコード全体の文字列",
    )
    description: str = Field(
        ...,
        description="Submissionの概要や動作確認結果をMarkdown形式でまとめた文章",
    )


__all__ = ["AnalyzerOutput", "ScriptEntry", "SubmitterOutput"]

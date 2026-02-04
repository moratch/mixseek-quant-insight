"""Unit tests for output_models."""

import pytest
from pydantic import ValidationError

from quant_insight.agents.local_code_executor.output_models import (
    AnalyzerOutput,
    ScriptEntry,
    SubmitterOutput,
)


@pytest.mark.unit
class TestScriptEntry:
    """ScriptEntryのユニットテスト。"""

    def test_create_with_required_fields(self):
        """必須フィールドで正しく作成される。"""
        entry = ScriptEntry(
            file_name="analysis.py",
            code="print('hello')",
        )

        assert entry.file_name == "analysis.py"
        assert entry.code == "print('hello')"

    def test_requires_file_name(self):
        """file_nameは必須フィールド。"""
        with pytest.raises(ValidationError) as exc_info:
            ScriptEntry(code="print('hello')")  # type: ignore[call-arg]
        assert "file_name" in str(exc_info.value)

    def test_requires_code(self):
        """codeは必須フィールド。"""
        with pytest.raises(ValidationError) as exc_info:
            ScriptEntry(file_name="test.py")  # type: ignore[call-arg]
        assert "code" in str(exc_info.value)

    def test_code_accepts_multiline_string(self):
        """codeは複数行の文字列を受け入れる。"""
        multiline_code = """import pandas as pd

def analyze():
    return pd.DataFrame()
"""
        entry = ScriptEntry(file_name="analysis.py", code=multiline_code)
        assert entry.code == multiline_code

    def test_model_validate_from_dict(self):
        """辞書からmodel_validateで作成できる。"""
        data = {
            "file_name": "test.py",
            "code": "print('test')",
        }
        entry = ScriptEntry.model_validate(data)

        assert entry.file_name == "test.py"
        assert entry.code == "print('test')"


@pytest.mark.unit
class TestAnalyzerOutput:
    """AnalyzerOutputのユニットテスト。"""

    def test_create_with_required_fields(self):
        """必須フィールドで正しく作成される。"""
        scripts = [
            ScriptEntry(file_name="script1.py", code="print('hello')"),
            ScriptEntry(file_name="script2.py", code="print('world')"),
        ]
        output = AnalyzerOutput(
            scripts=scripts,
            report="# 分析レポート\n\n分析結果の説明",
        )

        assert len(output.scripts) == 2
        assert output.scripts[0].file_name == "script1.py"
        assert output.scripts[1].file_name == "script2.py"
        assert output.report == "# 分析レポート\n\n分析結果の説明"

    def test_requires_scripts(self):
        """scriptsは必須フィールド。"""
        with pytest.raises(ValidationError) as exc_info:
            AnalyzerOutput(report="レポート")  # type: ignore[call-arg]
        assert "scripts" in str(exc_info.value)

    def test_requires_report(self):
        """reportは必須フィールド。"""
        with pytest.raises(ValidationError) as exc_info:
            AnalyzerOutput(scripts=[])  # type: ignore[call-arg]
        assert "report" in str(exc_info.value)

    def test_scripts_accepts_empty_list(self):
        """scriptsは空リストを受け入れる。"""
        output = AnalyzerOutput(scripts=[], report="レポート")
        assert output.scripts == []

    def test_scripts_accepts_multiple_entries(self):
        """scriptsは複数のエントリを受け入れる。"""
        scripts = [
            ScriptEntry(file_name="a.py", code="a"),
            ScriptEntry(file_name="b.py", code="b"),
            ScriptEntry(file_name="c.py", code="c"),
        ]
        output = AnalyzerOutput(scripts=scripts, report="レポート")
        assert len(output.scripts) == 3

    def test_report_accepts_multiline_string(self):
        """reportは複数行の文字列を受け入れる。"""
        multiline_report = """# 分析レポート

## 概要
分析結果の概要

## 詳細
詳細な分析結果
"""
        output = AnalyzerOutput(scripts=[], report=multiline_report)
        assert output.report == multiline_report

    def test_model_validate_from_dict(self):
        """辞書からmodel_validateで作成できる。"""
        data = {
            "scripts": [
                {"file_name": "script.py", "code": "print('test')"},
            ],
            "report": "分析レポート",
        }
        output = AnalyzerOutput.model_validate(data)

        assert len(output.scripts) == 1
        assert output.scripts[0].file_name == "script.py"
        assert output.scripts[0].code == "print('test')"
        assert output.report == "分析レポート"

    def test_model_dump(self):
        """model_dumpで辞書に変換できる。"""
        output = AnalyzerOutput(
            scripts=[ScriptEntry(file_name="script.py", code="print('test')")],
            report="分析レポート",
        )

        dumped = output.model_dump()

        assert dumped == {
            "scripts": [{"file_name": "script.py", "code": "print('test')"}],
            "report": "分析レポート",
        }


@pytest.mark.unit
class TestSubmitterOutput:
    """SubmitterOutputのユニットテスト。"""

    def test_create_with_required_fields(self):
        """必須フィールドで正しく作成される。"""
        output = SubmitterOutput(
            submission="def generate_signal(): pass",
            description="# Submission概要\n\n動作確認結果",
        )

        assert output.submission == "def generate_signal(): pass"
        assert output.description == "# Submission概要\n\n動作確認結果"

    def test_requires_submission(self):
        """submissionは必須フィールド。"""
        with pytest.raises(ValidationError) as exc_info:
            SubmitterOutput(  # type: ignore[call-arg]
                description="説明",
            )
        assert "submission" in str(exc_info.value)

    def test_requires_description(self):
        """descriptionは必須フィールド。"""
        with pytest.raises(ValidationError) as exc_info:
            SubmitterOutput(  # type: ignore[call-arg]
                submission="code",
            )
        assert "description" in str(exc_info.value)

    def test_submission_accepts_multiline_code(self):
        """submissionは複数行のコードを受け入れる。"""
        multiline_code = """import polars as pl

def generate_signal(data: pl.DataFrame) -> pl.DataFrame:
    # シグナル生成ロジック
    return data
"""
        output = SubmitterOutput(
            submission=multiline_code,
            description="説明",
        )
        assert output.submission == multiline_code

    def test_description_accepts_multiline_markdown(self):
        """descriptionは複数行のMarkdownを受け入れる。"""
        multiline_desc = """# Submission概要

## 動作確認
- テスト1: 成功
- テスト2: 成功

## 注意事項
特になし
"""
        output = SubmitterOutput(
            submission="code",
            description=multiline_desc,
        )
        assert output.description == multiline_desc

    def test_model_validate_from_dict(self):
        """辞書からmodel_validateで作成できる。"""
        data = {
            "submission": "def generate_signal(): pass",
            "description": "Submission説明",
        }
        output = SubmitterOutput.model_validate(data)

        assert output.submission == "def generate_signal(): pass"
        assert output.description == "Submission説明"

    def test_model_dump(self):
        """model_dumpで辞書に変換できる。"""
        output = SubmitterOutput(
            submission="def generate_signal(): pass",
            description="Submission説明",
        )

        dumped = output.model_dump()

        assert dumped == {
            "submission": "def generate_signal(): pass",
            "description": "Submission説明",
        }

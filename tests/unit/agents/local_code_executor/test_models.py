"""Unit tests for LocalCodeExecutorConfig model."""

import pytest
from pydantic import ValidationError

from quant_insight.agents.local_code_executor.models import (
    LocalCodeExecutorConfig,
    OutputModelConfig,
)


@pytest.mark.unit
class TestOutputModelConfig:
    """OutputModelConfigのユニットテスト。"""

    def test_create_with_required_fields(self):
        """必須フィールドで正しく作成される。"""
        config = OutputModelConfig(
            module_path="quant_insight.agents.local_code_executor.output_models",
            class_name="AnalyzerOutput",
        )

        assert config.module_path == "quant_insight.agents.local_code_executor.output_models"
        assert config.class_name == "AnalyzerOutput"

    def test_requires_module_path(self):
        """module_pathは必須フィールド。"""
        with pytest.raises(ValidationError) as exc_info:
            OutputModelConfig(class_name="AnalyzerOutput")  # type: ignore[call-arg]
        assert "module_path" in str(exc_info.value)

    def test_requires_class_name(self):
        """class_nameは必須フィールド。"""
        with pytest.raises(ValidationError) as exc_info:
            OutputModelConfig(  # type: ignore[call-arg]
                module_path="quant_insight.agents.local_code_executor.output_models"
            )
        assert "class_name" in str(exc_info.value)

    def test_model_validate_from_dict(self):
        """辞書からmodel_validateで作成できる。"""
        data = {
            "module_path": "quant_insight.agents.local_code_executor.output_models",
            "class_name": "SubmitterOutput",
        }
        config = OutputModelConfig.model_validate(data)

        assert config.module_path == "quant_insight.agents.local_code_executor.output_models"
        assert config.class_name == "SubmitterOutput"

    def test_model_dump(self):
        """model_dumpで辞書に変換できる。"""
        config = OutputModelConfig(
            module_path="quant_insight.agents.local_code_executor.output_models",
            class_name="AnalyzerOutput",
        )

        dumped = config.model_dump()

        assert dumped == {
            "module_path": "quant_insight.agents.local_code_executor.output_models",
            "class_name": "AnalyzerOutput",
        }


@pytest.mark.unit
class TestLocalCodeExecutorConfig:
    """LocalCodeExecutorConfigのユニットテスト。"""

    def test_default_values(self):
        """デフォルト値が正しく設定される。"""
        config = LocalCodeExecutorConfig()

        assert config.available_data_paths == []
        assert config.timeout_seconds == 120
        assert config.max_output_chars is None
        assert config.output_model is None

    def test_create_with_all_fields(self):
        """全フィールド指定で正しく作成される。"""
        output_model = OutputModelConfig(
            module_path="quant_insight.agents.local_code_executor.output_models",
            class_name="AnalyzerOutput",
        )
        config = LocalCodeExecutorConfig(
            available_data_paths=["data/input", "data/output"],
            timeout_seconds=300,
            max_output_chars=5000,
            output_model=output_model,
        )

        assert config.available_data_paths == ["data/input", "data/output"]
        assert config.timeout_seconds == 300
        assert config.max_output_chars == 5000
        assert config.output_model == output_model

    def test_timeout_seconds_must_be_positive(self):
        """timeout_secondsは正の整数である必要がある。"""
        with pytest.raises(ValidationError) as exc_info:
            LocalCodeExecutorConfig(timeout_seconds=0)
        assert "greater than 0" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            LocalCodeExecutorConfig(timeout_seconds=-1)
        assert "greater than 0" in str(exc_info.value)

    def test_timeout_seconds_accepts_positive_int(self):
        """timeout_secondsは正の整数を受け入れる。"""
        config_1 = LocalCodeExecutorConfig(timeout_seconds=1)
        assert config_1.timeout_seconds == 1

        config_60 = LocalCodeExecutorConfig(timeout_seconds=60)
        assert config_60.timeout_seconds == 60

        config_300 = LocalCodeExecutorConfig(timeout_seconds=300)
        assert config_300.timeout_seconds == 300

    def test_max_output_chars_accepts_none(self):
        """max_output_charsはNoneを受け入れる。"""
        config = LocalCodeExecutorConfig(max_output_chars=None)
        assert config.max_output_chars is None

    def test_max_output_chars_accepts_positive_int(self):
        """max_output_charsは正の整数を受け入れる。"""
        config_100 = LocalCodeExecutorConfig(max_output_chars=100)
        assert config_100.max_output_chars == 100

        config_10000 = LocalCodeExecutorConfig(max_output_chars=10000)
        assert config_10000.max_output_chars == 10000

    def test_model_validate_from_dict(self):
        """辞書からmodel_validateで作成できる。"""
        data = {
            "available_data_paths": ["data/input"],
            "timeout_seconds": 60,
            "max_output_chars": 1000,
        }
        config = LocalCodeExecutorConfig.model_validate(data)

        assert config.available_data_paths == ["data/input"]
        assert config.timeout_seconds == 60
        assert config.max_output_chars == 1000

    def test_model_dump(self):
        """model_dumpで辞書に変換できる。"""
        config = LocalCodeExecutorConfig(
            available_data_paths=["data/input"],
            timeout_seconds=60,
            max_output_chars=1000,
        )

        dumped = config.model_dump()

        assert dumped == {
            "available_data_paths": ["data/input"],
            "timeout_seconds": 60,
            "max_output_chars": 1000,
            "output_model": None,
            "implementation_context": None,
        }

    def test_json_schema_extra_examples(self):
        """json_schema_extraにexamplesが含まれる。"""
        schema = LocalCodeExecutorConfig.model_json_schema()

        assert "examples" in schema
        assert len(schema["examples"]) > 0
        example = schema["examples"][0]
        assert "available_data_paths" in example
        assert "timeout_seconds" in example

    def test_output_model_accepts_output_model_config(self):
        """output_modelはOutputModelConfigを受け入れる。"""
        output_model = OutputModelConfig(
            module_path="quant_insight.agents.local_code_executor.output_models",
            class_name="AnalyzerOutput",
        )
        config = LocalCodeExecutorConfig(output_model=output_model)
        assert config.output_model == output_model

    def test_output_model_accepts_none(self):
        """output_modelはNoneを受け入れる。"""
        config = LocalCodeExecutorConfig(output_model=None)
        assert config.output_model is None

    def test_output_model_from_dict(self):
        """output_modelを辞書から作成できる。"""
        data = {
            "available_data_paths": ["data/input"],
            "timeout_seconds": 60,
            "output_model": {
                "module_path": "quant_insight.agents.local_code_executor.output_models",
                "class_name": "SubmitterOutput",
            },
        }
        config = LocalCodeExecutorConfig.model_validate(data)

        assert config.output_model is not None
        assert config.output_model.module_path == "quant_insight.agents.local_code_executor.output_models"
        assert config.output_model.class_name == "SubmitterOutput"

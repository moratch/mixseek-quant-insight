"""Unit tests for ReturnDefinition model."""

import pytest
from pydantic import ValidationError

from quant_insight.models.return_definition import ReturnDefinition


class TestReturnDefinitionValidation:
    """Test ReturnDefinition validation."""

    def test_default_values(self) -> None:
        """Test default values for ReturnDefinition."""
        rd = ReturnDefinition()
        assert rd.window == 1
        assert rd.method == "close2close"

    def test_valid_close2close(self) -> None:
        """Test valid close2close method."""
        rd = ReturnDefinition(window=5, method="close2close")
        assert rd.window == 5
        assert rd.method == "close2close"

    def test_valid_open2close(self) -> None:
        """Test valid open2close method."""
        rd = ReturnDefinition(window=3, method="open2close")
        assert rd.window == 3
        assert rd.method == "open2close"

    def test_window_must_be_positive(self) -> None:
        """Test window must be >= 1."""
        with pytest.raises(ValidationError) as exc_info:
            ReturnDefinition(window=0)
        assert "greater than or equal to 1" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            ReturnDefinition(window=-1)
        assert "greater than or equal to 1" in str(exc_info.value)

    def test_method_pattern_validation(self) -> None:
        """Test method must match pattern."""
        with pytest.raises(ValidationError) as exc_info:
            ReturnDefinition(method="invalid_method")
        assert "String should match pattern" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            ReturnDefinition(method="close2open")
        assert "String should match pattern" in str(exc_info.value)


class TestReturnDefinitionSerialization:
    """Test ReturnDefinition serialization."""

    def test_model_dump(self) -> None:
        """Test model_dump returns correct dict."""
        rd = ReturnDefinition(window=7, method="open2close")
        data = rd.model_dump()
        assert data == {"window": 7, "method": "open2close"}

    def test_model_dump_json(self) -> None:
        """Test model_dump_json returns valid JSON."""
        rd = ReturnDefinition(window=2, method="close2close")
        json_str = rd.model_dump_json()
        assert '"window":2' in json_str or '"window": 2' in json_str
        assert '"method":"close2close"' in json_str or '"method": "close2close"' in json_str

"""Unit tests for CLI api serve feature flag (P3-e)."""

from __future__ import annotations

import importlib
import sys
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

runner = CliRunner()


@pytest.mark.unit
class TestFeatureFlagDisabled:
    """Test #12: ENABLE_P3_API not set → server refuses to start."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_fastapi(self) -> None:
        pytest.importorskip("fastapi")

    def test_feature_flag_disabled(self, monkeypatch, tmp_path):
        """API serve refuses without ENABLE_P3_API=1."""
        from quant_insight.cli.main import app

        monkeypatch.delenv("ENABLE_P3_API", raising=False)
        monkeypatch.setenv("MIXSEEK_WORKSPACE", str(tmp_path))

        result = runner.invoke(app, ["api", "serve", "--workspace", str(tmp_path)])
        assert result.exit_code != 0
        assert "ENABLE_P3_API" in result.output


@pytest.mark.unit
class TestCliWithoutFastapi:
    """Test #13: CLI works normally when fastapi is not installed."""

    def test_api_subcommand_hidden_without_fastapi(self, monkeypatch, tmp_path):
        """When fastapi is not importable, 'api' subcommand is absent but CLI still works."""
        monkeypatch.setenv("MIXSEEK_WORKSPACE", str(tmp_path))

        # Temporarily remove fastapi/uvicorn from sys.modules and make them unimportable
        saved_modules: dict[str, object] = {}
        for mod_name in list(sys.modules):
            if mod_name == "fastapi" or mod_name.startswith("fastapi."):
                saved_modules[mod_name] = sys.modules.pop(mod_name)
            if mod_name == "uvicorn" or mod_name.startswith("uvicorn."):
                saved_modules[mod_name] = sys.modules.pop(mod_name)

        # Also remove cached cli.commands.api and cli.main to force re-import
        for mod_name in list(sys.modules):
            if "quant_insight.cli" in mod_name:
                saved_modules[mod_name] = sys.modules.pop(mod_name)

        original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

        def _mock_import(name: str, *args: object, **kwargs: object) -> object:
            if name in ("fastapi", "uvicorn") or name.startswith("fastapi.") or name.startswith("uvicorn."):
                raise ModuleNotFoundError(name=name)
            return original_import(name, *args, **kwargs)

        try:
            with patch("builtins.__import__", side_effect=_mock_import):
                # Re-import main to trigger the try/except for api_app
                import quant_insight.cli.main as main_mod

                importlib.reload(main_mod)
                app_reloaded = main_mod.app

                # CLI --help should work
                result = runner.invoke(app_reloaded, ["--help"])
                assert result.exit_code == 0

                # 'api' should NOT appear in the commands section of help output
                help_lower = result.output.lower()
                if "commands:" in help_lower:
                    commands_section = help_lower.split("commands:")[-1]
                    assert "api" not in commands_section

                # Other subcommands should still be present
                assert "screening" in result.output
        finally:
            # Restore all saved modules
            sys.modules.update(saved_modules)
            # Reload main to restore normal state
            importlib.reload(importlib.import_module("quant_insight.cli.main"))

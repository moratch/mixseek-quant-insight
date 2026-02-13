"""API server CLI commands.

This module imports fastapi and uvicorn at module level.
If [api] extra is not installed, the import of this module will fail
with ModuleNotFoundError, which is caught in main.py to gracefully
hide the `api` subcommand.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

import typer
import uvicorn

from quant_insight.api.app import create_app

api_app = typer.Typer(name="api", help="API server management")


@api_app.command("serve")
def serve(
    workspace: Annotated[
        str,
        typer.Option(
            "--workspace",
            "-w",
            envvar="MIXSEEK_WORKSPACE",
            help="Workspace path (or set MIXSEEK_WORKSPACE env var)",
        ),
    ],
    host: Annotated[
        str,
        typer.Option("--host", help="Bind host"),
    ] = "127.0.0.1",
    port: Annotated[
        int,
        typer.Option("--port", help="Bind port"),
    ] = 8000,
) -> None:
    """Start the API server (requires ENABLE_P3_API=1)."""
    if os.environ.get("ENABLE_P3_API") != "1":
        typer.echo("API is disabled. Set ENABLE_P3_API=1 to enable.")
        raise typer.Exit(1)

    app = create_app(workspace=Path(workspace))
    uvicorn.run(app, host=host, port=port)

"""Local Code Executor Agent用ツール関数。

Article 9（データ精度）およびArticle 16（型安全性）に準拠。
並列実行時のスレッド安全性を確保するため、subprocessによるプロセス分離を使用。
"""

import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

from pydantic_ai import FunctionToolset, RunContext

from quant_insight.agents.local_code_executor.models import LocalCodeExecutorConfig
from quant_insight.constants import DEFAULT_PRELOADED_MODULES
from quant_insight.models.preloaded_module import PreloadedModule
from quant_insight.storage import get_implementation_store


def _get_workspace_path() -> Path:
    """環境変数からMIXSEEK_WORKSPACEパスを取得。

    Returns:
        ワークスペースディレクトリのパス。

    Raises:
        ValueError: MIXSEEK_WORKSPACEが設定されていない場合。
    """
    workspace = os.environ.get("MIXSEEK_WORKSPACE")
    if not workspace:
        raise ValueError("環境変数MIXSEEK_WORKSPACEが設定されていません")
    return Path(workspace)


def _resolve_data_path(relative_path: str) -> Path:
    """相対パスをMIXSEEK_WORKSPACE配下のフルパスに変換。

    Args:
        relative_path: MIXSEEK_WORKSPACEからの相対パス。

    Returns:
        絶対パス。
    """
    workspace = _get_workspace_path()
    return workspace / relative_path


def list_available_data(ctx: RunContext[LocalCodeExecutorConfig]) -> str:
    """設定されたパス内の利用可能なデータファイル一覧を取得。

    Returns:
        Markdown形式のファイル・ディレクトリ一覧。
    """
    config = ctx.deps
    result_lines: list[str] = ["# 利用可能なデータ\n"]

    if not config.available_data_paths:
        return "データパスが設定されていません。"

    for relative_path in config.available_data_paths:
        full_path = _resolve_data_path(relative_path)
        result_lines.append(f"## {relative_path}\n")

        if not full_path.exists():
            result_lines.append("- *パスが存在しません*\n")
            continue

        if full_path.is_file():
            size = full_path.stat().st_size
            result_lines.append(f"- `{full_path.name}` ({size:,} bytes)\n")
        elif full_path.is_dir():
            try:
                entries = sorted(full_path.iterdir())
                if not entries:
                    result_lines.append("- *空のディレクトリ*\n")
                else:
                    for entry in entries[:50]:  # 50件に制限
                        if entry.is_dir():
                            result_lines.append(f"- `{entry.name}/` (ディレクトリ)\n")
                        else:
                            size = entry.stat().st_size
                            result_lines.append(f"- `{entry.name}` ({size:,} bytes)\n")
                    if len(entries) > 50:
                        result_lines.append(f"- ... 他{len(entries) - 50}件\n")
            except PermissionError:
                result_lines.append("- *アクセス権限がありません*\n")

    return "".join(result_lines)


def list_preloaded_modules(ctx: RunContext[LocalCodeExecutorConfig]) -> str:
    """実行環境でプレimport済みのモジュール一覧を取得。

    Returns:
        プレimport済みモジュールの一覧（使用例付き）。
    """
    # configは使用しない
    _ = ctx
    result_lines: list[str] = ["# プレimport済みモジュール\n\n"]

    if not DEFAULT_PRELOADED_MODULES:
        return "プレimportされたモジュールはありません。"

    result_lines.append("以下のモジュールがコード実行時に自動的にimportされます:\n\n")

    for module in DEFAULT_PRELOADED_MODULES:
        if module.alias:
            result_lines.append(f"- `{module.name}` as `{module.alias}`\n")
        else:
            result_lines.append(f"- `{module.name}`\n")

    return "".join(result_lines)


def _build_wrapper_script(
    user_code: str,
    preloaded_modules: list[PreloadedModule],
    available_paths: list[str],
) -> str:
    """実行環境をセットアップするラッパースクリプトを生成。

    Args:
        user_code: 実行するユーザーのPythonコード。
        preloaded_modules: プレimportするモジュールリスト。
        available_paths: 利用可能なデータパスのリスト（文字列）。

    Returns:
        セットアップとユーザーコードを含む完全なPythonスクリプト。
    """
    # ユーザーコードをスクリプトに埋め込むためにエスケープ
    escaped_code = user_code.replace("\\", "\\\\").replace('"""', r"\"\"\"")

    # import文を構築
    import_lines = []
    for module in preloaded_modules:
        module_name = module.name
        # aliasが指定されていればそれを使用、なければモジュール名をそのまま使用
        alias = module.alias if module.alias else module_name
        import_block = (
            f"try:\n"
            f"    import {module_name} as {alias}\n"
            f"except ImportError as e:\n"
            f"    print(f'警告: {module_name}のimportに失敗: {{e}}')"
        )
        import_lines.append(import_block)

    imports_code = "\n".join(import_lines)

    # ラッパースクリプトを構築（テンプレート使用）
    template = textwrap.dedent("""\
        # -*- coding: utf-8 -*-
        # モジュールをプレimport
        {imports_code}

        # ユーザーコードを実行
        try:
            exec(\"\"\"{escaped_code}\"\"\")
        except Exception:
            import traceback
            traceback.print_exc()
        """)
    return template.format(
        imports_code=imports_code,
        available_paths=available_paths,
        escaped_code=escaped_code,
    )


async def read_script(ctx: RunContext[LocalCodeExecutorConfig], file_name: str) -> str:
    """DuckDBからPythonスクリプトの内容を読み込む。

    Args:
        file_name: 読み込むファイル名（.py拡張子を含めること）。

    Returns:
        ファイルの内容（文字列）。

    Raises:
        ValueError: implementation_contextが設定されていない場合。
        FileNotFoundError: ファイルが存在しない場合。
    """
    config = ctx.deps
    if config.implementation_context is None:
        raise ValueError("implementation_contextが設定されていません")

    impl_ctx = config.implementation_context

    # シングルトンストアを取得し、非同期版で読み込み
    store = get_implementation_store()
    code = await store.read_script(
        execution_id=impl_ctx.execution_id,
        team_id=impl_ctx.team_id,
        round_number=impl_ctx.round_number,
        file_name=file_name,
    )

    if code is None:
        raise FileNotFoundError(f"スクリプトが見つかりません: {file_name}")

    return code


async def list_scripts(ctx: RunContext[LocalCodeExecutorConfig]) -> str:
    """DuckDBに保存されているスクリプト一覧を取得。

    Returns:
        Markdown形式のスクリプト一覧。

    Raises:
        ValueError: implementation_contextが設定されていない場合。
    """
    config = ctx.deps
    if config.implementation_context is None:
        raise ValueError("implementation_contextが設定されていません")

    impl_ctx = config.implementation_context

    # シングルトンストアを取得し、非同期版で一覧取得
    store = get_implementation_store()
    scripts = await store.list_scripts(
        execution_id=impl_ctx.execution_id,
        team_id=impl_ctx.team_id,
        round_number=impl_ctx.round_number,
    )

    if not scripts:
        return "保存されているスクリプトはありません。"

    result_lines: list[str] = ["# 保存済みスクリプト\n\n"]
    for script in scripts:
        result_lines.append(f"- `{script['file_name']}` ({script['created_at']})\n")

    return "".join(result_lines)


def execute_python_code(ctx: RunContext[LocalCodeExecutorConfig], code: str) -> str:
    """サブプロセスでPythonコードを実行し、stdout/stderr出力を返す。

    同一プロセス内で複数エージェントが並列実行される場合のスレッド安全性を確保するため、
    subprocessによるプロセス分離を使用。

    プレimportされたモジュールとAVAILABLE_DATA_PATHSが実行環境で利用可能。

    Args:
        code: 実行するPythonコード文字列。

    Returns:
        実行結果のstdoutおよびstderr出力。
    """
    config = ctx.deps

    # すべての利用可能パスを解決
    available_paths = [str(_resolve_data_path(p)) for p in config.available_data_paths]

    # ラッパースクリプトを生成（共通モジュールを使用）
    wrapper_script = _build_wrapper_script(code, DEFAULT_PRELOADED_MODULES, available_paths)

    # エラーメッセージ改善のため一時ファイルに書き出し
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(wrapper_script)
        script_path = f.name

    try:
        # セキュリティ対策: 必要最小限の環境変数のみを渡す
        minimal_env = {
            "PATH": os.environ.get("PATH", ""),
        }

        # サブプロセスで実行
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=config.timeout_seconds,
            cwd=str(_get_workspace_path()),
            env=minimal_env,
        )

        # stdoutとstderrを結合
        output = result.stdout
        if result.stderr:
            output += result.stderr

    except subprocess.TimeoutExpired:
        return f"エラー: 実行が{config.timeout_seconds}秒でタイムアウトしました"
    except Exception as e:
        return f"エラー: コード実行に失敗しました: {e}"
    finally:
        # 一時ファイルをクリーンアップ
        try:
            os.unlink(script_path)
        except OSError:
            pass

    # max_output_charsが設定されている場合は出力を切り詰め
    if config.max_output_chars is not None and len(output) > config.max_output_chars:
        output = output[: config.max_output_chars] + f"\n... （{config.max_output_chars}文字を超えたため切り詰め）"

    return output if output else "（出力なし）"


# FunctionToolsetを作成
local_code_executor_toolset = FunctionToolset(
    tools=[
        list_available_data,
        list_preloaded_modules,
        execute_python_code,
        read_script,
    ]
)


__all__ = [
    "local_code_executor_toolset",
    "list_available_data",
    "list_preloaded_modules",
    "execute_python_code",
    "read_script",
]

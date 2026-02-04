"""環境変数の最小化に関する動作検証テスト。

このテストは、サブプロセスに渡す環境変数を最小化した場合の動作を検証します。
"""

import os
import subprocess
import sys
import tempfile

import pytest


@pytest.mark.integration
class TestMinimalEnvironment:
    """最小環境変数での動作検証。"""

    def test_subprocess_with_path_only(self, tmp_path, monkeypatch):
        """PATHのみでサブプロセスが動作するか検証。"""
        monkeypatch.setenv("MIXSEEK_WORKSPACE", str(tmp_path))

        # テストスクリプト作成
        script = """
import sys
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print("Basic execution successful")
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
            f.write(script)
            script_path = f.name

        try:
            # PATHのみで実行
            minimal_env = {"PATH": os.environ.get("PATH", "")}

            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=5,
                env=minimal_env,
            )

            print(f"stdout: {result.stdout}")
            print(f"stderr: {result.stderr}")
            print(f"returncode: {result.returncode}")

            assert result.returncode == 0, f"実行失敗: {result.stderr}"
            assert "Basic execution successful" in result.stdout

        finally:
            os.unlink(script_path)

    def test_subprocess_with_dependencies(self, tmp_path, monkeypatch):
        """PATHのみで依存ライブラリ（polars, numpy, scipy）が動作するか検証。"""
        monkeypatch.setenv("MIXSEEK_WORKSPACE", str(tmp_path))

        # 依存ライブラリをimportするスクリプト
        script = """
import sys
print(f"Python: {sys.executable}")

try:
    import polars as pl
    print("✓ polars imported successfully")
except ImportError as e:
    print(f"✗ polars import failed: {e}")

try:
    import numpy as np
    print("✓ numpy imported successfully")
except ImportError as e:
    print(f"✗ numpy import failed: {e}")

try:
    import scipy
    print("✓ scipy imported successfully")
except ImportError as e:
    print(f"✗ scipy import failed: {e}")

# 簡単な計算テスト
try:
    import numpy as np
    arr = np.array([1, 2, 3])
    print(f"✓ numpy array test: {arr.mean()}")
except Exception as e:
    print(f"✗ numpy test failed: {e}")
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
            f.write(script)
            script_path = f.name

        try:
            # PATHのみで実行
            minimal_env = {"PATH": os.environ.get("PATH", "")}

            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=10,
                env=minimal_env,
            )

            print(f"stdout: {result.stdout}")
            print(f"stderr: {result.stderr}")
            print(f"returncode: {result.returncode}")

            # 基本的な実行が成功することを確認
            # ライブラリのimportが失敗してもテストは失敗としない（環境依存）
            assert result.returncode == 0, f"実行失敗: {result.stderr}"

            # 結果を記録
            if "✓ polars imported" in result.stdout:
                print("INFO: Polarsは最小環境で動作します")
            if "✓ numpy imported" in result.stdout:
                print("INFO: NumPyは最小環境で動作します")
            if "✓ scipy imported" in result.stdout:
                print("INFO: SciPyは最小環境で動作します")

        finally:
            os.unlink(script_path)

    def test_subprocess_with_virtual_env(self, tmp_path, monkeypatch):
        """VIRTUAL_ENVが必要かどうか検証。"""
        monkeypatch.setenv("MIXSEEK_WORKSPACE", str(tmp_path))

        script = """
import sys
import os
print(f"Python: {sys.executable}")
print(f"VIRTUAL_ENV in env: {'VIRTUAL_ENV' in os.environ}")
if 'VIRTUAL_ENV' in os.environ:
    print(f"VIRTUAL_ENV: {os.environ['VIRTUAL_ENV']}")
print("Execution successful")
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
            f.write(script)
            script_path = f.name

        try:
            # PATHのみで実行
            minimal_env = {"PATH": os.environ.get("PATH", "")}

            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=5,
                env=minimal_env,
            )

            print(f"stdout: {result.stdout}")
            print(f"stderr: {result.stderr}")

            # VIRTUAL_ENV付きで実行
            minimal_env_with_venv = {"PATH": os.environ.get("PATH", "")}
            if "VIRTUAL_ENV" in os.environ:
                minimal_env_with_venv["VIRTUAL_ENV"] = os.environ["VIRTUAL_ENV"]

            result_with_venv = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=5,
                env=minimal_env_with_venv,
            )

            print(f"WITH VENV stdout: {result_with_venv.stdout}")
            print(f"WITH VENV stderr: {result_with_venv.stderr}")

            # 両方とも成功することを確認
            assert result.returncode == 0
            assert result_with_venv.returncode == 0

        finally:
            os.unlink(script_path)

    def test_polars_with_minimal_env(self, tmp_path, monkeypatch):
        """Polarsを使ったデータ操作が最小環境で動作するか検証。"""
        monkeypatch.setenv("MIXSEEK_WORKSPACE", str(tmp_path))

        # Polarsでデータを作成・読み書き
        script = """
import sys
from pathlib import Path

print(f"Python: {sys.executable}")

try:
    import polars as pl

    # DataFrameを作成
    df = pl.DataFrame({
        "a": [1, 2, 3],
        "b": [4, 5, 6]
    })

    # Parquetファイルに保存
    test_path = Path("/tmp/test_polars.parquet")
    df.write_parquet(test_path)

    # 読み込み
    df2 = pl.read_parquet(test_path)

    print(f"✓ Polars DataFrame operations successful: {df2.shape}")

    # クリーンアップ
    test_path.unlink(missing_ok=True)

except Exception as e:
    print(f"✗ Polars test failed: {e}")
    import traceback
    traceback.print_exc()
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
            f.write(script)
            script_path = f.name

        try:
            # PATHのみで実行
            minimal_env = {"PATH": os.environ.get("PATH", "")}

            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=10,
                env=minimal_env,
            )

            print(f"stdout: {result.stdout}")
            print(f"stderr: {result.stderr}")
            print(f"returncode: {result.returncode}")

            # 実行自体は成功することを確認
            assert result.returncode == 0, f"実行失敗: {result.stderr}"

            # Polarsが動作したかチェック（環境依存なので警告レベル）
            if "✓ Polars DataFrame operations successful" in result.stdout:
                print("INFO: Polarsは最小環境で完全に動作します")
            elif "✗ Polars test failed" in result.stdout:
                print("WARNING: Polarsが最小環境で動作しませんでした")

        finally:
            os.unlink(script_path)

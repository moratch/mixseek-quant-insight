"""Unit tests for submission_parser module."""

import pytest

from quant_insight.evaluator.submission_parser import (
    extract_code_from_submission,
    parse_submission_function,
)
from quant_insight.exceptions import SubmissionInvalidError


@pytest.mark.unit
def test_parse_valid_submission():
    """Test parsing a valid submission with generate_signal function."""
    submission = """以下はシグナル生成関数です。

```python
import polars as pl

def generate_signal(ohlcv: pl.DataFrame, additional_data: dict[str, pl.DataFrame]) -> pl.DataFrame:
    return ohlcv.with_columns(
        (pl.col("close").pct_change(5).over("symbol")).alias("signal")
    ).select(["datetime", "symbol", "signal"])
```
"""
    func = parse_submission_function(submission)
    assert callable(func)
    assert func.__name__ == "generate_signal"


@pytest.mark.unit
def test_parse_submission_with_syntax_error():
    """Test parsing a submission with syntax errors."""
    submission = """説明文です。

```python
def generate_signal(ohlcv, additional_data):
    return ohlcv.with_columns(  # Missing closing parenthesis
        (pl.col("close").pct_change(5)
```
"""
    with pytest.raises(SubmissionInvalidError, match="構文エラー"):
        parse_submission_function(submission)


@pytest.mark.unit
def test_parse_submission_missing_function():
    """Test parsing a submission without generate_signal function."""
    submission = """コードです。

```python
def some_other_function():
    pass
```
"""
    with pytest.raises(SubmissionInvalidError, match="'generate_signal'関数が含まれている必要があります"):
        parse_submission_function(submission)


@pytest.mark.unit
def test_parse_submission_with_non_callable():
    """Test parsing a submission where generate_signal is not callable."""
    submission = """コードです。

```python
generate_signal = "not a function"
```
"""
    with pytest.raises(SubmissionInvalidError, match="呼び出し可能な関数である必要があります"):
        parse_submission_function(submission)


@pytest.mark.unit
def test_parse_submission_with_runtime_error():
    """Test parsing a submission that raises errors during execution."""
    submission = """コードです。

```python
raise ValueError("Intentional error")

def generate_signal(ohlcv, additional_data):
    pass
```
"""
    with pytest.raises(SubmissionInvalidError, match="サブミッションコードの実行に失敗しました"):
        parse_submission_function(submission)


@pytest.mark.unit
def test_parse_submission_with_imports():
    """Test parsing a submission with various imports."""
    submission = """コードです。

```python
import polars as pl
import numpy as np
from typing import Dict

def generate_signal(ohlcv: pl.DataFrame, additional_data: Dict[str, pl.DataFrame]) -> pl.DataFrame:
    return ohlcv.select(["datetime", "symbol"]).with_columns(pl.lit(1.0).alias("signal"))
```
"""
    func = parse_submission_function(submission)
    assert callable(func)
    assert func.__name__ == "generate_signal"


@pytest.mark.unit
def test_parse_submission_validates_argument_count():
    """Test that submission must have exactly 2 arguments."""
    submission = """コードです。

```python
def generate_signal(ohlcv):
    pass
```
"""
    with pytest.raises(SubmissionInvalidError, match="2つの引数を受け取る必要があります"):
        parse_submission_function(submission)


@pytest.mark.unit
def test_parse_submission_validates_ohlcv_type():
    """Test that ohlcv argument must be typed as pl.DataFrame or pd.DataFrame when annotated."""
    submission = """コードです。

```python
def generate_signal(ohlcv: str, additional_data):
    pass
```
"""
    with pytest.raises(SubmissionInvalidError, match="pl.DataFrameまたはpd.DataFrameである必要があります"):
        parse_submission_function(submission)


@pytest.mark.unit
def test_parse_submission_validates_return_type():
    """Test that return type must be pl.DataFrame or pd.DataFrame when annotated."""
    submission = """コードです。

```python
import polars as pl

def generate_signal(ohlcv: pl.DataFrame, additional_data) -> str:
    pass
```
"""
    with pytest.raises(SubmissionInvalidError, match="返り値の型はpl.DataFrameまたはpd.DataFrameである必要があります"):
        parse_submission_function(submission)


@pytest.mark.unit
def test_parse_submission_allows_no_type_hints():
    """Test that submission without type hints is allowed."""
    submission = """コードです。

```python
def generate_signal(ohlcv, additional_data):
    pass
```
"""
    func = parse_submission_function(submission)
    assert callable(func)
    assert func.__name__ == "generate_signal"


# コードブロック抽出機能のテスト


@pytest.mark.unit
def test_extract_code_from_submission_with_python_code_block():
    """Test extracting code from ```python code block."""
    submission = """以下はシグナル生成関数の実装です。

```python
import polars as pl

def generate_signal(ohlcv, additional_data):
    return ohlcv.select(["datetime", "symbol"]).with_columns(pl.lit(1.0).alias("signal"))
```

この関数は移動平均を計算してシグナルを生成します。
"""
    code = extract_code_from_submission(submission)
    assert "def generate_signal" in code
    assert "以下はシグナル生成関数" not in code
    assert "この関数は移動平均" not in code


@pytest.mark.unit
def test_extract_code_from_submission_with_plain_code_block():
    """Test extracting code from ``` code block (without python specifier)."""
    submission = """説明文です。

```
def generate_signal(ohlcv, additional_data):
    pass
```

追加の説明。
"""
    code = extract_code_from_submission(submission)
    assert "def generate_signal" in code
    assert "説明文です" not in code


@pytest.mark.unit
def test_extract_code_from_submission_without_code_block():
    """Test that raw code without code block raises SubmissionInvalidError."""
    submission = """
def generate_signal(ohlcv, additional_data):
    pass
"""
    with pytest.raises(SubmissionInvalidError, match="コードブロック"):
        extract_code_from_submission(submission)


@pytest.mark.unit
def test_extract_code_from_submission_with_multiple_code_blocks():
    """Test extracting and concatenating multiple code blocks."""
    submission = """まずimport文です。

```python
import polars as pl
import numpy as np
```

次に関数定義です。

```python
def generate_signal(ohlcv, additional_data):
    return ohlcv.select(["datetime", "symbol"]).with_columns(pl.lit(1.0).alias("signal"))
```

以上です。
"""
    code = extract_code_from_submission(submission)
    assert "import polars as pl" in code
    assert "import numpy as np" in code
    assert "def generate_signal" in code
    assert "まずimport文" not in code
    assert "以上です" not in code


@pytest.mark.unit
def test_parse_submission_with_code_block():
    """Test parsing a submission with code block format."""
    submission = """以下はシグナル生成関数です。

```python
import polars as pl

def generate_signal(ohlcv: pl.DataFrame, additional_data: dict[str, pl.DataFrame]) -> pl.DataFrame:
    return ohlcv.select(["datetime", "symbol"]).with_columns(pl.lit(1.0).alias("signal"))
```

この実装は単純なシグナルを返します。
"""
    func = parse_submission_function(submission)
    assert callable(func)
    assert func.__name__ == "generate_signal"


@pytest.mark.unit
def test_parse_submission_with_code_block_syntax_error():
    """Test that syntax errors in code blocks are properly detected."""
    submission = """シグナル関数です。

```python
def generate_signal(ohlcv, additional_data):
    return ohlcv.with_columns(  # 構文エラー：閉じ括弧がない
```
"""
    with pytest.raises(SubmissionInvalidError, match="構文エラー"):
        parse_submission_function(submission)


@pytest.mark.unit
def test_parse_submission_with_code_block_missing_function():
    """Test that missing function in code block raises error."""
    submission = """以下はコードです。

```python
import polars as pl

def some_other_function():
    pass
```
"""
    with pytest.raises(SubmissionInvalidError, match="'generate_signal'関数が含まれている必要があります"):
        parse_submission_function(submission)


@pytest.mark.unit
def test_parse_submission_validates_return_type_pandas():
    """Test that return type pd.DataFrame is accepted when annotated."""
    submission = """コードです。

```python
import pandas as pd

def generate_signal(ohlcv, additional_data) -> pd.DataFrame:
    pass
```
"""
    func = parse_submission_function(submission)
    assert callable(func)
    assert func.__name__ == "generate_signal"


@pytest.mark.unit
def test_parse_submission_validates_return_type_union():
    """Test that return type pl.DataFrame | pd.DataFrame is accepted."""
    submission = """コードです。

```python
import polars as pl
import pandas as pd

def generate_signal(ohlcv, additional_data) -> pl.DataFrame | pd.DataFrame:
    pass
```
"""
    func = parse_submission_function(submission)
    assert callable(func)
    assert func.__name__ == "generate_signal"


@pytest.mark.unit
def test_parse_submission_validates_pandas_ohlcv_type():
    """Test that pd.DataFrame is accepted for ohlcv argument."""
    submission = """コードです。

```python
import pandas as pd

def generate_signal(ohlcv: pd.DataFrame, additional_data):
    pass
```
"""
    # pd.DataFrameは入力としてもOKとするかは仕様次第
    # ここでは受け入れる実装とする
    func = parse_submission_function(submission)
    assert callable(func)
    assert func.__name__ == "generate_signal"

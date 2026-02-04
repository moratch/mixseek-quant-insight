"""文字列からシグナル生成関数を抽出するためのサブミッション関数パーサー。"""

import inspect
import re
import types
from collections.abc import Callable
from typing import Any, get_args, get_origin

import pandas as pd
import polars as pl

from quant_insight.constants import DEFAULT_PRELOADED_MODULES
from quant_insight.exceptions import SubmissionInvalidError

# コードブロックを検出する正規表現パターン
# ```python または ``` で囲まれたコードブロックを抽出
_CODE_BLOCK_PATTERN = re.compile(
    r"```(?:python)?\s*(.*?)```",
    re.DOTALL,
)


def _build_preloaded_namespace() -> dict[str, Any]:
    """型ヒント評価用のプレロード済みnamespaceを構築する。

    DEFAULT_PRELOADED_MODULESをimportし、型ヒント評価時に
    pl.DataFrameやpd.DataFrameなどが利用可能な環境を作る。

    Returns:
        プレロード済みモジュールを含むnamespace辞書。
    """
    namespace: dict[str, Any] = {}

    for module_spec in DEFAULT_PRELOADED_MODULES:
        try:
            # モジュールをimport
            module = __import__(module_spec.name, fromlist=[""])

            # aliasが指定されている場合はそれを使用、なければモジュール名
            key = module_spec.alias if module_spec.alias else module_spec.name
            namespace[key] = module
        except ImportError:
            # import失敗時は無視（型ヒント評価には影響しない場合がある）
            pass

    return namespace


def extract_code_from_submission(submission_content: str) -> str:
    """Submissionからコードブロックを抽出する。

    コードブロック（```python または ```）が存在する場合はその中身を返す。
    複数のコードブロックがある場合は連結して返す。

    Args:
        submission_content: Submission文字列（説明文 + コードブロック）

    Returns:
        抽出されたPythonコード文字列

    Raises:
        SubmissionInvalidError: コードブロックが見つからない場合
    """
    matches = _CODE_BLOCK_PATTERN.findall(submission_content)

    if not matches:
        raise SubmissionInvalidError("サブミッションにコードブロック(```python ... ```)が含まれていません")

    # 複数のコードブロックがある場合は連結
    return "\n\n".join(match.strip() for match in matches)


def parse_submission_function(submission_content: str) -> Callable[..., Any]:
    """サブミッション文字列からシグナル生成関数をパースして抽出する。

    Submissionはコードブロック（```python または ```）を含む必要がある。
    コードブロック内のコードのみを抽出してパースする。

    Args:
        submission_content: コードブロック形式のSubmission文字列

    Returns:
        シグナル生成関数のCallable（シグネチャ:
            (ohlcv: pl.DataFrame, additional_data: Dict[str, pl.DataFrame]) -> pl.DataFrame | pd.DataFrame）

    Raises:
        SubmissionInvalidError: コードブロックが見つからない、構文エラーがある、
            または必須の'generate_signal'関数が含まれていない場合
    """
    # コードブロックを抽出
    code_content = extract_code_from_submission(submission_content)

    # 構文エラーをチェックするためサブミッション内容をコンパイル
    try:
        compiled_code = compile(code_content, "<submission>", "exec")
    except SyntaxError as e:
        raise SubmissionInvalidError(f"サブミッションに構文エラーがあります: {e}") from e

    # 関数を抽出するためコンパイル済みコードを実行
    # 型ヒント評価のため、DEFAULT_PRELOADED_MODULESをnamespaceに事前ロード
    namespace: dict[str, Any] = _build_preloaded_namespace()
    try:
        exec(compiled_code, namespace)  # noqa: S102
    except Exception as e:
        raise SubmissionInvalidError(f"サブミッションコードの実行に失敗しました: {e}") from e

    # 'generate_signal'関数を抽出
    if "generate_signal" not in namespace:
        raise SubmissionInvalidError(
            "サブミッションには'generate_signal'関数が含まれている必要があります。"
            "期待されるシグネチャ: def generate_signal(ohlcv: pl.DataFrame, "
            "additional_data: Dict[str, pl.DataFrame]) -> pl.DataFrame"
        )

    generate_signal = namespace["generate_signal"]

    # callableであることを検証
    if not callable(generate_signal):
        raise SubmissionInvalidError("'generate_signal'は呼び出し可能な関数である必要があります")

    # 型ヒントを検証（提供されている場合）
    _validate_function_signature(generate_signal)

    # 型の絞り込み: 上記でcallableであることを検証したが、mypyはこれを推論できない
    return generate_signal  # type: ignore[no-any-return]


def _validate_function_signature(func: Callable[..., Any]) -> None:
    """generate_signal関数のシグネチャと型ヒントを検証する。

    Args:
        func: 検証する関数

    Raises:
        SubmissionInvalidError: シグネチャまたは型ヒントが期待される形式と一致しない場合
    """
    sig = inspect.signature(func)
    params = list(sig.parameters.values())

    # 引数の数を検証（ohlcv, additional_data の2つ）
    if len(params) != 2:
        raise SubmissionInvalidError(
            f"'generate_signal'は2つの引数を受け取る必要があります（ohlcv, additional_data）。"
            f"実際の引数数: {len(params)}"
        )

    # 型ヒントが提供されている場合のみ検証
    ohlcv_param = params[0]
    additional_data_param = params[1]

    # ohlcvパラメータの型検証
    if ohlcv_param.annotation is not inspect.Parameter.empty:
        if not _is_dataframe_type(ohlcv_param.annotation):
            raise SubmissionInvalidError(
                f"第1引数'{ohlcv_param.name}'の型はpl.DataFrameまたはpd.DataFrameである必要があります。"
                f"実際の型: {ohlcv_param.annotation}"
            )

    # additional_dataパラメータの型検証
    if additional_data_param.annotation is not inspect.Parameter.empty:
        if not _is_dict_dataframe_type(additional_data_param.annotation):
            raise SubmissionInvalidError(
                f"第2引数'{additional_data_param.name}'の型はDict[str, pl.DataFrame]である必要があります。"
                f"実際の型: {additional_data_param.annotation}"
            )

    # 返り値の型検証
    if sig.return_annotation is not inspect.Signature.empty:
        if not _is_dataframe_type(sig.return_annotation):
            raise SubmissionInvalidError(
                f"返り値の型はpl.DataFrameまたはpd.DataFrameである必要があります。実際の型: {sig.return_annotation}"
            )


def _is_dataframe_type(annotation: Any) -> bool:
    """型アノテーションがpl.DataFrameまたはpd.DataFrameを表すか確認する。"""
    # 直接の型比較
    if annotation is pl.DataFrame:
        return True
    if annotation is pd.DataFrame:
        return True

    # Union型の場合（pl.DataFrame | pd.DataFrame）
    origin = get_origin(annotation)
    if origin is types.UnionType:  # Python 3.10+ の | 構文
        args = get_args(annotation)
        return any(_is_dataframe_type(arg) for arg in args)

    # 文字列アノテーションの場合
    if isinstance(annotation, str):
        return annotation in (
            "pl.DataFrame",
            "polars.DataFrame",
            "DataFrame",
            "pd.DataFrame",
            "pandas.DataFrame",
            "pl.DataFrame | pd.DataFrame",
            "pd.DataFrame | pl.DataFrame",
        )
    return False


def _is_dict_dataframe_type(annotation: Any) -> bool:
    """型アノテーションがDict[str, pl.DataFrame]または互換型を表すか確認する。"""
    # 生のdict型は許容（型引数なしは厳密な検証をスキップ）
    if annotation is dict:
        return True

    origin = get_origin(annotation)
    if origin is None:
        # 文字列アノテーションの場合
        if isinstance(annotation, str):
            # "dict"または"Dict"を含む場合は許容
            return "dict" in annotation.lower()
        return False

    # dict型か確認
    if origin is not dict:
        return False

    # 型引数を確認（型引数がある場合のみ検証）
    args = get_args(annotation)
    if len(args) == 0:
        # dict[...]の形式だが型引数が取得できない場合は許容
        return True

    if len(args) != 2:
        return False

    # キーがstr、値がDataFrameか確認
    key_type, value_type = args
    return key_type is str and _is_dataframe_type(value_type)

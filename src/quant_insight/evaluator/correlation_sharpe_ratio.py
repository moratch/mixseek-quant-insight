"""バックテストベースのシグナル評価のための相関シャープレシオevaluator。"""

import traceback
from pathlib import Path

import polars as pl
from mixseek.evaluator.exceptions import EvaluatorAPIError
from mixseek.evaluator.metrics.base import BaseMetric
from mixseek.models.evaluation_result import MetricScore

from quant_insight.evaluator.backtest_loop import BacktestLoop
from quant_insight.evaluator.submission_parser import parse_submission_function
from quant_insight.exceptions import SubmissionFailedError, SubmissionInvalidError
from quant_insight.utils.config import get_test_data_paths


class CorrelationSharpeRatio(BaseMetric):
    """Spearman順位相関とシャープレシオを使用したバックテストevaluator。

    このメトリックはシグナル生成関数を以下の手順で評価する:
    1. サブミッション文字列をパースしてgenerate_signal関数を抽出
    2. Time Series APIスタイルのバックテストループを実行
    3. 各イテレーションでSpearman順位相関を計算
    4. 相関系列からシャープレシオを計算
    5. シャープレシオをメトリックスコアとして返す（正規化なし）

    このメトリックはFR-013で指定されているとおり、テストデータのみにアクセスする。

    データパスはMIXSEEK_WORKSPACE環境変数から自動的に解決される:
    - ohlcv: $MIXSEEK_WORKSPACE/data/inputs/ohlcv/test.parquet
    - returns: $MIXSEEK_WORKSPACE/data/inputs/returns/test.parquet
    - additional_data: $MIXSEEK_WORKSPACE/data/inputs/{name}/test.parquet
      （competition.tomlで定義された追加データセット）

    Note:
        mixseek-coreのEvaluator.evaluate()はBaseMetricに対してkwargsを渡さないため、
        データパスはコンストラクタ内でget_test_data_paths()を呼び出して取得する。
    """

    # Submission起因エラー時のスコア
    # この値は評価結果が無効であることを示す
    INVALID_SUBMISSION_SCORE = -100.0

    def __init__(self) -> None:
        """CorrelationSharpeRatioメトリックを初期化する。

        MIXSEEK_WORKSPACE環境変数からテストデータパスを解決する。
        competition.tomlが存在する場合は追加データセットも自動的に検出する。

        Note:
            MIXSEEK_WORKSPACE未設定の場合、evaluate()呼び出し時にエラーとなる。
        """
        # データパスは遅延初期化（evaluate時にMIXSEEK_WORKSPACEが設定されている必要がある）
        self._ohlcv_path: Path | None = None
        self._returns_path: Path | None = None
        self._additional_data_paths: dict[str, Path] | None = None

    def _format_exception_with_traceback(self, exc: BaseException) -> str:
        """例外からtraceback情報を含む詳細文字列を生成する。

        Args:
            exc: 例外オブジェクト

        Returns:
            traceback情報を含む文字列
        """
        lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
        return "".join(lines)

    def _create_submission_failed_result(self, error_msg: str, exc: BaseException | None = None) -> MetricScore:
        """Submission起因のエラー時に返すMetricScoreを生成する。

        Args:
            error_msg: エラーメッセージ
            exc: 例外オブジェクト（tracebackを含める場合）

        Returns:
            無効スコアと説明的なコメントを含むMetricScore
        """
        comment = f"[無効な評価結果] Submissionは失敗しました。エラー内容を確認して修正してください: {error_msg}"

        if exc is not None:
            tb_str = self._format_exception_with_traceback(exc)
            comment += f"\n\n詳細トレースバック:\n{tb_str}"

        return MetricScore(
            metric_name=self.__class__.__name__,
            score=self.INVALID_SUBMISSION_SCORE,
            evaluator_comment=comment,
        )

    async def evaluate(
        self,
        user_query: str,
        submission: str,
        **_kwargs: object,
    ) -> MetricScore:
        """バックテストを使用してシグナル生成関数を評価する。

        Args:
            user_query: ユーザークエリ（バックテスト評価では無視される）
            submission: Python関数定義の文字列
            **_kwargs: 未使用（mixseek-coreとの互換性のため保持）

        Returns:
            MetricScore（以下を含む）:
                - metric_name: "CorrelationSharpeRatio"
                - score: シャープレシオ（正規化なし、負の値も可）
                - evaluator_comment: イテレーション数と平均相関を含む統計

        Note:
            - user_queryはバックテスト評価で使用されないため無視される
            - データパスはMIXSEEK_WORKSPACE環境変数から自動的に解決される
            - competition.tomlが存在する場合は追加データセットも自動的に検出される

        Raises:
            EvaluatorAPIError: システム側のエラー（データロード失敗など）が発生した場合
        """
        # user_queryはバックテスト評価では使用しない
        _ = user_query

        # サブミッションをパースしてgenerate_signal関数を抽出
        # パースエラーはSubmission起因
        try:
            signal_func = parse_submission_function(submission)
        except SubmissionInvalidError as e:
            return self._create_submission_failed_result(f"サブミッションのパースエラー: {e}", exc=e)

        # テストデータをロード
        # データロードエラーはシステム起因 → EvaluatorAPIErrorをraise
        try:
            ohlcv, returns, additional_data = self._load_test_data()
        except Exception as e:
            raise EvaluatorAPIError(
                f"テストデータのロードに失敗しました: {e}",
                metric_name=self.__class__.__name__,
            ) from e

        # バックテストループを実行
        try:
            backtest_loop = BacktestLoop(ohlcv, returns, additional_data)
            result = backtest_loop.run(signal_func)
        except (SubmissionFailedError, SubmissionInvalidError) as e:
            # Submission起因のエラー
            return self._create_submission_failed_result(f"バックテスト実行エラー: {e}", exc=e)
        except EvaluatorAPIError:
            # System起因のエラーはそのまま伝播
            raise

        # BacktestResultをMetricScoreに変換
        if result.status == "failed":
            # バックテスト失敗はSubmission起因
            return self._create_submission_failed_result(f"バックテスト失敗: {result.error_message}")

        # 成功: sharpe_ratioをスコアとして使用（正規化なし）
        comment = (
            f"バックテスト完了。"
            f"総イテレーション数: {result.total_iterations}, "
            f"有効イテレーション数: {result.valid_iterations}"
        )
        if result.mean_correlation is not None:
            comment += f", 平均相関: {result.mean_correlation:.4f}"
        if result.std_correlation is not None:
            comment += f", 相関標準偏差: {result.std_correlation:.4f}"

        return MetricScore(
            metric_name=self.__class__.__name__,
            score=result.sharpe_ratio,
            evaluator_comment=comment,
        )

    def _ensure_data_paths_initialized(self) -> None:
        """データパスが初期化されていることを確認する。

        遅延初期化: 最初のevaluate()呼び出し時にデータパスを解決する。

        Raises:
            ValueError: MIXSEEK_WORKSPACE環境変数が設定されていない場合
        """
        if self._ohlcv_path is None:
            ohlcv_path, returns_path, additional_paths = get_test_data_paths()
            self._ohlcv_path = ohlcv_path
            self._returns_path = returns_path
            self._additional_data_paths = additional_paths

    def _load_test_data(self) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, pl.DataFrame]]:
        """設定されたパスからテストデータをロードする。

        データパスはMIXSEEK_WORKSPACE環境変数から自動的に解決される:
        - ohlcv: $MIXSEEK_WORKSPACE/data/inputs/ohlcv/test.parquet
        - returns: $MIXSEEK_WORKSPACE/data/inputs/returns/test.parquet
        - additional_data: competition.tomlで定義された追加データセット

        Returns:
            (ohlcv, returns, additional_data)のタプル

        Raises:
            Exception: データのロードに失敗した場合
        """
        # データパスを確実に初期化
        self._ensure_data_paths_initialized()

        # 型ガード（_ensure_data_paths_initializedで初期化済み）
        assert self._ohlcv_path is not None
        assert self._returns_path is not None
        assert self._additional_data_paths is not None

        # OHLCVとリターンをロード
        ohlcv = pl.read_parquet(self._ohlcv_path)
        returns = pl.read_parquet(self._returns_path)

        # 追加データをロード
        additional_data: dict[str, pl.DataFrame] = {}
        for name, path in self._additional_data_paths.items():
            if path.exists():
                additional_data[name] = pl.read_parquet(path)

        return ohlcv, returns, additional_data

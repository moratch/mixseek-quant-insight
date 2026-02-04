"""Time Series API形式評価のためのバックテストループ実装。"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from mixseek.evaluator.exceptions import EvaluatorAPIError
from scipy.stats import spearmanr

from quant_insight.exceptions import SubmissionFailedError, SubmissionInvalidError
from quant_insight.models.backtest_result import BacktestResult, IterationResult


@dataclass(frozen=True)
class CorrelationStatistics:
    """相関系列から計算される統計情報。"""

    mean: float | None
    std: float | None
    sharpe_ratio: float
    valid_count: int


class BacktestLoop:
    """シグナル評価のためのTime Series APIバックテストループ。

    このクラスはKaggle Time Series APIスタイルのバックテストを実装する:
    1. データは日時順に段階的に公開される
    2. シグナル関数は現在の日時までのデータのみを受け取る
    3. シグナルは実際のリターンに対して評価される
    4. Spearman順位相関が各イテレーションで計算される
    5. 相関のシャープレシオが最終スコアとなる

    ループはOHLCVデータ内のユニークな日時値を反復処理することで、
    任意の時間粒度（日次、時間次など）を自動的に処理する。
    """

    def __init__(
        self,
        ohlcv: pl.DataFrame,
        returns: pl.DataFrame,
        additional_data: dict[str, pl.DataFrame] | None = None,
    ) -> None:
        """バックテストループを初期化する。

        Args:
            ohlcv: OHLCV DataFrame（カラム: datetime, symbol, open, high, low, close, volume）
            returns: リターンDataFrame（カラム: datetime, symbol, return_value）
            additional_data: オプションの追加データセット（各データセットにdatetimeカラムを含む）

        Note:
            datetime型カラムはマイクロ秒精度（μs）に統一される。
            これはpandas（ナノ秒精度）とpolars（マイクロ秒精度）間の
            型不一致によるjoinエラーを防ぐため。
        """
        # datetime精度をマイクロ秒に統一（pandas由来のnsデータ対策）
        self.ohlcv = self._normalize_datetime_precision(ohlcv)
        self.returns = self._normalize_datetime_precision(returns)
        self.additional_data = {
            name: self._normalize_datetime_precision(df) for name, df in (additional_data or {}).items()
        }

    def run(self, signal_func: Callable[..., Any]) -> BacktestResult:
        """バックテストループを実行する。

        Args:
            signal_func: シグナル生成関数（シグネチャ:
                (ohlcv: pl.DataFrame, additional_data: Dict[str, pl.DataFrame]) -> pl.DataFrame | pd.DataFrame
                pd.DataFrameが返された場合、pl.from_pandas()で変換
                出力DataFrameはカラム: datetime, symbol, signal を持つ必要がある）

        Returns:
            イテレーション結果とシャープレシオを含むBacktestResult

        Raises:
            SubmissionFailedError: シグナル関数が例外を発生させた場合（Submission起因）
            SubmissionInvalidError: シグナル形式が不正な場合（Submission起因）
            EvaluatorAPIError: データ処理でシステムエラーが発生した場合（System起因）
        """
        # ユニークなdatetime値を取得（ソート済み） - これがイテレーション粒度を決定する
        # OHLCVデータの問題はシステム起因（データロード側の責務）
        try:
            datetimes = self.ohlcv.select("datetime").unique().sort("datetime")["datetime"].to_list()
        except Exception as e:
            raise EvaluatorAPIError(f"OHLCVデータからdatetime値の抽出に失敗しました: {e}") from e

        iteration_results: list[IterationResult] = []
        evaluation_started_at = datetime.now()

        for i, current_datetime in enumerate(datetimes):
            try:
                # 現在の日時までのデータをフィルタ（Time Series API制約）
                available_ohlcv = self.ohlcv.filter(pl.col("datetime") <= current_datetime)

                # 追加データも同様にフィルタ
                available_additional = {
                    name: df.filter(pl.col("datetime") <= current_datetime)
                    for name, df in self.additional_data.items()
                }

                # シグナルを生成
                try:
                    signal_df = signal_func(available_ohlcv, available_additional)
                except Exception as e:
                    raise SubmissionFailedError(
                        f"イテレーション{i} ({current_datetime})でシグナル関数が例外を発生させました: {e}",
                        iteration=i,
                        datetime=current_datetime,
                    ) from e

                # pd.DataFrameの場合はPolarsに変換
                signal_df = self._normalize_signal_df(signal_df)

                # シグナル形式と型を検証
                self._validate_signal_format(signal_df)

                # 現在の日時のシグナルとリターンを取得
                current_signals = signal_df.filter(pl.col("datetime") == current_datetime)
                current_returns = self.returns.filter(pl.col("datetime") == current_datetime)

                # このイテレーションの順位相関を計算
                rank_correlation = self._calculate_rank_correlation(current_signals, current_returns)

                iteration_results.append(
                    IterationResult(
                        datetime=current_datetime,
                        rank_correlation=rank_correlation,
                        error_message=None,
                    )
                )

            except (SubmissionFailedError, SubmissionInvalidError):
                # サブミッションエラーを即座に再raise（Submission起因）
                raise
            except EvaluatorAPIError:
                # システムエラーを即座に再raise（System起因）
                raise
            except Exception as e:
                # 予期しないエラー - データ処理の問題はシステム起因として扱う
                # （シグナル関数の例外は上で明示的にSubmissionFailedErrorに変換済み）
                raise EvaluatorAPIError(
                    f"イテレーション{i} ({current_datetime})でシステムエラーが発生しました: {e}",
                ) from e

        # 相関系列から統計を計算
        stats = self._calculate_correlation_statistics(iteration_results)

        # 有効なイテレーションが0件の場合はサブミッション失敗
        if stats.valid_count == 0:
            raise SubmissionFailedError(
                f"有効な相関を計算できるイテレーションがありません。"
                f"全{len(datetimes)}イテレーションで相関計算に失敗しました。"
                f"シグナル生成関数が有効なシグナルを出力しているか確認してください。"
            )

        mean_correlation = stats.mean
        std_correlation = stats.std
        sharpe_ratio = stats.sharpe_ratio

        evaluation_completed_at = datetime.now()

        return BacktestResult(
            status="completed",
            error_message=None,
            iteration_results=iteration_results,
            sharpe_ratio=sharpe_ratio,
            total_iterations=len(datetimes),
            valid_iterations=stats.valid_count,
            mean_correlation=mean_correlation,
            std_correlation=std_correlation,
            evaluation_started_at=evaluation_started_at,
            evaluation_completed_at=evaluation_completed_at,
        )

    def _normalize_datetime_precision(self, df: pl.DataFrame) -> pl.DataFrame:
        """DataFrameのdatetime型カラムをマイクロ秒精度に統一する。

        pandasはナノ秒精度（ns）、polarsはマイクロ秒精度（μs）をデフォルトとするため、
        join操作時に型不一致エラーを防ぐために精度を統一する。

        Args:
            df: 対象のDataFrame

        Returns:
            datetime精度がマイクロ秒に統一されたDataFrame
        """
        # datetime型カラムを検出してマイクロ秒精度に変換
        cast_exprs: list[pl.Expr] = []
        for col_name, dtype in zip(df.columns, df.dtypes):
            # Datetime型かどうかを判定（timezone付きも含む）
            if isinstance(dtype, pl.Datetime):
                # 既にマイクロ秒精度の場合はスキップ
                if dtype.time_unit != "us":
                    cast_exprs.append(pl.col(col_name).dt.cast_time_unit("us"))

        if not cast_exprs:
            return df

        return df.with_columns(cast_exprs)

    def _normalize_signal_df(self, signal_df: pl.DataFrame | pd.DataFrame) -> pl.DataFrame:
        """シグナルDataFrameをPolarsに正規化する。

        Args:
            signal_df: シグナル生成関数の戻り値（pl.DataFrame または pd.DataFrame）

        Returns:
            pl.DataFrame: 正規化されたPolars DataFrame（datetime精度はμs）

        Raises:
            SubmissionInvalidError: DataFrameでない場合
        """
        if isinstance(signal_df, pd.DataFrame):
            df = pl.from_pandas(signal_df)
        elif isinstance(signal_df, pl.DataFrame):
            df = signal_df
        else:
            raise SubmissionInvalidError(
                f"戻り値はpl.DataFrameまたはpd.DataFrameでなければなりません。受け取った型: {type(signal_df).__name__}"
            )

        # datetime型カラムをマイクロ秒精度に統一
        return self._normalize_datetime_precision(df)

    def _validate_signal_format(self, signal_df: pl.DataFrame) -> None:
        """シグナルDataFrameのカラム存在と型を検証する。

        Args:
            signal_df: 検証するシグナルDataFrame

        Raises:
            SubmissionInvalidError: カラムが不足または型が不正な場合
        """
        required_columns = {"datetime", "symbol", "signal"}
        missing = required_columns - set(signal_df.columns)
        if missing:
            raise SubmissionInvalidError(f"シグナルDataFrameに必須カラムがありません: {sorted(missing)}")

        # datetime型チェック
        dt_type = signal_df.schema["datetime"]
        if not isinstance(dt_type, pl.Datetime):
            raise SubmissionInvalidError(f"'datetime'カラムの型はDatetimeである必要があります（実際: {dt_type}）")

        # symbol型チェック
        sym_type = signal_df.schema["symbol"]
        if sym_type not in (pl.Utf8, pl.String):
            raise SubmissionInvalidError(f"'symbol'カラムの型はUtf8/Stringである必要があります（実際: {sym_type}）")

        # signal型チェック（数値型）
        sig_type = signal_df.schema["signal"]
        if not sig_type.is_numeric():
            raise SubmissionInvalidError(f"'signal'カラムの型は数値型である必要があります（実際: {sig_type}）")

    def _calculate_rank_correlation(
        self,
        signals: pl.DataFrame,
        returns: pl.DataFrame,
    ) -> float | None:
        """シグナルとリターン間のSpearman順位相関を計算する。

        Args:
            signals: DataFrame（カラム: datetime, symbol, signal）
            returns: DataFrame（カラム: datetime, symbol, return_value）

        Returns:
            Spearman相関係数、計算不可能な場合None

        Note:
            - シグナルのNaN値は平均値で埋められる
            - リターンのNaN値はそのシンボルを除外する
            - 有効なデータポイントが2未満の場合、Noneを返す
        """
        # datetimeとsymbolでシグナルとリターンをjoin
        merged = signals.join(returns, on=["datetime", "symbol"], how="inner")

        if len(merged) < 2:
            return None

        # numpy配列を抽出
        signal_values = merged["signal"].to_numpy()
        return_values = merged["return_value"].to_numpy()

        # シグナルのNaNを処理: 平均値で埋める
        signal_mean = float(np.nanmean(signal_values))
        signal_values = np.where(np.isnan(signal_values), signal_mean, signal_values)

        # リターンのNaNを処理: そのシンボルを除外
        valid_mask = ~np.isnan(return_values)
        if valid_mask.sum() < 2:
            return None

        signal_values = signal_values[valid_mask]
        return_values = return_values[valid_mask]

        # Spearman相関を計算
        try:
            correlation, _ = spearmanr(signal_values, return_values)
            # すべての値が同一の場合、spearmanrはnanを返す可能性がある
            return None if np.isnan(correlation) else float(correlation)
        except Exception:
            return None

    def _calculate_correlation_statistics(self, iteration_results: list[IterationResult]) -> CorrelationStatistics:
        """相関系列から統計情報（平均、標準偏差、シャープレシオ）を計算する。

        Args:
            iteration_results: イテレーション結果のリスト

        Returns:
            相関統計情報（mean, std, sharpe_ratio, valid_count）

        Note:
            - None以外の相関のみ使用される
            - 有効なイテレーションがない場合、meanとstdはNone、sharpe_ratioは0.0
            - 標準偏差が0の場合、sharpe_ratioは0.0
        """
        correlations = [r.rank_correlation for r in iteration_results if r.rank_correlation is not None]
        valid_count = len(correlations)

        if valid_count == 0:
            return CorrelationStatistics(mean=None, std=None, sharpe_ratio=0.0, valid_count=0)

        mean_corr = float(np.mean(correlations))
        std_corr = float(np.std(correlations, ddof=1)) if valid_count > 1 else None

        sharpe_ratio = 0.0
        if std_corr is not None and std_corr != 0.0:
            sharpe_ratio = mean_corr / std_corr

        return CorrelationStatistics(
            mean=mean_corr,
            std=std_corr,
            sharpe_ratio=sharpe_ratio,
            valid_count=valid_count,
        )

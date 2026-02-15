"""Return builder for calculating return series from OHLCV data."""

import polars as pl


class ReturnBuilder:
    """リターン計算ロジック（API非依存）.

    FR-007: pct_changeによるリターン計算
    FR-008: window幅と方式を設定可能
    FR-016: TargetBuildはOHLCVデータからリターン系列を計算

    TODO: 調整済みclose（分割・配当調整済み）を使用したリターン計算をサポートする
    （現在は単純なclose価格を使用）
    """

    def calculate_returns(
        self,
        ohlcv: pl.DataFrame,
        window: int = 1,
        method: str = "close2close",
    ) -> pl.DataFrame:
        """リターン系列を計算（未来リターン）.

        Args:
            ohlcv: OHLCVデータ
                必須カラム: datetime, symbol, open, high, low, close, volume
            window: リターン計算のwindow幅（日数/バー数）
            method: 計算方式
                - "close2close": 翌window期間のリターン
                  window=1: 翌営業日のclose2closeリターン
                  window=2: 翌々営業日のclose2closeリターン
                - "open2close": 翌openからwindow期間後のcloseまでのリターン
                  window=1: 翌営業日のopen→翌営業日のclose
                  window=2: 翌営業日のopen→翌々営業日のclose
                - "daytrade_market": 翌日の日中リターン (open[t+1] → close[t+1])
                  window=1のみ。全銘柄のmarket return（選択バイアスなし）

        Returns:
            pl.DataFrame: 以下のカラムを持つDataFrame
                - datetime: datetime型（エントリー日時）
                - symbol: str型
                - return_value: float型（未来の実現リターン）

        Raises:
            ValueError: 不正なmethod指定時、またはdaytrade_marketでwindow≠1

        Note:
            バックテストでは、各datetimeにエントリーした場合の未来の実現リターンを計算する。
            そのため、shift(-window)を使用して未来の価格を参照する。

            close2close vs open2close:
            - close2close: シグナル生成時点(t)のclose価格でエントリーを前提。
              シンプルで直感的だが、厳密にはlook-ahead bias（t時点のcloseを見てから
              t時点のcloseでエントリーすることは現実には不可能）がある。
              シミュレーション上の慣習として許容される。
            - open2close: 翌日(t+1)のopen価格でエントリーを前提。
              より現実的な取引シミュレーション。シグナルは前日closeまでの情報で
              生成され、翌日の寄付きで約定する想定。

            daytrade_market:
            - 翌日の日中リターン: (close[t+1] - open[t+1]) / open[t+1]
            - open2close(window=1)と数値的に同値だが、意図を明示するために分離
            - 全銘柄の market return を計算（選択バイアスなし）
            - 境界NaN: 最終行は翌日データなし（全銘柄一律、close2closeと同じ挙動）
        """
        # daytrade_market は window==1 を強制（ReturnDefinition 経由しない直接呼び出し経路の保護）
        if method == "daytrade_market" and window != 1:
            msg = f"daytrade_market requires window=1, got {window}"
            raise ValueError(msg)

        # データをdatetime, symbolでソート（shift操作の前提）
        ohlcv_sorted = ohlcv.sort(["symbol", "datetime"])

        if method == "close2close":
            # close2close: t時点のclose → (t+window)時点のclose
            # シグナル生成時点の価格でエントリーする想定（シミュレーション上の慣習）
            # 厳密にはlook-ahead biasあり（詳細はdocstringのNote参照）
            return ohlcv_sorted.with_columns(
                (pl.col("close").shift(-window).over("symbol") / pl.col("close") - 1).alias("return_value")
            ).select(["datetime", "symbol", "return_value"])
        elif method == "open2close":
            # open2close: (t+1)時点のopen → (t+window)時点のclose
            # 翌日の寄付きでエントリーする想定（より現実的）
            return (
                ohlcv_sorted.with_columns(
                    next_open=pl.col("open").shift(-1).over("symbol"),  # 翌日のopen
                    future_close=pl.col("close").shift(-window).over("symbol"),  # window日後のclose
                )
                .with_columns(
                    ((pl.col("future_close") - pl.col("next_open")) / pl.col("next_open")).alias("return_value")
                )
                .select(["datetime", "symbol", "return_value"])
            )
        elif method == "daytrade_market":
            # daytrade_market: 翌日の日中リターン (close[t+1] - open[t+1]) / open[t+1]
            # 全銘柄の market return（選択バイアスなし）
            # 境界NaN: 最終行は翌日データなし（全銘柄一律）
            return (
                ohlcv_sorted.with_columns(
                    next_open=pl.col("open").shift(-1).over("symbol"),
                    next_close=pl.col("close").shift(-1).over("symbol"),
                )
                .with_columns(
                    ((pl.col("next_close") - pl.col("next_open")) / pl.col("next_open")).alias("return_value")
                )
                .select(["datetime", "symbol", "return_value"])
            )
        else:
            raise ValueError(f"Unknown method: {method}")

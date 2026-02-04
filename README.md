# mixseek-quant-insight

Mixseek Quant Insightは、Mixseekのコンペティションアーキテクチャを、クオンツ的なアプローチによるシグナル生成に拡張したプラグインです。

## Overview

mixseek-quant-insightは、[mixseek-core](https://github.com/mixseek/mixseek-core)フレームワークを拡張し、
複数のAIエージェントチームが株価シグナル生成戦略を競い合うコンペティションシステムを提供します。

### Key Features

- **シグナル生成コンペティション**: 複数チームがそれぞれ独自のシグナル生成関数を開発・提出
- **Time Series APIバックテスト**: KaggleのTimeseries API形式にインスパイアされたバックテスト評価
- **カスタムデータソース**: 任意の時系列データを組み込み可能
- **カスタムエージェント**: 任意のエージェントチームやメンバー構成をカスタマイズ可能

## Quick Start

詳細なセットアップ手順については [docs/quickstart.md](docs/quickstart.md) を参照してください。

## Customize

エージェント構成、データソースなどのカスタマイズ方法については [docs/config_guide.md](docs/config_guide.md) を参照してください。

## Evaluation Metrics

### 評価の流れ

1. **Private Test期間の分離**: エージェントからは見えない未来のテスト期間を設定してデータが分離されます
2. **逐次的なシグナル計算**: 各評価日において、エージェントが提出したシグナル生成関数が逐次的に実行されます
3. **クロスセクショナル評価**: 各日の全銘柄に対するシグナルと実際のリターンを比較します

### Scoring Method

```
日次評価:
  各日t: rank_corr(t) = Spearman相関(シグナル(t), リターン(t))
                        ↑ 全銘柄のクロスセクショナルな順位相関

最終スコア:
  Sharpe Ratio = mean(rank_correlations) / std(rank_correlations)
                 ↑ 時系列方向の安定性を評価
```

- **順位相関（Spearman）**: 各日において、シグナルの順位と実際のリターンの順位の相関を計算します。これにより、シグナルがリターンの大小関係をどれだけ正しく予測できているかを測定します。
- **期間シャープレシオ**: 日次の順位相関を時系列で集計し、その平均を標準偏差で割ったシャープレシオが最終スコアとなります。これにより、単に平均的に良い予測ができるだけでなく、安定して良い予測ができるエージェントが高く評価されます。

## Licence

Apache License 2.0 - 詳細は[LICENSE](LICENSE)を参照してください。
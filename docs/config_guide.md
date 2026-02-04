# Config Guide: 設定のカスタマイズ

このガイドでは、quant-insightの主要なカスタマイズポイントを説明します。

## 目次

1. [カスタムデータの追加](#1-カスタムデータの追加)
2. [LLMモデルの変更](#2-llmモデルの変更)
3. [チームの追加・編集](#3-チームの追加編集)
4. [メンバーエージェントの追加・編集](#4-メンバーエージェントの追加編集)

---

## 1. カスタムデータの追加

J-Quants API以外のデータ（ファンダメンタルズ、センチメント等）をコンペティションに追加できます。

### 1.1 データの配置

カスタムデータを `$MIXSEEK_WORKSPACE/data/inputs/raw/` に配置します。

```bash
$MIXSEEK_WORKSPACE/data/inputs/raw/
├── ohlcv.parquet      # J-Quantsから取得済み（必須）
├── returns.parquet    # build-returnsで生成（必須）
├── master.parquet     # J-Quantsから取得済み
└── fundamentals.parquet  # ← カスタムデータ
```

### 1.2 データ形式の要件

| 要件 | 説明 |
|------|------|
| ファイル形式 | Parquet または CSV |
| 必須カラム | `datetime`（または指定した日時カラム）、`symbol` |
| datetime型 | datetime型、または解析可能な日付文字列（YYYY-MM-DD） |
| ソート | datetime昇順を推奨 |

**カスタムデータの例（fundamentals.parquet）:**

```
datetime        symbol    pe_ratio    pb_ratio    roa
2024-01-04      1234      15.5        1.2         0.08
2024-01-04      1235      18.2        1.5         0.10
2024-01-05      1234      15.6        1.2         0.09
...
```

### 1.3 competition.tomlへの登録

`$MIXSEEK_WORKSPACE/configs/competition.toml` にデータセットを追加します。

```toml
[competition]
name = "My Competition"

# 必須データセット
[[competition.data]]
name = "ohlcv"
datetime_column = "datetime"

[[competition.data]]
name = "returns"
datetime_column = "datetime"

# オプション：マスタデータ
[[competition.data]]
name = "master"
datetime_column = "datetime"

# カスタムデータを追加
[[competition.data]]
name = "fundamentals"
datetime_column = "datetime"  # カラム名が異なる場合は指定

[[competition.data]]
name = "sentiment"
datetime_column = "date"  # 例: 日時カラムが "date" の場合
```

### 1.4 データの分割

登録後、`split`コマンドで自動的に train/valid/test に分割されます。

```bash
quant-insight data split --config $MIXSEEK_WORKSPACE/configs/competition.toml
```

分割後の構造:
```
$MIXSEEK_WORKSPACE/data/inputs/
├── ohlcv/
│   ├── train.parquet
│   ├── valid.parquet
│   └── test.parquet
├── fundamentals/        # カスタムデータも分割される
│   ├── train.parquet
│   ├── valid.parquet
│   └── test.parquet
└── ...
```

### 1.5 エージェントからのデータアクセス

分割後のデータは、エージェント設定の `available_data_paths` で指定することでアクセス可能になります。

```toml
# configs/agents/members/train_analyzer.toml
[agent.metadata.tool_settings.local_code_executor]
available_data_paths = [
    "data/inputs/ohlcv/train.parquet",
    "data/inputs/master/train.parquet",
    "data/inputs/returns/train.parquet",
    "data/inputs/fundamentals/train.parquet",  # カスタムデータを追加
]
```

---

## 2. LLMモデルの変更

### 2.1 サポートされているプロバイダー

| プロバイダー | プレフィックス | 環境変数 |
|-------------|---------------|----------|
| Google AI (Gemini) | `google-gla:` | `GOOGLE_API_KEY` |
| OpenAI | `openai:` | `OPENAI_API_KEY` |
| Anthropic (Claude) | `anthropic:` | `ANTHROPIC_API_KEY` |
| Grok (X.AI) | `grok:` | `XAI_API_KEY` |
| Google Vertex AI | `google-vertex:` | Google Cloud認証 |

### 2.2 モデル指定の形式

```
{プロバイダー}:{モデル名}
```

**モデル名の例:**

```toml
# Google AI (Gemini)
model = "google-gla:gemini-3-flash-preview"
model = "google-gla:gemini-2.5-pro"

# OpenAI
model = "openai:gpt-5.2"

# Anthropic (Claude)
model = "anthropic:claude-sonnet-4-5-20250929"

# Grok
model = "grok:grok-4-1-fast-reasoning"
```

### 2.3 環境変数の設定

`.env` ファイルまたは環境変数でAPIキーを設定します。

```bash
# .env ファイル
GOOGLE_API_KEY=your-google-api-key
OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
XAI_API_KEY=your-xai-api-key
```

### 2.4 チーム/メンバー設定でのモデル変更

**チームリーダーのモデル変更（teams/*.toml）:**

```toml
[team.leader]
model = "openai:gpt-5.2"  # ← ここを変更
```

**メンバーエージェントのモデル変更（members/*.toml）:**

```toml
[agent]
type = "custom"
name = "train-analyzer"
model = "anthropic:claude-sonnet-4-5-20250929"  # ← ここを変更
```

### 2.5 temperatureの調整

| 値 | 特性 | 用途 |
|----|------|------|
| 0.0 | 決定的（同じ入力→同じ出力） | 再現性が必要な分析 |
| 0.5 | 中程度の多様性 | バランスの取れた生成 |
| 1.0 | 高い多様性 | 創造的な仮説生成 |

```toml
[agent]
model = "openai:gpt-5.2"
temperature = 0.5  # 0.0〜2.0の範囲
```

---

## 3. チームの追加・編集

### 3.1 チーム設定ファイルの構造

```toml
# configs/agents/teams/my_team.toml

[team]
team_id = "my-custom-team"           # 一意のID
team_name = "My Custom Analysis Team"  # 表示名

[team.leader]
model = "google-gla:gemini-2.5-flash"
temperature = 0.5
system_instruction = """
あなたは株価シグナル生成コンペティションのチームリーダーです。

# 目標
リターンとの順位相関が最も高いシグナルを生成すること。

# 戦略
1. train-analyzerにデータ分析を依頼
2. 分析結果を元にsubmission-creatorにシグナル関数の実装を依頼
3. 結果を評価し、改善を繰り返す

# 注意事項
- メンバーはセッション間でコンテキストを保持しません
- 必要な情報は毎回明示的に指示してください
"""

[[team.members]]
config = "configs/agents/members/train_analyzer.toml"

[[team.members]]
config = "configs/agents/members/submission_creator.toml"
```

### 3.2 リーダー指示のカスタマイズ

リーダーの `system_instruction` で戦略を定義します。

**分析重視のリーダー:**

```toml
[team.leader]
system_instruction = """
あなたは慎重な分析を重視するチームリーダーです。

# 方針
- 複数の仮説を検証してから最終的なシグナルを決定
- データの特性を十分に理解してから実装に進む
- 過学習を避けるため、シンプルな手法を優先
"""
```

**実験重視のリーダー:**

```toml
[team.leader]
temperature = 1.0  # 高めの温度で多様なアプローチを試す
system_instruction = """
あなたは実験的なアプローチを重視するチームリーダーです。

# 方針
- 複数の異なるアイデアを並行して試す
- 失敗を恐れず、新しい手法に挑戦
- 各ラウンドで前回と異なるアプローチを試行
"""
```

### 3.3 orchestrator.tomlへの登録

新しいチームを `orchestrator.toml` に追加します。

```toml
[orchestrator]
min_rounds = 3
max_rounds = 5
timeout_per_team_seconds = 3600
evaluator_config = "configs/evaluator.toml"

# 既存チーム
[[orchestrator.teams]]
config = "configs/agents/teams/minimal_gemini_deterministic.toml"

# 新しいチームを追加
[[orchestrator.teams]]
config = "configs/agents/teams/my_team.toml"
```

### 3.4 複数チームの並列実行

複数のチームが並列に実行され、最終的にスコアでランキングされます。

```toml
[[orchestrator.teams]]
config = "configs/agents/teams/team_gemini.toml"

[[orchestrator.teams]]
config = "configs/agents/teams/team_openai.toml"

[[orchestrator.teams]]
config = "configs/agents/teams/team_claude.toml"
```

---

## 4. メンバーエージェントの追加・編集

### 4.1 メンバー設定ファイルの構造

```toml
# configs/agents/members/my_analyzer.toml

[agent]
type = "custom"
name = "my-analyzer"                    # 一意の名前
model = "google-gla:gemini-2.5-flash"
temperature = 0.5
description = "独自の分析手法でデータを分析するエージェント"

[agent.system_instruction]
text = """
あなたはクオンツアナリストです。

# 役割
提供されたデータを分析し、株価リターンを予測するための特徴量やパターンを発見します。

# 利用可能なデータ
- ohlcv: 日足の株価データ（open, high, low, close, volume）
- master: 銘柄マスタ（業種、市場区分等）
- returns: 翌日リターン

# 出力形式
分析結果をMarkdownレポートとして出力してください。
"""

[agent.plugin]
path = "src/quant_insight/agents/local_code_executor/agent.py"
agent_class = "LocalCodeExecutorAgent"

[agent.metadata]

[agent.metadata.tool_settings.local_code_executor]
available_data_paths = [
    "data/inputs/ohlcv/train.parquet",
    "data/inputs/master/train.parquet",
    "data/inputs/returns/train.parquet",
]
timeout_seconds = 300

[agent.metadata.tool_settings.local_code_executor.output_model]
module_path = "quant_insight.agents.local_code_executor.output_models"
class_name = "AnalyzerOutput"
```

### 4.2 主要な設定項目

| 項目 | 説明 |
|------|------|
| `name` | エージェントの一意識別子（英数字、ハイフン、アンダースコア） |
| `model` | 使用するLLMモデル |
| `temperature` | 出力の多様性（0.0〜2.0） |
| `description` | チームリーダーに表示される説明 |
| `system_instruction.text` | エージェントへの詳細な指示 |
| `available_data_paths` | アクセス可能なデータファイル |
| `timeout_seconds` | コード実行のタイムアウト |

### 4.3 system_instructionのカスタマイズ

エージェントの振る舞いは `system_instruction` で詳細に制御できます。

**テクニカル分析特化:**

```toml
[agent.system_instruction]
text = """
あなたはテクニカル分析の専門家です。

# 分析手法
以下のテクニカル指標を活用してください：
- 移動平均（SMA, EMA）
- RSI（相対力指数）
- MACD
- ボリンジャーバンド

# 注意事項
- ファンダメンタルズは考慮しない
- 価格パターンと出来高の関係に注目
"""
```

**ファンダメンタルズ分析特化:**

```toml
[agent.system_instruction]
text = """
あなたはファンダメンタルズ分析の専門家です。

# 分析手法
- PER、PBRなどのバリュエーション指標
- 業種別の特性分析
- 市場区分（プライム/スタンダード/グロース）の傾向

# データ活用
masterデータから業種・市場情報を取得し、
セクター別のリターン傾向を分析してください。
"""
```

### 4.4 データアクセス制御

エージェントごとにアクセス可能なデータを制限できます。

```toml
# train-analyzer: trainデータのみアクセス可能
[agent.metadata.tool_settings.local_code_executor]
available_data_paths = [
    "data/inputs/ohlcv/train.parquet",
    "data/inputs/master/train.parquet",
    "data/inputs/returns/train.parquet",
]

# submission-creator: validデータのみアクセス可能（リターンは不可）
[agent.metadata.tool_settings.local_code_executor]
available_data_paths = [
    "data/inputs/ohlcv/valid.parquet",
    "data/inputs/master/valid.parquet",
    # returns/valid.parquet は意図的に除外（リーク防止）
]
```

### 4.5 チームへの組み込み

作成したメンバーエージェントをチーム設定に追加します。

```toml
# configs/agents/teams/my_team.toml

[[team.members]]
config = "configs/agents/members/my_analyzer.toml"

[[team.members]]
config = "configs/agents/members/submission_creator.toml"
```

### 4.6 出力モデルの選択

| クラス名 | 用途 | 出力内容 |
|----------|------|----------|
| `AnalyzerOutput` | データ分析 | `scripts`（実行コード）、`report`（分析レポート） |
| `SubmitterOutput` | シグナル生成 | `submission`（シグナル関数）、`description`（説明） |

```toml
# 分析エージェント用
[agent.metadata.tool_settings.local_code_executor.output_model]
module_path = "quant_insight.agents.local_code_executor.output_models"
class_name = "AnalyzerOutput"

# 提出エージェント用
[agent.metadata.tool_settings.local_code_executor.output_model]
module_path = "quant_insight.agents.local_code_executor.output_models"
class_name = "SubmitterOutput"
```

---

## 設定ファイルの関係図

```
orchestrator.toml
├── evaluator_config → evaluator.toml
└── teams[]
    └── config → teams/*.toml
                 ├── team.leader（リーダー設定）
                 └── team.members[]
                     └── config → members/*.toml
                                  └── agent（メンバー設定）

competition.toml（独立）
├── data[]（データセット定義）
├── data_split（分割設定）
└── return_definition（リターン計算設定）
```

---

## 次のステップ

- サンプル設定（`examples/`）を参考に、独自のチーム構成を設計
- 異なるモデルやtemperature設定で実験
- カスタムデータを追加して分析の幅を拡大

# クイックスタート: AIエージェントによる金融コンペティション

このガイドでは、quant-insightを使用してAIエージェントチームによる株価シグナル生成コンペティションを実行する手順を説明します。

## 前提条件

- Docker
- J-Quants API アカウント（[https://jpx-jquants.com/](https://jpx-jquants.com/)）
- Google AI API キー（Geminiモデル使用時）

## 1. Docker イメージのビルド

まず、最小構成のDockerイメージをビルドします。

```bash
cd dockerfiles/minimal
make build
```

ビルドには数分かかる場合があります。完了すると `mixseek-quant-insight/minimal:latest` イメージが作成されます。

## 2. 環境変数の設定

`.env.template` をコピーして `.env` ファイルを作成し、必要な環境変数を設定します。

```bash
cd dockerfiles/minimal
cp .env.template .env
```

`.env` ファイルを編集し、以下の環境変数を設定してください：

```bash
# ワークスペースパス
# コンテナ内では /app がプロジェクトルートにマウントされます
# ワークスペースは /app/workspace などを推奨
MIXSEEK_WORKSPACE=/app/workspace

# J-Quants API キー
JQUANTS_API_KEY=your-jquants-api-key

# Google AI API キー（Geminiモデル使用時）
GEMINI_API_KEY=your-gemini-api-key
```

**注意**: ワークスペースディレクトリは `quant-insight setup` で自動作成されます。

### J-Quants API キーの取得

1. [J-Quants](https://jpx-jquants.com/) にアクセス
2. アカウントを作成（無料プランあり）
3. API キーを発行

### Google AI API キーの取得

1. [Google AI Studio](https://aistudio.google.com/) にアクセス
2. API キーを作成

## 3. コンテナの起動とシェルへの接続

```bash
# コンテナをバックグラウンドで起動
make run

# コンテナ内のシェルに接続
make bash
```

以降のコマンドはすべてコンテナ内で実行します。

## 4. 環境のセットアップ

`quant-insight setup` コマンドで環境を一括セットアップします。

```bash
quant-insight setup
```

このコマンドは以下の処理を行います：

1. **ワークスペース構造の作成**: `$MIXSEEK_WORKSPACE` 配下に必要なディレクトリを作成
2. **サンプル設定のコピー**: `examples/` 配下の設定ファイルを `$MIXSEEK_WORKSPACE/configs/` にコピー
3. **データベースの初期化**: 実行結果を保存するDuckDBを初期化

実行後、以下のディレクトリ構造が作成されます：

```
$MIXSEEK_WORKSPACE/
├── configs/
│   ├── competition.toml       # コンペティション設定
│   ├── orchestrator.toml      # オーケストレータ設定
│   ├── evaluator.toml         # 評価設定
│   └── agents/
│       ├── members/           # メンバーエージェント設定
│       └── teams/             # チーム設定
└── mixseek.db                 # 実行結果DB
```

## 5. データのビルド

J-Quants APIからデータを取得し、コンペティション用に加工します。

### 5.1 OHLCVデータの取得

```bash
quant-insight data fetch-jquants --plan free --universe prime
```

**オプション:**
- `--plan`: J-Quantsプラン（`free`/`light`/`standard`/`premium`）
- `--universe`: 対象ユニバース（`prime`/`standard`/`growth`/`all`）
- `--start-date`: 開始日（YYYY-MM-DD、デフォルト: 2年前）
- `--end-date`: 終了日（YYYY-MM-DD、デフォルト: 12週間前）

取得されるデータ：
- `$MIXSEEK_WORKSPACE/data/raw/ohlcv.parquet`: 日足OHLCVデータ
- `$MIXSEEK_WORKSPACE/data/raw/master.parquet`: 銘柄マスタデータ

### 5.2 データ分割設定の調整

取得したデータの期間に合わせて、`competition.toml` のデータ分割設定を調整します。

`[competition.data_split]` セクションの `train_end` と `valid_end` を、取得したデータ期間に合わせて設定してください：

```toml
# 例: Period: 2024-01-01 to 2025-12-31
[competition.data_split]
train_end = "2024-12-31T23:59:59"  # 最初の1年をtrainに
valid_end = "2025-06-30T23:59:59"  # 残りをvalid/testに分割
purge_rows = 1
```

**注意**: `train_end` と `valid_end` は取得したデータの範囲内で設定してください。範囲外の日付を指定すると、一部のデータセットが空になり、エラーとなります。

### 5.3 リターンの計算

```bash
quant-insight data build-returns --config $MIXSEEK_WORKSPACE/configs/competition.toml
```

`competition.toml` の `[competition.return_definition]` セクションの設定に従ってリターンを計算します：
- `window`: リターン計算の期間（日数）
- `method`: 計算方法（`open2close` または `close2close`）

出力：
- `$MIXSEEK_WORKSPACE/data/raw/returns.parquet`

### 5.4 データの分割

```bash
quant-insight data split --config $MIXSEEK_WORKSPACE/configs/competition.toml
```

5.2で設定した `train_end` と `valid_end` に基づいて、データをtrain/valid/testに分割します。

出力構造：
```
$MIXSEEK_WORKSPACE/data/inputs/
├── ohlcv/
│   ├── train.parquet
│   ├── valid.parquet
│   └── test.parquet
├── returns/
│   ├── train.parquet
│   ├── valid.parquet
│   └── test.parquet
└── master/
    ├── train.parquet
    ├── valid.parquet
    └── test.parquet
```

## 6. コンペティションの実行

データの準備が完了したら、`mixseek exec` コマンドでコンペティションを開始します。

```bash
mixseek exec "株価リターンを予測する面白いシグナル関数を生成してください" \
    --config $MIXSEEK_WORKSPACE/configs/orchestrator.toml
```

**引数:**
- 第1引数: ユーザープロンプト（エージェントへの指示）
- `--config`: オーケストレータ設定ファイルのパス

### 実行の流れ

1. **チーム並列実行**: `orchestrator.toml` で定義された各チームが並列に動作
2. **エージェント協調**: 各チーム内でリーダーがメンバーエージェントを呼び出し
   - `train-analyzer`: 訓練データを分析
   - `submission-creator`: シグナル関数を実装
3. **シグナル生成**: 各チームがシグナル生成関数を提出
4. **バックテスト評価**:
   - シグナルとリターンの順位相関（Spearman）を計算
   - 相関系列のシャープレシオでスコアリング
5. **リーダーボード表示**: チームのスコアランキングを表示

### 実行結果の確認

実行結果は `$MIXSEEK_WORKSPACE/mixseek.db`（DuckDB）に保存されます。

```bash
# 特定の実行結果をMarkdown形式でエクスポート
quant-insight export logs <execution_id> \
    --config $MIXSEEK_WORKSPACE/configs/orchestrator.toml
```

`execution_id` は `mixseek exec` 実行時に表示されます。

## 次のステップ

- **設定のカスタマイズ**: 各設定ファイルの詳細については[Config Guide](config.md)を参照してください。
  - LLMモデルの変更（OpenAI、Claude、Grok対応）
  - チームの追加とリーダー指示のカスタマイズ
  - メンバーエージェントの追加と役割定義
- **カスタムデータの追加**: ファンダメンタルズやセンチメントなど、独自のデータを`data/inputs/raw/`に配置して活用できます。詳細は[Config Guide - カスタムデータの追加](config.md#1-カスタムデータの追加)を参照。

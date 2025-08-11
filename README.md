# Stock Predictor
# 株式予測AI

## 概要 (Overview)

TransformerアーキテクチャとGrouped-Query Attention (GQA) を使用した株式価格予測AIシステムです。5分足データから次の5分間の終値を予測し、デイトレード戦略をサポートします。デフォルトでトヨタ自動車 (7203.T) を対象としていますが、他の銘柄にも対応可能です。

An AI stock prediction system using Transformer architecture with Grouped-Query Attention (GQA). Predicts next 5-minute closing prices from candlestick data to support day trading strategies. Currently configured for Toyota Motor Corporation (7203.T) but extensible to other stocks.

## 主な特徴 (Key Features)

- **高効率アーキテクチャ**: GQA採用により高速推論を実現
- **多変量時系列**: 対象銘柄 + 市場指数データ
- **リアルタイム推論**: ローカルマシンでの高速予測
- **包括的評価**: バックテスト機能による取引シミュレーション
- **可視化ダッシュボード**: 詳細な結果分析とチャート
- **📱 Webアプリ**: Streamlitベースの使いやすいUI
- **🔧 拡張可能**: 設定変更で他の銘柄にも対応
- **📈 テクニカル指標**: RSI、MACD、ボリンジャーバンド、ストキャスティクス、一目均衡表
- **📊 高度なチャート**: インタラクティブなテクニカル分析ダッシュボード
- **🎯 売買シグナル**: 複数指標による総合的なシグナル生成

## プロジェクト構造 (Project Structure)

```
Stock prediction/
├── requirements.txt          # 依存関係
├── config.py                # 設定ファイル
├── main.py                  # メイン実行スクリプト
├── app.py                   # 📱 Streamlitアプリ
├── run_app.py               # アプリ起動スクリプト
├── data_handler.py          # データ取得・前処理
├── dataset.py               # PyTorchデータセット
├── model.py                 # GQA Transformerモデル
├── train.py                 # モデル学習
├── evaluate.py              # モデル評価・バックテスト
├── inference.py             # リアルタイム推論
├── visualize.py             # 結果可視化
├── technical_indicators.py  # 📈 テクニカル指標計算
├── TECHNICAL_INDICATORS.md  # テクニカル指標ドキュメント
├── data/                    # 処理済みデータ
├── models/                  # 学習済みモデル
├── checkpoints/             # 学習チェックポイント
└── results/                 # 結果・ログ
```
├── model.py                 # GQA Transformerモデル
├── train.py                 # モデル学習
├── evaluate.py              # モデル評価・バックテスト
├── inference.py             # リアルタイム推論
├── visualize.py             # 結果可視化
├── data/                    # 処理済みデータ
├── models/                  # 学習済みモデル
├── checkpoints/             # 学習チェックポイント
└── results/                 # 結果・ログ
```

## セットアップ (Setup)

### 1. 依存関係のインストール
```bash
pip install -r requirements.txt
```

### 2. モデル学習（初回のみ）
```bash
python main.py
# 選択肢 1 を選んで完全パイプラインを実行
```

### 3. アプリケーション起動
```bash
python run_app.py
```

## 使用方法 (Usage)

### Webアプリケーション（推奨）
```bash
python run_app.py
```
http://localhost:8501 でアクセス

**主な機能**:
- リアルタイム価格チャート
- 5分足予測とシグナル表示  
- 分析ダッシュボード
- 予測履歴参照
- 表示設定カスタマイズ

### コマンドライン実行

#### 完全パイプライン
```bash
python main.py  # 選択肢 1
```

#### 個別コンポーネント実行

#### データ準備
```bash
python data_handler.py
```

#### モデル学習
```bash
python train.py
```

#### モデル評価
```bash
python evaluate.py
```

#### リアルタイム推論
```bash
python inference.py
```

#### 結果可視化
```bash
python visualize.py
```

## モデル仕様 (Model Specifications)

### アーキテクチャ
- **ベース**: Transformer Encoder
- **注意機構**: Grouped-Query Attention (GQA)
- **モデル次元**: 128
- **ヘッド数**: 8 (Query), 2 (Key-Value)
- **層数**: 6
- **ドロップアウト**: 0.1

### データ仕様
- **予測対象**: 設定可能（デフォルト: 7203.T トヨタ自動車）
- **市場参照**: 設定可能（デフォルト: 1321.T 日経225ETF）
- **時間軸**: 5分足
- **入力長**: 48ステップ (4時間分)
- **予測期間**: 1ステップ先 (次の5分)

### 特徴量
- **対象銘柄**: OHLCV (始値、高値、安値、終値、出来高)
- **市場データ**: 市場指数のOHLCV
- **正規化**: MinMaxScaler適用

## 評価指標 (Evaluation Metrics)

### 予測精度
- **RMSE**: 価格予測の二乗平均平方根誤差
- **MAE**: 平均絶対誤差
- **方向性精度**: 価格変動方向の予測正確率
- **R²スコア**: 決定係数

### 取引性能
- **総リターン**: 初期資本に対する利益率
- **シャープ比**: リスク調整後リターン
- **最大ドローダウン**: 最大損失期間
- **勝率**: 利益取引の割合

## 設定のカスタマイズ (Configuration)

主要な設定は `config.py` で変更できます：

```python
# 予測対象銘柄（変更可能）
TARGET_SYMBOL = "7203.T"  # デフォルト: トヨタ自動車
MARKET_SYMBOL = "1321.T"  # デフォルト: 日経225ETF

# データパラメータ
DATA_INTERVAL = "5m"      # 5分足
DATA_PERIOD = "60d"       # 60日間

# モデルパラメータ
SEQUENCE_LENGTH = 48      # 入力系列長
MODEL_DIM = 128          # モデル次元
NUM_HEADS = 8            # 注意ヘッド数
```

## テクニカル指標 (Technical Indicators)

### 実装済み指標

#### RSI (相対力指数)
- **期間**: 14
- **売買判定**: 30以下で買い、70以上で売り
- **用途**: 買われすぎ・売られすぎ判定

#### MACD (移動平均収束拡散法)
- **パラメータ**: 高速EMA(12)、低速EMA(26)、シグナル(9)
- **売買判定**: MACDラインとシグナルラインのクロス
- **用途**: トレンド転換点の検出

#### ボリンジャーバンド
- **期間**: 20日移動平均
- **標準偏差**: 2.0
- **用途**: 価格のレンジ分析、反転ポイント検出

#### ストキャスティクス
- **パラメータ**: %K(14)、%D(3)
- **売買判定**: 20以下で買い、80以上で売り
- **用途**: オシレーター系の過熱感判定

#### 一目均衡表
- **パラメータ**: 転換線(9)、基準線(26)、先行スパンB(52)
- **売買判定**: 価格と雲の位置関係
- **用途**: 総合的なトレンド分析

### テクニカル指標の使用方法

#### Streamlitアプリで使用
```bash
python run_app.py
```
- 「テクニカル指標分析」セクションで各指標を表示/非表示切り替え
- 個別指標の詳細ビューも選択可能
- リアルタイムでシグナル状況を確認

#### コマンドラインで使用
```bash
python visualize.py
# オプション7を選択: Technical indicators
```

#### プログラムで使用
```python
from technical_indicators import TechnicalIndicators
from data_handler import StockDataHandler

# データ取得
data_handler = StockDataHandler()
data = data_handler.download_stock_data("7203.T")

# 指標計算
ti = TechnicalIndicators()
result = ti.calculate_all_indicators(data)
signals = ti.generate_signals(result)

# シグナル確認
summary = ti.get_signal_summary(signals)
print(summary)  # {'RSI': 'BUY', 'MACD': 'HOLD', ...}
```

### 複合シグナル
- **Strong Buy**: 2つ以上の指標が買いシグナル
- **Strong Sell**: 2つ以上の指標が売りシグナル  
- **Hold**: 中立またはシグナルが混在

詳細は [TECHNICAL_INDICATORS.md](TECHNICAL_INDICATORS.md) を参照してください。

## 使用例 (Usage Examples)

### 初回セットアップ
```bash
pip install -r requirements.txt
python main.py  # 選択肢 1 でモデル学習
python run_app.py  # アプリ起動
```

### 日常使用
```bash
python run_app.py  # Webアプリ
# または
python inference.py  # コマンドライン予測
```

## 注意事項 (Important Notes)

### データの制限
- yfinanceの5分足データは直近60日間のみ取得可能
- 市場時間外（平日9:00-15:00以外）はリアルタイムデータが更新されない

### 投資判断について
- **このシステムは教育・研究目的のためのものです**
- **実際の投資判断には使用しないでください**
- **金融投資にはリスクが伴います**

### システム要件
- Python 3.8以上
- メモリ: 最低4GB RAM推奨
- ストレージ: 500MB以上の空き容量
- インターネット接続（データ取得用）

## トラブルシューティング (Troubleshooting)

### よくある問題

1. **データ取得エラー**
   - インターネット接続を確認
   - yfinanceのAPIレート制限の可能性

2. **CUDA関連エラー**
   - CPU推論に自動フォールバック
   - 設定確認: `config.DEVICE = "cpu"`

3. **メモリ不足**
   - バッチサイズを削減: `config.BATCH_SIZE = 16`
   - シーケンス長を短縮: `config.SEQUENCE_LENGTH = 24`

## ライセンス (License)

MIT License - 詳細は [LICENSE](LICENSE) ファイルを参照してください。

## 免責事項 (Disclaimer)

このソフトウェアは投資助言ではありません。実際の取引での損失について開発者は一切の責任を負いません。投資は自己責任で行ってください。
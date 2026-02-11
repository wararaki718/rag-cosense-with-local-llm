# RAG 評価基盤 (RAGAS)

このプロジェクトでは、Ragas を使用して RAG パイプラインの精度（Faithfulness, Relevancy 等）を定量的に評価します。

## 構成
- `eval/dataset_generator.py`: Elasticsearch からドキュメントを読み込み、合成テストデータセットを作成します。
- `eval/evaluate.py`: テストデータセットを用いて `app-api` にリクエストを送り、Ragas で回答精度を評価します。
- `eval/evaluator_config.py`: 評価に使用する LLM (Gemma 3) と Embedding モデルの設定です。

## セットアップ

### Docker を使用する場合 (推奨)
Docker Compose を使用して、依存関係を手動でインストールすることなく評価を実行できます。

### ローカル環境で実行する場合
1. **依存関係のインストール**
   ```bash
   pip install -r eval/requirements.txt
   ```
...
   ```

## 評価手順

### Docker を使用する場合

1. **合成テストデータセットの作成**
   ```bash
   docker compose run --rm eval python dataset_generator.py
   ```

2. **評価の実行**
   ```bash
   docker compose run --rm eval python evaluate.py
   ```

### ローカル環境で実行する場合

1. **合成テストデータセットの作成**
...
```bash
python eval/dataset_generator.py
```

2. **評価の実行**
```bash
python eval/evaluate.py
```
実行完了後、評価結果のサマリーが表示され、詳細は `eval/evaluation_results.csv` に保存されます。

## 評価指標 (Metrics)
- **Faithfulness**: 回答が提供されたコンテキストのみに基づいているか（情報の正確性）。
- **Answer Relevancy**: 回答が質問に対して適切かつ関連性があるか。
- **Context Precision**: 検索されたコンテキストが回答の生成にどの程度有用か。
- **Context Recall**: 正解を生成するために必要な情報がコンテキストに含まれているか。

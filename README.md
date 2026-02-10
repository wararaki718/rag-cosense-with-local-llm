# RAG Cosense with Local LLM

Scrapbox (Cosense) のデータをソースとし、SPLADE による疎ベクトル検索と Elasticsearch、そしてローカル LLM (Gemma 3) を組み合わせた RAG (Retrieval-Augmented Generation) システムです。

## 🚀 特徴

- **疎ベクトル検索 (Sparse Vector Search)**: SPLADE (`naver/splade-cocondenser-ensembledistil`) を使用して、キーワードベースよりも精度の高い検索を実現。
- **ハイブリッド検索**: Elasticsearch の `rank_features` を活用。
- **ローカル LLM**: Ollama を介して Gemma 3 を使用し、プライバシーを保ちつつ回答を生成。
- **マイクロサービス構成**: GPU リソースを消費するベクトル化サービスと、軽量なアプリケーション API を分離。

## 🏗️ アーキテクチャ

1.  **splade-api**: テキストを `{token: weight}` 形式の疎ベクトルに変換。
2.  **indexer**: Scrapbox のエクスポートデータ（JSON）を読み込み、チャンク分割・ベクトル化して Elasticsearch に登録。
3.  **app-api**: RAG フローの司令塔。検索、プロンプト構築、LLM への推論依頼を制御。
4.  **web-ui**: Streamlit によるチャットインターフェース。

---

## 🛠️ セットアップと実行方法 (How to Run)

### 1. 前提条件 (Prerequisites)

- [Docker](https://www.docker.com/) & Docker Compose
- [Ollama](https://ollama.com/) (ローカルで実行)
- Scrapbox のエクスポートデータ (`.json`)

### 2. モデルの準備 (Ollama)

Ollama で Gemma 3 モデルを取得しておきます。
```bash
ollama run gemma3
```

### 3. 環境変数の設定

各サービスの環境変数を設定します（デフォルト設定は `docker-compose.yml` に記載されています）。
必要に応じて `.env` ファイルを作成してください。

```bash
cp .env.example .env
```

### 4. サービスの起動

Docker Compose を使用して、Elasticsearch、splade-api、app-api、web-ui を一括で起動します。

```bash
docker compose up --build
```
※ `splade-api` は初回起動時にモデルのダウンロードを行うため、時間がかかる場合があります。

### 5. データのインデックス作成

Docker を使用してインデックス作成を実行します。`.env` ファイルに `SCRAPBOX_PROJECT`（および必要に応じて `SCRAPBOX_SID`）が設定されていることを確認してください。

```bash
docker compose run --rm indexer --project your-project-name
```

もしローカルの Python 環境で実行したい場合は以下の通りです：
```bash
cd indexer
pip install -r requirements.txt
python index_data.py --project your-project-name
```

### 6. Web UI へのアクセス

ブラウザで以下の URL にアクセスします。
- **Web UI**: `http://localhost:8501`

---

## 📄 ライセンス

[LICENSE](LICENSE) を参照してください。

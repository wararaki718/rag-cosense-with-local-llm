# Implementation Prompts for RAG Cosense System

[architecture.md](../architecture.md) の設計に基づき、システムを段階的に構築するための GitHub Copilot 向けプロンプト集です。

---

### Step 0: プロジェクト構成の作成
まず、全体のディレクトリ構造を定義させます。

**Prompt:**
> [architecture.md](../architecture.md) の設計に基づき、Python をベースとしたマイクロサービス構成のプロジェクトディレクトリ構造を提案してください。
> 以下のコンポーネントを独立したディレクトリとして含めてください：
> - `splade-api`: SPLADE モデルによるベクトル化サービス
> - `indexer`: Scrapbox からのデータ取得と ES へのインデックス作成スクリプト
> - `app-api`: RAG ロジックを制御するメイン API
> - `web-ui`: Streamlit もしくは Next.js による UI

---

### Step 1: SPLADE API Service (GPU/疎ベクトル変換)
SPLADE モデルを FastAPI でラップします。

**Prompt:**
> `splade-api/` 内に、FastAPI を使用した疎ベクトル変換 API を作成してください。
> - `naver/splade-cocondenser-ensemblev2` モデルを Hugging Face からロードすること
> - 入力テキストを受け取り、Elasticsearch の `rank_features` 形式に適した `{token: weight}` の辞書形式を返すエンドポイント `/encode` を作成してください
> - GPU が利用可能な場合は GPU を使用するようにしてください

---

### Step 2: Data Indexer (データ注入パイプライン)
Scrapbox のデータを取得し、インデックスを作成します。

**Prompt:**
> `indexer/` 内に、Scrapbox のデータを Elasticsearch に登録するスクリプトを作成してください。
> - Scrapbox の JSON データを読み込むか API から取得すること
> - `langchain` の `RecursiveCharacterTextSplitter` 等を使用してテキストをチャンク分割すること
> - 各チャンクを `splade-api` (localhost:8001/encode) に送信して疎ベクトルを取得すること
> - Elasticsearch のインデックスを作成し、`rank_features` 型のフィールドを含むマッピングを定義してデータを投入してください

---

### Step 3: App API (RAG エンジン)
検索、プロンプト構築、Gemma 3 呼び出しを行う中核部分です。

**Prompt:**
> `app-api/` 内に、メインの RAG ロジックを実装した FastAPI アプリを作成してください。
> 以下のフローを `/query` エンドポイントで実装してください：
> 1. ユーザーのクエリを `splade-api` でベクトル化
> 2. Elasticsearch で `rank_features` を用いた類似度検索を実行
> 3. 検索結果の上位数件をコンテキストとして抽出し、プロンプトを構築
> 4. Ollama (localhost:11434) で稼働している `gemma3` モデルにプロンプトを送信し、回答をストーミング形式で返す
> - 非同期処理 (httpx) を使用してください

---

### Step 4: Web UI (フロントエンド)
ユーザーインターフェースです。

**Prompt:**
> `web-ui/` 内に、Streamlit を使用したチャット UI を作成してください。
> - ユーザーが質問を入力するチャット入力欄
> - `app-api` (localhost:8000/query) を呼び出し、回答をリアルタイムで表示すること
> - 参考にした Scrapbox ページのリンク（メタデータ）をサイドバーや回答の下に表示すること

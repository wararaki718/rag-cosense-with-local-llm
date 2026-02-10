# Project Context: RAG Cosense with Local LLM

このプロジェクトは、Scrapbox (Cosense) のデータをソースとし、SPLADE による疎ベクトル変換と Elasticsearch を活用した RAG (Retrieval-Augmented Generation) システムです。

## 技術スタック (Technology Stack)
- **Language**: Python 3.10+
- **Search Engine**: Elasticsearch (疎ベクトル検索に `rank_features` を使用)
- **Main Framework**: FastAPI (App API & SPLADE API)
- **LLM Service**: Gemma 3 (Ollama / vLLM を介して公開)
- **Sparse Vector Model**: SPLADE (`naver/splade-cocondenser-ensembledistil`)
- **UI Framework**: Streamlit (もしくは Next.js)

## 構成コンポーネント (Component Architecture)
1. **splade-api**: テキストを `{token: weight}` 形式の疎ベクトルに変換する API。
2. **indexer**: Scrapbox からデータを取得し、チャンク分割、ベクトル化を経て Elasticsearch へ登録するバッチプロセス。
3. **app-api**: RAG フローの司令塔。クエリのベクトル化、ES 検索、プロンプト構築、Gemma 3 への推論依頼を制御。
4. **web-ui**: ユーザーとのチャットインターフェース。

## 重要な実装ガイドライン (Key Implementation Guidelines)
- **疎ベクトル検索**: Elasticsearch の `rank_features` フィールドを使い、SPLADE から返される辞書形式のベクトルを格納・検索すること。
- **マイクロサービス設計**: 各サービスは独立して動作し、API 経由で通信する。特に GPU を利用する `splade-api` と `Gemma 3 API` は分離して管理する。
- **レスポンス品質**:
    - LLM の回答は `StreamingResponse` を使用して、Web UI まで透過的にストリーミングすること。
    - 検索結果の出典（Scrapbox ページタイトル、URL）を必ずメタデータとして保持・表示すること。
- **パフォーマンスと信頼性**:
    - サービス間通信には `httpx` (async) を使用すること。
    - ネットワーク遅延や GPU リソースの競合を考慮したエラーハンドリングを実装すること。

## コーディングスタイル
- 型ヒント (Type Hints) を明示的に使用すること。
- docstring は Google スタイルで記述すること。
- 非同期処理 (async/await) を積極的に活用すること。

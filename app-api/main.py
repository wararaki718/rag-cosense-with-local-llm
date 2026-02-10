import json
from typing import AsyncGenerator, Dict, List

import httpx
from elasticsearch import AsyncElasticsearch
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from loguru import logger

# --- Configuration ---
class Settings(BaseSettings):
    elasticsearch_url: str = "http://localhost:9200"
    splade_api_url: str = "http://localhost:8001/encode"
    ollama_url: str = "http://localhost:11434/api/generate"
    index_name: str = "scrapbox-index"
    llm_model: str = "gemma3"
    
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()

# --- Schemas ---
class QueryRequest(BaseModel):
    user_query: str = Field(..., description="The user's question")
    top_k: int = Field(default=5, description="Number of documents to retrieve")

# --- App Initialization ---
app = FastAPI(title="Cosense RAG App API")
es_client = AsyncElasticsearch(settings.elasticsearch_url)

# --- Service Logic ---

async def get_query_vector(text: str) -> Dict[str, float]:
    """Retrieves the sparse vector representation of the query from SPLADE API.
    
    Args:
        text: User query string.
        
    Returns:
        A dictionary mapping tokens to their importance weights.
        
    Raises:
        HTTPException: If the SPLADE API call fails.
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            logger.info(f"Vectorizing query via {settings.splade_api_url}: {text[:50]}...")
            response = await client.post(
                settings.splade_api_url, 
                json={"text": text}
            )
            response.raise_for_status()
            data = response.json()
            # Extract the sparse_vector from the response
            return data.get("sparse_vector", {})
        except httpx.HTTPStatusError as e:
            logger.error(f"SPLADE API status error: {e.response.status_code} - {e.response.text}")
            raise HTTPException(status_code=503, detail=f"Vectorization service error: {e.response.status_code}")
        except httpx.RequestError as e:
            logger.error(f"SPLADE API connection error: {e}")
            raise HTTPException(status_code=503, detail="Vectorization service unreachable")
        except Exception as e:
            logger.exception(f"Unexpected error calling SPLADE API")
            raise HTTPException(status_code=500, detail=str(e))

async def search_documents(query_vector: Dict[str, float], top_k: int) -> List[Dict]:
    """Performs a sparse vector search on Elasticsearch using rank_features.
    
    Args:
        query_vector: Token-weight dictionary.
        top_k: Number of results to return.
        
    Returns:
        List of document dictionaries containing content and metadata.
    """
    if not query_vector:
        return []

    # Build rank_feature query clauses for each token
    should_clauses = [
        {"rank_feature": {"field": f"sparse_vector.{token}", "boost": weight}}
        for token, weight in query_vector.items()
    ]

    body = {
        "query": {
            "bool": {
                "should": should_clauses
            }
        },
        "size": top_k
    }

    try:
        logger.info(f"Searching Elasticsearch index: {settings.index_name}")
        response = await es_client.search(index=settings.index_name, body=body)
        hits = response["hits"]["hits"]
        
        results = []
        for hit in hits:
            source = hit["_source"]
            results.append({
                "title": source.get("title", "Untitled"),
                "content": source.get("content", ""),
                "url": source.get("url", ""),
                "score": hit["_score"]
            })
        return results
    except Exception as e:
        logger.error(f"Elasticsearch search error: {e}")
        return []

def build_prompt(query: str, contexts: List[Dict]) -> str:
    """Constructs the augmented prompt for Gemma 3.
    
    Args:
        query: User's question.
        contexts: List of retrieved document snippets.
        
    Returns:
        Formatted prompt string.
    """
    if not contexts:
        return f"ユーザーの質問: {query}\n\n関連する情報がナレッジベースで見つかりませんでした。一般的な知識で回答してください。"

    context_text = "\n\n".join([
        f"--- ソース: {idx+1}. {c['title']} ({c['url']}) ---\n{c['content']}"
        for idx, c in enumerate(contexts)
    ])

    return f"""あなたはScrapbox（Cosense）の情報を基に回答するAIアシスタントです。
以下のコンテキスト情報を使用して、ユーザーの質問に日本語で分かりやすく、かつ正確に答えてください。
回答には、参考にしたソースのタイトルを必ず含めてください。

# コンテキスト情報:
{context_text}

# ユーザーの質問:
{query}

# 回答:
"""

async def generate_response_stream(prompt: str, contexts: List[Dict]) -> AsyncGenerator[str, None]:
    """Streams the response from Ollama/Gemma 3, prefixing with metadata.
    
    Args:
        prompt: The constructed prompt.
        contexts: The supporting documents retrieved.
    """
    # Send metadata first as a JSON object separated by a custom delimiter
    metadata = {
        "type": "metadata",
        "sources": [
            {"title": c["title"], "url": c["url"], "score": c["score"]} 
            for c in contexts
        ]
    }
    yield json.dumps(metadata) + "\n---\n"

    payload = {
        "model": settings.llm_model,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": 0.7,
            "top_p": 0.9,
        }
    }

    async with httpx.AsyncClient(timeout=None) as client:
        try:
            async with client.stream("POST", settings.ollama_url, json=payload) as response:
                if response.status_code != 200:
                    yield f"Error: LLM service returned {response.status_code}"
                    return

                async for line in response.aiter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                        if "response" in chunk:
                            yield chunk["response"]
                        if chunk.get("done"):
                            break
                    except json.JSONDecodeError:
                        continue
        except httpx.RequestError as e:
            logger.error(f"Ollama connection error: {e}")
            yield f"Error: Failed to connect to LLM service."

# --- API Endpoints ---

@app.post("/query")
async def query(request: QueryRequest):
    """Entry point for the RAG pipeline."""
    try:
        # 1. Vectorize query via SPLADE API
        query_vector = await get_query_vector(request.user_query)
        
        # 2. Retrieve relevant documents from Elasticsearch
        contexts = await search_documents(query_vector, request.top_k)
        
        # 3. Build prompt with context
        prompt = build_prompt(request.user_query, contexts)
        
        # 4. Stream response from Gemma 3
        return StreamingResponse(
            generate_response_stream(prompt, contexts),
            media_type="text/event-stream"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected error in query pipeline")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """Health check endpoint."""
    es_status = "connected" if await es_client.ping() else "disconnected"
    return {"status": "up", "elasticsearch": es_status}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

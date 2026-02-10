import json
import os
import argparse
import httpx
import asyncio
import urllib.parse
from typing import List, Dict, Any
from elasticsearch import AsyncElasticsearch, helpers
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

# Configuration
ES_URL = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
SPLADE_API_URL = os.getenv("SPLADE_API_URL", "http://localhost:8001/encode")
INDEX_NAME = "scrapbox-index"
SCRAPBOX_PROJECT = os.getenv("SCRAPBOX_PROJECT")
SCRAPBOX_SID = os.getenv("SCRAPBOX_SID")

# Use AsyncElasticsearch
es = AsyncElasticsearch(ES_URL)

async def create_index() -> None:
    """Create Elasticsearch index with rank_features mapping."""
    mapping = {
        "mappings": {
            "properties": {
                "title": {"type": "text", "analyzer": "keyword"},
                "content": {"type": "text"},
                "url": {"type": "keyword"},
                "sparse_vector": {"type": "rank_features"},
                "chunk_id": {"type": "integer"}
            }
        }
    }
    try:
        if await es.indices.exists(index=INDEX_NAME):
            print(f"Index {INDEX_NAME} already exists.")
            return
        await es.indices.create(index=INDEX_NAME, body=mapping)
        print(f"Created index {INDEX_NAME}")
    except Exception as e:
        print(f"Error creating index: {e}")
        raise

async def get_sparse_vector(text: str) -> Dict[str, float]:
    """Call SPLADE API to get sparse vector."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(SPLADE_API_URL, json={"text": text})
            response.raise_for_status()
            data = response.json()
            return data.get("sparse_vector", {})
        except Exception as e:
            print(f"Error calling SPLADE API: {e}")
            return {}

async def fetch_scrapbox_pages(project_name: str) -> List[Dict[str, str]]:
    """Fetch all pages from Scrapbox API."""
    cookies = {}
    if SCRAPBOX_SID:
        cookies["connect.sid"] = SCRAPBOX_SID
    async with httpx.AsyncClient(cookies=cookies, timeout=30.0) as client:
        print(f"Fetching page list for project: {project_name}")
        list_url = f"https://scrapbox.io/api/pages/{project_name}?limit=1000"
        try:
            response = await client.get(list_url)
            response.raise_for_status()
            pages_data = response.json().get("pages", [])
        except Exception as e:
            print(f"Error fetching page list: {e}")
            return []
        docs = []
        for i, page in enumerate(pages_data):
            title = page.get("title")
            print(f"[{i+1}/{len(pages_data)}] Fetching content for: {title}")
            safe_title = urllib.parse.quote(title)
            page_url = f"https://scrapbox.io/api/pages/{project_name}/{safe_title}"
            try:
                page_res = await client.get(page_url)
                page_res.raise_for_status()
                full_page = page_res.json()
                lines = [line.get("text", "") for line in full_page.get("lines", [])]
                content = "\n".join(lines)
                public_url = f"https://scrapbox.io/{project_name}/{title.replace(' ', '_')}"
                docs.append({"title": title, "content": content, "url": public_url})
                await asyncio.sleep(0.1)
            except Exception as e:
                print(f"Failed to fetch page '{title}': {e}")
        return docs

async def index_documents(documents: List[Dict[str, str]]) -> None:
    """Chunk, vectorize and index documents into Elasticsearch."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", "。", "、", " ", ""])
    actions = []
    
    async def run_batch(batch_actions):
        try:
            await helpers.async_bulk(es, batch_actions)
            print(f"Indexed {len(batch_actions)} chunks...")
        except helpers.BulkIndexError as e:
            print(f"BulkIndexError: {len(e.errors)} documents failed.")
            for error in e.errors:
                print(f"Error detail: {json.dumps(error, indent=2)}")
            raise

    for doc in documents:
        chunks = text_splitter.split_text(doc["content"])
        print(f"Processing '{doc['title']}' ({len(chunks)} chunks)...")
        for i, chunk_text in enumerate(chunks):
            text_to_encode = f"{doc['title']}\n{chunk_text}"
            sparse_vector = await get_sparse_vector(text_to_encode)
            if not sparse_vector: continue
            
            # Filter for rank_features: keys must be strings, values > 0
            filtered_vector = {k: v for k, v in sparse_vector.items() if v > 0}
            if not filtered_vector: continue

            action = {
                "_index": INDEX_NAME,
                "_source": {
                    "title": doc["title"],
                    "content": chunk_text,
                    "url": doc["url"],
                    "sparse_vector": filtered_vector,
                    "chunk_id": i
                }
            }
            actions.append(action)
            if len(actions) >= 50:
                await run_batch(actions)
                actions = []
    if actions:
        await run_batch(actions)
    print("Successfully finished indexing all documents.")

async def main():
    parser = argparse.ArgumentParser(description="Index Scrapbox via API.")
    parser.add_argument("--project", help="Scrapbox project name")
    args = parser.parse_args()
    project = args.project or SCRAPBOX_PROJECT
    if not project:
        print("Error: project not specified.")
        return
    
    # Wait for Elasticsearch to be ready (Retry logic)
    max_retries = 20
    retry_interval = 5
    connected = False
    
    print(f"Connecting to Elasticsearch at {ES_URL}...")
    for i in range(max_retries):
        try:
            if await es.ping():
                connected = True
                break
        except Exception as e:
            print(f"Connection attempt {i+1} failed: {e}")
        print(f"[{i+1}/{max_retries}] Waiting for Elasticsearch...")
        await asyncio.sleep(retry_interval)

    if not connected:
        print(f"Error: Could not connect to Elasticsearch at {ES_URL} after {max_retries} retries.")
        await es.close()
        return

    try:
        await create_index()
        documents = await fetch_scrapbox_pages(project)
        if documents:
            await index_documents(documents)
            print(f"Successfully finished indexing {len(documents)} pages.")
        else:
            print("No documents found to index.")
    finally:
        await es.close()

if __name__ == "__main__":
    asyncio.run(main())

import json
import os
import argparse
import httpx
import asyncio
from typing import List, Dict, Any
from elasticsearch import Elasticsearch, helpers
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

# Configuration
ES_URL = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
SPLADE_API_URL = os.getenv("SPLADE_API_URL", "http://localhost:8001/encode")
INDEX_NAME = "scrapbox-index"

es = Elasticsearch(ES_URL)

def create_index() -> None:
    """Create Elasticsearch index with rank_features mapping.
    
    This function defines the mapping for the index, ensuring that 'sparse_vector'
    is configured correctly for SPLADE vectors.
    """
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
    if es.indices.exists(index=INDEX_NAME):
        print(f"Index {INDEX_NAME} already exists.")
        return
    
    es.indices.create(index=INDEX_NAME, body=mapping)
    print(f"Created index {INDEX_NAME}")

async def get_sparse_vector(text: str) -> Dict[str, float]:
    """Call SPLADE API to get sparse vector.

    Args:
        text (str): The text to be vectorized.

    Returns:
        Dict[str, float]: A dictionary of token weights.
    """
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(SPLADE_API_URL, json={"text": text})
            response.raise_for_status()
            data = response.json()
            return data.get("sparse_vector", {})
        except Exception as e:
            print(f"Error calling SPLADE API for text '{text[:20]}...': {e}")
            return {}

def process_scrapbox_json(file_path: str) -> List[Dict[str, str]]:
    """Load and parse Scrapbox JSON export.

    Args:
        file_path (str): Path to the Scrapbox JSON file.

    Returns:
        List[Dict[str, str]]: A list of documents containing title, content, and url.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    project_name = data.get("name")
    pages = data.get("pages", [])
    
    docs = []
    for page in pages:
        title = page.get("title")
        lines = page.get("lines", [])
        # Some lines might be empty or start with whitespace
        content = "\n".join(lines)
        # Simple URL construction (Scrapbox uses underscores for spaces)
        safe_title = title.replace(' ', '_')
        url = f"https://scrapbox.io/{project_name}/{safe_title}"
        
        docs.append({
            "title": title,
            "content": content,
            "url": url
        })
    return docs

async def index_documents(documents: List[Dict[str, str]]) -> None:
    """Chunk, vectorize and index documents into Elasticsearch.

    Args:
        documents (List[Dict[str, str]]): List of documents to index.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "、", " ", ""]
    )
    
    actions: List[Dict[str, Any]] = []
    
    # Process documents one by one
    for doc in documents:
        chunks = text_splitter.split_text(doc["content"])
        print(f"Processing '{doc['title']}' ({len(chunks)} chunks)...")
        
        for i, chunk_text in enumerate(chunks):
            # Combined text (Title + Chunk) provides better context for sparse vectorization
            text_to_encode = f"{doc['title']}\n{chunk_text}"
            sparse_vector = await get_sparse_vector(text_to_encode)
            
            if not sparse_vector:
                print(f"Warning: Empty sparse vector for chunk {i} of '{doc['title']}'")
                continue
            
            action = {
                "_index": INDEX_NAME,
                "_source": {
                    "title": doc["title"],
                    "content": chunk_text,
                    "url": doc["url"],
                    "sparse_vector": sparse_vector,
                    "chunk_id": i
                }
            }
            actions.append(action)
            
            # Batch indexing for efficiency
            if len(actions) >= 50:
                helpers.bulk(es, actions)
                print(f"Indexed {len(actions)} chunks...")
                actions = []
    
    if actions:
        helpers.bulk(es, actions)
        print(f"Indexed final {len(actions)} chunks.")
    
    print(f"Successfully finished indexing all documents.")

async def main():
    parser = argparse.ArgumentParser(description="Index Scrapbox data into Elasticsearch with SPLADE vectors.")
    parser.add_argument("--file", required=True, help="Path to Scrapbox JSON export file")
    args = parser.parse_args()
    
    if not os.path.exists(args.file):
        print(f"Error: File '{args.file}' not found.")
        return

    create_index()
    documents = process_scrapbox_json(args.file)
    await index_documents(documents)

if __name__ == "__main__":
    asyncio.run(main())

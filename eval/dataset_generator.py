import os
import asyncio
import pandas as pd
from elasticsearch import AsyncElasticsearch
from langchain_core.documents import Document
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from evaluator_config import get_evaluator_llm, get_evaluator_embeddings
from dotenv import load_dotenv

load_dotenv()

ES_URL = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
INDEX_NAME = "scrapbox-index"

async def fetch_documents_from_es(limit=50):
    """Fetch documents from Elasticsearch and convert to LangChain format."""
    es = AsyncElasticsearch(ES_URL)
    try:
        query = {"query": {"match_all": {}}, "size": limit}
        response = await es.search(index=INDEX_NAME, body=query)
        hits = response["hits"]["hits"]
        
        docs = []
        for hit in hits:
            source = hit["_source"]
            metadata = {
                "title": source.get("title"),
                "url": source.get("url"),
                "chunk_id": source.get("chunk_id")
            }
            content = source.get("content", "")
            if content:
                docs.append(Document(page_content=content, metadata=metadata))
        
        await es.close()
        return docs
    except Exception as e:
        print(f"Error fetching from ES: {e}")
        await es.close()
        return []

async def generate_testset():
    print("Fetching documents from Elasticsearch...")
    documents = await fetch_documents_from_es(limit=20)
    
    if not documents:
        print("No documents found in Elasticsearch. Please run the indexer first.")
        return

    print(f"Generating synthetic testset from {len(documents)} document chunks...")
    
    generator_llm = get_evaluator_llm()
    generator_embeddings = get_evaluator_embeddings()
    
    generator = TestsetGenerator.from_langchain(
        generator_llm,
        generator_llm, # Ragas uses LLM for both generation and critique sometimes
        generator_embeddings
    )

    # Define the distribution of question types
    distributions = {
        simple: 0.5,
        reasoning: 0.25,
        multi_context: 0.25
    }

    testset = generator.generate_with_langchain_docs(
        documents, 
        test_size=10, 
        distributions=distributions
    )

    df = testset.to_pandas()
    # Save in the same directory as this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "synthetic_testset.csv")
    df.to_csv(output_path, index=False)
    print(f"Testset saved to {output_path}")

if __name__ == "__main__":
    asyncio.run(generate_testset())

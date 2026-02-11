import os
from langchain_ollama import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Configuration for local Gemma 3 via Ollama
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "gemma3") 

def get_evaluator_llm():
    """Returns a raw LangChain LLM instance for Gemma 3."""
    return ChatOllama(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
    )

def get_evaluator_embeddings():
    """Returns a raw LangChain Embeddings instance."""
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )

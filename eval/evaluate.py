import os
import json
import asyncio
import pandas as pd
import httpx
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from evaluator_config import get_evaluator_llm, get_evaluator_embeddings
from dotenv import load_dotenv

load_dotenv()

APP_API_URL = os.getenv("APP_API_URL", "http://localhost:8000/query")

async def get_rag_response(question: str):
    """Calls app-api and extracts contexts and full answer."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            payload = {"user_query": question, "top_k": 3}
            contexts = []
            full_answer = ""
            
            async with client.stream("POST", APP_API_URL, json=payload) as response:
                if response.status_code != 200:
                    return None, None
                
                async for chunk in response.aiter_text():
                    if "---" in chunk and not full_answer:
                        # Split metadata and start of answer
                        parts = chunk.split("---", 1)
                        try:
                            meta_data = json.loads(parts[0].strip())
                            if meta_data.get("type") == "metadata":
                                contexts = [s["content"] for s in meta_data.get("sources", [])]
                            if len(parts) > 1:
                                full_answer += parts[1].strip()
                        except:
                            pass
                    else:
                        full_answer += chunk
            
            return full_answer.strip(), contexts
        except Exception as e:
            print(f"Error calling App API: {e}")
            return None, None

async def run_evaluation():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    testset_path = os.path.join(script_dir, "synthetic_testset.csv")
    if not os.path.exists(testset_path):
        print(f"Testset not found at {testset_path}. Please run dataset_generator.py first.")
        return

    df_test = pd.read_csv(testset_path)
    print(f"Loaded {len(df_test)} cases from testset.")

    questions = df_test["question"].tolist()
    ground_truths = df_test["ground_truth"].tolist()
    
    answers = []
    contexts_list = []

    print("Running RAG pipeline for all questions...")
    for q in questions:
        print(f"Querying: {q}")
        ans, ctx = await get_rag_response(q)
        answers.append(ans if ans else "")
        contexts_list.append(ctx if ctx else [])

    # Prepare dataset for Ragas
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths
    }
    dataset = Dataset.from_dict(data)

    print("Evaluating with Ragas...")
    
    evaluator_llm = get_evaluator_llm()
    evaluator_embeddings = get_evaluator_embeddings()

    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        ],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings
    )

    print("\n--- Evaluation Results ---")
    print(result)
    
    # Save results
    result_df = result.to_pandas()
    output_path = os.path.join(script_dir, "evaluation_results.csv")
    result_df.to_csv(output_path, index=False)
    print(f"\nDetailed results saved to {output_path}")

if __name__ == "__main__":
    asyncio.run(run_evaluation())

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForMaskedLM, AutoTokenizer
from typing import Dict
import uvicorn
import os

app = FastAPI(
    title="SPLADE Vectorization API",
    description="API to convert text into sparse vectors for Elasticsearch rank_features."
)

# Model configuration
MODEL_NAME = os.getenv("MODEL_NAME", "naver/splade-cocondenser-ensemblev2")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Loading SPLADE model '{MODEL_NAME}' on {device}...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME).to(device)
    model.eval()
except Exception as e:
    print(f"Error loading model: {e}")
    raise e

class EncodeRequest(BaseModel):
    text: str

class EncodeResponse(BaseModel):
    sparse_vector: Dict[str, float]

@app.post("/encode", response_model=EncodeResponse)
async def encode(request: EncodeRequest) -> EncodeResponse:
    """
    Encode text into a sparse vector (token: weight) format for Elasticsearch.

    Args:
        request (EncodeRequest): The request containing the text to encode.

    Returns:
        EncodeResponse: A dictionary where keys are tokens and values are weights.
            Dots in tokens are replaced with underscores for Elasticsearch compatibility.
    """
    try:
        inputs = tokenizer(request.text, return_tensors="pt", truncation=True, max_length=512).to(device)
        
        with torch.no_grad():
            logits = model(**inputs).logits
            
        # SPLADE representation: max aggregation over tokens for each dimension
        # log(1 + relu(w)) is the standard SPLADE weight calculation
        weights = torch.max(
            torch.log1p(torch.relu(logits)) * inputs.attention_mask.unsqueeze(-1), 
            dim=1
        ).values.squeeze()
        
        # Extract indices where weights are non-zero
        indices = weights.nonzero().squeeze().cpu().tolist()
        # Convert to list for mapping
        weights_list = weights.cpu().tolist()
        
        # Map token IDs to actual tokens and their weights
        id_to_token = {v: k for k, v in tokenizer.get_vocab().items()}
        
        sparse_vector = {}
        # Ensure indices is iterable even if only one token is active
        if isinstance(indices, int):
            indices = [indices]
            
        for idx in indices:
            token = id_to_token[idx]
            weight = float(weights_list[idx])
            if weight > 0:
                # Elasticsearch rank_features keys cannot contain dots '.'
                # Replace with underscores '_' to avoid indexing errors
                safe_token = token.replace(".", "_")
                sparse_vector[safe_token] = round(weight, 4)
                
        return EncodeResponse(sparse_vector=sparse_vector)
    except Exception as e:
        print(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "device": str(device), "model": MODEL_NAME}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)

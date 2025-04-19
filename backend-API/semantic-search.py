from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from datasets import Dataset

app = FastAPI()

# Define input model for search
class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

# Global variables
model = None
index = None
data = None
claims = []

@app.on_event("startup")
def load_resources():
    global model, index, data, claims

    # Load CSV data
    df = pd.read_csv("./df_combine.csv")
    dataset = Dataset.from_pandas(df)

    # Keep only necessary columns
    columns_to_keep = ["title", "claims", "ipc"]
    columns_to_remove = set(dataset.column_names).symmetric_difference(columns_to_keep)
    dataset = dataset.remove_columns(columns_to_remove)

    # Filter out short claims
    dataset = dataset.map(lambda x: {"claims_length": len(x["claims"].split())})
    dataset = dataset.filter(lambda x: x["claims_length"] > 15)

    claims = dataset["claims"]

    # Load model
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(claims, convert_to_numpy=True)

    # Build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

@app.post("/search")
def search(request: SearchRequest):
    query_embedding = model.encode([request.query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, request.top_k)
    
    results = []
    for i, score in zip(indices[0], distances[0]):
        results.append({
            "claim": claims[i],
            "score": float(score)
        })

    return {"query": request.query, "results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

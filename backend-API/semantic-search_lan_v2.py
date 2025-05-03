from fastapi import FastAPI
from pydantic import BaseModel

import pandas as pd
import numpy as np

import sqlite3
import sqlite_vec 
from sqlite_vec import serialize_float32

from transformers import AutoTokenizer, AutoModel
from llama_cpp import Llama

import torch
import os

#===
# Supporting Functions
#===
def cls_pooling(model_output):
    return model_output.last_hidden_state[:, 0]

def get_embeddings(text_list, imp_tokenizer, imp_model):
    encoded_input = imp_tokenizer(
        text_list, padding=True, truncation=True, return_tensors="pt"
    )
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    model_output = imp_model(**encoded_input)
    return cls_pooling(model_output)

#===
# Environemnt setup
#===
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
device = torch.device("cpu")

#===
# # FASTAPI object, input type, etc.
#===
app = FastAPI()

# Define input model for search
class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


# Global variables
all_dataset = None
model = None
tokenizer = None
all_index_faiss = None
input_gen_ai = None
gen_model = None
db_name = 'G:/PhD/EPO2025/SQLite_Tutorial/epo.db'

@app.on_event("startup")
def load_resources():
    # global model, index, data, claims
    global device, all_dataset, model, tokenizer, all_index_faiss, gen_model

    device = "cpu"
    
    #====
    #  Load model for embeddings
    #====
    token_ckpt = "sadickam/sdg-classification-bert"
    model_ckpt = "G:/PhD/EPO2025/Shared/current_batch" 
    tokenizer = AutoTokenizer.from_pretrained(token_ckpt)
    model = AutoModel.from_pretrained(model_ckpt)

    #===
    # Load model for gen ai
    #===
    model_pre_downloaded_path = "G:/PhD/EPO2025/Shared/models--MaziyarPanahi--Mistral-7B-Instruct-v0.3-GGUF/snapshots/ce89f595755a4bf2e2e05d155cc43cb847c78978/Mistral-7B-Instruct-v0.3.Q4_K_M-002.gguf"
    gen_model = Llama(model_path=model_pre_downloaded_path, n_ctx= 2048*8)

@app.post("/search")
async def search(request: SearchRequest):
    global input_gen_ai, db_name
    question_embedding = get_embeddings([request.query], tokenizer, model).cpu().detach().numpy()

    try:
        with sqlite3.connect(db_name) as conn:

            # load the `sqlite-vec` extention into the connected db
            ## NOTE:
            ## must load the `sqlite-vec` extention everytime connect to the db, 
            ## in order to use the vec table created using extension `sqlte-vec` and `sqlite-vec` functions
            conn.enable_load_extension(True) # start loading extensions
            sqlite_vec.load(conn)
            conn.enable_load_extension(True) # end loading extensions

            # Query to the created vec table
            query = question_embedding.tolist()
            print(type(query))
            rows = conn.execute(
            f"""
            SELECT
                rowid,
                distance
            FROM vec_items
            WHERE embedding MATCH ?
            ORDER BY distance
            LIMIT {str(request.top_k)}
            """,
            [serialize_float32(query[0])],
            ).fetchall()

            # print(f"{'='*10}")
            # print(rows)

            # Check data in the mata table
            results = []
            for i, tuple in enumerate(rows):
                meta_query = f"""
                    SELECT
                        title,
                        claims
                    FROM meta_data_embeddings
                    WHERE rowid={int(tuple[0])}
                    """
                cursor = conn.cursor()
                res = cursor.execute(meta_query).fetchall()

                results.append(
                        {
                            "TITLE": res[0][0],
                            "DISTANCE": float(tuple[1]),
                            "CLAIMS": res[0][1]
                        }
                    )

    except sqlite3.OperationalError as e:
        print(e)

    input_gen_ai = {"query": request.query, "relevant_docs": results} 
    # NOTE:
    # JSON encoder for returned object: https://fastapi.tiangolo.com/advanced/response-directly/
    # All of the returned objects MUST be converted to known python standard objects.
    return input_gen_ai 

@app.post("/answer")
async def answer(request: SearchRequest):
    global input_gen_ai, gen_model

    context = ""
    for idx in range(len(input_gen_ai["relevant_docs"])):
        print(idx)
        temp = f"""
        Title: {input_gen_ai["relevant_docs"][idx]["TITLE"]}
        Context: {input_gen_ai["relevant_docs"][idx]["CLAIMS"]}
        """
        context = context + temp
        

    rag_prompt = f"""
    Context information is below.
    ---------------------
    {context}
    ---------------------
    Given the context information and not prior knowledge, answer the query.
    Query: {request.query}
    Answer:
    """
    
    output = gen_model(
        rag_prompt,
        max_tokens=512,
        temperature=1,
        top_p=0.95,
        echo=False,
        stop=["#"],
    )
    
    reply_prompt = f"""
    Context information is below.
    ---------------------
    {context}
    ---------------------
    Given the context information and not prior knowledge, answer the query.
    Query: {request.query}
    Answer: {output["choices"][0]["text"].strip()}
    """

    return {"reply": reply_prompt}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

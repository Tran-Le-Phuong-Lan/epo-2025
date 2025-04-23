from fastapi import FastAPI
from pydantic import BaseModel

import pandas as pd
import numpy as np
import faiss
from datasets import Dataset
from datasets import load_from_disk
from datasets import concatenate_datasets

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

@app.on_event("startup")
def load_resources():
    # global model, index, data, claims
    global device, all_dataset, model, tokenizer, all_index_faiss

    device = "cpu"
    #====
    #  Load data
    #====
    # check how many datasets is available
    PATH = './data_embeddings'
    dir_names = []
    for root, dirnames, filenames in os.walk(PATH):
        if dirnames != []: 
            for idx, dirname in enumerate(dirnames):
                dir_names.append(dirname)

    # load the data into program
    load_all_dataset = []
    for idx, names in enumerate(dir_names):
        dataset_name = PATH + '/'+ names
        loaded_dataset = load_from_disk(dataset_name)
        load_all_dataset.append(loaded_dataset)
    
    all_dataset = concatenate_datasets(load_all_dataset)

    #====
    #  Add FAISS index
    #====
    # get the stored embeddings 
    all_emds = np.array(all_dataset['embeddings'])
    # add faiss index
    all_index_faiss = faiss.IndexFlatL2(all_emds.shape[1]) 
    all_index_faiss.add(all_emds)
    
    #====
    #  Load model for embeddings
    #====
    token_ckpt = "sadickam/sdg-classification-bert"
    model_ckpt = "../current_batch" 
    tokenizer = AutoTokenizer.from_pretrained(token_ckpt)
    model = AutoModel.from_pretrained(model_ckpt)

    #===
    # Load model for gen ai
    #===
    model_pre_downloaded_path = "C:/Users/20245580/.cache/huggingface/hub/models--MaziyarPanahi--Mistral-7B-Instruct-v0.3-GGUF/snapshots/ce89f595755a4bf2e2e05d155cc43cb847c78978/Mistral-7B-Instruct-v0.3.Q4_K_M.gguf"
    gen_model = Llama(model_path=model_pre_downloaded_path, n_ctx= 2048*4)

@app.post("/search")
async def search(request: SearchRequest):
    question_embedding = get_embeddings([request.query], tokenizer, model).cpu().detach().numpy()

    [distance_res], [index_res] = all_index_faiss.search(question_embedding, request.top_k)

    results = []
    for i, d in zip(index_res, distance_res):
        # results.append(i)
        results.append(
            {
                "TITLE": all_dataset['title'][i],
                "DISTANCE": float(d),
                "CLAIMS": all_dataset['claims'][i]
            }
        )

    input_gen_ai = {"query": request.query, "relevant_docs": results} 
    # NOTE:
    # JSON encoder for returned object: https://fastapi.tiangolo.com/advanced/response-directly/
    # All of the returned objects MUST be converted to known python standard objects.
    return input_gen_ai 

@app.get("/answer")
async def answer(request: SearchRequest):
    global input_gen_ai, llm

    context = ""
    for idx in range(len(input_gen_ai["relevant_docs"])):
        temp = f"""
        Title: {input_gen_ai["relevant_docs"][idx]["TITLE"]}
        Context: {input_gen_ai["relevant_docs"][idx]["CLAIMS"]}
        """
        context = context + temp
        

    retrieved_chunk = context

    rag_prompt = f"""
    Context information is below.
    ---------------------
    {retrieved_chunk}
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
    {retrieved_chunk}
    ---------------------
    Given the context information and not prior knowledge, answer the query.
    Query: {request.query}
    Answer: {output["choices"][0]["text"].strip()}
    """
    output_text = output["choices"][0]["text"].strip()


    return {"reply": reply_prompt}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

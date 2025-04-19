from fastapi import FastAPI
from pydantic import BaseModel

import pandas as pd
import numpy as np
import faiss
from datasets import Dataset
from datasets import load_from_disk
from datasets import concatenate_datasets

from transformers import AutoTokenizer, AutoModel
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

# Environemnt setup
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
device = torch.device("cpu")

# FASTAPI app
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

@app.on_event("startup")
def load_resources():
    # global model, index, data, claims
    global device, all_dataset, model, tokenizer, all_index_faiss

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
    num_dat = len(dir_names)
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
    #  Load model
    #====
    token_ckpt = "sadickam/sdg-classification-bert"
    model_ckpt = "../current_batch" 
    tokenizer = AutoTokenizer.from_pretrained(token_ckpt)
    model = AutoModel.from_pretrained(model_ckpt)

@app.post("/search")
def search(request: SearchRequest):
    question_embedding = get_embeddings(request.query, tokenizer, model).cpu().detach().numpy()

    [distance_res], [index_res] = all_index_faiss.search(question_embedding, request.top_k)

    results = []
    for i, d in zip(index_res, distance_res):
        results.append(
            {
                "TITLE": all_dataset['title'][i],
                "DISTANCE": d,
                "CLAIMS": all_dataset['claims'][i]
            }
        )

    return {"query": request.query, "results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

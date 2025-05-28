from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import numpy as np

import sqlite3
import sqlite_vec 
from sqlite_vec import serialize_float32

from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from mistralai import Mistral

import torch
import os
import requests
from fastapi.middleware.cors import CORSMiddleware
import json
import string

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
# os.environ["SSL_CERT_FILE"] = "C:/Users/20245580/AppData/Local/anaconda3/envs/workspace_1/Library/ssl/cert.pem"

device = torch.device("cpu")

#===
# # FASTAPI object, input type, etc.
#===
app = FastAPI()

# CORS Configuration for Next.js
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "*"],  # 👈 Next.js dev server origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

###### DATA MODELS ######
# Define input model for search
class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

# Also used for answer generatio on the bot
class ClassifyRequest(BaseModel):
     description : str

# Also used for answer generatio on the bot
class EmbeddingRequest(BaseModel):
     query_to_embed : str



# Global variables
model = None
tokenizer = None
input_gen_ai = None

db_name = './database/embed_trial.db'

client = None
api_key = "T0sAC36z31CWIsTmUjU8dFN03XXf7OiI" # (Lan) free api key for free MISTRAL AI Model
mistral_model = "open-codestral-mamba"

@app.on_event("startup")
def load_resources():
    # global model, index, data, claims
    global device, model, tokenizer, api_key, mistral_model, client, classify_model, classify_tokenizer

    device = "cpu"
    
   

    #===
    # Load model for gen ai
    #===
    client = Mistral(api_key=api_key)

    #===
    # Load model for SDG classification
    #===
    classify_MODEL_DIR = "./sdg_Classification_v1/single_dense"
    classify_tokenizer = AutoTokenizer.from_pretrained(classify_MODEL_DIR)
    classify_model = AutoModelForSequenceClassification.from_pretrained(classify_MODEL_DIR)

    #====
    #  Load model for embeddings
    #====    
    global embed_token_ckpt, embed_model_ckpt, embed_classify_tokenizer, embed_classify_model   
    embed_token_ckpt = "anferico/bert-for-patents"
    embed_model_ckpt = "./sdg_Classification_v1/single_dense"
    embed_classify_tokenizer = AutoTokenizer.from_pretrained(embed_token_ckpt)
    embed_classify_model = AutoModel.from_pretrained(embed_model_ckpt)

    #====
    #  Load transformer model for embeddings
    #====    
    global transformer_model
    transformer_model = SentenceTransformer(embed_model_ckpt)


# Define common stopwords and punctuation
SPECIAL_TOKENS = {"[CLS]", "[SEP]", "[PAD]"}
STOPWORDS = {"the", "a", "an", "in", "to", "of", "and", "or", "on", "at", "for", "by", "with"}
PUNCTUATION = set(string.punctuation)

def is_meaningful_token(token: str) -> bool:
    return (
        token.lower() not in STOPWORDS and
        token not in SPECIAL_TOKENS and
        not all(c in PUNCTUATION for c in token)
    )

def classify_text_with_explanation(text, threshold=0.3, max_length=512):
    inputs = classify_tokenizer(
        text, return_tensors="pt", truncation=True, padding=True, max_length=max_length
    )
    
    with torch.no_grad():
        outputs = classify_model(**inputs, output_attentions=True)
        logits = outputs.logits
        probs = torch.sigmoid(logits).cpu().numpy()[0]
        attentions = outputs.attentions  # (num_layers, batch_size, num_heads, seq_len, seq_len)

    # Get label mapping
    id2label = (classify_model.config.id2label if hasattr(classify_model.config, "id2label")
                else {i: f"sdg_{i}" for i in range(len(probs))})

    # Predictions above threshold
    results = [(id2label[i], float(prob)) for i, prob in enumerate(probs) if prob >= threshold]
    results.sort(key=lambda x: x[1], reverse=True)

    # Extract attention from last layer, from [CLS] token
    last_layer_attention = attentions[-1][0]  # (num_heads, seq_len, seq_len)
    cls_attention = last_layer_attention[:, 0, :]  # (num_heads, seq_len)
    avg_attention = cls_attention.mean(dim=0)  # (seq_len,)

    # Convert tokens and assign attention
    tokens = classify_tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    token_attention = [
        (token, float(attn)) for token, attn in zip(tokens, avg_attention)
        if is_meaningful_token(token)
    ]

    # Sort and take top 5 most meaningful tokens
    top_tokens = sorted(token_attention, key=lambda x: x[1], reverse=True)[:5]

    return {
        "predictions": results,
        "explanation": {
            "most_influential_tokens": top_tokens
        }
    }


##########    Embed enpoint for RAG bot    ##########

@app.post("/embed-query/")
def embed_and_return_query(payload: EmbeddingRequest):
    question = payload.query_to_embed

    # Step 1: Embed the query
    question_embedding = get_embeddings([question], embed_classify_tokenizer, embed_classify_model).cpu().detach().numpy()

    # Step 2: Convert to list

    # Step 3: Return

    return { "Query embedded": question_embedding.tolist() }

from sentence_transformers import SentenceTransformer

@app.post("/embed-query-w-sentence-transformers/")
def embed_and_return_query(payload: EmbeddingRequest):
    question = payload.query_to_embed

    # Step 1: Embed the query
    embedding = transformer_model.encode(question)

    # Step 2: Convert to list

    # Step 3: Return

    return { "Query embedded": embedding.tolist() }



import psycopg2
from datetime import datetime
# PostgreSQL connection config
DB_CONFIG = {
    'dbname': 'mauricio.rodriguez',
    'user': 'mauricio.rodriguez',
    'password': '',
    'host': 'localhost',
    'port': 5432,
}


##########    New Chatbot Endpoint    ##########
### NEW RAG ENDPOINT MOCKUP
@app.post("/send-message-bot/")
async def send_message_bot(request : ClassifyRequest):
    return {"answer" : "Message from backend sent to bot successfully"}

# Dictionary for SDG labels - Classification
SDG_LABELS = {
    "LABEL_1": "No Poverty",
    "LABEL_2": "Zero Hunger",
    "LABEL_3": "Good Health and Well-being",
    "LABEL_4": "Quality Education",
    "LABEL_5": "Gender Equality",
    "LABEL_6": "Clean Water and Sanitation",
    "LABEL_7": "Affordable and Clean Energy",
    "LABEL_8": "Decent Work and Economic Growth",
    "LABEL_9": "Industry, Innovation and Infrastructure",
    "LABEL_10": "Reduced Inequalities",
    "LABEL_11": "Sustainable Cities and Communities",
    "LABEL_12": "Responsible Consumption and Production",
    "LABEL_13": "Climate Action",
    "LABEL_14": "Life Below Water",
    "LABEL_15": "Life on Land",
    "LABEL_16": "Peace, Justice and Strong Institutions",
    "LABEL_17": "Partnerships for the Goals",
}

@app.post("/sdg-classification/")
async def classify_sdg(request : ClassifyRequest):
    output = classify_text_with_explanation(request.description)

    readable_results = [
        {
            "label": label,
            "name": SDG_LABELS.get(label.upper(), "Unknown"),
            "confidence": round(score, 3)
        }
        for label, score in output["predictions"]
    ]

    return {
        "query": request.description,
        "sdgs": readable_results,
        "explanation": {
            "most_influential_tokens": output["explanation"]["most_influential_tokens"]
        }
    }



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

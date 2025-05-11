from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import numpy as np

import sqlite3
import sqlite_vec 
from sqlite_vec import serialize_float32

from transformers import AutoTokenizer, AutoModel
from mistralai import Mistral

import torch
import os

import requests

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
os.environ["SSL_CERT_FILE"] = "C:/Users/20245580/AppData/Local/anaconda3/envs/workspace_1/Library/ssl/cacert.pem"
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
model = None
tokenizer = None
input_gen_ai = None

db_name = 'C:/Users/20245580/Documents/Others/EPO2025/epo_data/embed_trial.db'

client = None
api_key = "T0sAC36z31CWIsTmUjU8dFN03XXf7OiI" # (Lan) free api key for free MISTRAL AI Model
mistral_model = "open-codestral-mamba"

@app.on_event("startup")
def load_resources():
    # global model, index, data, claims
    global device, model, tokenizer, api_key, mistral_model, client

    device = "cpu"
    
    #====
    #  Load model for embeddings
    #====
    token_ckpt = "sadickam/sdg-classification-bert"
    model_ckpt = "C:/Users/20245580/Documents/Others/EPO2025/EPO-CodeFest-2025/current_batch" 
    tokenizer = AutoTokenizer.from_pretrained(token_ckpt)
    model = AutoModel.from_pretrained(model_ckpt)

    #===
    # Load model for gen ai
    #===
    client = Mistral(api_key=api_key)

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
    global input_gen_ai, api_key, mistral_model, client

    print(type(input_gen_ai))

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
    """
    
    chat_response = client.chat.complete(
        model= mistral_model,
        messages = [
            {
                "role": "user",
                "content": rag_prompt,
            },
        ]
    )

    reply_prompt = f"""
    Context information is below.
    ---------------------
    {context}
    ---------------------
    Given the context information and not prior knowledge, answer the query.
    Query: {request.query}
    Answer: {chat_response.choices[0].message.content}
    """

    return {"reply": reply_prompt}

##### New requests

MISTRAL_API_KEY = "mockkey1234"

# Required for Dashboard - SDG Patent Distribution
# fake_sdg_dg = {
#             1 : {
#             "sdg_name": "No Poverty",
#              "count" : 245 },
#             2 : {
#             "sdg_name": "Climate Action",
#              "count" : 384 
#             },
#             3 : {
#             "sdg_description": "Quality Education",
#              "count" : 178 
#             }
# }

# Required for Dashboard - Patent by country map
mock_sdg_by_country_db ={
    1 : {
        "countryCode": "USA",
        "total": 3245,
        "growth": 12.5,
        "sdgs": { "1": 187, "2": 224, "3": 398, "6": 256, "7": 587, "9": 476, "11": 298, "12": 267, "13": 552 },
    },
    2 : {
        "countryCode": "CHN",
        "total": 2876,
        "growth": 18.7,
        "sdgs": { "1": 154, "2": 187, "3": 343, "6": 198, "7": 498, "9": 432, "11": 276, "12": 245, "13": 543 },
    },
    3 : {
        "countryCode": "JPN",
        "total": 1923,
        "growth": 8.3,
        "sdgs": { "1": 167, "2": 198, "3": 356, "6": 187, "7": 376, "9": 343, "11": 187, "12": 156, "13": 253 },
    }
}

mock_sdg_rag_insights_db = {
    1 : {
    "type": "key",
    "title": "Key Insight",
    "content":
      "Cross-analysis of patent data reveals that technologies addressing multiple SDGs simultaneously show 2.7x higher adoption rates. Energy storage technologies in particular appear in 4 different SDG categories, indicating their foundational role in sustainable development.",
  },
  2 : {
    "type": "trend",
    "title": "Emerging Trend",
    "content":
      "Patent data indicates a significant shift toward biomimetic approaches across multiple technology categories. These nature-inspired designs show a 43% annual growth rate and appear particularly prominent in water filtration and materials science patents.",
  },
  3 : {
    "type": "gap",
    "title": "Gap Analysis",
    "content":
      "Despite strong growth in clean energy patents, there is a notable gap in technologies addressing the integration of these solutions in developing regions. Only 12% of patents explicitly address deployment challenges in resource-constrained environments.",
  }
}

mock_sdg_rag_insights_by_sdg_id_db = {
    1: [
    {
      "id": "1",
      "type": "key",
      "title": "Key Insight",
      "content":
        "Patents in poverty reduction focus primarily on agricultural technology and microfinance systems, with a 32% increase in filings over the past 5 years.",
      "sdgId": "1",
      "icon": "Zap",
    },
  ],
  6: [
    {
      "id": "1",
      "type": "key",
      "title": "Key Insight",
      "content":
        "Water purification technologies dominate this SDG, with membrane filtration systems showing the highest growth rate at 28% annually.",
      "sdgId": "6",
      "icon": "Zap",
    },
  ],
  7: [
    {
      "id": "1",
      "type": "key",
      "title": "Key Insight",
      "content":
        "Energy storage patents have overtaken generation technologies, indicating a market shift toward grid stabilization and renewable integration.",
      "sdgId": "7",
      "icon": "Zap",
    },
  ],
  13: [
    {
      "id": "1",
      "type": "key",
      "title": "Key Insight",
      "content":
        "Carbon capture technologies show the highest cross-sector integration, appearing in energy, manufacturing, and transportation patent portfolios.",
      "sdgId": "13",
      "icon": "Zap",
    },
  ],
}

mock_sdg_by_id_with_technologies = {
    7: {
  "Solar Photovoltaics": [
    {
      "id": "US20250123456",
      "title": "Perovskite-Silicon Tandem Solar Cell with Enhanced Stability",
      "abstract":
        "A novel perovskite-silicon tandem solar cell architecture with improved stability and efficiency exceeding 30%. The invention includes a modified interfacial layer that prevents ion migration and enhances long-term performance under real-world conditions.",
      "date": "2025-03-15",
      "organization": "SunTech Innovations",
      "relevance": 92,
      "ragSummary":
        "This patent directly addresses SDG 7.2 by significantly improving solar cell efficiency and stability, which are key barriers to widespread adoption. The technology could reduce solar energy costs by approximately 35% when manufactured at scale.",
      "url": "#",
    },
    {
      "id": "US20250123789",
      "title": "Self-Healing Photovoltaic Module with Integrated Diagnostics",
      "abstract":
        "A self-healing photovoltaic module that can detect and repair microcracks and connection failures autonomously. The system includes integrated diagnostic sensors and a novel polymer-based healing mechanism that extends module lifetime by up to 50%.",
      "date": "2025-02-28",
      "organization": "RenewCorp",
      "relevance": 87,
      "ragSummary":
        "By extending solar panel lifetime and reducing maintenance costs, this innovation addresses SDG 7.3 (energy efficiency) and 12.5 (waste reduction). The self-healing capability could prevent up to 30% of premature panel replacements.",
      "url": "#",
    },
    {
      "id": "EP20250056789",
      "title": "Transparent Solar Cells for Building-Integrated Photovoltaics",
      "abstract":
        "A transparent solar cell technology with selective harvesting of ultraviolet and infrared light while maintaining visible transparency. The invention enables windows and building facades to generate electricity without compromising aesthetics or natural lighting.",
      "date": "2025-01-10",
      "organization": "GlassPower Systems",
      "relevance": 85,
      "ragSummary":
        "This technology enables integration of solar power generation into urban environments (SDG 11.6) while supporting renewable energy adoption (SDG 7.2). The dual-use nature of the technology improves land-use efficiency in densely populated areas.",
      "url": "#",
    },
  ],
  "Wind Energy": [
    {
      "id": "US20250187654",
      "title": "Floating Offshore Wind Turbine with Dynamic Stabilization",
      "abstract":
        "A floating offshore wind turbine platform with an active stabilization system that adapts to changing wave and wind conditions. The design enables deployment in deep water locations with extreme weather conditions while maintaining optimal energy production.",
      "date": "2025-04-02",
      "organization": "OceanWind Technologies",
      "relevance": 94,
      "ragSummary":
        "This patent directly supports SDG 7.2 by expanding viable locations for offshore wind deployment. The technology could unlock an estimated 1,700 GW of additional offshore wind potential in regions previously considered unsuitable.",
      "url": "#",
    },
    {
      "id": "EP20250078912",
      "title": "Bird-Safe Wind Turbine with Acoustic Deterrent System",
      "abstract":
        "A wind turbine design incorporating an advanced acoustic deterrent system that reduces bird and bat collisions by over 90%. The system uses targeted sound frequencies and machine learning to identify and deter approaching wildlife without affecting energy production.",
      "date": "2025-03-18",
      "organization": "EcoTurbine Ltd",
      "relevance": 88,
      "ragSummary":
        "This innovation addresses a key environmental concern with wind energy (SDG 15.5 - biodiversity) while supporting renewable energy expansion (SDG 7.2). The technology could significantly reduce opposition to wind farm development in sensitive ecological areas.",
      "url": "#",
    },
  ],
  "Water Filtration": [
    {
      "id": "US20250134567",
      "title": "Biomimetic Membrane for Low-Energy Water Purification",
      "abstract":
        "A biomimetic membrane inspired by natural water channels in cell membranes, enabling highly efficient water purification with minimal energy input. The technology removes contaminants including microplastics, heavy metals, and organic pollutants while consuming 80% less energy than conventional reverse osmosis.",
      "date": "2025-02-12",
      "organization": "AquaPure Solutions",
      "relevance": 96,
      "ragSummary":
        "This patent directly addresses SDG 6.1 (safe drinking water) and 6.4 (water-use efficiency) by dramatically reducing the energy required for water treatment. The technology is particularly relevant for water-stressed regions with limited energy resources.",
      "url": "#",
    },
  ],
  "Carbon Capture": [
    {
      "id": "US20250156789",
      "title": "Direct Air Capture System with Enhanced Sorbent Regeneration",
      "abstract":
        "A direct air capture system utilizing a novel sorbent material and regeneration process that reduces energy requirements by 65% compared to conventional approaches. The system can be powered entirely by renewable energy and produces concentrated CO2 suitable for utilization or storage.",
      "date": "2025-01-25",
      "organization": "CarbonTech Solutions",
      "relevance": 95,
      "ragSummary":
        "This technology directly supports SDG 13 (Climate Action) by enabling cost-effective negative emissions. The reduced energy requirement addresses a key barrier to DAC deployment at scale, making it viable for meeting climate targets in hard-to-abate sectors.",
      "url": "#",
    },
  ],
  "Energy Storage": [
    {
      "id": "US20250112345",
      "title": "Solid-State Battery with Silicon-Carbon Composite Anode",
      "abstract":
        "A solid-state battery design incorporating a silicon-carbon composite anode and a novel solid electrolyte interface that enables fast charging, high energy density, and enhanced safety. The battery maintains 90% capacity after 2000 cycles and eliminates the risk of thermal runaway.",
      "date": "2025-03-30",
      "organization": "EnerStore Inc",
      "relevance": 93,
      "ragSummary":
        "This innovation addresses key barriers to renewable energy integration (SDG 7.2) by providing safe, high-density energy storage. The extended cycle life also supports SDG 12 (responsible consumption) by reducing battery replacement frequency and associated resource use.",
      "url": "#",
    },
  ],
}
}
########## DASHBOARD ##########
# Funcs
def find_sdg(x, y): 
        y = y.split(',')
        # print(y)
        for idx, elem in enumerate(y):
                # print(elem, x)
                # print(type(x), type(elem))
                if x == elem:
                        # print("find")
                        return 1
        # print("not find")
        return 0

def find_tech(x, y): 
        y = y.split(';')
        # print(y)
        for idx, elem in enumerate(y):
                # print(elem, x)
                # print(type(x), type(elem))
                if x == elem:
                        # print("find")
                        return 1
        # print("not find")
        return 0


# Global variables
sdg_label_name_mapping ={
    '1' : 'sdg name for label 1',
    '2' : 'sdg name for label 2',
    '3' : 'sdg name for label 3',
    '4' : 'sdg name for label 4',
    '5' : 'sdg name for label 5',
    '6' : 'sdg name for label 6',
    '7' : 'sdg name for label 7',
    '8' : 'sdg name for label 8',
    '9' : 'sdg name for label 8',
    '10' : 'sdg name for label 10',
    '11' : 'sdg name for label 11',
    '12' : 'sdg name for label 12'
}

# For SDG Patent Distribution - Dashboard
fake_sdg_dg = None
@app.get("/sdg-patents-distribution")
async def get_all_sdg_patents_distribution():
    global sdg_label_name_mapping, db_name, fake_sdg_dg

    total_num_sdg = 12
    fake_sdg_dg = {}
    try:
        with sqlite3.connect(db_name) as conn:

            # load the `sqlite-vec` extention into the connected db
            ## NOTE:
            ## must load the `sqlite-vec` extention everytime connect to the db, 
            ## in order to use the vec table created using extension `sqlte-vec` and `sqlite-vec` functions
            conn.enable_load_extension(True) # start loading extensions
            sqlite_vec.load(conn)
            conn.enable_load_extension(True) # end loading extensions

            conn.create_function('find_smth', 2, find_sdg)

            for sdg_num in range(1, total_num_sdg+1):
              sql_cn_meta = f"""
                  SELECT COUNT(*) FROM meta_data_embeddings
                  WHERE find_smth('{str(int(sdg_num))}', sdg_labels)
              """
              cur = conn.cursor()
              res = cur.execute(sql_cn_meta)
              len_sdg = res.fetchall()[0][0]
              print(f"{sdg_num} has {len_sdg} rows in meta table")
              fake_sdg_dg.update({ int(sdg_num): {
                                      "sdg_name": sdg_label_name_mapping[str(int(sdg_num))],
                                      "count" : int(len_sdg) }})

    except sqlite3.OperationalError as e:
        print(e)
    return fake_sdg_dg

# For SDG Patent Distribution by SDG Number - Dashboard
@app.get("/sdg-patents-distribution/{sdg_id}")
async def get_sdg_patent_distribution_by_id(sdg_id : int | None = None):
    global fake_sdg_dg
    return fake_sdg_dg.get(sdg_id)

# For SDG Distribution by country Map - Dashboard
@app.get("/sdg-by-country/")
async def get_patents_distribution_by_country():

    return mock_sdg_by_country_db

# For RAG Insights accross all SDGs - Dashboard
# Here we can set up a function to run RAG insights on all Patents/SDGs once per day/week, since it's a compute heavy process
@app.get("/sdg-rag-insights/")
async def get_all_sdg_rag_insights():

    return mock_sdg_rag_insights_db

# For RAG Insights for specific SDG by ID - Dashboard
# Setup function to pre-compute RAG once per day on a single SDG category
@app.get("/sdg-rag-insights-by-sdg-id/{sdg_id}")
async def get_sdg_rag_insight_by_id(sdg_id : int | None = None):

    return mock_sdg_rag_insights_by_sdg_id_db.get(sdg_id)


##########    SDG    ##########

# For SDG by ID and technologies 
# Get all technologies for a selected SDG
# In this request sdg_id is required
# tech parameter is optional, and will return the technologies associated with a specific SDG_id by name...
# ... e.g. Get all associated technologies for sdg_id = 7 and tech ="Solar Photovoltaics"
@app.get("/sdg-by-id-and-technologies/{sdg_id}/")
async def get_sdg_related_tech_by_sdg_id(sdg_id : int | None = None, tech : str | None = None):

    results = []
    
    if sdg_id not in mock_sdg_by_id_with_technologies:
        raise HTTPException(status_code=404, detail="SDG Not Found")
    
    sdg_data = mock_sdg_by_id_with_technologies.get(sdg_id)
    results = sdg_data

    if tech:
        tech_data = sdg_data.get(tech)
        if not tech_data:
            raise HTTPException(status_code=404, detail="Technology not found under this SDG")
        results = tech_data

    return results
    



##########    Chatbot    ##########
@app.post("/send-message-bot/")
async def send_message_bot(request : str):
    
    # Format payload for Mistral API
    mistral_url = "https://api.mistral.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "mistral-tiny",  # Or mistral-small, mistral-medium
        "messages": [
            {"role": "user", "content": request.user_message}
        ],
        "temperature": 0.7,
        "max_tokens": 200
    }

    response = requests.post(mistral_url, headers=headers, json=payload)

    if response.status_code == 200:
        result = response.json()
        return {"reply": result["choices"][0]["message"]["content"]}
    else:
        return {
            "error": f"Mistral API failed",
            "status_code": response.status_code,
            "details": response.text
        }

#####

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

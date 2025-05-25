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

os.environ["SSL_CERT_FILE"] = "C:/Users/20245580/AppData/Local/anaconda3/envs/workspace_1/Library/ssl/cert.pem"

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

class ClassifyRequest(BaseModel):
     description : str



# Global variables
model = None
tokenizer = None
input_gen_ai = None


db_name = 'C:/Users/20245580/Documents/Others/EPO2025/epo_data/embed_final.db'

client = None
api_key = "T0sAC36z31CWIsTmUjU8dFN03XXf7OiI" # (Lan) free api key for free MISTRAL AI Model
mistral_model = "open-codestral-mamba"

@app.on_event("startup")
def load_resources():
    # global model, index, data, claims
    global device, model, tokenizer, api_key, mistral_model, client, classify_model, classify_tokenizer

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

    #===
    # Load model for SDG classification
    #===
    # classify_MODEL_DIR = "./sdg_Classification_v1/single_dense"
    # classify_tokenizer = AutoTokenizer.from_pretrained(classify_MODEL_DIR)
    # classify_model = AutoModelForSequenceClassification.from_pretrained(classify_MODEL_DIR)

def classify_text(text, threshold=0.3, max_length=512):
    inputs = classify_tokenizer(
        text, return_tensors="pt", truncation=True, padding=True, max_length=max_length
    )
    with torch.no_grad():
        logits = classify_model(**inputs).logits
        probs = torch.sigmoid(logits).cpu().numpy()[0]
    if hasattr(classify_model.config, "id2label") and classify_model.config.id2label:
        id2label = {int(k): v for k, v in classify_model.config.id2label.items()}
    else:
        id2label = {i: f"sdg_{i}" for i in range(len(probs))}
    results = [(id2label[i], float(prob)) for i, prob in enumerate(probs) if prob >= threshold]
    # Remove before production
    print("Unsorted results: ", results)
    results.sort(key=lambda x: x[1], reverse=True)
    # Remove before production
    print("Sorted results: ", results)
    return results


# Will search closest patents from embeddings
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
                        claims,
                        pub_num
                    FROM meta_data_embeddings
                    WHERE rowid={int(tuple[0])}
                    """
                cursor = conn.cursor()
                res = cursor.execute(meta_query).fetchall()

                results.append(
                        {
                            "TITLE": res[0][0],
                            "DISTANCE": float(tuple[1]),
                            "PUBLICATION_NUMBER": res[0][2],
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
        Publication number: {input_gen_ai["relevant_docs"][idx]["PUBLICATION_NUMBER"]}
        Context: {input_gen_ai["relevant_docs"][idx]["CLAIMS"]}
        """
        context = context + temp
        

    rag_prompt = f"""
    Context information belonging to European Patent Office database is below.
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

# mock_sdg_rag_insights_db = {
#     1 : {
#     "type": "key",
#     "title": "Key Insight",
#     "content":
#       "Cross-analysis of patent data reveals that technologies addressing multiple SDGs simultaneously show 2.7x higher adoption rates. Energy storage technologies in particular appear in 4 different SDG categories, indicating their foundational role in sustainable development.",
#   },
#   2 : {
#     "type": "trend",
#     "title": "Emerging Trend",
#     "content":
#       "Patent data indicates a significant shift toward biomimetic approaches across multiple technology categories. These nature-inspired designs show a 43% annual growth rate and appear particularly prominent in water filtration and materials science patents.",
#   },
#   3 : {
#     "type": "gap",
#     "title": "Gap Analysis",
#     "content":
#       "Despite strong growth in clean energy patents, there is a notable gap in technologies addressing the integration of these solutions in developing regions. Only 12% of patents explicitly address deployment challenges in resource-constrained environments.",
#   }
# }



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
                # print(f"y: {y}")
                # print(f"elem {elem}, x {x}")
                # print(type(x), type(elem))
                if x == elem:
                        # print("find")
                        return 1
        # print("not find")
        return 0


# Global variables
sdg_label_name_mapping ={
    '1' : 'No Poverty',
    '2' : 'Zero Hunger',
    '3' : 'Good Health and Well-being',
    '4' : 'Quality Education',
    '5' : 'Gender Equality',
    '6' : 'Clean Water and Sanitation',
    '7' : 'Affordable and Clean Energy',
    '8' : 'Decent Work and Economic Growth',
    '9' : 'Industry, Innovation, and Infrastructure',
    '10' : 'Reduced Inequalities',
    '11' : 'Sustainable Cities and Communities',
    '12' : 'Responsible Consumption and Production',
    '13' : 'Climate Action',
    '14' : 'Life Below Water',
    '15' : 'Life on Land',
    '16' : 'Peace, Justice, and Strong Institutions',
    '17' : 'Partnerships for the Goals'
}

# For SDG Patent Distribution - Dashboard
fake_sdg_dg = None
@app.get("/sdg-patents-distribution")
async def get_all_sdg_patents_distribution():
    global sdg_label_name_mapping, db_name, fake_sdg_dg

    total_num_sdg = 17
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
            
            cur = conn.cursor()
            for sdg_num in range(1, total_num_sdg+1):
              sql_cn_meta = f"""
                  SELECT COUNT(*) FROM main_table_wihout_split_claims
                  WHERE find_smth('{str(int(sdg_num))}', sdg_labels)
              """
              res = cur.execute(sql_cn_meta)
              len_sdg = res.fetchall()[0][0]
              print(f"{sdg_num} has {len_sdg} rows in meta table")
              fake_sdg_dg.update({ int(sdg_num): {
                                      "id" : str(sdg_num),
                                      "name": sdg_label_name_mapping[str(int(sdg_num))],
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
# mock_sdg_rag_insights_db = {
#     1 : {
#     "type": "key",
#     "title": "Key Insight",
#     "content":
#       "Cross-analysis of patent data reveals that technologies addressing multiple SDGs simultaneously show 2.7x higher adoption rates. Energy storage technologies in particular appear in 4 different SDG categories, indicating their foundational role in sustainable development.",
#   },
#   2 : {
#     "type": "trend",
#     "title": "Emerging Trend",
#     "content":
#       "Patent data indicates a significant shift toward biomimetic approaches across multiple technology categories. These nature-inspired designs show a 43% annual growth rate and appear particularly prominent in water filtration and materials science patents.",
#   },
#   3 : {
#     "type": "gap",
#     "title": "Gap Analysis",
#     "content":
#       "Despite strong growth in clean energy patents, there is a notable gap in technologies addressing the integration of these solutions in developing regions. Only 12% of patents explicitly address deployment challenges in resource-constrained environments.",
#   }
# }
@app.get("/sdg-rag-insights/")
async def get_all_sdg_rag_insights():
    global fake_sdg_dg, client, mistral_model

    statistic_context = ""
    for sdg_num, val in fake_sdg_dg.items():
        print(sdg_num, val)
        temp = f"""
          There are {val['count']} patents in the database related to Sustainable Development Goal: {val['sdg_name']}
          """
        statistic_context = statistic_context + temp

    # KEY INSIGHT
    rag_prompt = f"""
    The statistics of patent database is given as followings:
    ---------------------
    {statistic_context}
    ---------------------
    Given the statistic distribution of the database categorized by 17 Sustainable Development Goals, 
    please, give the essential insight of how the patent database related to the Sustainable Development Goal defined by the United Nations:
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
    key_insight = chat_response.choices[0].message.content
    # print(f"Key insight: {key_insight}\n{'='*10}\n")

    # EMERGING TREND
    rag_prompt_trend = f"""
    The statistics of patent database is given as followings:
    ---------------------
    {statistic_context}
    ---------------------
    Given the statistic distribution of the database categorized by 17 Sustainable Development Goals, 
    please, give the emerging trend of 17 Sustainable Development Goals defined by the United Nations within the patent database :
    """

    chat_response = client.chat.complete(
    model= mistral_model,
    messages = [
            {
                "role": "user",
                "content": rag_prompt_trend,
            },
        ]
    )
    emerge_trend = chat_response.choices[0].message.content
    # print(f"Key insight: {emerge_trend}\n{'='*10}\n")

    # GAP ANALYSIS
    rag_prompt_gap = f"""
    The statistics of patent database is given as followings:
    ---------------------
    {statistic_context}
    ---------------------
    Given the statistic distribution of the database categorized by 17 Sustainable Development Goals, 
    please, answer the question: what is the most underated Sustainable Development Goal where not such research has been conducted?
    """

    chat_response = client.chat.complete(
    model= mistral_model,
    messages = [
            {
                "role": "user",
                "content": rag_prompt_gap,
            },
        ]
    )
    gap_analysis = chat_response.choices[0].message.content
    # print(f"Key insight: {gap_analysis}\n{'='*10}\n")

    mock_sdg_rag_insights_db = {
      1 : {
      "type": "key",
      "title": "Key Insight",
      "content":{key_insight},
      },
      2 : {
        "type": "trend",
        "title": "Emerging Trend",
        "content":{emerge_trend},
      },
      3 : {
        "type": "gap",
        "title": "Gap Analysis",
        "content":{gap_analysis},
      }
    }
    return mock_sdg_rag_insights_db

# For RAG Insights for specific SDG by ID - Dashboard
# Setup function to pre-compute RAG once per day on a single SDG category
# mock_sdg_rag_insights_by_sdg_id_db = {
#     1: [
#     {
#       "id": "1",
#       "type": "key",
#       "title": "Key Insight",
#       "content":
#         "Patents in poverty reduction focus primarily on agricultural technology and microfinance systems, with a 32% increase in filings over the past 5 years.",
#       "sdgId": "1",
#       "icon": "Zap",
#     },
#   ],
#   6: [
#     {
#       "id": "1",
#       "type": "key",
#       "title": "Key Insight",
#       "content":
#         "Water purification technologies dominate this SDG, with membrane filtration systems showing the highest growth rate at 28% annually.",
#       "sdgId": "6",
#       "icon": "Zap",
#     },
#   ],
#   7: [
#     {
#       "id": "1",
#       "type": "key",
#       "title": "Key Insight",
#       "content":
#         "Energy storage patents have overtaken generation technologies, indicating a market shift toward grid stabilization and renewable integration.",
#       "sdgId": "7",
#       "icon": "Zap",
#     },
#   ],
#   13: [
#     {
#       "id": "1",
#       "type": "key",
#       "title": "Key Insight",
#       "content":
#         "Carbon capture technologies show the highest cross-sector integration, appearing in energy, manufacturing, and transportation patent portfolios.",
#       "sdgId": "13",
#       "icon": "Zap",
#     },
#   ],
# }
@app.get("/sdg-rag-insights-by-sdg-id/{sdg_id}")
async def get_sdg_rag_insight_by_id(sdg_id : int | None = None):
    global db_name, sdg_label_name_mapping, client, mistral_model

    # Find the patents related to input sdg id
    print(f"input sdg_id: {sdg_id}")
    try:
        with sqlite3.connect(db_name) as conn:

            # load the `sqlite-vec` extention into the connected db
            ## NOTE:
            ## must load the `sqlite-vec` extention everytime connect to the db, 
            ## in order to use the vec table created using extension `sqlte-vec` and `sqlite-vec` functions
            conn.enable_load_extension(True) # start loading extensions
            sqlite_vec.load(conn)
            conn.enable_load_extension(True) # end loading extensions

            conn.create_function('find_related_sdg', 2, find_sdg)

            cur = conn.cursor()

            # Get total number of data (rows) in the database
            meta_query = f"""
                  SELECT
                          title,
                          sdg_labels
                  FROM main_table_wihout_split_claims
                  WHERE find_related_sdg('{str(int(sdg_id))}', sdg_labels);
              """
            res = cur.execute(meta_query)
            result = res.fetchall()
            # print(result)
    except sqlite3.OperationalError as e:
        print(e)
    
    
    # ask the Gen ai for insights
    statistic_context = ""
    for title, _ in result:
        # print(sdg_num, val)
        temp = f"""
          title: {title}
          """
        statistic_context = statistic_context + temp

    # KEY INSIGHT
    rag_prompt = f"""
    There are {len(result)} patents in the database related to {sdg_label_name_mapping[str(int(sdg_id))]}. Their title are the following:
    ---------------------
    {statistic_context}
    ---------------------
    Given the context of all patents related to {sdg_label_name_mapping[str(int(sdg_id))]} in the database,
    please, give the essential insight of how this {sdg_label_name_mapping[str(int(sdg_id))]} is reflected currently in the databse:
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
    key_insight = chat_response.choices[0].message.content

    return [
              {
                "id": "1",
                "type": "key",
                "title": "Key Insight",
                "content":
                  key_insight,
                "sdgId": str(int(sdg_id)),
                "icon": "Zap",
              },
            ]


##########    SDG    ##########

# ⚠️ The following data must be pre-computed from the final database
technology_data_by_sdg = {
    "1": [
        {"id": "Microfinance Platforms", "name": "BEMicrofinance Platforms", "count": 64, "growth": 12},
        {"id": "Affordable Housing", "name": "BEAffordable Housing", "count": 83, "growth": 9},
    ],
    "2": [
        {"id": "Precision Agriculture", "name": "BEPrecision Agriculture", "count": 148, "growth": 27},
        {"id": "Crop Yield Optimization", "name": "BECrop Yield Optimization", "count": 132, "growth": 22},
    ],
    "3": [
        {"id": "Medical Diagnostics", "name": "BEMedical Diagnostics", "count": 174, "growth": 35},
        {"id": "Vaccine Technologies", "name": "BEVaccine Technologies", "count": 142, "growth": 28},
    ],
    "4": [
        {"id": "Remote Learning Platforms", "name": "BERemote Learning Platforms", "count": 96, "growth": 21},
        {"id": "EdTech Tools", "name": "BEEdTech Tools", "count": 110, "growth": 18},
    ],
    "5": [
        {"id": "Women’s Health", "name": "BEWomen’s Health", "count": 89, "growth": 17},
        {"id": "Gender Safety Tech", "name": "BEGender Safety Tech", "count": 64, "growth": 12},
    ],
    "6": [
        {"id": "Water Filtration", "name": "BEWater Filtration", "count": 134, "growth": 19},
        {"id": "Desalination", "name": "BEDesalination", "count": 112, "growth": 22},
        {"id": "Water Monitoring", "name": "BEWater Monitoring", "count": 98, "growth": 17},
        {"id": "Wastewater Treatment", "name": "BEWastewater Treatment", "count": 124, "growth": 15},
    ],
    "7": [
        {"id": "Solar Photovoltaics", "name": "BESolar Photovoltaics", "count": 156, "growth": 24},
        {"id": "Wind Energy", "name": "BEWind Energy", "count": 142, "growth": 18},
        {"id": "Energy Storage", "name": "BEEnergy Storage", "count": 124, "growth": 32},
        {"id": "Smart Grid", "name": "BESmart Grid", "count": 98, "growth": 15},
        {"id": "Hydrogen Production", "name": "BEHydrogen Production", "count": 87, "growth": 28},
    ],
    "8": [
        {"id": "Workplace Automation", "name": "BEWorkplace Automation", "count": 101, "growth": 19},
        {"id": "Sustainable Business Models", "name": "BESustainable Business Models", "count": 84, "growth": 16},
    ],
    "9": [
        {"id": "Smart Manufacturing", "name": "BESmart Manufacturing", "count": 128, "growth": 26},
        {"id": "Industrial IoT", "name": "BEIndustrial IoT", "count": 147, "growth": 23},
    ],
    "10": [
        {"id": "Accessibility Tech", "name": "BEAccessibility Tech", "count": 73, "growth": 14},
        {"id": "Financial Inclusion", "name": "BEFinancial Inclusion", "count": 92, "growth": 19},
    ],
    "11": [
        {"id": "Smart Cities", "name": "BESmart Cities", "count": 106, "growth": 21},
        {"id": "Urban Mobility", "name": "BEUrban Mobility", "count": 89, "growth": 18},
    ],
    "12": [
        {"id": "Recycling Technologies", "name": "BERecycling Technologies", "count": 112, "growth": 20},
        {"id": "Circular Economy Solutions", "name": "BECircular Economy Solutions", "count": 97, "growth": 15},
    ],
    "13": [
        {"id": "Carbon Capture", "name": "BECarbon Capture", "count": 118, "growth": 42},
        {"id": "Climate Modeling", "name": "BEClimate Modeling", "count": 87, "growth": 23},
        {"id": "Emissions Reduction", "name": "BEEmissions Reduction", "count": 132, "growth": 31},
        {"id": "Climate Adaptation", "name": "BEClimate Adaptation", "count": 76, "growth": 19},
    ],
    "14": [
        {"id": "Marine Monitoring", "name": "BEMarine Monitoring", "count": 83, "growth": 16},
        {"id": "Sustainable Fishing", "name": "BESustainable Fishing", "count": 69, "growth": 14},
    ],
    "15": [
        {"id": "Biodiversity Sensors", "name": "BEBiodiversity Sensors", "count": 94, "growth": 17},
        {"id": "Reforestation Drones", "name": "BEReforestation Drones", "count": 77, "growth": 20},
    ],
    "16": [
        {"id": "Blockchain for Governance", "name": "BEBlockchain for Governance", "count": 65, "growth": 13},
        {"id": "Secure Voting Tech", "name": "BESecure Voting Tech", "count": 58, "growth": 11},
    ],
    "17": [
        {"id": "Collaboration Platforms", "name": "BECollaboration Platforms", "count": 91, "growth": 15},
        {"id": "Data Sharing Frameworks", "name": "BEData Sharing Frameworks", "count": 74, "growth": 12},
    ],
}


# For SDG by ID - General View
# ✅ in SDG/SDG_ID 
@app.get("/technologies-by-sdg-id/{sdg_id}")
async def get_technologies_by_sdg_id(sdg_id: str):
    if sdg_id in technology_data_by_sdg:
        return technology_data_by_sdg[sdg_id]
    raise HTTPException(status_code=404, detail="No data found for the given SDG ID")

# For SDG by ID and technologies 
# Get all technologies for a selected SDG
# In this request sdg_id is required
# tech parameter is optional, and will return the technologies associated with a specific SDG_id by name...
# ... e.g. Get all associated technologies for sdg_id = 7 and tech ="Solar Photovoltaics"
@app.get("/sdg-by-id-and-technologies/{sdg_id}")
async def get_sdg_related_tech_by_sdg_id(sdg_id : int, tech : str | None = None):
    global db_name
    # get (by brute force) all distance related to the sdg label
    # tech = 'Chemical Processes'
    print(f"input sdg_id: {sdg_id}")
    print(f"input tech: {tech}")
    # print(f"limit search: {limit_search}")
    defined_question = f"what are the patents related to {sdg_label_name_mapping[str(int(sdg_id))]} and {tech}?" 
    question_embedding = get_embeddings([defined_question], tokenizer, model).cpu().detach().numpy()
    try:
        with sqlite3.connect(db_name) as conn:

            # load the `sqlite-vec` extention into the connected db
            ## NOTE:
            ## must load the `sqlite-vec` extention everytime connect to the db, 
            ## in order to use the vec table created using extension `sqlte-vec` and `sqlite-vec` functions
            conn.enable_load_extension(True) # start loading extensions
            sqlite_vec.load(conn)
            conn.enable_load_extension(True) # end loading extensions

            conn.create_function('find_related_sdg', 2, find_sdg)
            conn.create_function('find_related_tech', 2, find_tech)

            cur = conn.cursor()

            # Get total number of data (rows) in the database
            sql_cn_meta = f"""
                SELECT COUNT(*) FROM meta_data_embeddings
            """
            res = cur.execute(sql_cn_meta)
            len_embed_table = res.fetchall()[0][0]
            
            query = question_embedding.tolist()
            q_query_distance = conn.execute(
              f"""
              SELECT rowid, vec_distance_cosine(embedding, ?) AS score
              FROM vec_items
              ORDER BY score ASC
              LIMIT {str(len_embed_table)}
              """,
            [serialize_float32(query[0])],
            ).fetchall()

            # try
            query_list = [f"find_related_sdg ('{str(int(sdg_id))}', sdg_labels)", f"find_related_tech ('{tech}', tech_labels)"]
            meta_query_try = f"""
                              SELECT
                                      rowid,
                                      pub_num,
                                      sdg_labels
                              FROM meta_data_embeddings"""

            for idx, value in enumerate(query_list):
                if idx == 0:
                    meta_query_try = meta_query_try + f"\nWHERE {value}"
                else:
                    meta_query_try = meta_query_try + f"\nOR {value}"
            # try
            
            # Get all patent related to the queried sdg_id and tech label
            if tech == None:
              meta_query = f"""
                  SELECT
                          rowid,
                          pub_num,
                          sdg_labels
                  FROM meta_data_embeddings
                  WHERE find_related_sdg('{str(int(sdg_id))}', sdg_labels);
              """
            else:
              meta_query = meta_query_try
            meta_data_sdg = cur.execute(meta_query).fetchall()
            # print(len(meta_data_sdg),
            #       meta_data_sdg[0:20])
            
            # Find the sdg_id related patent in the distance query
            np_q_query_distance = np.array(q_query_distance)
            np_meta_data_sdg = np.array(meta_data_sdg)

            res = {}
            if tech == None:
                  print(f"number of the patents matching the id: {np_meta_data_sdg.size}")
            else:
                  print(f"number of the patents matching the id and tech: {np_meta_data_sdg.size}")

            if np_meta_data_sdg.size == 0:
                  if tech != None:
                      print(f"No result for combination of sdg_id: {sdg_id} and tech: {tech}")
                  else:
                      print(f"No result for sdg_id: {sdg_id}")
            else:
                  np_meta_data_sdg_float = np_meta_data_sdg[:,0].astype(float)

                  indices = np.where(np.isin(np_q_query_distance[:,0], np_meta_data_sdg_float))[0]

                  limit_search = 5
                  # print(np.shape(np_q_query_distance))
                  q_dis_sdg = np_q_query_distance[indices[0:limit_search], :]
                  # print(q_dis_sdg,
                  #       np.shape(q_dis_sdg))
                  
                  for idx, elem in enumerate(q_dis_sdg):
                      # print(f"from q_dis_sdg: {str(q_dis_sdg[idx,0])}")
                      # print(f"elem: {str(elem)}")
                      # print(f"elem first: {int(elem[0])}")
                      # print(f"elem second: {float(elem[1])}")
                      meta_query = f"""
                          SELECT
                                  rowid,
                                  pub_num,
                                  sdg_labels,
                                  title,
                                  claims,
                                  tech_labels
                          FROM meta_data_embeddings
                          WHERE rowid == {int(elem[0])}
                      """

                      meta_data_sdg_dis = cur.execute(meta_query).fetchall()
                      # print(meta_data_sdg_dis)
                      # print(f"\
                      #       rowid: {meta_data_sdg_dis[0][0]}\n\
                      #       pub_num: {meta_data_sdg_dis[0][1]}\n\
                      #       sdg: {meta_data_sdg_dis[0][2]}\n\
                      #       tech_labels: {meta_data_sdg_dis[0][5]}\n\
                      #       dist: {float(elem[1])}\n\
                      #       title: {meta_data_sdg_dis[0][3]}\n\
                      #       claim: {meta_data_sdg_dis[0][4]}\n\
                      #       ")
                      res.update({
                          int(idx):{
                                "pub_num": int(meta_data_sdg_dis[0][1]),
                                "sdg_id": str(meta_data_sdg_dis[0][2]),
                                "tech_labels": meta_data_sdg_dis[0][5],
                                "dist": float(elem[1]),
                                "title": meta_data_sdg_dis[0][3],
                                "claim": meta_data_sdg_dis[0][4]
                          }
                      })
    except sqlite3.OperationalError as e:
        print(e)

    return res
    
##########    Trends    ##########

mockTrendData = {
    "5y": [
        {"label": "2020", "value": 287, "color": "#38bdf8"},
        {"label": "2021", "value": 356, "color": "#38bdf8"},
        {"label": "2022", "value": 423, "color": "#38bdf8"},
        {"label": "2023", "value": 512, "color": "#38bdf8"},
        {"label": "2024", "value": 587, "color": "#38bdf8"},
    ],
    "10y": [
        {"label": "2015", "value": 245, "color": "#38bdf8"},
        {"label": "2016", "value": 267, "color": "#38bdf8"},
        {"label": "2017", "value": 289, "color": "#38bdf8"},
        {"label": "2018", "value": 312, "color": "#38bdf8"},
        {"label": "2019", "value": 356, "color": "#38bdf8"},
        {"label": "2020", "value": 423, "color": "#38bdf8"},
        {"label": "2021", "value": 487, "color": "#38bdf8"},
        {"label": "2022", "value": 542, "color": "#38bdf8"},
        {"label": "2023", "value": 587, "color": "#38bdf8"},
        {"label": "2024", "value": 623, "color": "#38bdf8"},
    ],
    "15y": [
        {"label": "2010", "value": 156, "color": "#38bdf8"},
        {"label": "2011", "value": 187, "color": "#38bdf8"},
        {"label": "2012", "value": 201, "color": "#38bdf8"},
        {"label": "2013", "value": 215, "color": "#38bdf8"},
        {"label": "2014", "value": 232, "color": "#38bdf8"},
        {"label": "2015", "value": 245, "color": "#38bdf8"},
        {"label": "2016", "value": 267, "color": "#38bdf8"},
        {"label": "2017", "value": 289, "color": "#38bdf8"},
        {"label": "2018", "value": 312, "color": "#38bdf8"},
        {"label": "2019", "value": 356, "color": "#38bdf8"},
        {"label": "2020", "value": 423, "color": "#38bdf8"},
        {"label": "2021", "value": 487, "color": "#38bdf8"},
        {"label": "2022", "value": 542, "color": "#38bdf8"},
        {"label": "2023", "value": 587, "color": "#38bdf8"},
        {"label": "2024", "value": 623, "color": "#38bdf8"},
    ]
}

@app.get("/trends/")
async def get_trends(timeframe : str):
    return mockTrendData[timeframe]


##########    New Chatbot Endpoint    ##########
@app.post("/send-message-bot/")
async def send_message_bot(request : str):
    return None

@app.post("/sdg_classification/")
async def classify_sdg(request : ClassifyRequest):
     results = classify_text(request.description)
     return {
          "results" : str(results)
          # "[('LABEL_2', 0.9185042977333069), ('LABEL_7', 0.6150209307670593)]"
     }

# This should actually be "/search/"
@app.post("/relevant_patents/")
async def get_relevant_patents(request : str):
     
     return {
          "message" : "Here are the top 5 patents closest to your description",
          "relevant_docs": 
        [
        {
            "TITLE": "Method and apparatus for flue-gas cleaning",
            "DISTANCE": 6.496407508850098,
            "CLAIMS": "water thus separated, whereby watersoluble substances in the fluegases are separated in said prior separation stage, which prior separation stage 1 is connected to a collecting means 12 for collecting the water fed to the prior",
          "sdg_result" : "5",
          "confidence" : "99"
        },
        {
            "TITLE": "METHOD OF MEASURING WATER CONTENT",
            "DISTANCE": 7.981537342071533,
            "CLAIMS": "method of measurement of water content of a liquid, in which method the properties of the liquid are measured by a first measurement",
        }
        ]
     }
    

#####

#### ==============
## Tryout filter layer + automatic query search
### ================
# Will search closest patents from embeddings

import json
import re

# Funcs
def find_sdg_v2(x, y): 
        y = y.split(',')
        # print(y)
        for idx, elem in enumerate(y):
                # print(elem, x)
                # print(type(x), type(elem))
                matches = re.findall(r'\b' + x + r'\b', elem)
                if len(matches) != 0:
                        # print("find")
                        return 1
        # print("not find")
        return 0

def find_tech_v2(x, y): 
        y = y.split(',')
        # print(y)
        for idx, elem in enumerate(y):
                # print(f"y: {y}")
                # print(f"elem {elem}, x {x}")
                # print(type(x), type(elem))
                matches = re.findall(r'\b' + x + r'\b', elem)
                if len(matches) != 0:
                        # print("find")
                        return 1
        # print("not find")
        return 0

def find_author(x, y): 
        y = y.split(';')
        # print(y)
        for idx, elem in enumerate(y):
                # print(f"y: {y}")
                # print(f"elem {elem}, x {x}")
                # print(type(x), type(elem))
                matches = re.findall(r'\b' + x + r'\b', elem)
                if len(matches) != 0:
                        # print("find")
                        return 1
        # print("not find")
        return 0

def find_ipc(x, y): 
        y = y.split(',')
        # print(y)
        for idx, elem in enumerate(y):
                # print(f"y: {y}")
                # print(f"elem {elem}, x {x}")
                # print(type(x), type(elem))
                matches = re.findall(r'\b' + x + r'\b', elem)
                if len(matches) != 0:
                        # print("find")
                        return 1
        # print("not find")
        return 0

def find_country(x, y): 
        y = y.split(',')
        # print(y)
        for idx, elem in enumerate(y):
                # print(f"y: {y}")
                # print(f"elem {elem}, x {x}")
                # print(type(x), type(elem))
                matches = re.findall(r'\b' + x + r'\b', elem)
                if len(matches) != 0:
                        # print("find")
                        return 1
        # print("not find")
        return 0

@app.post("/search_v2")
async def search_v2(request: SearchRequest):
    global input_gen_ai, db_name, api_key, mistral_model, client

    user_query_message = request.query

    # =====
    # Filter the information for query
    # =====
    chatGPTmess = f"""
Extract the following information from the input text and return a JSON object. 
If a value is not explicitly found or cannot be inferred with high confidence 
(e.g., due to low semantic similarity), set its value to null.
For key "sdg" below, the value must be only integer numbers such as 6, 9023, etc.
For key "publication_number" below, the value must be only an integer number such as 6, 9023, etc.
For key "country" below, The value must be only the name of a country (e.g., "France", "Brazil"), and not include abbreviations, country codes, regions, or additional text.
For key "author" below, The value must be only the names of people or organizations (e.g., "David", "Fraunhofer"), and not include abbreviations or initilas of names or additional text.

Keys to extract: 
- "sdg"
- "country"
- "technology"
- "author"
- "publication_number"
- "ipc"

Input text:
===
{user_query_message}
===

Return the result as a JSON object with null for any missing or uncertain fields.
"""
    
    chat_response = client.chat.complete(
        model= mistral_model,
        messages = [
        {
            "role": "user",
            "content": chatGPTmess,
        }
        ],
        temperature= 0.1,
        max_tokens= 256000,
        # random_seed= 123,
        # response_format= { "type": "json_object" }
    )

    user_query_json = chat_response.choices[0].message.content

    user_query_json = json.loads(user_query_json)

    print(f"returned json object: {user_query_json}")

    print(f"returned json object type: {type(user_query_json)}")

    # ======
    # Query
    # ======
    question_embedding = get_embeddings([user_query_message], tokenizer, model).cpu().detach().numpy()
    try:
        with sqlite3.connect(db_name) as conn:

            # load the `sqlite-vec` extention into the connected db
            ## NOTE:
            ## must load the `sqlite-vec` extention everytime connect to the db, 
            ## in order to use the vec table created using extension `sqlte-vec` and `sqlite-vec` functions
            conn.enable_load_extension(True) # start loading extensions
            sqlite_vec.load(conn)
            conn.enable_load_extension(True) # end loading extensions

            conn.create_function('find_related_sdg', 2, find_sdg_v2)
            conn.create_function('find_related_tech', 2, find_tech_v2)
            conn.create_function('find_related_author', 2, find_author)
            conn.create_function('find_related_ipc', 2, find_ipc)
            conn.create_function('find_related_country', 2, find_country)

            cur = conn.cursor()

            # Get total number of data (rows) in the database
            sql_cn_meta = f"""
                SELECT COUNT(*) FROM meta_data_embeddings
            """
            res = cur.execute(sql_cn_meta)
            len_embed_table = res.fetchall()[0][0]
            
            # Get all cosine distance of all embeddings compared to the user input
            query = question_embedding.tolist()
            q_query_distance = conn.execute(
              f"""
              SELECT rowid, vec_distance_cosine(embedding, ?) AS score
              FROM vec_items
              ORDER BY score ASC
              LIMIT {str(len_embed_table)}
              """,
            [serialize_float32(query[0])],
            ).fetchall()

            # Prepare query_list
            key_func_maps = {
                "sdg": 'find_related_sdg',
                "country": 'find_related_country',
                "technology": 'find_related_tech',
                "author": 'find_related_author',
                "publication_number": 'pub_num == ',
                "ipc": 'find_related_ipc'
            }
# meta_table column names:
# pub_num
# title
# claims
# sdg_labels
# tech_labels
# country
# ipc
# author
            key_column_maps= {
                "sdg": 'sdg_labels',
                "country": 'country',
                "technology": 'tech_labels',
                "author": 'author',
                "publication_number": 'pub_num',
                "ipc": 'ipc'
            }

            
            query_list_v2 = []
            for key, value in user_query_json.items():
                if value != None:
                    if key != "publication_number":
                      query_list_v2.append(key_func_maps[key] + f"('{str(value)}', {key_column_maps[key]})")
                    else:
                      query_list_v2.append(key_func_maps[key] + f"{str(value)}")
            print(f"query_list_v2: {query_list_v2}")

            
            # sdg_id = 6
            # tech = "Chemical"
            # query_list = [f"find_related_sdg ('{str(int(sdg_id))}', sdg_labels)", f"find_related_tech ('{tech}', tech_labels)"]
            # print(f"query_list: {query_list}")

            # Query to meta table
            meta_query_try = f"""
                              SELECT
                                      rowid,
                                      pub_num,
                                      sdg_labels
                              FROM meta_data_embeddings"""

            for idx, value in enumerate(query_list_v2):
                if idx == 0:
                    meta_query_try = meta_query_try + f"\nWHERE {value}"
                else:
                    meta_query_try = meta_query_try + f"\nOR {value}"
            # try
            
            # Get all patent related to the user queried
            meta_query = meta_query_try

            meta_data_sdg = cur.execute(meta_query).fetchall()
            # print(len(meta_data_sdg),
            #       meta_data_sdg[0:20])
            
            # Find the sdg_id related patent in the distance query
            np_q_query_distance = np.array(q_query_distance)
            np_meta_data_sdg = np.array(meta_data_sdg)

            res = {}
            limit_search = 2 # by default
            if np_meta_data_sdg.size == 0:
                  print(f"no result for the query information")
                  print(f"return most relevant docs")
                  q_dis_sdg = np_q_query_distance[0:limit_search, :]


            else:
                  np_meta_data_sdg_float = np_meta_data_sdg[:,0].astype(float)

                  indices = np.where(np.isin(np_q_query_distance[:,0], np_meta_data_sdg_float))[0]

                  limit_search = np_meta_data_sdg.size
                  # print(np.shape(np_q_query_distance))
                  q_dis_sdg = np_q_query_distance[indices[0:limit_search], :]
                  # print(q_dis_sdg,
                  #       np.shape(q_dis_sdg))
                  
            for idx, elem in enumerate(q_dis_sdg):
                      # print(f"from q_dis_sdg: {str(q_dis_sdg[idx,0])}")
                      # print(f"elem: {str(elem)}")
                      # print(f"elem first: {int(elem[0])}")
                      # print(f"elem second: {float(elem[1])}")
                      meta_query = f"""
                          SELECT
                                  rowid,
                                  pub_num,
                                  sdg_labels,
                                  title,
                                  claims,
                                  tech_labels,
                                  author,
                                  country
                          FROM meta_data_embeddings
                          WHERE rowid == {int(elem[0])}
                      """

                      meta_data_sdg_dis = cur.execute(meta_query).fetchall()
                      # print(meta_data_sdg_dis)
                      # print(f"\
                      #       rowid: {meta_data_sdg_dis[0][0]}\n\
                      #       pub_num: {meta_data_sdg_dis[0][1]}\n\
                      #       sdg: {meta_data_sdg_dis[0][2]}\n\
                      #       tech_labels: {meta_data_sdg_dis[0][5]}\n\
                      #       dist: {float(elem[1])}\n\
                      #       title: {meta_data_sdg_dis[0][3]}\n\
                      #       claim: {meta_data_sdg_dis[0][4]}\n\
                      #       ")
                      res.update({
                          int(idx):{
                                "pub_num": int(meta_data_sdg_dis[0][1]),
                                "sdg_id": str(meta_data_sdg_dis[0][2]),
                                "tech_labels": meta_data_sdg_dis[0][5],
                                "author": meta_data_sdg_dis[0][6],
                                "country": meta_data_sdg_dis[0][7],
                                "dist": float(elem[1]),
                                "title": meta_data_sdg_dis[0][3],
                                "claim": meta_data_sdg_dis[0][4]
                          }
                      })
    except sqlite3.OperationalError as e:
        print(e)

    input_gen_ai = {"query": request.query, "relevant_docs": res} 
    # NOTE:
    # JSON encoder for returned object: https://fastapi.tiangolo.com/advanced/response-directly/
    # All of the returned objects MUST be converted to known python standard objects.
    return input_gen_ai 


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

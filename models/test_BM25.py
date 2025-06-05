import os 
import sys
sys.path.append(".")
import requests
import pandas as pd 
import csv 
from tqdm import tqdm 
from models.repllama_server_API import generate_with_repllama, generate_with_repllama_batched
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np 
import json 


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(SCRIPT_DIR, "../.env")

from dotenv import load_dotenv
load_dotenv(ENV_PATH)


def bm25_search(query: str, k: int = 5, verbose=False):
    
    """
    Sends a GET request to the BM25 search API and prints the results.
    """
    # Define the API URL (update if running on a different machine)
    BASE_URL = "http://localhost:8012"
    url = f"{BASE_URL}/search"
    params = {"query": query, "k": k}

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise error for non-200 status codes

        data = response.json()
        
        if verbose is True:
            print("\n🔍 Search Query:", query)
            print("📌 Top Results:")
            print(data["results"][0])
            print(data["results"][1])
            print(data["results"][2])
            for i, (content, doc_id, score) in enumerate(zip(*data["results"]), 1):
                print(f"{i}. [ID: {doc_id}] (Score: {score:.2f})")
                print(f"   {content[:200]}...\n")  # Show only first 200 characters
        
        
        return data["results"][0], data["results"][1], data["results"][2]

    except requests.exceptions.RequestException as e:
        print("❌ API Request Failed:", e)



def get_doc_content(doc_id):
    
    with open(f'../theorems_corpus/theorem_{doc_id}.jsonl', 'r') as file:
        data = [json.loads(line) for line in file]
        data = data[0]
        # Get the id of the document and the text of the theorem
        content = data['contents']
    
    return content     



###### Code for adding BM25, Repllama retrieved results for the lexical queries 
###### and filtering using round trip consistency.

# df_path = "NuminaMath_LLMoutputs_forlexicalqueries.csv"
# df = pd.read_csv(df_path)
# print(df.columns)
# print(len(df))
# output_file = "NuminaMath_LLMoutputs_bm25_repllama_added.csv"

# # Create the output CSV file with headers
# with open(output_file, mode='w', newline='', encoding='utf-8') as f:
#     writer = csv.writer(f)
#     # Write header: original columns + new columns
#     writer.writerow(list(df.columns) + ['BM25_top100_docids', 'BM25_top100_docs', 'flag_in_top20', 'Repllama_top10_docids', 'Repllama_top10_docs'])


# # # Get repllama doc embeddings 
# repllama_doc_emb_path = os.environ.get("REPLLAMA_THEOREMT_DOC_EMB_CACHE", "")
# doc_emb = np.load(repllama_doc_emb_path)


# k = 20  # Parameter for round-trip consistency 
# for i in tqdm(range(len(df))):
#     query = df["generated_LLM"][i]
    
#     ##### BM25 retrieval part #####
#     # Code for Round trip consistency 
#     top_k_docs, doc_ids, doc_scores = bm25_search(query, k=100)
#     flag = str(df["theorem_id"][i]) in doc_ids[:k]
    
#     ##### Repllama retrieval part #####
#     query_emb = generate_with_repllama(f"Query: {query}</s>")
#     scores_repllama = cosine_similarity(query_emb, doc_emb)
#     scores_repllama = scores_repllama.tolist()

#     # Get the top k retrieved docs for augmentation 
#     doc_scores_repllama = {}
#     count = 0 
#     for j in range(len(scores_repllama[0])):
#         s = scores_repllama[0][j]
#         doc_scores_repllama[j] = s 
#         count = count + 1

#     # Get the scores and corresponding doc_ids of the top 5 repllama retrieved docs 
#     top_k_docs_repllama = sorted(doc_scores_repllama.items(), key=lambda x: x[1], reverse=True)[:10]

#     # Extract only the doc_ids from the top k
#     top_k_doc_ids_repllama = [doc_id for doc_id, score in top_k_docs_repllama]
#     doc_ids_repllama = [str(doc_) for doc_ in top_k_doc_ids_repllama]
    
#     # Get the doc content 
#     doc_content = [get_doc_content(id_) for id_ in doc_ids_repllama]
    
#     # Compose the row with new columns
#     row = list(df.iloc[i]) + [doc_ids, top_k_docs, flag, doc_ids_repllama, doc_content]

#     # Append the row to the CSV
#     with open(output_file, mode='a', newline='', encoding='utf-8') as f:
#         writer = csv.writer(f)
#         writer.writerow(row)
  
    
#     print(f"Is the gold doc in BM25 top {k} results?", flag)
#     print("-"*80)


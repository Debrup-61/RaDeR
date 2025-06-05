import sys
sys.path.append(".")

import os
import json
import subprocess
import asyncio
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException
from pyserini.search.lucene import LuceneSearcher

app = FastAPI()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(SCRIPT_DIR, "../index_bm25")

# Initialize a Pyserini searcher.
lucene_bm25_searcher = LuceneSearcher(INDEX_PATH)   # Path to prebuilt BM25 index 

# Create a ThreadPoolExecutor for blocking search calls.
executor = ThreadPoolExecutor(max_workers=10)

def send_query(query, k):
    response = get(f"http://localhost:8012/search?query={query}&k={k}")
    response.raise_for_status()
    return response.json()


def get_contents(hits):
    top_documents = []
    top_doc_ids = []
    top_doc_scores = [] 

    for i in range(len(hits)):
        
        with open(f'../theorems_corpus/theorem_{str(hits[i].docid)}.jsonl', 'r') as file:
            data = [json.loads(line) for line in file]
            data = data[0]
        
        # Get the id of the document and the text of the theorem
        id_doc = data['id']
        content = data['contents']
        
        top_documents.append(content)
        top_doc_ids.append(id_doc)
        top_doc_scores.append(hits[i].score)

    return top_documents, top_doc_ids, top_doc_scores
    
def perform_search(query, top_k=5):
    hits = lucene_bm25_searcher.search(query, k=top_k)
    return get_contents(hits)


@app.get("/search")
async def search_endpoint(query: str, k: int = 10):
    """
    An asynchronous endpoint that offloads the blocking search call
    to a separate thread, allowing multiple queries to be processed concurrently.
    """
    loop = asyncio.get_event_loop()
    try:
        results = await loop.run_in_executor(executor, perform_search, query, k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    return {"results": results}

@app.on_event("shutdown")
def shutdown_event():
    executor.shutdown(wait=True)

# Add this section at the end of the file
if __name__ == "__main__":
    import uvicorn
    
    # Get port from command line or use default
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8012, help="Port to run the BM25 server on")
    args = parser.parse_args()
    
    print(f"Starting BM25 server on port {args.port}...")
    uvicorn.run(app, host="0.0.0.0", port=args.port)
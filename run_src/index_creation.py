from datasets import load_dataset
import json 
from pyserini.search.faiss import FaissSearcher
from pyserini.search.lucene import LuceneSearcher

from tqdm import tqdm

'''
BM25 Index creation command:

python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input theorems_corpus/ \
  --index index_bm25 \
  --generator DefaultLuceneDocumentGenerator \
  --threads 1 \
  --storeRaw
'''

def make_json_theorems():
    print("Loading dataset...")
    ds = load_dataset("xlangai/BRIGHT", "documents")
    print("Dataset loaded, creating jsonl files...")
    theoremqa_theorems = ds["theoremqa_theorems"]
    for i in tqdm(range(len(theoremqa_theorems))):
        d = theoremqa_theorems[i]
        id_ = d["id"]
        # Modify the key 'content' to 'contents'
        d["contents"] = d.pop("content")
        # Replace the newlines with space 
        d["contents"] = d["contents"].replace("\n", " ")

        with open(f"theorems_corpus/theorem_{id_}.jsonl", "a") as file:
            json_line = json.dumps(d)      # Convert dictionary to JSON string
            file.write(json_line + "\n")   # Write JSON string to file with a newline



def get_contents(hits, log=False, question_id=None):
   
    
    top_documents = []
    top_doc_ids = []
    top_doc_scores = [] 

    # top_docs_dict = {
    #         "question_id": question_id,
    #         "top_docs": []
    #     }

    # Path to the JSONL file
    # output_file = "top_docs_scores_BM25.jsonl"


    for i in range(len(hits)):
        
        '''
        if log is True:
            print(f'{i+1:2} {hits[i].docid:7} {hits[i].score:.5f}')
        '''
        
        with open(f'theorems_corpus/theorem_{str(hits[i].docid)}.jsonl', 'r') as file:
            data = [json.loads(line) for line in file]
            data = data[0]
        
        # Get the id of the document and the text of the theorem
        id_doc = data['id']
        content = data['contents']
        
        top_documents.append(content)
        top_doc_ids.append(id_doc)
        top_doc_scores.append(hits[i].score)
    
    return top_documents, top_doc_ids, top_doc_scores
    


def search_bm25(query, path_to_index="index_bm25", top_k=5, log=False, question_id=None):
    searcher = LuceneSearcher(path_to_index)
    hits = searcher.search(query, k=top_k)
    return get_contents(hits,question_id=question_id)



if __name__ == "__main__":

    q = "Pythagoras Theorem"
    print(search_bm25(q, path_to_index="index_bm25", top_k=5, log=True))
   
from models.RaDeR_retriever_server_API import generate_with_qwen_batched
from models.repllama_server_API import generate_with_repllama_batched
import argparse
import os
import json
from tqdm import tqdm
from datasets import load_dataset
import numpy as np 
from huggingface_hub import login
from transformers import AutoTokenizer

def main(args):
    # doc_pairs = load_dataset('xlangai/bright', 'documents',cache_dir=args.cache_dir)[args.task]

    # Get the ids and the content for the documents
    documents = []
    doc_ids = []
    
    if args.task == "msmarco":
        docs = load_dataset('irds/msmarco-passage', 'docs', cache_dir=args.cache_dir)
        print("No of documents in MS-MARCO passage:", len(docs))
        for record in docs:
            documents.append(record['text'])
            doc_ids.append(record['doc_id'])
            
        print("MS-MARCO documents load completed!")

    else:
        doc_pairs = load_dataset('xlangai/bright', 'documents',cache_dir=args.cache_dir)[args.task]
        count = 0 
        for dp in doc_pairs:
            documents.append(dp['content'])
            doc_ids.append(dp['id'])

        print("BRIGHT document load completed!")    



    save_path = os.path.join(args.cache_dir, args.model_name, args.task, 'long_False_1')
    # Create the directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)


    # save documents to memmap file
    if args.task == "msmarco":
        doc_cache_file = os.path.join(args.cache_dir, args.model_name, args.task, 'long_False_1', 'MSMARCO_passage_documents.npy')
    else:
        doc_cache_file = os.path.join(args.cache_dir, args.model_name, args.task, 'long_False_1', 'BRIGHT_documents.npy')
    
    if not os.path.exists(doc_cache_file):
        print("Saving to", doc_cache_file)
        # save documents to file
        np.save(doc_cache_file, np.array(documents, dtype=object))

        # save doc ids as well
        if args.task == "msmarco":
            doc_id_file = os.path.join(args.cache_dir, args.model_name, args.task, 'long_False_1', 'MSMARCO_passage_doc_ids.npy')
        else:
            doc_id_file = os.path.join(args.cache_dir, args.model_name, args.task, 'long_False_1', 'BRIGHT_doc_ids.npy')
        
        np.save(doc_id_file, np.array(doc_ids, dtype=object)) 
        
    if args.model_name=="RepLLama":
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_fast=True, cache_dir=os.environ.get("HUGGINGFACE_CACHE_DIR",""))
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_hf_dir)
    
    eos_token = tokenizer.eos_token
    print("EOS token:", eos_token)
    # create embeddings and store in memmap file
    emb_cache_file = os.path.join(args.cache_dir, args.model_name, args.task, 'long_False_1', f'0.npy')
    if not os.path.exists(emb_cache_file):
        print("Creating document embeddings...")
        if args.model_name == "RepLLama":
            doc_emb = generate_with_repllama_batched(inputs=[f"document: {document}{eos_token}" for document in documents], max_threads=16)
        else:    
            doc_emb = generate_with_qwen_batched(tokenizer=tokenizer, inputs=[f"document: {document}{eos_token}" for document in documents], max_threads=16, model_hf_path = args.model_hf_dir, eos_token=eos_token)
        print(f"Saving to {emb_cache_file}...")
        np.save(emb_cache_file, doc_emb)
        print("Save completed!")
        del doc_emb

    del documents
    del doc_ids

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Initialize the retrieval server with documents")
    
    # Adding arguments
    parser.add_argument('--cache_dir', type=str, default='BRIGHT_cache/doc_emb', help="Directory for caching datasets")
    parser.add_argument('--model_name', type=str, default='MATH_qpartialsol_and_LLMquery_full', help="Name of the trained_retriever model")
    parser.add_argument('--model_hf_dir', type=str, default='RaDeR/merged_retriever_Qwen-2.5-7B-Instruct_MATH_questionpartialsol_and_LLMquery_full', help="Huggingface repo of trained_retriever model")
    parser.add_argument('--task', type=str, required=True, help="The specific split of the BRIGHT dataset")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    main(args)

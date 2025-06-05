import os.path
import os 
import time
import torch
import json
# import cohere
import numpy as np
#import vertexai
import pytrec_eval
#import tiktoken
#import voyageai
from tqdm import tqdm,trange
import torch.nn.functional as F
#from gritlm import GritLM
#from openai import OpenAI
from transformers import AutoTokenizer, AutoModel
#from InstructorEmbedding import INSTRUCTOR
from transformers import BertModel, BertTokenizer
#from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from peft import PeftModel, PeftConfig
from huggingface_hub import login
from tqdm import tqdm 
# from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
from torchmetrics.functional.pairwise import pairwise_cosine_similarity
import concurrent.futures
from transformers import AutoTokenizer, AutoModelForCausalLM

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(SCRIPT_DIR, ".env")

from dotenv import load_dotenv
load_dotenv(ENV_PATH)

def process_batch(batch_indices, batch_docs, tokenizer, model, device):
    inputs = tokenizer(batch_docs, padding=True, truncation=True, return_tensors='pt')  # Batch tokenization
    inputs = {key: val.to(device) for key, val in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        
        # Take embeddings from the last hidden state of the last token
        embeddings = outputs.last_hidden_state[:, -1, :]
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)  # Normalize
        embeddings = embeddings.cpu().numpy()
        
    return batch_indices, embeddings

def process_documents_parallel(documents, tokenizer, model, device, batch_size=8):
    doc_emb = [None] * len(documents)  # Initialize a list with the same length as documents

    # Split documents into batches
    batches = [(list(range(i, min(i + batch_size, len(documents)))), documents[i:i + batch_size])
               for i in range(0, len(documents), batch_size)]

    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
        # Create futures for processing each batch
        futures = [executor.submit(process_batch, batch_indices, batch_docs, tokenizer, model, device)
                   for batch_indices, batch_docs in batches]

        # Process the results as they complete
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            batch_indices, embeddings = future.result()
            # Place the embeddings back in the correct order in doc_emb
            for idx, embedding in zip(batch_indices, embeddings):
                doc_emb[idx] = embedding

    return doc_emb


def process_document(idx, text, tokenizer, model, device):
    inputs = tokenizer(f"document: {text}<|im_end|>", return_tensors='pt')  # max_length=4096
    inputs = {key: val.to(device) for key, val in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        
        embeddings = outputs.last_hidden_state[:, -1, :]  # Take the last hidden state
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)  # Normalize
        embeddings = embeddings.cpu().numpy()
        
    return idx, embeddings

# def process_documents_parallel(documents, tokenizer, model, device):
#     doc_emb = [None] * len(documents)  # Initialize a list with the same length as documents

#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         # Create futures for processing each document
#         futures = [executor.submit(process_document, i, doc, tokenizer, model, device) 
#                    for i, doc in enumerate(documents)]

#         # Process the results as they complete
#         for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
#             idx, embeddings = future.result()
#             doc_emb[idx] = embeddings  # Place the result in the correct index

#     return doc_emb


def get_model(peft_model_name):
    huggingface_token = os.environ.get("HUGGING_FACE_HUB_TOKEN","")   #Set huggingface token 
    
    # Login to huggingface 
    login(token=huggingface_token)
    
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the model configuration and weights
    config = PeftConfig.from_pretrained(peft_model_name)
    print("Base model load started")
    base_model = AutoModel.from_pretrained(config.base_model_name_or_path, device_map="auto" if torch.cuda.is_available() else None)
    print("Base model load completed!")
    model = PeftModel.from_pretrained(base_model, peft_model_name)
    print("PEFT Model loaded!")
    # Merge LoRA weights and unload
    model = model.merge_and_unload()
    model.eval()
    
    # Move model to device
    #model.to(device)
    return model

def encode_texts(texts, tokenizer, model, prefix):
    """
    Encode a batch of texts (queries or passages).

    Args:
        texts (list of str): List of texts to encode.
        tokenizer: Tokenizer for text encoding.
        model: Model for embedding generation.
        prefix (str): Prefix for text type ('query' or 'passage').

    Returns:
        torch.Tensor: Normalized embeddings for the input texts.
    """
    #tokenizer.pad_token = tokenizer.eos_token 
    #tokenizer.add_special_tokens({"pad_token":"<pad>"})
    #print("Tokenizer max length", tokenizer.model_max_length)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embeddings_list = []
    for text in texts:
        inputs = tokenizer(f"{prefix}: {text}</s>", return_tensors='pt')  # max_length=4096
        inputs = {key: val.to(device) for key, val in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state[:, -1, :]  # Take the last hidden state
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)  # Normalize
            embeddings_list.append(embeddings)
            #print(f"{prefix} Last hidden state shape", (outputs.last_hidden_state).shape)

    embeddings = torch.cat(embeddings_list, dim=0)        
    return embeddings



def print_high_ndcg_queries(scores, output_file, threshold=0):
    """
    Print query IDs where NDCG@10 > threshold.

    Parameters:
        scores (dict): A dictionary containing scores per query.
        threshold (float): The threshold for NDCG@10.
    """
    high_ndcg_queries = [query_id for query_id, metrics in scores.items() if metrics.get("ndcg_cut_10", 0) > threshold]

    # Write query IDs to the output file
    with open(output_file, "w") as f:
        for query_id in high_ndcg_queries:
            f.write(query_id + "\n")

    print(f"Stored {len(high_ndcg_queries)} query IDs in {output_file}.")


def average_pooling_contriever(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings

def mean_pooling(last_hidden_state, attention_mask):
    """
    Applies mean pooling to the last hidden state using the attention mask.

    Args:
        last_hidden_state (torch.Tensor): The token embeddings from the model.
                                          Shape: (batch_size, sequence_length, hidden_size)
        attention_mask (torch.Tensor): The attention mask indicating non-padding tokens.
                                       Shape: (batch_size, sequence_length)

    Returns:
        torch.Tensor: Pooled embeddings. Shape: (batch_size, hidden_size)
    """
    # Expand attention mask to match hidden state dimensions
    mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    # Sum token embeddings along the sequence length, weighted by the attention mask
    sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
    # Divide by the number of valid (non-padding) tokens to get the mean
    sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)  # Avoid division by zero
    return sum_embeddings / sum_mask


def cut_text(text,tokenizer,threshold):
    text_ids = tokenizer(text)['input_ids']
    if len(text_ids) > threshold:
        text = tokenizer.decode(text_ids[:threshold])
    return text

def cut_text_openai(text,tokenizer,threshold=6000):
    token_ids = tokenizer.encode(text)
    if len(token_ids) > threshold:
        text = tokenizer.decode(token_ids[:threshold])
    return text

def get_embedding_google(texts,task,model,dimensionality=768):
    success = False
    while not success:
        try:
            new_texts = []
            for t in texts:
                if t.strip()=='':
                    print('empty content')
                    new_texts.append('empty')
                else:
                    new_texts.append(t)
            texts = new_texts
            inputs = [TextEmbeddingInput(text, task) for text in texts]
            kwargs = dict(output_dimensionality=dimensionality) if dimensionality else {}
            embeddings = model.get_embeddings(inputs, **kwargs)
            success = True
        except Exception as e:
            print(e)
    return [embedding.values for embedding in embeddings]

def get_embedding_openai(texts, openai_client,tokenizer,model="text-embedding-3-large"):
    texts =[json.dumps(text.replace("\n", " ")) for text in texts]
    success = False
    threshold = 6000
    count = 0
    cur_emb = None
    exec_count = 0
    while not success:
        exec_count += 1
        if exec_count>5:
            print('execute too many times')
            exit(0)
        try:
            emb_obj = openai_client.embeddings.create(input=texts, model=model).data
            cur_emb = [e.embedding for e in emb_obj]
            success = True
        except Exception as e:
            print(e)
            count += 1
            threshold -= 500
            if count>4:
                print('openai cut',count)
                exit(0)
            new_texts = []
            for t in texts:
                new_texts.append(cut_text_openai(text=t, tokenizer=tokenizer,threshold=threshold))
            texts = new_texts
    if cur_emb is None:
        raise ValueError("Fail to embed, openai")
    return cur_emb

TASK_MAP = {
    'biology': 'Biology',
    'earth_science': 'Earth Science',
    'economics': 'Economics',
    'psychology': 'Psychology',
    'robotics': 'Robotics',
    'stackoverflow': 'Stack Overflow',
    'sustainable_living': 'Sustainable Living',
}

def add_instruct_concatenate(texts,task,instruction):
    return [instruction.format(task=task)+t for t in texts]

def add_instruct_list(texts,task,instruction):
    return [[instruction.format(task=task),t] for t in texts]

def mean_pooling(last_hidden_state, attention_mask):
    """
    Applies mean pooling to the last hidden state using the attention mask.

    Args:
        last_hidden_state (torch.Tensor): The token embeddings from the model.
                                          Shape: (batch_size, sequence_length, hidden_size)
        attention_mask (torch.Tensor): The attention mask indicating non-padding tokens.
                                       Shape: (batch_size, sequence_length)

    Returns:
        torch.Tensor: Pooled embeddings. Shape: (batch_size, hidden_size)
    """
    # Expand attention mask to match hidden state dimensions
    mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    # Sum token embeddings along the sequence length, weighted by the attention mask
    sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
    # Divide by the number of valid (non-padding) tokens to get the mean
    sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)  # Avoid division by zero
    
    return sum_embeddings / sum_mask


def last_token_pool(last_hidden_states,attention_mask):
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]



def get_scores(query_ids,doc_ids,scores,excluded_ids):
    assert len(scores)==len(query_ids),f"{len(scores)}, {len(query_ids)}"
    assert len(scores[0])==len(doc_ids),f"{len(scores[0])}, {len(doc_ids)}"
    emb_scores = {}
    for query_id,doc_scores in zip(query_ids,scores):
        cur_scores = {}
        assert len(excluded_ids[query_id])==0 or (isinstance(excluded_ids[query_id][0], str) and isinstance(excluded_ids[query_id], list))
        for did,s in zip(doc_ids,doc_scores):
            cur_scores[str(did)] = s
        for did in set(excluded_ids[str(query_id)]):
            if did!="N/A":
                cur_scores.pop(did)
        cur_scores = sorted(cur_scores.items(),key=lambda x:x[1],reverse=True)[:1000]
        emb_scores[str(query_id)] = {}
        for pair in cur_scores:
            emb_scores[str(query_id)][pair[0]] = pair[1]
    return emb_scores


@torch.no_grad()
def retrieval_sf_qwen_e5(queries,query_ids,documents,doc_ids,task,model_id,instructions,cache_dir,excluded_ids,long_context,**kwargs):
    
    print("In Retrieval func!")

    if model_id=='sf':
        tokenizer = AutoTokenizer.from_pretrained('salesforce/sfr-embedding-mistral')
        model = AutoModel.from_pretrained('salesforce/sfr-embedding-mistral',device_map="auto").eval()
        max_length = kwargs.get('doc_max_length',4096)
    elif model_id=='qwen':
        tokenizer = AutoTokenizer.from_pretrained('alibaba-nlp/gte-qwen1.5-7b-instruct', trust_remote_code=True)
        model = AutoModel.from_pretrained('alibaba-nlp/gte-qwen1.5-7b-instruct', device_map="auto", trust_remote_code=True).eval()
        max_length = kwargs.get('doc_max_length',8192)
    elif model_id=='qwen2':
        tokenizer = AutoTokenizer.from_pretrained('alibaba-nlp/gte-qwen2-7b-instruct', trust_remote_code=True)
        model = AutoModel.from_pretrained('alibaba-nlp/gte-qwen2-7b-instruct', device_map="auto", trust_remote_code=True).eval()
        max_length = kwargs.get('doc_max_length',8192)
    elif model_id=='e5':
        tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-mistral-7b-instruct')
        model = AutoModel.from_pretrained('intfloat/e5-mistral-7b-instruct', device_map="auto").eval()
        max_length = kwargs.get('doc_max_length',4096)
    else:
        raise ValueError(f"The model {model_id} is not supported")
    model = model.eval()
    queries = add_instruct_concatenate(texts=queries,task=task,instruction=instructions['query'])
    batch_size = kwargs.get('encode_batch_size',1)

    doc_emb = None
    
    cache_dir_path = os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}")
    os.makedirs(os.path.dirname(cache_dir_path), exist_ok=True)
    print("Cache dir:", cache_dir_path)

    cache_path = os.path.join(cache_dir, 'doc_emb', model_id , task, f"long_{long_context}_{batch_size}", f'0.npy')
    
    if os.path.isfile(cache_path):
        print("Exists!")
        # already exists so we can just load it
        doc_emb = np.load(cache_path, allow_pickle=True)
    
    for start_idx in trange(0,len(documents),batch_size):
        assert doc_emb is None or doc_emb.shape[0] % batch_size == 0, f"{doc_emb % batch_size} reminder in doc_emb"
        if doc_emb is not None and doc_emb.shape[0] // batch_size > start_idx:
            continue

        batch_dict = tokenizer(documents[start_idx:start_idx+batch_size], max_length=max_length, padding=True, truncation=True, return_tensors='pt').to(model.device)
        outputs = model(**batch_dict)
        embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask']).cpu()
        # doc_emb[start_idx] = embeddings
        doc_emb = embeddings if doc_emb is None else np.concatenate((doc_emb, np.array(embeddings)), axis=0)

        # save the embeddings every 1000 iters, you can adjust this as needed
        if (start_idx + 1) % 1000 == 0:
            np.save(cache_path, doc_emb)
        
    np.save(cache_path, doc_emb)

    doc_emb = torch.tensor(doc_emb)
    print("doc_emb shape:",doc_emb.shape)
    doc_emb = F.normalize(doc_emb, p=2, dim=1)
    query_emb = []
    for start_idx in trange(0, len(queries), batch_size):
        batch_dict = tokenizer(queries[start_idx:start_idx + batch_size], max_length=max_length, padding=True,
                               truncation=True, return_tensors='pt').to(model.device)
        outputs = model(**batch_dict)
        embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask']).cpu().tolist()
        query_emb += embeddings
    query_emb = torch.tensor(query_emb)
    print("query_emb shape:", query_emb.shape)
    query_emb = F.normalize(query_emb, p=2, dim=1)
    
    query_emb = query_emb.float()  # or .to(torch.float32)
    doc_emb = doc_emb.float()

    scores = (query_emb @ doc_emb.T) * 100
    scores = scores.tolist()
    return get_scores(query_ids=query_ids,doc_ids=doc_ids,scores=scores,excluded_ids=excluded_ids)


def retrieval_bm25(queries,query_ids,documents,doc_ids,excluded_ids,long_context,**kwargs):
    from pyserini import analysis
    from gensim.corpora import Dictionary
    from gensim.models import LuceneBM25Model
    from gensim.similarities import SparseMatrixSimilarity
    analyzer = analysis.Analyzer(analysis.get_lucene_analyzer())
    corpus = [analyzer.analyze(x) for x in documents]
    dictionary = Dictionary(corpus)
    model = LuceneBM25Model(dictionary=dictionary, k1=0.9, b=0.4)
    bm25_corpus = model[list(map(dictionary.doc2bow, corpus))]
    bm25_index = SparseMatrixSimilarity(bm25_corpus, num_docs=len(corpus), num_terms=len(dictionary),
                                        normalize_queries=False, normalize_documents=False)
    all_scores = {}
    bar = tqdm(queries, desc="BM25 retrieval")
    for query_id, query in zip(query_ids, queries):
        bar.update(1)
        query = analyzer.analyze(query)
        bm25_query = model[dictionary.doc2bow(query)]
        
        # if query_id == 'TheoremQA_mingyin/convexity1.json':
        #     print("Query:", query)
        #     print(f"Final bm25 query {query_id}:", bm25_query)
        
        similarities = bm25_index[bm25_query].tolist()
        all_scores[str(query_id)] = {}
        for did, s in zip(doc_ids, similarities):
            all_scores[str(query_id)][did] = s
        for did in set(excluded_ids[str(query_id)]):
            if did!="N/A":
                all_scores[str(query_id)].pop(did)
        cur_scores = sorted(all_scores[str(query_id)].items(),key=lambda x:x[1],reverse=True)[:1000]
        all_scores[str(query_id)] = {}
        for pair in cur_scores:
            all_scores[str(query_id)][pair[0]] = pair[1]
    return all_scores

@torch.no_grad()
def retrieval_mathbert(queries,query_ids,documents,doc_ids,task,instructions,model_id,cache_dir,excluded_ids,long_context,**kwargs):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the MATHBERT model from huggingface
    tokenizer = BertTokenizer.from_pretrained('tbs17/MathBERT', output_hidden_states=True)
    model = BertModel.from_pretrained("tbs17/MathBERT")
   
    # Move the model to the GPU
    model.to(device)
    
    # Append instructions before queries 
    queries = add_instruct_concatenate(texts=queries,task=task,instruction=instructions['query'])
    batch_size = kwargs.get('batch_size', 128)
    
    # Check if documents are already encoded 
    if not os.path.isdir(os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}")):
        os.makedirs(os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}"))
    
    cur_cache_file = os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}", f'0.npy')
    
    if os.path.isfile(cur_cache_file):
        doc_emb = np.load(cur_cache_file,allow_pickle=True)
    else:
        doc_emb = []
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i+batch_size]
            print("Batch docs", batch_docs)

            # Tokenize the batch of documents
            encoded_input = tokenizer(batch_docs, padding=True, max_length=512, truncation=True, return_tensors='pt')
            
            # Move the tokenized inputs to the GPU
            encoded_input = {key: val.to(device) for key, val in encoded_input.items()}

            # Generate embeddings for the batch
            with torch.no_grad():
                outputs = model(**encoded_input)

            # Use the [CLS] token's hidden state as the embedding
            #embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

            # Use mean pooling of the final layer 
            embeddings = mean_pooling(outputs, encoded_input['attention_mask']).cpu().numpy()

            # Append batch embeddings to the list
            doc_emb.extend(embeddings)

        # Convert to numpy array and save
        doc_emb = np.array(doc_emb)
        np.save(cur_cache_file, doc_emb)

    print("Shape of doc emb", doc_emb.shape)
    # Encode queries 
    # Tokenize and encode queries in batches
    query_emb = []
    for i in range(0, len(queries), batch_size):
        batch_queries = queries[i:i+batch_size]
        print("Batch Queries:", batch_queries)

        # Tokenize the batch of queries
        encoded_input = tokenizer(batch_queries, max_length=512, padding=True, truncation=True, return_tensors='pt')

        # Move the tokenized inputs to the GPU
        encoded_input = {key: val.to(device) for key, val in encoded_input.items()}

        # Generate embeddings for the batch
        outputs = model(**encoded_input)

        # Use the [CLS] token's hidden state as the embedding
        #embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings = mean_pooling(outputs, encoded_input['attention_mask']).cpu().numpy()

        # Append batch embeddings to the list
        query_emb.extend(embeddings)

    # Convert to numpy array
    query_emb = np.array(query_emb)
    print("Shape of query emb", query_emb.shape)

    # Find cosine similarity between doc_emb and query_emb
    scores = cosine_similarity(query_emb, doc_emb)
    print("Scores shape", scores.shape)
    scores = scores.tolist()
    return get_scores(query_ids=query_ids,doc_ids=doc_ids,scores=scores,excluded_ids=excluded_ids)


@torch.no_grad()
def retrieval_repllama_doc(queries,query_ids,documents,doc_ids,task,instructions,model_id,cache_dir,excluded_ids,long_context,**kwargs):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    huggingface_token = os.environ.get("HUGGING_FACE_HUB_TOKEN","")   #Set huggingface token 
    
    batch_size = kwargs.get('batch_size',8)
    ablation =  kwargs.get('ablation','all')
    
    # Login to huggingface 
    login(token=huggingface_token)

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
    model = get_model('castorini/repllama-v1-7b-lora-doc')
    
    # Append instructions before queries 
    #queries = add_instruct_concatenate(texts=queries,task=task,instruction=instructions['query'])
    
    # Check if documents are already encoded 
    if not os.path.isdir(os.path.join(cache_dir, 'doc_emb', model_id + f"_{ablation}", task, f"long_{long_context}_{batch_size}")):
        os.makedirs(os.path.join(cache_dir, 'doc_emb', model_id + f"_{ablation}", task, f"long_{long_context}_{batch_size}"))
        print("Made new cache document directory !")
    
    cur_cache_file = os.path.join(cache_dir, 'doc_emb', model_id + f"_{ablation}", task, f"long_{long_context}_{batch_size}", f'0.npy')
    
    if os.path.isfile(cur_cache_file):
        doc_emb = np.load(cur_cache_file,allow_pickle=True)
    else:
        doc_emb = []
        for i in tqdm(range(0,len(documents))): #len(documents)
            text = documents[i]
            inputs = tokenizer(f"Document: {text}</s>", return_tensors='pt')  # max_length=4096
            inputs = {key: val.to(device) for key, val in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                embeddings = outputs.last_hidden_state[:, -1, :]  # Take the last hidden state
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)  # Normalize
                embeddings = embeddings.cpu().numpy()
                doc_emb.extend(embeddings)
                #print(f"{prefix} Last hidden state shape", (outputs.last_hidden_state).shape)
        
        # Convert to numpy array and save
        doc_emb = np.array(doc_emb)
        np.save(cur_cache_file, doc_emb)
        print("Document embeddings stored")

    print("Shape of doc emb", doc_emb.shape)
    
    # Encode queries 
    # Tokenize and encode queries in batches
    if type(queries[0]) != list:      #Check if the queries are individual queries or set of candidate theorems
        query_emb = []
        for i in tqdm(range(0, len(queries))):
            text = queries[i]
            if i==0:
                print("Text after augmentation by Qwen-2.5-32B-instruct", text)
            
            inputs = tokenizer(f"query: {text}</s>", return_tensors='pt')  # max_length=4096
            inputs = {key: val.to(device) for key, val in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                embeddings = outputs.last_hidden_state[:, -1, :]  # Take the last hidden state
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)  # Normalize
                embeddings = embeddings.cpu().numpy()
                query_emb.extend(embeddings)
        
        # Convert to numpy array
        query_emb = np.array(query_emb)
        print("Shape of query emb", query_emb.shape)

        # Find cosine similarity between doc_emb and query_emb
        scores = cosine_similarity(query_emb, doc_emb)
        print("Scores shape", scores.shape)
        scores = scores.tolist()
        return get_scores(query_ids=query_ids,doc_ids=doc_ids,scores=scores,excluded_ids=excluded_ids)

    else:   #Use each candidate theorem as a seperate query 

        numpy_scores = np.empty((0, len(doc_ids)))  # Shape (0, |D|)
        candidate_theorems_grounded_fname = "grounded_candidate_theorems.jsonl"
        for i in tqdm(range(0, len(queries))):
            candidate_theorems = queries[i]
            query_emb = []
            
            if len(candidate_theorems) == 0:
                print(f"No candidate theorems for example {i}")
            
            for theorem in candidate_theorems:
                                         
                print(f"Candidate theorem for example {i}: ", theorem)      #Print candidate theorems for example 0
                
                inputs = tokenizer(f"Query: {theorem}</s>", return_tensors='pt')  # max_length=4096
                inputs = {key: val.to(device) for key, val in inputs.items()}
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    embeddings = outputs.last_hidden_state[:, -1, :]  # Take the last hidden state
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)  # Normalize
                    embeddings = embeddings.cpu().numpy()
                    query_emb.extend(embeddings)   

            query_emb = np.array(query_emb)
            
            if query_emb.ndim == 1:  # If it's 1D, reshape it to (1, -1)
                query_emb = query_emb.reshape(1, -1)
                print("No of dimensions is 1 !")
                print(f"Shape of query emb for example {i}", query_emb.shape)

            # Find cosine similarity between doc_emb and query_emb for current question
            print("Query embedding shape",query_emb.shape)
            scores = cosine_similarity(query_emb, doc_emb)
            print("Scores shape of example: ",i, scores.shape)

            max_scores = np.max(scores, axis=0, keepdims=True)  # Shape (1, |D|)
            numpy_scores = np.concatenate((numpy_scores, max_scores), axis=0)    

            scores = scores.tolist()  
            query_id = query_ids[i]             #Id of the current question
            
            # Initialize the result dictionary
            result = {"query_id": query_id, "candidate_theorems": []}
            
            # Populate the dictionary with top 100 docs for each LLM generated theorem
            for idx, theorem in enumerate(candidate_theorems):
                
                # Get scores for the current theorem and sort by descending score
                top_indices = np.argsort(scores[idx])[::-1][:100]
                top_docs = [{"doc_id": doc_ids[i], "score": scores[idx][i]} for i in top_indices]
                
                # Append the theorem and its top docs to the result
                result["candidate_theorems"].append({
                    "theorem": theorem,
                    "top_docs": top_docs
                })

            # if i==0:
            #     print("Candidate theorems pool for 1st example", result["candidate_theorems"])

            with open(candidate_theorems_grounded_fname, "a") as jsonl_file:
                jsonl_file.write(json.dumps(result) + "\n")
        
        print("Final Scores shape", numpy_scores.shape)
        final_scores = numpy_scores.tolist()
        
        return get_scores(query_ids=query_ids,doc_ids=doc_ids,scores=final_scores,excluded_ids=excluded_ids)


@torch.no_grad()
def retrieval_repllama(queries,query_ids,documents,doc_ids,task,instructions,model_id,cache_dir,excluded_ids,long_context,**kwargs):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    huggingface_token = os.environ.get("HUGGING_FACE_HUB_TOKEN","")   #Set huggingface token 
    batch_size = kwargs.get('batch_size',1)

    # Login to huggingface 
    login(token=huggingface_token)

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
    model = get_model('castorini/repllama-v1-7b-lora-passage')
    
    # Append instructions before queries 
    #queries = add_instruct_concatenate(texts=queries,task=task,instruction=instructions['query'])
    
    # Check if documents are already encoded 
    if not os.path.isdir(os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}")):
        os.makedirs(os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}"))
    
    cur_cache_file = os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}", f'0.npy')
    
    if os.path.isfile(cur_cache_file):
        doc_emb = np.load(cur_cache_file,allow_pickle=True)
    else:
        doc_emb = []
        for i in tqdm(range(0, len(documents))): #len(documents)
            text = documents[i]
            inputs = tokenizer(f"Document: {text}</s>", return_tensors='pt')  # max_length=4096
            inputs = {key: val.to(device) for key, val in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                embeddings = outputs.last_hidden_state[:, -1, :]  # Take the last hidden state
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)  # Normalize
                embeddings = embeddings.cpu().numpy()
                doc_emb.extend(embeddings)
                #print(f"{prefix} Last hidden state shape", (outputs.last_hidden_state).shape)
        
        # Convert to numpy array and save
        doc_emb = np.array(doc_emb)
        np.save(cur_cache_file, doc_emb)

    print("Shape of doc emb", doc_emb.shape)
    
    # Encode queries 
    # Tokenize and encode queries in batches
    query_emb = []
    for i in tqdm(range(0, len(queries))):
        text = queries[i]
        inputs = tokenizer(f"Query: {text}</s>", return_tensors='pt')  # max_length=4096
        inputs = {key: val.to(device) for key, val in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state[:, -1, :]  # Take the last hidden state
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)  # Normalize
            embeddings = embeddings.cpu().numpy()
            query_emb.extend(embeddings)
     
    # Convert to numpy array
    query_emb = np.array(query_emb)
    print("Shape of query emb", query_emb.shape)

    # Find cosine similarity between doc_emb and query_emb
    scores = cosine_similarity(query_emb, doc_emb)
    print("Scores shape", scores.shape)
    scores = scores.tolist()
    return get_scores(query_ids=query_ids,doc_ids=doc_ids,scores=scores,excluded_ids=excluded_ids)


@torch.no_grad()
def retrieval_trainedmodel(queries,query_ids,documents,doc_ids,task,instructions,model_id,cache_dir,excluded_ids,long_context,trainedmodel_name, trainedmodel_hf, **kwargs):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    huggingface_token = os.environ.get("HUGGING_FACE_HUB_TOKEN","")
    batch_size = kwargs.get('batch_size',1)
    
    # Login to huggingface 
    login(token=huggingface_token)

    # Load the tokenizer and model
    #tokenizer = AutoTokenizer.from_pretrained('Alibaba-NLP/gte-Qwen2-7B-instruct')
    #tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/DeepSeek-R1-Distill-Qwen-7B')
    # tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B-Instruct')
    # tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-14B-Instruct')

    if trainedmodel_name == "qwen2.5_7b_instruct":
        model = AutoModelForCausalLM.from_pretrained(trainedmodel_hf, device_map="auto")
        model.eval()
    else:
        model = get_model(trainedmodel_hf)
    
    print("Model type", model)
    
    # Append instructions before queries 
    queries = add_instruct_concatenate(texts=queries,task=task,instruction=instructions['query'])
    print("Example query:", queries[0])
    
    # Check if documents are already encoded 
    if not os.path.isdir(os.path.join(cache_dir, 'doc_emb', trainedmodel_name, task, f"long_{long_context}_{batch_size}")):
        os.makedirs(os.path.join(cache_dir, 'doc_emb', trainedmodel_name, task, f"long_{long_context}_{batch_size}"))
    
    cur_cache_file = os.path.join(cache_dir, 'doc_emb', trainedmodel_name, task, f"long_{long_context}_{batch_size}", f'0.npy')
    
    if os.path.isfile(cur_cache_file):
        doc_emb = np.load(cur_cache_file,allow_pickle=True)
    else:
        doc_emb = []
        #doc_emb = process_documents_parallel(documents, tokenizer, model, device)
        #doc_emb = process_documents_parallel(documents, tokenizer, model, device, batch_size=2)

        for i in tqdm(range(0, len(documents))): #len(documents)
            text = documents[i]
            #inputs = tokenizer(f"document: {text}<|im_end|>", return_tensors='pt')  # max_length=4096
            #inputs = tokenizer(f"document: {text}<|endoftext|>", return_tensors='pt')  # max_length=4096
            inputs = tokenizer(f"document: {text}{tokenizer.eos_token}", return_tensors='pt')
            
            
            inputs = {key: val.to(device) for key, val in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                
                if i==0:
                    print("Doc outputs shape", outputs.last_hidden_state.shape)
                
                
                embeddings = outputs.last_hidden_state[:, -1, :]  # Take the last hidden state
                
                
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)  # Normalize
                embeddings = embeddings.cpu().numpy()
                doc_emb.extend(embeddings)
                #print(f"{prefix} Last hidden state shape", (outputs.last_hidden_state).shape)
        
        # Convert to numpy array and save
        doc_emb = np.array(doc_emb)
        np.save(cur_cache_file, doc_emb)

    print("Shape of doc emb", doc_emb.shape)
    
    #nan_indices = np.argwhere(np.isnan(doc_emb))
    #print("NaNs found at total number indices:", len(nan_indices))
    
    # Encode queries 
    # Tokenize and encode queries in batches
    
    # Append instructions before queries 
    # queries = add_instruct_concatenate(texts=queries,task=task,instruction=instructions['query'])
    #print("Example query:", queries[0])

    query_emb = []
    for i in tqdm(range(0, len(queries))):
        text = queries[i]
        #inputs = tokenizer(f"query: {text}<|im_end|>", return_tensors='pt')  # max_length=4096
        #inputs = tokenizer(f"query: {text}<|endoftext|>", return_tensors='pt')
        inputs = tokenizer(f"query: {text}{tokenizer.eos_token}", return_tensors='pt')
        #inputs = tokenizer(f"{text}<|im_end|>", return_tensors='pt')
        inputs = {key: val.to(device) for key, val in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            
            # if i==0:
            #     print("Query outputs shape", outputs.last_hidden_state.shape)
            
            if trainedmodel_name == "qwen2.5_7b_instruct":
                last_hidden = outputs.hidden_states[-1]  # Last layer
                # Get last token representation
                embeddings = last_hidden.mean(dim=1)  #Mean pooling
                #embeddings = last_hidden[:, -1, :]  # Shape: (1, hidden_dim)

            else:
                embeddings = outputs.last_hidden_state[:, -1, :]  # Take the last hidden state
            
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)  # Normalize
            embeddings = embeddings.cpu().numpy()
            query_emb.extend(embeddings)
     
    # Convert to numpy array
    query_emb = np.array(query_emb)
    print("Query emb", query_emb)
    print("Shape of query emb", query_emb.shape)
    
    print("First doc embedding:", doc_emb[0,:])

    # Find cosine similarity between doc_emb and query_emb
    scores = cosine_similarity(query_emb, doc_emb)
    print("Scores shape", scores.shape)
    scores = scores.tolist()
    return get_scores(query_ids=query_ids,doc_ids=doc_ids,scores=scores,excluded_ids=excluded_ids)


@torch.no_grad()
def retrieval_distilroberta(queries,query_ids,documents,doc_ids,task,instructions,model_id,cache_dir,excluded_ids,long_context,**kwargs):
   
    model = SentenceTransformer('sentence-transformers/all-distilroberta-v1')
   
    batch_size = kwargs.get('batch_size',128)
    if not os.path.isdir(os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}")):
        os.makedirs(os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}"))
    cur_cache_file = os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}", f'0.npy')
    if os.path.isfile(cur_cache_file):
        doc_emb = np.load(cur_cache_file,allow_pickle=True)
    else:
        doc_emb = model.encode(documents, show_progress_bar=True, batch_size=batch_size, normalize_embeddings=True)
        np.save(cur_cache_file, doc_emb)
    
    print("Type of doc emb", type(doc_emb))
    print("Shape of doc emb", doc_emb.shape)
    query_emb = model.encode(queries,show_progress_bar=True,batch_size=batch_size, normalize_embeddings=True)
    print("Type of query emb", type(query_emb))
    print("Shape of query emb", query_emb.shape)
    scores = cosine_similarity(query_emb, doc_emb)
    print("Shape of scores", scores.shape)
    scores = scores.tolist()
    return get_scores(query_ids=query_ids,doc_ids=doc_ids,scores=scores,excluded_ids=excluded_ids)



@torch.no_grad()
def retrieval_sbert_bge(queries,query_ids,documents,doc_ids,task,instructions,model_id,cache_dir,excluded_ids,long_context,**kwargs):
    if model_id=='bge':
        model = SentenceTransformer('BAAI/bge-large-en-v1.5')
        queries = add_instruct_concatenate(texts=queries,task=task,instruction=instructions['query'])
        print("Query example:", queries[0])

    elif model_id=='sbert':
        model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    else:
        raise ValueError(f"The model {model_id} is not supported")
    batch_size = kwargs.get('batch_size',128)
    if not os.path.isdir(os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}")):
        os.makedirs(os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}"))
    cur_cache_file = os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}", f'0.npy')
    if os.path.isfile(cur_cache_file):
        doc_emb = np.load(cur_cache_file,allow_pickle=True)
    else:
        doc_emb = model.encode(documents, show_progress_bar=True, batch_size=batch_size, normalize_embeddings=True)
        np.save(cur_cache_file, doc_emb)
    
    print("Type of doc emb", type(doc_emb))
    print("Shape of doc emb", doc_emb.shape)
    query_emb = model.encode(queries,show_progress_bar=True,batch_size=batch_size, normalize_embeddings=True)
    print("Type of query emb", type(query_emb))
    print("Shape of query emb", query_emb.shape)
    scores = cosine_similarity(query_emb, doc_emb)
    print("Shape of scores", scores.shape)
    scores = scores.tolist()
    return get_scores(query_ids=query_ids,doc_ids=doc_ids,scores=scores,excluded_ids=excluded_ids)

@torch.no_grad()
def retrieval_dpr(queries,query_ids,documents,doc_ids,task,instructions,model_id,cache_dir,excluded_ids,long_context,**kwargs):
    passage_encoder = SentenceTransformer("facebook-dpr-ctx_encoder-single-nq-base")
    batch_size = kwargs.get('batch_size',128)
    if not os.path.isdir(os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}")):
        os.makedirs(os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}"))
    cur_cache_file = os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}", f'0.npy')
    if os.path.isfile(cur_cache_file):
        doc_emb = np.load(cur_cache_file,allow_pickle=True)
    else:
        doc_emb = passage_encoder.encode(documents, show_progress_bar=True, batch_size=batch_size, normalize_embeddings=True)
        np.save(cur_cache_file, doc_emb)
    
    print("Type of doc emb", type(doc_emb))
    print("Shape of doc emb", doc_emb.shape)
    query_encoder = SentenceTransformer("facebook-dpr-question_encoder-single-nq-base")
    
    query_emb = query_encoder.encode(queries,show_progress_bar=True,batch_size=batch_size, normalize_embeddings=True)
    print("Type of query emb", type(query_emb))
    print("Shape of query emb", query_emb.shape)
    scores = cosine_similarity(query_emb, doc_emb)
    print("Shape of scores", scores.shape)
    scores = scores.tolist()
    return get_scores(query_ids=query_ids,doc_ids=doc_ids,scores=scores,excluded_ids=excluded_ids)


@torch.no_grad()
def retrieval_contriever(queries,query_ids,documents,doc_ids,task,instructions,model_id,cache_dir,excluded_ids,long_context,**kwargs):
    # model = SentenceTransformer('nishimoto/contriever-sentencetransformer')
    # batch_size = kwargs.get('batch_size',128)
    # if not os.path.isdir(os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}")):
    #     os.makedirs(os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}"))
    # cur_cache_file = os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}", f'0.npy')
    # if os.path.isfile(cur_cache_file):
    #     doc_emb = np.load(cur_cache_file,allow_pickle=True)
    # else:
    #     doc_emb = model.encode(documents, show_progress_bar=True, batch_size=batch_size, normalize_embeddings=True)
    #     np.save(cur_cache_file, doc_emb)
    
    # print("Type of doc emb", type(doc_emb))
    # print("Shape of doc emb", doc_emb.shape)
    # query_emb = model.encode(queries,show_progress_bar=True,batch_size=batch_size, normalize_embeddings=True)
    # print("Type of query emb", type(query_emb))
    # print("Shape of query emb", query_emb.shape)
    # scores = cosine_similarity(query_emb, doc_emb)
    # print("Shape of scores", scores.shape)
    # scores = scores.tolist()
    # return get_scores(query_ids=query_ids,doc_ids=doc_ids,scores=scores,excluded_ids=excluded_ids)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')
    model = AutoModel.from_pretrained('facebook/contriever')
    # Move the model to the GPU
    model.to(device)
    
    # Append instructions before queries 
    queries = add_instruct_concatenate(texts=queries,task=task,instruction=instructions['query'])
    batch_size = kwargs.get('batch_size', 128)
    
    # Check if documents are already encoded 
    if not os.path.isdir(os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}")):
        os.makedirs(os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}"))
    
    cur_cache_file = os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}", f'0.npy')
    
    if os.path.isfile(cur_cache_file):
        doc_emb = np.load(cur_cache_file,allow_pickle=True)
    else:
        doc_emb = []
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i+batch_size]
            #print("Batch docs", batch_docs)

            # Tokenize the batch of documents
            encoded_input = tokenizer(batch_docs, padding=True, max_length=512, truncation=True, return_tensors='pt')
            
            # Move the tokenized inputs to the GPU
            encoded_input = {key: val.to(device) for key, val in encoded_input.items()}

            # Generate embeddings for the batch
            with torch.no_grad():
                outputs = model(**encoded_input)

            # Use the [CLS] token's hidden state as the embedding
            #embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

            # Use mean pooling of the final layer 
            embeddings = average_pooling_contriever(outputs[0], encoded_input['attention_mask']).cpu().numpy()

            # Append batch embeddings to the list
            doc_emb.extend(embeddings)

        # Convert to numpy array and save
        doc_emb = np.array(doc_emb)
        np.save(cur_cache_file, doc_emb)

    print("Shape of doc emb", doc_emb.shape)
    # Encode queries 
    # Tokenize and encode queries in batches
    query_emb = []
    for i in range(0, len(queries), batch_size):
        batch_queries = queries[i:i+batch_size]
        #print("Batch Queries:", batch_queries)

        # Tokenize the batch of queries
        encoded_input = tokenizer(batch_queries, max_length=512, padding=True, truncation=True, return_tensors='pt')

        # Move the tokenized inputs to the GPU
        encoded_input = {key: val.to(device) for key, val in encoded_input.items()}

        # Generate embeddings for the batch
        outputs = model(**encoded_input)

        # Use the [CLS] token's hidden state as the embedding
        #embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings = average_pooling_contriever(outputs[0], encoded_input['attention_mask']).cpu().numpy()

        # Append batch embeddings to the list
        query_emb.extend(embeddings)

    # Convert to numpy array
    query_emb = np.array(query_emb)
    print("Shape of query emb", query_emb.shape)

    # Find cosine similarity between doc_emb and query_emb
    scores = cosine_similarity(query_emb, doc_emb)
    print("Scores shape", scores.shape)
    scores = scores.tolist()
    return get_scores(query_ids=query_ids,doc_ids=doc_ids,scores=scores,excluded_ids=excluded_ids)


@torch.no_grad()
def retrieval_MATHBERTa(queries,query_ids,documents,doc_ids,task,instructions,model_id,cache_dir,excluded_ids,long_context,**kwargs):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained('witiko/mathberta')
    model = AutoModel.from_pretrained('witiko/mathberta')
    model.to(device)
    model = model.eval()
    max_length = kwargs.get('doc_max_length',512)
    queries = add_instruct_concatenate(texts=queries,task=task,instruction=instructions['query'])
    batch_size = kwargs.get('encode_batch_size',16)

    doc_emb = None
    cache_path = os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}.npy")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    if os.path.isfile(cache_path):
        # already exists so we can just load it
        doc_emb = np.load(cache_path, allow_pickle=True)
    
    for start_idx in trange(0,len(documents),batch_size):
        assert doc_emb is None or doc_emb.shape[0] % batch_size == 0, f"{doc_emb % batch_size} reminder in doc_emb"
        if doc_emb is not None and doc_emb.shape[0] // batch_size > start_idx:
            continue

        batch_dict = tokenizer(documents[start_idx:start_idx+batch_size], max_length=max_length, padding=True, truncation=True, return_tensors='pt').to(model.device)
        outputs = model(**batch_dict)
        #embeddings = outputs.pooler_output
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        #embeddings = mean_pooling(outputs.last_hidden_state, batch_dict['attention_mask']).cpu()
        # doc_emb[start_idx] = embeddings
        doc_emb = embeddings if doc_emb is None else np.concatenate((doc_emb, np.array(embeddings)), axis=0)

        # save the embeddings every 1000 iters, you can adjust this as needed
        if (start_idx + 1) % 1000 == 0:
            np.save(cache_path, doc_emb)
        
    np.save(cache_path, doc_emb)

    doc_emb = torch.tensor(doc_emb)
    print("doc_emb shape:",doc_emb.shape)
    doc_emb = F.normalize(doc_emb, p=2, dim=1)
    
    query_emb = []
    for start_idx in trange(0, len(queries), batch_size):
        batch_dict = tokenizer(queries[start_idx:start_idx + batch_size], max_length=max_length, padding=True,
                               truncation=True, return_tensors='pt').to(model.device)
        outputs = model(**batch_dict)
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().tolist()
        #embeddings = mean_pooling(outputs.last_hidden_state, batch_dict['attention_mask']).cpu().tolist()
        query_emb += embeddings

    query_emb = torch.tensor(query_emb)
    print("query_emb shape:", query_emb.shape)
    
    query_emb = F.normalize(query_emb, p=2, dim=1)
    scores = (query_emb @ doc_emb.T) * 100
    scores = scores.tolist()
    return get_scores(query_ids=query_ids,doc_ids=doc_ids,scores=scores,excluded_ids=excluded_ids)


    

@torch.no_grad()
def retrieval_arxivBERTmath(queries,query_ids,documents,doc_ids,task,instructions,model_id,cache_dir,excluded_ids,long_context,**kwargs):
    model = SentenceTransformer('math-similarity/Bert-MLM_arXiv-MP-class_arXiv')
    #queries = add_instruct_concatenate(texts=queries,task=task,instruction=instructions['query'])
    batch_size = kwargs.get('batch_size',128)
    
    # queries = add_instruct_list(texts=queries,task=task,instruction=instructions['query'])
    # documents = add_instruct_list(texts=documents,task=task,instruction=instructions['document'])

    query_embs = model.encode(queries,batch_size=batch_size,show_progress_bar=True,normalize_embeddings=True)
    if not os.path.isdir(os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}")):
        os.makedirs(os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}"))
    cur_cache_file = os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}", f'0.npy')
    if os.path.isfile(cur_cache_file):
        doc_embs = np.load(cur_cache_file,allow_pickle=True)
    else:
        doc_embs = model.encode(documents, show_progress_bar=True, batch_size=batch_size, normalize_embeddings=True)
        np.save(cur_cache_file, doc_embs)

    print(doc_embs.shape)    
    scores = cosine_similarity(query_embs, doc_embs)
    scores = scores.tolist()
    return get_scores(query_ids=query_ids,doc_ids=doc_ids,scores=scores,excluded_ids=excluded_ids)


@torch.no_grad()
def retrieval_instructor(queries,query_ids,documents,doc_ids,task,instructions,model_id,cache_dir,excluded_ids,long_context,**kwargs):
    if model_id=='inst-l':
        model = SentenceTransformer('hkunlp/instructor-large')
    elif model_id=='inst-xl':
        model = SentenceTransformer('hkunlp/instructor-xl')
    else:
        raise ValueError(f"The model {model_id} is not supported")
    model.set_pooling_include_prompt(False)

    batch_size = kwargs.get('batch_size',4)
    model.max_seq_length = kwargs.get('doc_max_length',2048)
    # queries = add_instruct_list(texts=queries,task=task,instruction=instructions['query'])
    # documents = add_instruct_list(texts=documents,task=task,instruction=instructions['document'])

    query_embs = model.encode(queries,batch_size=batch_size,show_progress_bar=True,prompt=instructions['query'].format(task=task),normalize_embeddings=True)
    if not os.path.isdir(os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}")):
        os.makedirs(os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}"))
    cur_cache_file = os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}", f'0.npy')
    if os.path.isfile(cur_cache_file):
        doc_embs = np.load(cur_cache_file,allow_pickle=True)
    else:
        doc_embs = model.encode(documents, show_progress_bar=True, batch_size=batch_size, normalize_embeddings=True,prompt=instructions['document'].format(task=task))
        np.save(cur_cache_file, doc_embs)

    print(doc_embs.shape)    
    scores = cosine_similarity(query_embs, doc_embs)
    scores = scores.tolist()
    return get_scores(query_ids=query_ids,doc_ids=doc_ids,scores=scores,excluded_ids=excluded_ids)


@torch.no_grad()
def retrieval_grit(queries,query_ids,documents,doc_ids,task,instructions,model_id,cache_dir,excluded_ids,long_context,**kwargs):
    customized_checkpoint = kwargs.get('checkpoint',None)
    if customized_checkpoint is None:
        customized_checkpoint = 'GritLM/GritLM-7B'
    else:
        print('use',customized_checkpoint)
    model = GritLM(customized_checkpoint, torch_dtype="auto", mode="embedding")
    query_instruction = instructions['query'].format(task=task)
    doc_instruction = instructions['document']
    query_max_length = kwargs.get('query_max_length',256)
    doc_max_length = kwargs.get('doc_max_length',2048)
    print("doc max length:",doc_max_length)
    print("query max length:", query_max_length)
    batch_size = kwargs.get('batch_size',1)
    if not os.path.isdir(os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}")):
        os.makedirs(os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}"))
    cur_cache_file = os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}", f'0.npy')
    ignore_cache = kwargs.pop('ignore_cache',False)
    if os.path.isfile(cur_cache_file):
        doc_emb = np.load(cur_cache_file, allow_pickle=True)
    elif ignore_cache:
        doc_emb = model.encode(documents, instruction=doc_instruction, batch_size=1, max_length=doc_max_length)
    else:
        doc_emb = model.encode(documents, instruction=doc_instruction, batch_size=1, max_length=doc_max_length)
        np.save(cur_cache_file, doc_emb)
    query_emb = model.encode(queries, instruction=query_instruction, batch_size=1, max_length=query_max_length)
    scores = pairwise_cosine_similarity(torch.from_numpy(query_emb), torch.from_numpy(doc_emb))
    scores = scores.tolist()
    assert len(scores) == len(query_ids), f"{len(scores)}, {len(query_ids)}"
    assert len(scores[0]) == len(documents), f"{len(scores[0])}, {len(documents)}"
    return get_scores(query_ids=query_ids,doc_ids=doc_ids,scores=scores,excluded_ids=excluded_ids)


def retrieval_openai(queries,query_ids,documents,doc_ids,task,model_id,cache_dir,excluded_ids,long_context,**kwargs):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    new_queries = []
    for q in queries:
        new_queries.append(cut_text_openai(text=q,tokenizer=tokenizer))
    queries = new_queries
    new_documents = []
    for d in documents:
        new_documents.append(cut_text_openai(text=d,tokenizer=tokenizer))
    documents = new_documents
    doc_emb = []
    batch_size = kwargs.get('batch_size',1024)
    # openai_client = OpenAI(api_key=kwargs['key'])
    openai_client = OpenAI()
    if not os.path.isdir(os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}")):
        os.makedirs(os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}"))
    for idx in trange(0,len(documents),batch_size):
        cur_cache_file = os.path.join(cache_dir,'doc_emb',model_id,task,f"long_{long_context}_{batch_size}",f'{idx}.json')
        if os.path.isfile(cur_cache_file):
            with open(cur_cache_file) as f:
                cur_emb = json.load(f)
        else:
            cur_emb = get_embedding_openai(texts=documents[idx:idx + batch_size],openai_client=openai_client,tokenizer=tokenizer)
            with open(cur_cache_file,'w') as f:
                json.dump(cur_emb,f,indent=2)
        doc_emb += cur_emb
    query_emb = []
    for idx in trange(0, len(queries), batch_size):
        cur_emb = get_embedding_openai(texts=queries[idx:idx + batch_size], openai_client=openai_client,
                                       tokenizer=tokenizer)
        query_emb += cur_emb
    scores = pairwise_cosine_similarity(torch.tensor(query_emb), torch.tensor(doc_emb))
    scores = scores.tolist()
    return get_scores(query_ids=query_ids,doc_ids=doc_ids,scores=scores,excluded_ids=excluded_ids)


def retrieval_cohere(queries,query_ids,documents,doc_ids,task,model_id,cache_dir,excluded_ids,long_context,**kwargs):
    query_emb = []
    doc_emb = []
    batch_size = kwargs.get('batch_size',8192)
    # cohere_client = cohere.Client(kwargs['key'])
    cohere_client = cohere.Client()
    if not os.path.isdir(os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}")):
        os.makedirs(os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}"))
    for idx in trange(0,len(documents),batch_size):
        cur_cache_file = os.path.join(cache_dir,'doc_emb',model_id,task,f"long_{long_context}_{batch_size}",f'{idx}.json')
        if os.path.isfile(cur_cache_file):
            with open(cur_cache_file) as f:
                cur_emb = json.load(f)
        else:
            success = False
            exec_count = 0
            cur_emb = []
            while not success:
                exec_count += 1
                if exec_count>5:
                    print('cohere execute too many times')
                    exit(0)
                try:
                    cur_emb = cohere_client.embed(texts=documents[idx:idx+batch_size], input_type="search_document",
                                                  model="embed-english-v3.0").embeddings

                    success = True
                except Exception as e:
                    print(e)
                    time.sleep(60)
            with open(cur_cache_file, 'w') as f:
                json.dump(cur_emb, f, indent=2)
        doc_emb += cur_emb
    for idx in trange(0, len(queries), batch_size):
        success = False
        exec_count = 0
        while not success:
            exec_count += 1
            if exec_count > 5:
                print('cohere query execute too many times')
                exit(0)
            try:
                cur_emb = cohere_client.embed(queries[idx:idx+batch_size], input_type="search_query",
                                              model="embed-english-v3.0").embeddings
                query_emb += cur_emb
                success = True
            except Exception as e:
                print(e)
                time.sleep(60)
    scores = (torch.tensor(query_emb) @ torch.tensor(doc_emb).T) * 100
    scores = scores.tolist()
    return get_scores(query_ids=query_ids,doc_ids=doc_ids,scores=scores,excluded_ids=excluded_ids)


def retrieval_voyage(queries,query_ids,documents,doc_ids,task,model_id,cache_dir,excluded_ids,long_context,**kwargs):
    tokenizer = AutoTokenizer.from_pretrained('voyageai/voyage')
    new_queries = []
    for q in queries:
        new_queries.append(cut_text(text=q,tokenizer=tokenizer,threshold=16000))
    queries = new_queries
    new_documents = []
    for d in tqdm(documents,desc='preprocess documents'):
        new_documents.append(cut_text(text=d,tokenizer=tokenizer,threshold=16000))
    documents = new_documents

    query_emb = []
    doc_emb = []

    batch_size = kwargs.get('batch_size',1)

    doc_emb = None
    doc_cache_path = os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}.npy")
    os.makedirs(os.path.dirname(doc_cache_path), exist_ok=True)
    if os.path.isfile(doc_cache_path):
        # already exists so we can just load it
        doc_emb = np.load(doc_cache_path, allow_pickle=True)

    # voyage_client = voyageai.Client(api_key=kwargs['key'])
    voyage_client = voyageai.Client()
    for i in trange(0,len(documents),batch_size):
        assert doc_emb is None or doc_emb.shape[0] % batch_size == 0, f"{doc_emb % batch_size} reminder in doc_emb"
        if doc_emb is not None and doc_emb.shape[0] // batch_size > i:
            continue
        
        success = False
        threshold = 16000
        cur_texts = documents[i:i+batch_size]
        count_over = 0
        exec_count = 0
        while not success:
            exec_count += 1
            if exec_count > 5:
                print('voyage document too many times')
                exit(0)
            try:
                cur_emb = voyage_client.embed(cur_texts, model="voyage-large-2-instruct", input_type="document").embeddings
                doc_emb = cur_emb if doc_emb is None else np.concatenate((doc_emb, np.array(cur_emb)), axis=0)
                if (i + 1) % 1000 == 0:
                    np.save(doc_cache_path, doc_emb)
                success = True
            except Exception as e:
                print(e)
                count_over += 1
                threshold = threshold-500
                if count_over>4:
                    print('voyage:',count_over)
                new_texts = []
                for t in cur_texts:
                    new_texts.append(cut_text(text=t,tokenizer=tokenizer,threshold=threshold))
                cur_texts = new_texts
                time.sleep(5)

    query_emb = []
    for i in trange(0,len(queries),batch_size):
        success = False
        threshold = 16000
        cur_texts = queries[i:i+batch_size]
        count_over = 0
        exec_count = 0
        while not success:
            exec_count += 1
            if exec_count > 5:
                print('voyage query execute too many times')
                exit(0)
            try:
                cur_emb = voyage_client.embed(cur_texts, model="voyage-large-2-instruct", input_type="query").embeddings
                query_emb += cur_emb
                success = True
            except Exception as e:
                print(e)
                count_over += 1
                threshold = threshold-500
                if count_over>4:
                    print('voyage:',count_over)
                new_texts = []
                for t in cur_texts:
                    new_texts.append(cut_text(text=t,tokenizer=tokenizer,threshold=threshold))
                cur_texts = new_texts
                time.sleep(60)
    scores = pairwise_cosine_similarity(torch.tensor(query_emb), torch.tensor(doc_emb))
    scores = scores.tolist()
    return get_scores(query_ids=query_ids,doc_ids=doc_ids,scores=scores,excluded_ids=excluded_ids)


def retrieval_google(queries,query_ids,documents,doc_ids,task,model_id,cache_dir,excluded_ids,long_context,**kwargs):
    model = TextEmbeddingModel.from_pretrained("text-embedding-preview-0409")
    query_emb = []
    # doc_emb = []
    batch_size = kwargs.get('batch_size',8)
    doc_emb = None
    cache_path = os.path.join(cache_dir, 'doc_emb', model_id, task, f"long_{long_context}_{batch_size}.npy")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    if os.path.isfile(cache_path):
        # already exists so we can just load it
        doc_emb = np.load(cache_path, allow_pickle=True)

    for start_idx in tqdm(range(0, len(documents), batch_size), desc='embedding'):
        assert doc_emb is None or doc_emb.shape[0] % batch_size == 0, f"{doc_emb % batch_size} reminder in doc_emb"
        if doc_emb is not None and doc_emb.shape[0] // batch_size > start_idx:
            continue
        
        cur_emb = get_embedding_google(
            texts=documents[start_idx:start_idx + batch_size], task='RETRIEVAL_DOCUMENT',
            model=model
        )
        doc_emb = cur_emb if doc_emb is None else np.concatenate((doc_emb, np.array(cur_emb)), axis=0)
        if (start_idx + 1) % 1000 == 0:
            np.save(cache_path, doc_emb)
    np.save(cache_path, doc_emb)
        
    for start_idx in tqdm(range(0,len(queries), batch_size),desc='embedding'):
        query_emb += get_embedding_google(texts=queries[start_idx:start_idx+ batch_size],task='RETRIEVAL_QUERY',model=model)
    scores = pairwise_cosine_similarity(torch.tensor(query_emb), torch.tensor(doc_emb))
    scores = scores.tolist()
    return get_scores(query_ids=query_ids,doc_ids=doc_ids,scores=scores,excluded_ids=excluded_ids)


RETRIEVAL_FUNCS = {
    'sf': retrieval_sf_qwen_e5,
    'qwen': retrieval_sf_qwen_e5,
    'qwen2': retrieval_sf_qwen_e5,
    'e5': retrieval_sf_qwen_e5,
    'bm25': retrieval_bm25,
    'sbert': retrieval_sbert_bge,
    'bge': retrieval_sbert_bge,
    'inst-l': retrieval_instructor,
    'inst-xl': retrieval_instructor,
    'grit': retrieval_grit,
    'cohere': retrieval_cohere,
    'voyage': retrieval_voyage,
    'openai': retrieval_openai,
    'google': retrieval_google,
    'mathbert': retrieval_mathbert,
    'arxivBERTmath': retrieval_arxivBERTmath,
    'MATHBERTa': retrieval_MATHBERTa,
    'contriever': retrieval_contriever,
    'dpr': retrieval_dpr,
    'distilroberta': retrieval_distilroberta,
    'repllama': retrieval_repllama,
    'repllama_doc': retrieval_repllama_doc,
    'trained_model': retrieval_trainedmodel

}

def calculate_retrieval_metrics(results, qrels, output_dir, k_values=[1, 5, 10, 25, 50, 100]):
    # https://github.com/beir-cellar/beir/blob/f062f038c4bfd19a8ca942a9910b1e0d218759d4/beir/retrieval/evaluation.py#L66
    # follow evaluation from BEIR, which is just using the trec eval
    ndcg = {}
    _map = {}
    recall = {}
    precision = {}
    mrr = {"MRR": 0}

    for k in k_values:
        ndcg[f"NDCG@{k}"] = 0.0
        _map[f"MAP@{k}"] = 0.0
        recall[f"Recall@{k}"] = 0.0
        precision[f"P@{k}"] = 0.0

    map_string = "map_cut." + ",".join([str(k) for k in k_values])
    ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
    recall_string = "recall." + ",".join([str(k) for k in k_values])
    precision_string = "P." + ",".join([str(k) for k in k_values])

    # https://github.com/cvangysel/pytrec_eval/blob/master/examples/simple_cut.py
    # qrels = {qid: {'pid': [0/1] (relevance label)}}
    # results = {qid: {'pid': float (retriever score)}}
    evaluator = pytrec_eval.RelevanceEvaluator(qrels,
                                               {map_string, ndcg_string, recall_string, precision_string, "recip_rank"})
    scores = evaluator.evaluate(results)

    
    with open(os.path.join(output_dir,"query_scores"), "w") as f:
        json.dump(scores, f, indent=4)
   
    print("Query ids with NDCG@10 = 0")
    print_high_ndcg_queries(scores, os.path.join(output_dir,"matched_queries.txt"), threshold=0)

    for query_id in scores.keys():
        for k in k_values:
            ndcg[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
            _map[f"MAP@{k}"] += scores[query_id]["map_cut_" + str(k)]
            recall[f"Recall@{k}"] += scores[query_id]["recall_" + str(k)]
            precision[f"P@{k}"] += scores[query_id]["P_" + str(k)]
        mrr["MRR"] += scores[query_id]["recip_rank"]

    for k in k_values:
        ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"] / len(scores), 5)
        _map[f"MAP@{k}"] = round(_map[f"MAP@{k}"] / len(scores), 5)
        recall[f"Recall@{k}"] = round(recall[f"Recall@{k}"] / len(scores), 5)
        precision[f"P@{k}"] = round(precision[f"P@{k}"] / len(scores), 5)
    mrr["MRR"] = round(mrr["MRR"] / len(scores), 5)

    output = {**ndcg, **_map, **recall, **precision, **mrr}
    print(output)
    return output

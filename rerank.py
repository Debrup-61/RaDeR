import copy
import os
import re
import time
import json
from tqdm import tqdm
import argparse
from datasets import load_dataset
import torch
from huggingface_hub import login
from retrievers import calculate_retrieval_metrics
import functools
import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel, PeftConfig
from accelerate import Accelerator
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, TaskType, get_peft_model

def get_model_new(base_model_path, lora_model_path):
    
    base_model = AutoModelForSequenceClassification.from_pretrained(base_model_path, num_labels=1, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    lora_config = LoraConfig.from_pretrained(lora_model_path)
    lora_model = PeftModel.from_pretrained(base_model, lora_model_path, config=lora_config)
    lora_model = lora_model.merge_and_unload()
    print("Lora Model loaded!")
    print(lora_model)
    return lora_model


def get_model(peft_model_name, tokenizer):


        
        # Check for GPU availability
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #print(f"Using device: {device}")
        
        config = PeftConfig.from_pretrained(peft_model_name)
        base_model = AutoModelForSequenceClassification.from_pretrained(config.base_model_name_or_path, num_labels=1, device_map="auto" if torch.cuda.is_available() else None)  #device_map="auto" if torch.cuda.is_available() else None
        #base_model.to('cuda:0')
        model = PeftModel.from_pretrained(base_model, peft_model_name)
        model = model.merge_and_unload()
        

        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id
        
        model.eval()
        print("Model device:", model.device)
        #model.to(device)  # Move model to device
        return model


def extract_program(a_string,lan='python',first_block_only=False):
    indices_object = re.finditer(pattern="```", string=a_string)
    indices = [index.start() for index in indices_object]
    contents = ''
    if len(indices) == 0:
        contents = a_string
    elif len(indices) % 2 == 0:
        for i in range(0, len(indices), 2):
            cur_str = a_string[indices[i]:indices[i + 1]]
            if cur_str.startswith(f"```{lan}"):
                cur_str = cur_str[len(f"```{lan}"):]
            elif cur_str.startswith(f"```\n{lan}"):
                cur_str = cur_str[len(f"```\n{lan}"):]
            elif cur_str.startswith("```"):
                cur_str = cur_str[len("```"):]
            contents += cur_str
            if first_block_only:
                break
    else:
        contents = a_string.replace(f"```{lan}", '').replace("```", '').replace(f"{lan}\n", '')
    lines = contents.strip().split('\n')
    if lines[-1].isidentifier():
        contents = '\n'.join(lines[:-1])
    return contents.replace(f"{lan}\n", '')


def call_api(func):
    count = 0
    while True:
        try:
            count += 1
            output = func()
            break
        except Exception as e:
            logger.info(f"Exception while using api: {e}")
            if "rate limit" in str(e).lower() or "rate_limit" in str(e).lower():
                logger.info("Rate limit exceeded, waiting 10 secs and retrying...")
                time.sleep(10)
            elif count < 5:
                logger.info("Encountered error, retrying...")
                time.sleep(5)
            else:
                logger.info("Skipping generation due to unknown error after 5 retries.")
                output = None
                break
    return output


def format_chat(message, include_system=True, system_message="You are a helpful assistant."):
    if include_system:
        chat = [{"role": "system", "content": system_message}, {"role": "user", "content": message}]
    else:
        chat = [{"role": "user", "content": message}]
    return chat


class ClaudeModel:

    def __init__(self, version):
        from anthropic import AnthropicVertex
        PROJECT_ID = "xxx"  # @param
        LOCATION = "xxx"  # @param
        self.model = AnthropicVertex(region=LOCATION, project_id=PROJECT_ID)
        self.version = version

    def rerank(self, docs, query, topk):
        doc_string = ''
        indices_map = {}
        for doc_idx,doc in enumerate(docs):
            assert isinstance(doc,list)
            doc_string += "[{}]. {}\n\n".format(doc_idx + 1, re.sub('\n+', ' ', doc[1]))
            indices_map[doc_idx + 1] = doc[0]
        cur_query = query.replace('\n','  ')
        prompt = (f'The following passages are related to query: {cur_query}\n\n'
                  f'{doc_string}'
                  f'First identify the essential problem in the query.\n'
                  f'Think step by step to reason about why each document is relevant or irrelevant.\n'
                  f'Rank these passages based on their relevance to the query.\n'
                  f'Please output the ranking result of passages as a list, where the first element is the id of the most relevant '
                  f'passage, the second element is the id of the second most element, etc.\n'
                  f'Please strictly follow the format to output a list of {topk} ids corresponding to the most relevant {topk} passages:\n'
                  f'```json\n'
                  f'[...]\n'
                  f'```')
        func = functools.partial(
            self.model.messages.create,
            max_tokens=2048,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=self.version,
            temperature=0.8,
            top_p=0.8
        )
        message = call_api(func)
        response = json.loads(message.model_dump_json(indent=2))
        ranks = extract_program(response['content'][0]['text'],lan='json')
        return [indices_map[r] for r in ranks]


class Reranker_models:
   
    def __init__(self, base_model_path, lora_model_path, batch_size=1):
        
        #self.accelerator = Accelerator()    # Initialize accelerator for multi-GPU support

        # Load the tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        
        print("Pad token id:", tokenizer.pad_token_id) 
        print("Unknown token id:", tokenizer.unk_token_id)
        self.tokenizer = tokenizer
        #print("Pad token ID:", self.tokenizer.pad_token_id)
        #print("Unknown token ID:", self.tokenizer.unk_token_id)
 
        self.model = get_model_new(base_model_path, lora_model_path)
        self.model.to("cuda")
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for inference.")
            self.model = nn.DataParallel(self.model)

        
        # if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        #     self.model = torch.nn.DataParallel(self.model, device_ids=[0, 1])
        
        #self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.batch_size = batch_size
        

    @torch.no_grad()
    def rerank(self, docs, query, topk):
        #inputs = [(query, doc["text"]) for doc in docs]
        #scores = self.model.predict(inputs, batch_size=self.batch_size)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #inputs_list = [f"query: {query} document: {doc['text']}" for doc in docs]
        eos_token = self.tokenizer.eos_token
        inputs_list = [f"query: {query} document: {doc['text']}{eos_token}" for doc in docs]
        
        scores = []
        for inp in inputs_list:
            inputs = self.tokenizer(inp, return_tensors='pt')
            inputs = {key: val.to(device) for key, val in inputs.items()}
            with torch.no_grad():
                #print("Input ids shape:", inputs["input_ids"].shape)
                outputs = self.model(**inputs)
                logits = outputs.logits
                #print("Logits shape", logits.shape)
                score = logits[0][0]
                scores.append(score) 
        
        ranking = {doc["id"]: score.item() for doc, score in zip(docs, scores)}
        ranking = dict(sorted(ranking.items(), key=lambda item: item[1], reverse=True)[:topk])
        return ranking    




class OpenAIModel:
    def __init__(self, model_name, temperature=0.8, top_p=0.8):
        import openai
        if "azure" in model_name:
            # env var: AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, and OPENAI_API_VERSION
            self.model = openai.AzureOpenAI()
            model_name = model_name[model_name.index("/")+1:]
        else:
            # make sure to set the OPENAI_API_KEY environment variable
            self.model = openai.OpenAI()
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = 2048

    def rerank(self, docs, query, topk):
        doc_string = ''
        indices_map = {}
        for doc_idx,doc in enumerate(docs):
            assert isinstance(doc,list)
            doc_string += "[{}]. {}\n\n".format(doc_idx + 1, re.sub('\n+', ' ', doc[1]))
            indices_map[doc_idx + 1] = doc[0]
        cur_query = query.replace('\n','  ')
        prompt = (f'The following passages are related to query: {cur_query}\n\n'
                  f'{doc_string}'
                  f'First identify the essential problem in the query.\n'
                  f'Think step by step to reason about why each document is relevant or irrelevant.\n'
                  f'Rank these passages based on their relevance to the query.\n'
                  f'Please output the ranking result of passages as a list, where the first element is the id of the most relevant '
                  f'passage, the second element is the id of the second most element, etc.\n'
                  f'Please strictly follow the format to output a list of {topk} ids corresponding to the most relevant {topk} passages, sorted from the most to least relevant passage. First think step by step and write the reasoning process, then output the ranking results as a list of ids in a json format.'
                  )
        inputs = format_chat(prompt, system_message="You are a helpful assistant")
        func = functools.partial(
            self.model.chat.completions.create,
            model=self.model_name,
            messages=inputs,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )

        def parse_json(text):
            matches = re.findall(r"(?:```json\s*)(.+)(?:```)", text, re.DOTALL)
            if len(matches) > 0:
                try:
                    return json.loads(matches[-1].strip())
                except:
                    return None
            return None

        output = call_api(func)
        if output is not None:
            response = parse_json(output.choices[0].message.content)
            if response is None:
                return None
            return [indices_map[r] for r in response if r in indices_map]
            # return output.choices[0].message.content
        return None


class STReranker:
    def __init__(self, model_name, batch_size=8):
        self.model = CrossEncoder(model_name)
        self.batch_size = batch_size

    @torch.no_grad()
    def rerank(self, docs, query, topk):
        inputs = [(query, doc["text"]) for doc in docs]
        scores = self.model.predict(inputs, batch_size=self.batch_size)
        ranking = {doc["id"]: score.item() for doc, score in zip(docs, scores)}
        ranking = dict(sorted(ranking.items(), key=lambda item: item[1], reverse=True)[:topk])
        return ranking


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True,
                        choices=['biology','earth_science','economics','pony','psychology','robotics','theoremqa_questions', "theoremqa_theorems",
                                 'stackoverflow','sustainable_living','aops','leetcode'])
    parser.add_argument('--long_context', action='store_true')
    parser.add_argument('--llm', type=str, default=None)
    parser.add_argument('--base_model_path', type=str, default=None)
    parser.add_argument('--lora_model_path', type=str, default=None)
    parser.add_argument('--score_file', type=str, default=None)
    parser.add_argument('--rerank_score_file', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--input_k', type=int)
    parser.add_argument('--k', type=int)
    args = parser.parse_args()

    

    if os.path.exists(args.rerank_score_file):
        print(f"Rerank score file {args.rerank_score_file} already exists.")
        exit()

    raw_examples = load_dataset('xlangai/bright', "examples")[args.task]
    examples = {}
    for e in raw_examples:
        examples[e['id']] = e
    if args.long_context:
        doc_pairs = load_dataset('xlangai/bright', 'long_documents')[args.task]
    else:
        doc_pairs = load_dataset('xlangai/bright', 'documents')[args.task]
    documents = {}
    for d in doc_pairs:
        documents[d['id']] = d['content']
    
    with open(args.score_file) as f:
        all_scores = json.load(f)
    
    new_scores = copy.deepcopy(all_scores)
    
    model = Reranker_models(args.base_model_path, args.lora_model_path, args.batch_size)

    for qid,scores in tqdm(all_scores.items()):
        docs = []
        sorted_scores = sorted(scores.items(),key=lambda x:x[1],reverse=True)[:args.input_k]
        for did, _ in sorted_scores:
            docs.append([did, documents[did]])

        
        ctxs = [{'id': did, 'text': documents[did]} for did, _ in sorted_scores]
        
        if qid not in examples:  
            print("Query id not part of BRIGHT!")
            continue
        
        cur_score = model.rerank(query=examples[qid]['query'], docs=ctxs, topk=args.k)
        new_scores[qid] = cur_score

    os.makedirs(os.path.dirname(args.rerank_score_file), exist_ok=True)
    with open(args.rerank_score_file, 'w') as f:
        json.dump(new_scores, f, indent=2)

    if args.long_context:
        key = 'gold_ids_long'
    else:
        key = 'gold_ids'
    ground_truth = {}
    for e in raw_examples:
        ground_truth[e['id']] = {}
        for gid in e[key]:
            ground_truth[e['id']][gid] = 1
        for i in e["excluded_ids"]:
            if i in documents:
                ground_truth[e['id']][i] = 0

    results = calculate_retrieval_metrics(results=new_scores, qrels=ground_truth, output_dir="outputs")
    with open(args.rerank_score_file.replace(".json", "_results.json"), 'w') as f:
        json.dump(results, f, indent=2)

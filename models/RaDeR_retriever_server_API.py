# Licensed under the MIT license.

import sys
sys.path.append(".")

import os
import time
from tqdm import tqdm
import concurrent.futures
import openai
import numpy as np
# import auto_tokenizer
from transformers import AutoTokenizer
import torch 
from tqdm import tqdm
from datasets import load_dataset
import numpy as np 
from huggingface_hub import login
from pylatexenc.latex2text import LatexNodes2Text
import re
import unicodedata

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(SCRIPT_DIR, "../.env")

from dotenv import load_dotenv
load_dotenv(ENV_PATH)


ret_client = openai.OpenAI(
    api_key="abc",
    base_url=os.environ.get("TRAINED_RETRIEVER_ENDPOINT", "http://localhost:8001/v1"),
)
#print("Retrieval client", ret_client)
#print(os.environ["RaDeR_MERGED_HUGGINGFACE_PATH"])


# Huggingface path to trained retriever model
model_hf_path = os.environ.get("RaDeR_MERGED_HUGGINGFACE_PATH", "RaDeR/merged_retriever_Qwen-2.5-7B-Instruct_MATH_questionpartialsol_and_LLMquery_full")
#print("RaDeR huggingface path", model_hf_path)

# Initialize tokenizer 
tokenizer = AutoTokenizer.from_pretrained(
    model_hf_path,
    use_fast=True,
    cache_dir=os.environ.get("HUGGINGFACE_CACHE_DIR","")
)

eos_token = tokenizer.eos_token


'''
Returns list of embeddings for the given input prompt. Should only pass in one string at a time, for multiple use batched.
'''
def generate_with_qwen(
    input_str: str,
    model_hf_path: str,
    eos_token: str,
    tokenizer
):
    
    ans, timeout = "", 5
    flag = 0
    while True:
        try:
            if not input_str :
                print("Input is False or invalid:", input_str)

            response = ret_client.embeddings.create(input=[input_str], model=model_hf_path)
            #print(response.data[0])
            #break
            ans = response.data[0].embedding
            num_nans = np.isnan(ans).sum()
            
            if num_nans > 0 :
                # print("String producing NaNs", input_str)
                # print("NaNs found for input!")
                # print("Number of NaNs", num_nans)
                # print("Initial embedding containing NaNs:", ans)
                
                if flag>=0:
                    if len(input_str) > 100:
                        input_str = input_str[:100]  # Truncate to first hundred characters
                        flag+=1
                        continue
                    else:
                        print("Zero embedding returned")
                        #return np.zeros((1,4096)) # For Llama hidden dim is 4096 
                        return np.zeros((1,3584))  # For Qwen-2.5-7B-Instruct hidden dim is 4096 
            break

        except openai.BadRequestError as e:
            print("Error with input:", type(input_str), input_str)
            print("Too long, concatenating...")
            token_ids = tokenizer.encode(input_str, max_length=3900, truncation=True)
            input_str = tokenizer.decode(token_ids) + str(eos_token)
            #print("Truncated input:", input_str)
            continue
        
        except Exception as e:
            print(e)
        
        if not ans:
            print(f"Will retry after {timeout} seconds ...")
            time.sleep(timeout)

            timeout = timeout * 2
            if timeout > 120:
                timeout = 1
    
    return np.array(ans, dtype=float).reshape(1, 3584)   # For Qwen-2.5-7B-Instruct hidden dim is 4096 
    #return np.array(ans, dtype=float).reshape(1, 4096)  # For Llama hidden dim is 4096 
   

def generate_with_qwen_batched(
    tokenizer,
    inputs: list,
    max_threads: int = 32,
    model_hf_path: str = "RaDeR/merged_retriever_Qwen-2.5-7B-Instruct_MATH_questionpartialsol_and_LLMquery_full",
    eos_token: str = "<|im_end|>",
    
):
    results = [None] * len(inputs)
    with concurrent.futures.ThreadPoolExecutor(max_threads) as executor:
        futures = {}
        for i, prompt in enumerate(inputs):
            futures[executor.submit(generate_with_qwen, prompt, model_hf_path, eos_token, tokenizer)] = i
        
        # make sure embeddings are stored in order, but it loads as they are completed
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(inputs)):
            results[futures[future]] = future.result()

    return np.vstack(results, dtype=float)

if __name__ == "__main__":
   print("")
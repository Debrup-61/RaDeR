# Licensed under the MIT license.

import sys
sys.path.append(".")
import os
import os
import time
from tqdm import tqdm
import concurrent.futures
import openai
import numpy as np
# import auto_tokenizer
from transformers import AutoTokenizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(SCRIPT_DIR, "../.env")

from dotenv import load_dotenv
load_dotenv(ENV_PATH)


client = openai.OpenAI(
    api_key="abc",
    base_url=os.environ.get("REPLLAMA_ENDPOINT", "http://localhost:8011/v1"),
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_fast=True, cache_dir=os.environ.get("HUGGINGFACE_CACHE_DIR",""))
eos_token = tokenizer.eos_token

'''
Returns list of embeddings for the given input prompt. Should only pass in one string at a time, for multiple use batch.
'''
def generate_with_repllama(
    input_str: str,
):

    ans, timeout = "", 5
   
    while not ans:
        try:
            response = client.embeddings.create(input=[input_str], model=os.environ.get("REPLLAMA_MERGED_HUGGINGFACE_PATH",""))
            ans = response.data[0].embedding

        except openai.BadRequestError as e:
            print("Error with input:", type(input_str), input_str)
            print("Too long, concatenating...")
            
            # If error code is 400, truncate the input to work properly
            # token_ids = tokenizer.encode(input, max_length=4095)
            # input = tokenizer.decode(token_ids)
            
            token_ids = tokenizer.encode(input_str, max_length=4090, truncation=True)
            input_str = tokenizer.decode(token_ids) + str(eos_token)
            print("Truncated input:", input_str)
            continue
        
        except Exception as e:
            print(e)
        
        if not ans:
            print(f"Will retry after {timeout} seconds ...")
            time.sleep(timeout)

            timeout = timeout * 2
            if timeout > 120:
                timeout = 1
    
    return np.array(ans, dtype=float).reshape(1, 4096)



def generate_with_repllama_batched(
    inputs: list,
    max_threads: int = 32,
):
    results = [None] * len(inputs)
    with concurrent.futures.ThreadPoolExecutor(max_threads) as executor:
        futures = {}
        for i, prompt in enumerate(inputs):
            futures[executor.submit(generate_with_repllama, prompt)] = i
        
        # make sure embeddings are stored in order, but it loads as they are completed
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(inputs)):
            results[futures[future]] = future.result()

    return np.vstack(results, dtype=float)

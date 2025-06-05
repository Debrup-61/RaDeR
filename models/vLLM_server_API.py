# Licensed under the MIT license.
import sys
sys.path.append(".")

import os
import os
import time
from tqdm import tqdm
import concurrent.futures
from openai import OpenAI
import concurrent.futures
import random 
import pandas as pd 

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(SCRIPT_DIR, "../.env")

from dotenv import load_dotenv
load_dotenv(ENV_PATH)


client = OpenAI(
    base_url=os.environ.get("VLLM_ENDPOINT", "http://localhost:8010/v1"),
    api_key="abc",
)

max_threads = 4
MODEL_CKPT = "Qwen/Qwen2.5-7B-Instruct"


def load_vLLM_server(model):
    return None, model


def generate_with_vLLM_server(
    prompt,
    model_ckpt=MODEL_CKPT,
    max_tokens=256,
    temperature=0.8,
    top_k=40,
    top_p=0.95,
    stop=["\n"],
):
    messages = [{"role": "user", "content": prompt}]
    parameters = {
        "model": model_ckpt,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "stop": stop,
        "seed": 1,
    }

    ans, timeout = "", 5
    while not ans:
        try:
            completion = client.chat.completions.create(messages=messages, **parameters)
            ans = completion.choices[0].message.content

        except Exception as e:
            print(e)
        if not ans:
                
            print(f"Will retry after {timeout} seconds ...")
            time.sleep(timeout)

            timeout = timeout * 2
            if timeout > 120:
                timeout = 1

    return ans



def generate_n_with_vLLM_server(
    prompt,
    n=1,
    model_ckpt=MODEL_CKPT,
    max_tokens=256,
    temperature=0.8,
    top_k=40,
    top_p=0.95,
    stop=["\n"],
    max_threads=3,
    disable_tqdm=True,
):
    
    messages = [{"role": "user", "content": prompt}]
    parameters = {
        "model": model_ckpt,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "stop": stop,
        "seed": 1,
        "n": n
    }

    ans, timeout = [], 5
    while not ans:
        try:
            completion = client.chat.completions.create(messages=messages, **parameters)
            ans = [completion.choices[i].message.content for i in range(n)]

        except Exception as e:
            print(e)
        if not ans:
                
            print(f"Will retry after {timeout} seconds ...")
            time.sleep(timeout)

            timeout = timeout * 2
            if timeout > 120:
                timeout = 1

    return ans



def generate_batch_with_vLLM_server(
    prompts,
    model_ckpt=MODEL_CKPT,
    max_tokens=256,
    temperature=0.8,
    top_k=40,
    top_p=0.95,
    stop=["\n"],
):
    messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
    parameters = {
        "model": model_ckpt,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "stop": stop,
        "seed": 1,
    }

    ans, timeout = [], 5
    while not ans:
        try:
            completion = client.chat.completions.create(messages=messages, **parameters)
            ans = [choice.message.content for choice in completion.choices]

        except Exception as e:
            print(e)
        if not ans:
            print(f"Will retry after {timeout} seconds ...")
            time.sleep(timeout)

            timeout = timeout * 2
            if timeout > 120:
                timeout = 1

    return ans





def generate_batch_with_vLLM_server_threads(
    prompts,
    model_ckpt=MODEL_CKPT,
    max_tokens=256,
    n=1,
    temperature=1,
    top_k=40,
    top_p=0.95,
    stop=None,
    max_threads=32,  # Adjust based on your server load
):
    def generate_single(prompt):
        messages = [{"role": "user", "content": prompt}]
        parameters = {
            "model": model_ckpt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stop": stop,
            #"seed": random.randint(1, 1e6),
            "n": n
        }

        ans, timeout = "", 5
        while not ans:
            try:
                completion = client.chat.completions.create(messages=messages, **parameters)
                ans = completion.choices[0].message.content
                #ans = [completion.choices[i].message.content for i in range(n)]
            except Exception as e:
                print(f"Error: {e}")
            
            if not ans:
                print(f"Retrying after {timeout} seconds ...")
                time.sleep(timeout)
                timeout = min(timeout * 2, 120)

        return ans

    # Parallel processing of multiple prompts
    with concurrent.futures.ThreadPoolExecutor(max_threads) as executor:
        results = list(executor.map(generate_single, prompts))

    return results




# ############### Code for generating lexical queries #################

# df_path = "NuminaMath_inputs_forlexicalqueries_prompt2.csv"
# output_csv_path = "NuminaMath_LLMoutputs_forlexicalqueries_prompt2.csv"

# df = pd.read_csv(df_path)
# total_rows = len(df)
# print("Total rows:", total_rows)

# # Load CSV headers
# df = pd.read_csv(df_path, nrows=1)  # Read only the header row
# columns = df.columns.tolist() + ["generated_LLM"]  # Add new column


# # Ensure output file has correct columns but no full copy
# if not os.path.exists(output_csv_path):
#     pd.DataFrame(columns=columns).to_csv(output_csv_path, index=False)

# count = 0 

# # Batch processing for lexical query generation 
# batch_size = 32

# for batch_df in tqdm(pd.read_csv(df_path, chunksize=batch_size), total=total_rows // batch_size, desc="Processing Batches", unit="batch"):
#     prompts = batch_df["input"].tolist()
#     responses = generate_batch_with_vLLM_server_threads(prompts)
    
#     # Append generated responses
#     batch_df["generated_LLM"] = responses

#     # Append batch to output CSV
#     batch_df.to_csv(output_csv_path, mode="a", header=False, index=False)

# print("Processing complete. Output saved at:", output_csv_path)
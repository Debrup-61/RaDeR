# Licensed under the MIT license.

import sys
sys.path.append(".")
import torch
from transformers import (
    GenerationConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria, 
    StoppingCriteriaList
)
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np

def load_HF_model(ckpt) -> tuple:
    
    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForCausalLM.from_pretrained(
        ckpt,
        torch_dtype="auto",
        device_map="auto",      # Mixtral: trust_remote_code=True, load_in_8bit=False
    )
    return tokenizer, model


def generate_with_HF_model(
    tokenizer, model, stop_tokens = [], input=None, num_return_sequences=1, temperature=0.8, top_p=0.95, top_k=40, num_beams=1, max_tokens=1024, system = "A chat between a curious user and an AI assistant." ,**kwargs
):
    try:
        # Generate with Qwen-2.5-32B-instruct
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": input}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
       
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        print("Tokenization complete!")

        # Generate 
        generation_config = GenerationConfig(
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )

        generated_ids = model.generate(
            model_inputs['input_ids'],
            generation_config=generation_config,
            max_new_tokens=max_tokens,    
            stop_strings=stop_tokens,
            tokenizer=tokenizer,
            num_return_sequences = num_return_sequences
        )
        
        processed_outputs = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(
            model_inputs['input_ids'].repeat_interleave(num_return_sequences, dim=0),  # Account for num_return_sequences
            generated_ids
            )
        ]

        # Decode all outputs into strings
        decoded_outputs = tokenizer.batch_decode(processed_outputs, skip_special_tokens=True)

        # index = output.find("\n")
        # if index != -1:  # Check if "\n" is found
        #     output = output[:index]
        # print("Final output", output)    

    except Exception as e:
        print(e)
    
    return decoded_outputs

if __name__ == "__main__":
    
    print("")
    #model_ckpt = "Qwen/Qwen2.5-Math-7B-Instruct"
    #model_ckpt = "meta-llama/Llama-3.1-8B-Instruct"
    # model_ckpt = "Qwen/Qwen2.5-32B-Instruct"
    # tokenizer, model = load_HF_model(model_ckpt)
    # print("Huggingface tokenizer and model loaded! ")
    
    # model_input = '''### Instruction: 
    # Mary is planning to bake exactly 10 cookies, and each cookie may be one of three different shapes -- triangle, circle, and square. Mary wants the cookie shapes to be a diverse as possible. What is the smallest possible count for the most common shape across the ten cookies?
    
    # ### Response:
    # Let's think step by step.
    # Step 1: To make the cookie shapes as diverse as possible, Mary should try to have an equal or nearly equal number of each shape.
    # Step 2: '''
    
    # hf_response=generate_with_HF_model(
    #                 tokenizer=tokenizer, 
    #                 model=model, 
    #                 stop_tokens=["\n","\n\n"],
    #                 input=model_input, 
    #                 max_tokens = 256,
    #                 num_return_sequences=3
    #             )            
                
    # #breakpoint()
    # print("Model outputs from huggingface pipeline:", hf_response)
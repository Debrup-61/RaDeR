# Licensed under the MIT license.

import sys

sys.path.append(".")

from typing import List, Dict

try:
    from models.vLLM_API import generate_with_vLLM_model
except:
    pass

try:
    from models.OpenAI_API import generate_n_with_OpenAI_model
except:
    pass

try:
    from models.HuggingFace_API import generate_with_HF_model
except:
    pass

try:
    from models.vLLM_server_API import generate_n_with_vLLM_server
except:
    pass

def extract_parts(text):
    # Split the string into sections by identifying the start of the last question
    last_question_start = text.rfind("### Question:")

    if last_question_start == -1:
        raise ValueError("No question found in the input string.")

    # Extract the system and model_input
    system = text[:last_question_start].strip()
    model_input = text[last_question_start:].strip()

    return system, model_input

class IO_System:
    """Input/Output system"""

    def __init__(self, args, tokenizer, model) -> None:
        self.api = args.api
        if self.api == "together":
            assert tokenizer is None and model is None
        elif self.api == "gpt3.5-turbo" or self.api == "vllm-server":
            assert tokenizer is None and isinstance(model, str)
        self.model_ckpt = args.model_ckpt
        self.temperature = args.temperature
        self.top_k = args.top_k
        self.top_p = args.top_p
        self.tokenizer = tokenizer
        self.model = model

        self.call_counter = 0
        self.token_counter = 0

    def generate(self, model_input, max_tokens: int, num_return: int, stop_tokens):
        if isinstance(model_input, str):
            
            # Extract the instructions and examples and put that into system prompt 
            if self.api == "huggingface":
                
                #system, model_input = extract_parts(model_input)
                hf_response=generate_with_HF_model(
                    tokenizer=self.tokenizer, 
                    model=self.model, 
                    input=model_input, 
                    num_return_sequences=num_return,
                    temperature=self.temperature, 
                    top_p=self.top_p, 
                    top_k=self.top_k, 
                    stop_tokens = stop_tokens,
                    max_tokens = max_tokens,
                    #system = system
                )
                
                io_output_list = hf_response
                self.call_counter += 1
                self.token_counter += 0


            elif self.api == "vllm":
                vllm_response = generate_with_vLLM_model(
                    self.model,
                    input=model_input,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    n=num_return,
                    max_tokens=max_tokens,
                    stop=stop_tokens,
                )
                io_output_list = [o.text for o in vllm_response[0].outputs]
                self.call_counter += 1
                self.token_counter += sum([len(o.token_ids) for o in vllm_response[0].outputs])
            elif self.api == "gpt3.5-turbo":
                gpt_response = generate_n_with_OpenAI_model(
                    prompt=model_input,
                    n=num_return,
                    model_ckpt=self.model,
                    max_tokens=max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    stop=["\n", "Answer"],
                )
                io_output_list = gpt_response
                self.call_counter += num_return
                self.token_counter += 0
            
            elif self.api == "vllm-server":
                server_response = generate_n_with_vLLM_server(
                    prompt=model_input,
                    n=num_return,
                    model_ckpt=self.model,
                    max_tokens=max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    stop=stop_tokens,
                )
                io_output_list = server_response
                self.call_counter += num_return
                self.token_counter += 0
            
            elif self.api == "debug":
                io_output_list = ["Debug: The answer is generated with debug mode, 233." for _ in range(num_return)]
            else:
                #raise NotImplementedError(f"API {self.api} is not implemented.")
                print(f"API {self.api} is not implemented.")
        
        elif isinstance(model_input, list):
            if self.api == "vllm":
                vllm_response = generate_with_vLLM_model(
                    self.model,
                    input=model_input,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    n=num_return,
                    max_tokens=max_tokens,
                    stop=stop_tokens,
                )
                io_output_list = [
                    [o.text for o in resp_to_single_input.outputs] for resp_to_single_input in vllm_response
                ]
                self.call_counter += 1
                self.token_counter += sum(
                    [
                        sum([len(o.token_ids) for o in resp_to_single_input.outputs])
                        for resp_to_single_input in vllm_response
                    ]
                )

            elif self.api == "huggingface":
                io_output_list = []
                for input in model_input:
                    #system, input = extract_parts(input)
                    hf_response=generate_with_HF_model(
                        tokenizer=self.tokenizer, 
                        model=self.model, 
                        input=input, 
                        num_return_sequences = 1, 
                        temperature=self.temperature, 
                        top_p=self.top_p, 
                        top_k=self.top_k, 
                        max_tokens=max_tokens,
                        stop_tokens = stop_tokens,
                        #system = system 
                    )
                    io_output_list.extend(hf_response)
                    self.call_counter += 1
                    self.token_counter += 0

            elif self.api == "gpt3.5-turbo":
                io_output_list = []
                for input in model_input:
                    gpt_response = generate_n_with_OpenAI_model(
                        prompt=input,
                        n=num_return,
                        model_ckpt=self.model,
                        max_tokens=max_tokens,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        top_k=self.top_k,
                        stop=["\n", "Answer"],
                    )
                    io_output_list.append(gpt_response)
                    self.call_counter += num_return
                    self.token_counter += 0
            
            elif self.api == "vllm-server":
                io_output_list = []
                for input in model_input:
                    server_response = generate_n_with_vLLM_server(
                        prompt=input,
                        n=num_return,
                        model_ckpt=self.model,
                        max_tokens=max_tokens,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        top_k=self.top_k,
                        stop=stop_tokens,
                    )
                    io_output_list.append(server_response)
                    self.call_counter += num_return
                    self.token_counter += 0
            
            elif self.api == "debug":
                io_output_list = [
                    ["Debug: The answer is generated with debug mode, 233." for _ in range(num_return)]
                    for _ in model_input
                ]
            else:
                #raise NotImplementedError(f"API {self.api} is not implemented.")
                print(f"API {self.api} is not implemented.")

        return io_output_list

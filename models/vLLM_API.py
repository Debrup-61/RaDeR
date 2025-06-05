# Licensed under the MIT license.
import sys
sys.path.append(".")

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import numpy as np
import math



def load_vLLM_model(model_ckpt, seed, tensor_parallel_size=1, half_precision=False, max_num_seqs=256):
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

    if half_precision:
        llm = LLM(
            model=model_ckpt,
            dtype="half",
            tensor_parallel_size=tensor_parallel_size,
            seed=seed,
            trust_remote_code=True,
            max_num_seqs=max_num_seqs,
            swap_space=16,
        )
    else:
        llm = LLM(
            model=model_ckpt,
            tensor_parallel_size=tensor_parallel_size,
            seed=seed,
            trust_remote_code=True,
            max_num_seqs=max_num_seqs,
            swap_space=16,
        )

    return tokenizer, llm


def generate_with_vLLM_model(
    model,
    input,
    temperature=0,
    top_p=0.95,
    top_k=40,
    repetition_penalty=1.1,
    n=1,
    max_tokens=512,
    min_tokens = 256,
    logprobs=1,
    stop=[],
):
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        n=n,
        logprobs=logprobs,
        max_tokens=max_tokens,
        min_tokens=min_tokens,
        stop=stop,
    )

    output = model.generate(input, sampling_params, use_tqdm=False)
    return output


if __name__ == "__main__":
    print("")
    
    # model_ckpt = "Qwen/Qwen2.5-7B-Instruct"
    # tokenizer, model = load_vLLM_model(model_ckpt, seed=42, tensor_parallel_size=1, half_precision=False)
    # print("VLLM tokenizer and model loaded !")
    
    # model_input = '''### Question: 
    # Find the quotient of the division $(3z^4-4z^3+5z^2-11z+2)/(2+3z)$.  
    # Intermediate solution: To find the quotient of the division (3z4-4z3+5z2-11z+2)÷(2+3z), we perform polynomial long division.
    # *Query for Retrieval*: 
    # Theorem 1: Polynomial Division Algorithm
    # Theorem statement: For any two polynomials P(z)(dividend) and D(z) (divisor), with deg(P(z)) ≥ deg (D(z)), there exist unique polynomials Q(z) (quotient) and R(z) (remainder) such that: P(z)=D(z)Q(z)+R(z)
    # Preconditions: 
    # (1) D(z)≠0 
    # (2) The degree of the dividend P(z) must be greater than or equal to the degree of the divisor D(z).
    # Why Preconditions are Satisfied in the Question: 
    # (1) The divisor D(z)=3z+2 is a linear polynomial and is non-zero as z not equal to -2/3.
    # (2) The degree of the dividend P(z)=3z^4 - 4z^3 + 5z^2 - 11z + 2 is 4, which is greater than the degree of the divisor (1).
    # Subject: Algebra (Polynomial Theory)


    # ### Question: 
    # Many states use a sequence of three letters followed by a sequence of three digits as their standard license-plate pattern. Given that each three-letter three-digit arrangement is equally likely, the probability that such a license plate will contain at least one palindrome (a three-letter arrangement or a three-digit arrangement that reads the same left-to-right as it does right-to-left) is $\dfrac{m}{n}$, where $m$ and $n$ are relatively prime positive integers. Find $m+n.$
    # Intermediate solution: 
    # To calculate the probability of a three-letter palindrome, we first note that there are 26 possible choices for each letter. The first and last letters must be the same, so there are 26 choices for the first and last letters, and 26 choices for the middle letter. 
    # *Query for Retrieval*: 
    # Theorem 1: Principle of Inclusion-Exclusion
    # Theorem Statement: For any two finite sets A and B, the size of their union is given by:
    # |A \cup B| = |A| + |B| - |A \cap B|
    # Preconditions:
    # (1) A and B are finite sets.
    # (2) A and B can overlap.
    # Why Preconditions are Satisfied in the Question:
    # (1) The sets A and B represent the events where the three-letter sequence and the three-digit sequence are palindromes, respectively. Both sets are finite because they are based on the finite number of possible sequences.
    # (2) It is possible for both the three-letter sequence and the three-digit sequence to be palindromes simultaneously.
    # Subject: Combinatorics (Set Theory)


    # ### Question: James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year? 
    # Intermediate solution: He writes each friend 3*2=6 pages a week.
    # *Query for Retrieval*:
    # Theorem 1: Basic Multiplication Principle
    # Theorem Statement: If there are m ways to do something and n ways to do another thing, then there are m*n to do both things.
    # Preconditions:
    # (1) There must be a clear count of options or occurrences for each action.
    # (2) Each action is independent of the other.
    # Why Preconditions are Satisfied in the Question:
    # (1) James writes 3-page letters to 2 different friends, and he does this twice a week. Each of these actions has a clear count: 3 pages per letter, 2 friends, and 2 times per week.
    # (2) The number of pages written per letter, the number of friends, and the frequency of writing are all independent of each other.
    # Subject: Arithmetic (Basic Operations)


    # ### Question: Mary is planning to bake exactly 10 cookies, and each cookie may be one of three different shapes -- triangle, circle, and square. Mary wants the cookie shapes to be a diverse as possible. What is the smallest possible count for the most common shape across the ten cookies?
    # Intermediate solution: To make the cookie shapes as diverse as possible, Mary should try to have an equal or nearly equal number of each shape.
    # *Query for Retrieval*:
    # '''
    
    # vllm_response = generate_with_vLLM_model(
    #                 model,
    #                 input=model_input,
    #                 n=1,
    #                 max_tokens=512,
    #                 stop=["### Question:","###","Question:","Question","\n\n"],
    #             )
    # #breakpoint()
    # io_output_list = [o.text for o in vllm_response[0].outputs]
    # print("Output by VLLM model", io_output_list[0])

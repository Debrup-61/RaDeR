from models.HuggingFace_API import load_HF_model
from common.arguments import get_parser, post_process_args, save_args
from models.IO_System import IO_System
from pathlib import Path
import glob 
import os 
import re 
import json 
import csv 
from tqdm import tqdm 
from common.utils import read_txt, read_json
import pandas as pd 
import random 

def has_retrieval_key(d):
    return any("retrieval" in key for key in d)

def get_queries(prompt, directory_path):
    

    directory = Path(directory_path)
    
    csv_file = os.path.join(directory_path, "generated_queries.csv")
    
    # Ensure CSV has headers if it does not exist
    if not os.path.exists(csv_file):
        with open(csv_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["question_number", "filepath", "solution_no", "retrieval_sol_no", "partial_solution", "LLMtheorem", "io_input", "generated_query"])
    
    files = list(directory.glob("Question ???? - Final Solutions*")) 

    for file in tqdm(files, desc="Processing Files", unit="file"):
        question_number = None 
        print("Name of file:", file.name)
        match = re.search(r"Question (\d+) - Final Solutions", file.name)
        if match:
            question_number = int(match.group(1))
        
        # Open and load JSON file
        with open(os.path.join(directory,str(file.name)), "r", encoding="utf-8") as f:
            data = json.load(f)  # Loads JSON into a Python dictionary
        
        question = (data[0]['trace']['0']['user_question'])
        #print(f"Question: {question}")
        total_solutions = 0 

        # Print or process the data
        for i in range(len(data)):
            solution_no = i 
            retrieval_sol_no = 1
            question = (data[i]['trace']['0']['user_question'])
            partial_solution = ""
            
            if not has_retrieval_key(data[i]['trace']['0']['ost_step']):  # Ignore if it is not a retrieval solution 
                continue
                
            
            if 'ost_step' not in data[i]['trace']['0']:  # Ignore if there are no OST steps
               continue

            else:
                ost_steps =  data[i]['trace']['0']['ost_step'] 
            
                for step_no in ost_steps:
                    
                    if 'retrieval' in step_no:
                        retrieval_no = step_no.split("_")[0]
                        #print("Retrieval no", retrieval_no)
                        theorem_name = ost_steps[retrieval_no + "_querygeneration"][0]
                        
                        step_sol = ost_steps[step_no][0] 
                        retriever_name = ost_steps[step_no][2] 
                       
                        print("Question No:", question_number)
                        print("Question:", question)
                        print("Partial solution:", partial_solution)
                        print("Theorem:", theorem_name)
                        #print("Retrieval:", step_sol)
                        print("Retrieval no:", retrieval_sol_no)
                        print("Retriever name", retriever_name)
                        ch = input("Press a key to continue:")
                        partial_solution += " " + ost_steps[step_no][0]
                        io_input = prompt + "\n\n" + f"**Question**:{question}\nPartial solution:{partial_solution}\n{theorem_name}\nPreconditions:" 
                        print("Input for query generation", io_input)
                        
                        generated_query = io.generate(
                                io_input,
                                num_return=1,
                                max_tokens=1024,
                                stop_tokens=["**Question**"],
                            )
                        print("Generated query:", generated_query)    
                        
                        with open(csv_file, mode="a", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow([question_number, directory_path + "/" + str(file.name), solution_no, retrieval_sol_no, partial_solution, theorem_name, io_input, generated_query[0]])


                        retrieval_sol_no += 1
                        print("-"*80)
                        
                    
                    elif 'querygeneration' in step_no:
                        print(step_no)
                        continue

                    else:
                       
                        if type(ost_steps[step_no]) == list:
                            partial_solution += " " + ost_steps[step_no][0]  
                            step_sol = ost_steps[step_no][0]

                        else:
                            partial_solution += " " + ost_steps[step_no]
                            step_sol = ost_steps[step_no]
                           

                        #print(f"OST Step: {step_no}" ,step_sol) 
                
            print("-"*70)
            #ch = input("Enter a key to continue:")
    
    print("Function exited!")

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    args = post_process_args(args)
    print("Arguments:", args)
    save_args(args)
    
    tokenizer, model = load_HF_model(args.model_ckpt)
    io = IO_System(args, tokenizer, model)

    prompt = read_txt("query_generation_prompts/prompt1.txt")
    print("Prompt for query generation:", prompt)
    
    get_queries(prompt = prompt, directory_path = args.answers_directory_path)  # Get the queries using Qwen-2.5-7B-Instruct model 
   
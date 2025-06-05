# Licensed under the MIT license.

import sys

sys.path.append(".")
from models.IO_System import IO_System
from common.utils import read_json, save_json
from eval_src.Evaluator import *

import warnings
import re 
warnings.filterwarnings("ignore")
from tqdm import tqdm
from argparse import ArgumentParser
from models.vLLM_server_API import load_vLLM_server
from models.vLLM_server_API import generate_n_with_vLLM_server
        
def extract_trace(data_item, num_votes):
    res = []
    for item in data_item:
        i = 0
        trace = item["trace"] if "trace" in item else item
        rollout_id = item["rollout_id"] if "rollout_id" in item else 0
        if num_votes != -1 and rollout_id >= num_votes:
            continue
        while str(i) in trace:
            i += 1
        if "direct_answer" in trace[str(i-1)]:
            res.append(trace[str(i-1)]["direct_answer"]["text"])
        elif trace[str(i-1)]["ost_step"] != {}:
            j = 1
            while str(j) in trace[str(i-1)]["ost_step"]:
                j += 1
            res.append(trace[str(i-1)]["ost_step"][str(j-1)])
        elif "subanswer" in trace[str(i-1)]:
            res.append(trace[str(i-1)]["subanswer"]["text"])
        else:
            import pdb; pdb.set_trace()
    return res


def extract_completions(data_item):
    res = []
    for item in data_item:
        res.append(data_item[item]["model_solution"])
    return res


def transform_string(s):
    return re.sub(r'Question 0*(\d+) - Final Solutions', r'Question \1 - Answer', s)
    


def eval_single_item_from_answer_sheets(example_id, answer_sheets_dir, evaluator: Evaluator, vote_pool_dir=None, num_votes=-1, io=None):
    data_item = {}
    
    #gold_example_id = transform_string(example_id)
    gold_example_id = example_id

    print("Example id solutions:", example_id)
    print("Gold answer example id:", gold_example_id)
    #ch = input("Enter a key to continue")
    
    args.api = "vllm-server"
    args.model_ckpt = "Qwen/Qwen2.5-7B-Instruct"
    args.temperature = 0.8
    args.top_k = 40
    args.top_p = 0.95
       

    tokenizer, model = load_vLLM_server("Qwen/Qwen2.5-7B-Instruct")
    io = IO_System(args, tokenizer, model)

    #print("Path to gold answer file",os.path.join(answer_sheets_dir, f"{example_id}.json"))
    question = read_json(os.path.join(answer_sheets_dir, f"{gold_example_id}.json"))["problem"]
    gold_answer = read_json(os.path.join(answer_sheets_dir, f"{gold_example_id}.json"))["gold_answer"]
    #print("Gold answer from BRIGHT 1", gold_answer)
    
    #gold_answer = extract_the_answer_is(read_json(os.path.join(answer_sheets_dir, f"{example_id}.json"))["solution"])
    data_1 = read_json(os.path.join(answer_sheets_dir, example_id + ".json"))
    data_2 = read_json(os.path.join(answer_sheets_dir, example_id.replace(" - Final Solutions", " - Rollout Solutions") + ".json"))
    
    #print("Extract trace function called")
    data_1 = extract_trace(data_1, num_votes) # data_1 + data_2
    data_2 = extract_trace(data_2, num_votes) # data_1 + data_2
    data_1_2 = data_1 + data_2
    
    #print("Find most confidence called!")
    #print("All model answers:", data_1_2)
    model_answer_1_2, _, _, _ = evaluator.find_most_confident_answer(data_1_2)
    print("Most confidence answer:", model_answer_1_2)
    print("Gold answer from BRIGHT: ", gold_answer)
    #print("Model answer", model_answer_1_2)
    #ch = input("Enter a key to continue:")
    
    #result_1_2 = evaluator.check_answers_equiv(model_answer_1_2, gold_answer)
    result_1_2 = evaluator.calculate_reward_goldanswer(model_answer_1_2, gold_answer, io, question)
    
    if result_1_2 != 1:
        result_1_2 = 0
    

    data_item["correct"] = result_1_2
    data_item["predict_answer"] = model_answer_1_2
    data_item["gold_answer"] = gold_answer

    #ch = input("Enter the key")

    return data_item


def eval_exp(exp_dir: str, dataset_name: str, num_votes: int = -1):
    answer_sheets_dir = os.path.join(exp_dir, "answer_sheets")
    vote_pool_dir = os.path.join(exp_dir, "vote_pool")

    example_ids = [f.replace(".json", "") for f in os.listdir(answer_sheets_dir) \
                   if f.endswith(" - Final Solutions.json")]
    
    
    # Initialize IO

    #print("Example ids:", example_ids)

    evaluator = eval(f"{dataset_name}Evaluator()")

    data_list = []
    for i in tqdm(range(len(example_ids))):
        example_id = example_ids[i]


        # try:
        dta = eval_single_item_from_answer_sheets(example_id, answer_sheets_dir, evaluator, vote_pool_dir, num_votes)
        data_list.append(dta)
        # except:
        #     print(f"Error in {example_id}")

    # Calculate accuracy
    accuracy = sum([item["correct"] for item in data_list]) / len(data_list)
    print(f"accuracy: {accuracy}")

    # Save eval results
    eval_result_dir = answer_sheets_dir.replace("run", "eval").replace("answer_sheets", "")
    os.makedirs(eval_result_dir, exist_ok=True)
    save_json(data_list, os.path.join(eval_result_dir, "eval_results.json"))
    analysis = {
        "accuracy": accuracy, 
        "num_tested": len(data_list),
        "num_correct": accuracy * len(data_list),
    }
    save_json(analysis, os.path.join(eval_result_dir, "analysis.json"))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--exp_dir_path", type=str, required=True)
    parser.add_argument("--num_votes", type=int, default=-1)
    args = parser.parse_args()

    eval_exp(args.exp_dir_path, args.dataset_name, num_votes=args.num_votes)

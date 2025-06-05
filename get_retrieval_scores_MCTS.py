import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
from argparse import ArgumentParser
import json
import re
from datasets import load_dataset
import pytrec_eval
import os 

def find_keys_with_id(data_dict, s):
    """
    Finds keys in a dictionary that contain the substring "id".
    
    Args:
        data_dict (dict): The dictionary to search.
        
    Returns:
        list: A list of keys containing s.
    """
    return [key for key in data_dict.keys() if s in key]

def max_ndcg_queries(scores, ids, output_file):
    """
    For each question, get the highest recall@10 query and its scores. 
    Parameters:
        scores (dict): A dictionary containing scores per query.
    """
    with open(output_file, "w") as f:

        new_scores = {}
        # For each id, find keys which have the id in it 
        for id_ in ids:
            list_keys = find_keys_with_id(scores,id_)
            #print("List keys:", list_keys)

            max_recall_query = -1
            max_key_query = None 
            for key in list_keys:
                if scores[key]['recall_10'] > max_recall_query:
                    max_recall_query = scores[key]['recall_10']
                    max_key_query = key
            
            if max_key_query is None:
                for key in list_keys:
                    print(scores[key]['recall_10'])


            max_score_query = scores[max_key_query]
            #print(max_score_query)

            # Add the query to the new_scores dictionary 
            new_scores[max_key_query] = max_score_query

            # Store the retrieval query id 
            f.write(max_key_query + "\n")
    
    return new_scores    


def calculate_retrieval_metrics(results, qrels, output_dir, ids, k_values=[1, 5, 10, 25, 50, 100]):
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
    with open(os.path.join(output_dir,"query_scores.json"), "w") as f:
        json.dump(scores, f, indent=4)
   
    #print("Query ids with NDCG@10 = 0")
    #print_high_ndcg_queries(scores, os.path.join(output_dir,"matched_queries.txt"), threshold=0)

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
    print("Average retrieval scores:", output)

    new_scores = max_ndcg_queries(scores, ids, os.path.join(output_dir,"max_retrieval_score_queries.txt"))
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
    
    for query_id in new_scores.keys():
        for k in k_values:
            ndcg[f"NDCG@{k}"] += new_scores[query_id]["ndcg_cut_" + str(k)]
            _map[f"MAP@{k}"] += new_scores[query_id]["map_cut_" + str(k)]
            recall[f"Recall@{k}"] += new_scores[query_id]["recall_" + str(k)]
            precision[f"P@{k}"] += new_scores[query_id]["P_" + str(k)]
        mrr["MRR"] += new_scores[query_id]["recip_rank"]

    for k in k_values:
        ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"] / len(new_scores), 5)
        _map[f"MAP@{k}"] = round(_map[f"MAP@{k}"] / len(new_scores), 5)
        recall[f"Recall@{k}"] = round(recall[f"Recall@{k}"] / len(new_scores), 5)
        precision[f"P@{k}"] = round(precision[f"P@{k}"] / len(new_scores), 5)
    mrr["MRR"] = round(mrr["MRR"] / len(new_scores), 5)

    output = {**ndcg, **_map, **recall, **precision, **mrr}
    print("Max retrieval scores:", output)

    return output



def get_retrieval_keys(d):
    """
    Check if a dictionary has keys of the form i_retrieval where i is an integer > 0,
    and return the matching keys.

    Args:
    d (dict): The dictionary to check.

    Returns:
    tuple: (bool, list) where the first element indicates if such keys exist,
           and the second element is a list of matching keys.
    """
    pattern = re.compile(r'^[1-9]\d*_retrieval$')  # Matches keys like 1_retrieval, 2_retrieval, etc.
    matching_keys = [key for key in d.keys() if pattern.match(key)]
    return matching_keys


def read_json_file(file_path):
    """
    Reads a JSON file and returns its content as a Python object.

    Args:
    file_path (str): Path to the JSON file.

    Returns:
    dict or list: The parsed JSON content as a Python object.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)  # Parse the JSON file
    return data

def get_id(input_string):
    match = re.search(r".*\.json", input_string)
    if match:
        result = match.group()
        return result
        #print(result)  
    else:
        return None    

def read_jsonl_as_list(file_path):
    """
    Reads a JSONL file and returns its content as a list of dictionaries.

    Args:
    file_path (str): Path to the JSONL file.

    Returns:
    list: A list of dictionaries, each representing a line in the JSONL file.
    """
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            # Parse each line as a JSON object
            data.append(json.loads(line.strip()))
    return data


def get_scores(args,cache="cache"):
    
    # Get the json file input
    #outputs = read_json_file(args.finalsol_path) 
    #print(outputs[1])
    retrieval_results = read_jsonl_as_list(args.finalsol_path)
    #print(retrieval_results[0].keys())

    # Get a dictionary of query_id -> gold_ids 
    examples = load_dataset('xlangai/bright', 'examples',cache_dir=cache)[args.task]
    ids = [examples[j]['id'] for j in range(len(examples))]
    final_scores_query = {}
    ground_truth = {}

    for i in range(len(retrieval_results)):
        
        retrieval_dict = retrieval_results[i] 
        for question_id in retrieval_dict:
            id_ = get_id(question_id) 
            q_text = retrieval_dict[question_id]["text"]
            scores_dict = retrieval_dict[question_id]["scores"]

            # Find the index of the id_ in examples 
            try:
                indx = ids.index(id_)  
            except ValueError:
                print("Element not found in the list.") # Element not in the list

            e = examples[indx]
            ground_truth[question_id] = {}
            #ground_truth[question_id + "_" + key] = {}
            for gid in e["gold_ids"]:
                ground_truth[question_id][gid] = 1
                #ground_truth[question_id + "_" + key][gid] = 1
            
            final_scores_query[question_id] = scores_dict
            #final_scores_query[question_id + "_" + key] = scores_dict 

    #print(final_scores_query)
    results = calculate_retrieval_metrics(results=final_scores_query, qrels=ground_truth, output_dir= args.output_dir, ids=ids)
    



if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--finalsol_path", type=str, default = "outputs_MCTS/run_outputs/BRIGHT/Qwen2.5-7B-Instruct/TESTRUN/retriever_score.jsonl")
    parser.add_argument("--output_dir", type=str, default = "outputs_MCTS/run_outputs/BRIGHT/Qwen2.5-7B-Instruct/TESTRUN/")
    parser.add_argument('--task', type=str, default='theoremqa_theorems', 
                        choices=['biology','earth_science','economics','pony','psychology','robotics',
                                 'stackoverflow','sustainable_living','aops','leetcode','theoremqa_theorems',
                                 'theoremqa_questions'])
    
    args = parser.parse_args()
    get_scores(args)



   
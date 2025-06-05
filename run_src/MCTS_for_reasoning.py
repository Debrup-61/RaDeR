# Licensed under the MIT license.
import random 
import sys
sys.path.append(".")
import json 
from sklearn.metrics.pairwise import cosine_similarity
from peft import PeftModel, PeftConfig
from huggingface_hub import login
from tqdm import tqdm 
# from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
from transformers import AutoTokenizer, AutoModel
from torchmetrics.functional.pairwise import pairwise_cosine_similarity
import re 
import numpy as np, os, random, json, math, wandb
from tqdm import trange
from typing import List, Dict, Tuple
from copy import deepcopy
from datasets import load_dataset
import torch 
from prompts.summarize_theorem import prompt_summarize
try:
    from rapidfuzz import fuzz, process
except:
    pass

from models.IO_System import IO_System
from common.utils import read_txt, read_json
from eval_src.Evaluator import Evaluator, GSM8KEvaluator
from MCTS_backbone import MCTS_Searcher, MCTS_Node
from run_src.rstar_utils import (
    Node_Type,
    GeneratorError,
    reach_terminal_subquestion,
    reach_terminal_ost_step,
    concat_subqs_and_subas,
    concat_ost_steps,
    concat_subqs_subas_as_ost_steps,
    concat_ost_steps_querygeneration,
    make_hint,
    make_response_prefix,
    split_user_question,
    print_tree_from_root,
    find_valid_solution_nodes,
    find_best_solution,
    stochastic_find_best_solution,
)

from models.repllama_server_API import generate_with_repllama, generate_with_repllama_batched
from models.RaDeR_retriever_server_API import generate_with_qwen
from models.test_BM25 import bm25_search


def get_candidate_theorems(text):
    # Regex to split the text into individual theorems
    theorem_split_pattern = r"(Theorem \d+:.*?)\s*(?=Theorem \d+:|$)"

    # Extract each theorem block
    theorem_blocks = re.findall(theorem_split_pattern, text, re.DOTALL)

    # Lists to store extracted components
    theorems = []
    preconditions_list = []
    why_preconditions_list = []
    subject_list = []

    for block in theorem_blocks:
        # Extract the theorem name and statement
        match = re.match(r"Theorem \d+: (.*?)\s*Theorem Statement:\s*(.*?)(?=\s*Preconditions:|\s*Why Preconditions|\s*Subject:|$)", block, re.DOTALL)
        if match:
            name = match.group(1).strip()
            statement = match.group(2).strip()
            theorems.append(f"Theorem Name: {name}\nTheorem Statement: {statement}")

        # Extract Preconditions
        preconditions_match = re.search(r"Preconditions:\s*(.*?)(?=\s*Why Preconditions|\s*Subject:|$)", block, re.DOTALL)
        if preconditions_match:
            preconditions_list.append(preconditions_match.group(1).strip())

        # Extract Why Preconditions are Satisfied
        why_preconditions_match = re.search(r"Why Preconditions are Satisfied:\s*(.*?)(?=\s*Subject:|$)", block, re.DOTALL)
        if why_preconditions_match:
            why_preconditions_list.append(why_preconditions_match.group(1).strip())

        # Extract Subject (Handles cases with or without newline)
        subject_match = re.search(r"Subject:\s*(.*)", block, re.DOTALL)
        if subject_match:
            subject_list.append(subject_match.group(1).strip())

    return theorems, preconditions_list, why_preconditions_list, subject_list


def remove_newlines(s):
    clean_text = s.replace("\n", " ")
    return clean_text

def clean(text): # Only use the statement not the entire proof with RAG 
   
   # Extract only the part before \begin{proof} if it exists
    match = re.search(r'(.*?)\\begin{proof}', text, re.DOTALL)
    if match:
        return remove_newlines(match.group(1).strip())
    else:
        return text.strip()
            
def verbose_print(s: str, verbose: bool):
    if verbose:
        print(s)


def retrieve_theorems(question,query,id_,documents, doc_emb, node_count, directory_path, top_k, retriever, threshold):
    
    # TODO: short fix
    if isinstance(query, list):
        query = query[0]
    
    # Randomly select proprtion of times BM25 is called, based on threshold  
    
    if random.random()<=threshold : 
            
        # TODO
        if retriever=="repllama":
            query_emb = generate_with_repllama(f"Query: {query}</s>")
        
        elif retriever=="RaDeR":
            query_emb = generate_with_qwen(f"query: {query}<|im_end|>")

        
        scores = cosine_similarity(query_emb, doc_emb)
        
        print("Scores shape", scores.shape)
        scores = scores.tolist()

        dict_retrieval = {}
        key = id_+ "_" + str(node_count)
        #print("Node count", node_count)
        #print("Key of dictionary", key)
        
        dict_retrieval[key] = {}
        retrieveddocs_scores = scores[0]
        scores_dict = {str(i): float(score) for i, score in enumerate(retrieveddocs_scores)}
            
        # Sort in descending order by retrieval scores 
        scores_dict = dict(sorted(scores_dict.items(), key=lambda item: item[1], reverse=True))
        dict_retrieval[key]= {}
        dict_retrieval[key]["query"] = query
        dict_retrieval[key]["scores"] = scores_dict
        
        
        retrieval_scores_path = directory_path + "/retriever_score.jsonl"
        #print("Scores dict", scores_dict)
        #print("Dictionary scores:", dict_retrieval)

        with open(retrieval_scores_path, "a") as jsonl_file:
            jsonl_file.write(json.dumps(dict_retrieval) + "\n")    


        # Get the top k retrieved docs for augmentation 
        doc_scores = {}
        count = 0 
        for k in range(len(scores[0])):
            s = scores[0][k]
            doc_scores[k] = s 
            count = count + 1

        # Get the top k scores and corresponding doc_ids
        top_k_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        # Extract only the doc_ids from the top k
        top_k_doc_ids = [doc_id for doc_id, score in top_k_docs]
        doc_ids = [str(doc_) for doc_ in top_k_doc_ids]

        # Get the contents of the sampled_doc_id
        if retriever=="RaDeR":
            return [clean(documents[sampled_doc_id]) for sampled_doc_id in top_k_doc_ids], dict_retrieval, doc_ids, "RaDeR"
        
        elif retriever=="repllama":
            return [clean(documents[sampled_doc_id]) for sampled_doc_id in top_k_doc_ids], dict_retrieval, doc_ids, "Repllama"

    
    else: # Use BM25 for the rest of the times 
        top_k_docs, doc_ids, doc_scores = bm25_search(query = candidate_theorems_query, k=top_k) 
        key = id_+ "_" + str(node_count)
        dict_retrieval = {}
        dict_retrieval[key] = {}
        dict_retrieval[key]["scores"] = {}
        dict_retrieval[key]["query"] = candidate_theorems_query
       
        for i in range(len(doc_ids)):
            dict_retrieval[key]["scores"][doc_ids[i]] = doc_scores[i]

        doc_ids = [str(doc_) for doc_ in doc_ids]    
        return [clean(doc) for doc in top_k_docs], dict_retrieval, doc_ids, "BM25"





class Generator:
    """Generator generates children nodes"""

    def __init__(self, args, tokenizer, model, evaluator: Evaluator) -> None:
        self.io = IO_System(args, tokenizer, model)
        print("IO_System initialized!")
        print(self.io)

        self.examples = load_dataset('xlangai/bright', 'examples',cache_dir=args.cache_dir)[args.task]
        
        self.directory_path = args.run_outputs_dir 
        self.evaluator = evaluator
        
        self.qg_nodes = args.qg_nodes
        self.max_qg_nodes = args.max_qg_nodes
        self.use_gold_documents = args.use_gold_documents
        self.topk_rt = args.topk_rt
        self.LLM_candidate_theorems = args.LLM_candidate_theorems
        self.retriever = args.retriever
        self.threshold = args.threshold 
        
        # Whether to use `self-reasoning` using generator LLM to not explore irrelevant theorems  
        self.retrieval_selfreasoning = args.retrieval_selfreasoning

        # Setting of calculating rewards
        self.bool_goldanswer_reward = args.bool_goldanswer_reward


        # Store the documents for retrieval
        doc_pairs = load_dataset('xlangai/bright', 'documents',cache_dir=args.cache_dir)[args.task]
        # Get the ids and the content for the documents
        doc_ids = []
        documents = []
        for dp in doc_pairs:
            doc_ids.append(dp['id'])
            documents.append(dp['content'])
        
        self.documents = documents    
        self.doc_ids = doc_ids     
        print("BRIGHT theorems corpus loaded!")
        
        
        self.doc_emb = np.load(os.path.join(args.cache_dir, "0.npy"), mmap_mode='r')
        
        
        # # Login to huggingface 
        # login(token=huggingface_token)

        # # Load the retriever tokenizer and model
        # self.retrieval_tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
        # self.retrieval_model= get_model('castorini/repllama-v1-7b-lora-doc')
        # print("Repllama Retrieval Model and tokenizer loaded!")


        self.id = {}
        self.num_subquestions = args.num_subquestions
        self.num_a1_steps = args.num_a1_steps
        self.num_votes = args.num_votes
        self.max_tokens = args.max_tokens
        self.enable_potential_score = args.enable_potential_score


        self.mcts_num_last_votes = args.mcts_num_last_votes

        with open(args.decompose_template_path, "r") as f:
            decompose_template = json.load(f)
            self.question_index = decompose_template["index"]

        self.decompose_prompt = read_txt(args.decompose_prompt_path)
        
        self.fewshot_querygen_prompt = read_txt(args.fewshot_querygen_prompt_path)
        self.fewshot_querygen_config = read_json(args.fewshot_querygen_config_path)

        self.fewshot_selfreason_relevance_prompt = read_txt(args.fewshot_selfreason_relevance_prompt_path)
        self.selfreason_relevance_config = read_json(args.selfreason_relevance_config_path)

        self.fewshot_retrievalreward_prompt_path = read_txt(args.fewshot_retrievalreward_prompt_path)
        self.fewshot_retrievalreward_config_path = read_json(args.fewshot_retrievalreward_config_path)

        self.fewshot_cot_prompt = read_txt(args.fewshot_cot_prompt_path)
        self.fewshot_cot_config = read_json(args.fewshot_cot_config_path)

        if not args.disable_a1:  # A1: Propose an one-step thought.
            self.fewshot_ost_prompt = read_txt(args.fewshot_ost_prompt_path)
            self.fewshot_ost_config = read_json(args.fewshot_ost_config_path)

        if not args.disable_a5:  # A5: Rephrase the question/sub-question.
            self.rephrasing_prompt_template = read_txt(args.rephrasing_prompt_template_path)
            self.decompose_prompt_rephrased = read_txt(args.decompose_prompt_rephrased_path)
            self.fewshot_cot_prompt_rephrased = read_txt(args.fewshot_cot_prompt_rephrased_path)
            self.fewshot_ost_prompt_rephrased = read_txt(args.fewshot_ost_prompt_rephrased_path)

    
    
    def _extract_from_cache(self, subquestion_list: List[str]):
        high_score_questions = []
        selected_answers = []
        values = []
        low_score_questions = []
        low_score_values = []
        low_score_answers_list = []
        unmatched_questions = []

        for subquestion in subquestion_list:
            best_match = process.extractOne(subquestion, self.reasoning_cache.keys(), scorer=fuzz.ratio)

            if best_match:
                best_question, best_score = best_match[0], best_match[1]
                similarity = best_score / 100
                cache_entry = self.reasoning_cache[best_question]
                score = cache_entry["score"]
                if similarity == 1:
                    if score >= 0.9:
                        high_score_questions.append(best_question)
                        selected_answers.append(cache_entry["selected_answer"])
                        values.append(score)
                    else:
                        low_score_questions.append(best_question)
                        low_score_values.append(score)
                        low_score_answers_list.append(cache_entry["answer_list"])
                else:
                    unmatched_questions.append(subquestion)
            else:
                unmatched_questions.append(subquestion)

        return {
            "high_score_questions": high_score_questions,
            "selected_answers": selected_answers,  # most likely answer corresponding to each subquestion
            "values": values,
            "low_score_questions": low_score_questions,
            "low_score_values": low_score_values,
            "low_score_answers_list": low_score_answers_list,
            "unmatched_questions": unmatched_questions,
        }
    
    def likelihood_answer(self, ost_step_list, index, gold_answer, question):
        
        if self.bool_goldanswer_reward is True:
            confidence = self.evaluator.calculate_reward_goldanswer(ost_step_list[index], gold_answer, self.io, question)
        
        else: # Find other elements of the list with same answer if not using gold answer of dataset 
            confidence = self.evaluator.find_confidence_of_answer(ost_step_list, index)
        
        return confidence   

    
    def _get_most_likely_answer(self, io_output_list: List[str], gold_answer: str, question: str) -> Tuple[str, float]:
        assert len(io_output_list) > 0

        if len(io_output_list) == 1:
            most_confident_answer_full_completion = io_output_list[0]
            if self.bool_goldanswer_reward is True:
                confidence = self.evaluator.calculate_reward_goldanswer(most_confident_answer_full_completion, gold_answer, self.io, question)
            else:
                confidence = 1     
        
        else:
            
            if self.bool_goldanswer_reward is True:
                most_confident_answer_full_completion = None
                max_confidence = -1
                for completion in io_output_list:
                    confidence = self.evaluator.calculate_reward_goldanswer(completion, gold_answer, self.io, question)
                    if confidence > max_confidence:
                        max_confidence = confidence
                        most_confident_answer_full_completion = completion
                
                return most_confident_answer_full_completion, confidence        
            
            else:
                _, most_confident_answer_full_completion, _, confidence = self.evaluator.find_most_confident_answer(
                    io_output_list
                )
                #assert confidence > 0
                return most_confident_answer_full_completion, confidence

        

   
    def _fewshot_cot_answer_question(self, question: str, paraphrased: bool, num_return: int, hint: str = None):
        fewshot_cot_prompt = self.fewshot_cot_prompt if not paraphrased else self.fewshot_cot_prompt_rephrased
        question += " " + hint if hint is not None else ""
        io_input = self.fewshot_cot_config["prompt_template"].format(examples=fewshot_cot_prompt, instruction=question)
        
        print("Generating direct answer using input context:", io_input)
        io_output_list = self.io.generate(
            io_input,
            num_return=num_return,
            max_tokens=self.max_tokens,
            stop_tokens=self.fewshot_cot_config["stop_tokens"],
        )
        
        cleaned_io_output_list = [io_output.strip() for io_output in io_output_list]  #! cleaning
        print("Generated direct answers: ", cleaned_io_output_list)
        return io_input, cleaned_io_output_list

    def generate_direct_answers(self, user_question: str, paraphrased: bool, hint: str, gold_answer: str, question: str, solution_trace: Dict[int, Dict[str, str]]):
        direct_answer_list, value_list = [], []

        #! few shot cot
        num_return = self.mcts_num_last_votes
        io_input, cleaned_io_output_list = self._fewshot_cot_answer_question(
            question=user_question, paraphrased=paraphrased, num_return=num_return, hint=hint
        )

        try:
            most_likely_answer, likelihood = self._get_most_likely_answer(cleaned_io_output_list, gold_answer, question)
        except Exception as e:
            raise GeneratorError(
                source="generate direct answer from: few shot cot",
                io_input=io_input,
                io_output_list=cleaned_io_output_list,
            )

        direct_answer_list.append(most_likely_answer)
        value_list.append(likelihood)
        print("Most likely Direct answer list", direct_answer_list)
        print("Value list", value_list)

        return direct_answer_list, value_list

    def generate_subquestions(
        self,
        user_question: str,
        solution_trace: Dict[int, Dict[str, str]],
        paraphrased: bool,
    ):
        subquestion_list, subanswer_list, value_list = [], [], []
        decompose_prompt = self.decompose_prompt if not paraphrased else self.decompose_prompt_rephrased

        #! generate subquestions
        existing_subquestions_and_subanswers, next_subquestion_id = concat_subqs_and_subas(
            solution_trace, self.question_index
        )
        io_input = (
            decompose_prompt
            + "\n\n"
            + f"Question {self.question_index}: {user_question}"
            + "\n"
            + existing_subquestions_and_subanswers
            + f"Question {self.question_index}.{next_subquestion_id}:"
        )
        print("Action A3: Generate Subquestion")
        io_output_list = self.io.generate(
            io_input,
            max_tokens=128,
            num_return=self.num_subquestions,
            stop_tokens=[
                "\n",
                "\n\n",
                "Answer",
                "Answer ",
                f"Answer {self.question_index}.{next_subquestion_id}",
                f"Answer {self.question_index}.{next_subquestion_id}:",
                f"Answer {self.question_index}.{next_subquestion_id}: ",
            ],
        )

        # subquestion_list = [io_output.split("?")[0] + "?" for io_output in io_output_list]  # cleaning, you might wanna modify this
        subquestion_list = [o.strip() for o in io_output_list]

        #! generate subanswers to the subquestions generated above
        io_input_list = []
        for subquestion in subquestion_list:
            io_input = (
                decompose_prompt
                + "\n\n"
                + f"Question {self.question_index}: {user_question}"
                + "\n"
                + existing_subquestions_and_subanswers
                + f"Question {self.question_index}.{next_subquestion_id}: "
                + subquestion
                + "\n"
                + f"Answer {self.question_index}.{next_subquestion_id}:"
            )
            io_input_list.append(io_input)

        if reach_terminal_subquestion(subquestion=subquestion, user_question=user_question):
            num_return = self.mcts_num_last_votes
        else:
            num_return = self.num_votes

        print("Action A3: Generate answer to subquestions")
        io_output_list = self.io.generate(
            io_input_list,
            max_tokens=512,
            num_return=num_return,
            stop_tokens=[
                "\n",
                "\n\n",
                f"Question {self.question_index}.{next_subquestion_id + 1}",
            ],
        )
        cleaned_io_output_list = [
            [io_output.strip() for io_output in io_output_group] for io_output_group in io_output_list
        ]

        for i, cleaned_io_output_group in enumerate(cleaned_io_output_list):
            try:
                most_likely_answer, likelihood = self._get_most_likely_answer(cleaned_io_output_group)
            except Exception as e:
                raise GeneratorError(
                    source="generate answer to subquestions",
                    io_input=io_input_list[i],
                    io_output_list=cleaned_io_output_group,
                )
            subanswer_list.append(most_likely_answer)
            value_list.append(likelihood)

        assert len(subquestion_list) == len(subanswer_list) == len(value_list)

        #! generate potential answer to the user question
        potential_answers_list: List[List[str]] = []
        if self.enable_potential_score:
            for subq, suba in zip(subquestion_list, subanswer_list):
                if reach_terminal_subquestion(subq, user_question):
                    potential_answers_list.append(None)
                else:
                    response_prefix = make_response_prefix(
                        solution_trace, Node_Type.SUBQUESTION, new_subq=subq, new_suba=suba
                    )
                    potential_score_input = "Question: " + user_question + "\nAnswer: " + response_prefix
                    
                    print("Generate potential answers for question")
                    potential_score_output = self.io.generate(
                        potential_score_input,
                        num_return=self.num_votes,
                        max_tokens=128,
                        stop_tokens=self.fewshot_cot_config["stop_tokens"],
                    )
                    potential_score_input2 = [
                        "Question: "
                        + user_question
                        + "\nAnswer: "
                        + response_prefix
                        + z
                        + "\nTherefore, the answer (arabic numerals) is"
                        for z in potential_score_output
                    ]
                    print("Clean potential answer:")
                    cleaned_io_output_list = self.io.generate(
                        potential_score_input2,
                        num_return=1,
                        max_tokens=128,
                        stop_tokens=self.fewshot_cot_config["stop_tokens"],
                    )
                    cleaned_io_output_list = [z[0] for z in cleaned_io_output_list]

                    potential_answers_list.append(
                        [self.evaluator.extract_answer_from_model_completion(o) for o in cleaned_io_output_list]
                    )
        else:
            potential_answers_list = [None] * len(subquestion_list)

        return subquestion_list, subanswer_list, value_list, potential_answers_list

    def generate_re_subanswers(
        self,
        user_question: str,
        solution_trace: Dict[int, Dict[str, str]],
        paraphrased: bool,
    ):
        re_subanswer_list, value_list = [], []

        user_question_context, _ = split_user_question(user_question)

        last_subquestion_id = int(sorted(solution_trace.keys())[-1])
        last_subquestion = solution_trace[last_subquestion_id]["subquestion"]

        #! few shot cot
        question = (
            f"{user_question_context} {last_subquestion}"
            if not paraphrased
            else f"{user_question_context} Question: {last_subquestion}"
        )
        print("Action A4, Regenerate answer to last subquestion:")
        io_input, cleaned_io_output_list = self._fewshot_cot_answer_question(
            question=question, paraphrased=paraphrased, num_return=self.num_votes
        )
        try:
            most_likely_answer, likelihood = self._get_most_likely_answer(cleaned_io_output_list)
        except Exception as e:
            raise GeneratorError(
                source="generate re-subanswers: few shot cot",
                io_input=io_input,
                io_output_list=cleaned_io_output_list,
            )
        re_subanswer_list.append(most_likely_answer)
        value_list.append(likelihood)

        #! generate potential answer to the user question
        potential_answers_list: List[List[str]] = []
        if self.enable_potential_score:
            solution_trace_copy = deepcopy(solution_trace)
            for re_suba in re_subanswer_list:
                solution_trace_copy[last_subquestion_id]["subanswer"] = {"text": re_suba}
                response_prefix = make_response_prefix(solution_trace_copy, Node_Type.SUBQUESTION)
                potential_score_input = "Question: " + user_question + "\nAnswer: " + response_prefix
                
                print("Generate potential answer to user question:")
                potential_score_output = self.io.generate(
                    potential_score_input,
                    num_return=self.num_votes,
                    max_tokens=128,
                    stop_tokens=self.fewshot_cot_config["stop_tokens"],
                )
                potential_score_input2 = [
                    "Question: "
                    + user_question
                    + "\nAnswer: "
                    + response_prefix
                    + z
                    + "\nTherefore, the answer (arabic numerals) is"
                    for z in potential_score_output
                ]
                
                print("Clean potential answer:")
                cleaned_io_output_list = self.io.generate(
                    potential_score_input2,
                    num_return=1,
                    max_tokens=128,
                    stop_tokens=self.fewshot_cot_config["stop_tokens"],
                )
                cleaned_io_output_list = [z[0] for z in cleaned_io_output_list]

                potential_answers_list.append(
                    [self.evaluator.extract_answer_from_model_completion(o) for o in cleaned_io_output_list]
                )
        else:
            potential_answers_list = [None] * len(re_subanswer_list)

        return re_subanswer_list, value_list, potential_answers_list

    def generate_rephrased_user_question(self, user_question: str):
        rephrased_user_question_list = []
        io_input = self.rephrasing_prompt_template
        io_input += "\n\n"
        io_input += "Original Question: " + user_question + "\n"
        io_input += "Rephrased Question: Given a list of conditions, please answer the question. Condition 1: "
        print("Action A5: Rephrase the question")
        io_output = self.io.generate(model_input=io_input, max_tokens=512, num_return=1, stop_tokens=["\n", "\n\n"])[0]
        io_output = "Given a list of conditions, please answer the question. Condition 1: " + io_output
        rephrased_user_question_list.append(io_output)

        #! generate potential answer to the user question
        potential_answers_list: List[List[str]] = []  # essentially direct answer list
        if self.enable_potential_score:
            response_prefix = make_response_prefix(None, None)
            potential_score_input = "Question: " + rephrased_user_question_list[0] + "\nAnswer: " + response_prefix
            print("Generate potential answer to user question:")
            potential_score_output = self.io.generate(
                potential_score_input,
                num_return=self.num_votes,
                max_tokens=128,
                stop_tokens=self.fewshot_cot_config["stop_tokens"],
            )
            potential_score_input2 = [
                "Question: "
                + rephrased_user_question_list[0]
                + "\nAnswer: "
                + response_prefix
                + z
                + "\nTherefore, the answer (arabic numerals) is"
                for z in potential_score_output
            ]
           
            print("Clean potential answer:")
            cleaned_io_output_list = self.io.generate(
                potential_score_input2, num_return=1, max_tokens=128, stop_tokens=self.fewshot_cot_config["stop_tokens"]
            )
            cleaned_io_output_list = [z[0] for z in cleaned_io_output_list]

            potential_answers_list.append(
                [self.evaluator.extract_answer_from_model_completion(o) for o in cleaned_io_output_list]
            )
        else:
            potential_answers_list = [None] * len(rephrased_user_question_list)

        return rephrased_user_question_list, potential_answers_list

    def generate_ost_step(
        self,
        user_question: str,
        solution_trace: Dict[int, Dict[str, str]],
        paraphrased: bool,
        parent_is_subquestion: bool,
    ):
        ost_step_list = []
        prompt_type = "new"
        if parent_is_subquestion:
            existing_ost_steps, next_ost_step_id = concat_subqs_subas_as_ost_steps(solution_trace)
        else:
            existing_ost_steps, next_ost_step_id = concat_ost_steps(solution_trace, prompt_type="new")
       
        
        ost_prompt = self.fewshot_ost_config["prompt_template"].format(
                examples=self.fewshot_ost_prompt if not paraphrased else self.fewshot_ost_prompt_rephrased,
                instruction=user_question,
            )
        io_input = ost_prompt + existing_ost_steps + "\n" + "Next step: " 
        print("Input for OST step generation: ", io_input)
        #print("Action A1: Generate a one step thought")
        
        io_output_list = self.io.generate(
            model_input=io_input, max_tokens=256, num_return=self.num_a1_steps, stop_tokens=["\n","\n\n"] 
        )
        ost_step_list = [io_output.strip() for io_output in io_output_list]
        

        #! generate potential answer to the user question
        potential_answers_list: List[List[str]] = []  # essentially direct answer list
        if self.enable_potential_score:
            for ost_step in ost_step_list:
                response_prefix = make_response_prefix(solution_trace, Node_Type.OST_STEP, new_ost_step=ost_step)

                potential_score_input = "Question: " + user_question + "\nAnswer: " + response_prefix
                #print("Generate potential answer to user question:")

                potential_score_output = self.io.generate(
                    potential_score_input,
                    num_return=self.num_votes,
                    max_tokens=128,
                    stop_tokens=self.fewshot_cot_config["stop_tokens"],
                )
                potential_score_input2 = [
                    "Question: "
                    + user_question
                    + "\nAnswer: "
                    + response_prefix
                    + z
                    + "\nTherefore, the answer (arabic numerals) is"
                    for z in potential_score_output
                ]
                
               
                #print("Clean potential answer")
                cleaned_io_output_list = self.io.generate(
                    potential_score_input2,
                    num_return=1,
                    max_tokens=128,
                    stop_tokens=self.fewshot_cot_config["stop_tokens"],
                )
                
                cleaned_io_output_list = [z[0] for z in cleaned_io_output_list]

                potential_answers_list.append(
                    [self.evaluator.extract_answer_from_model_completion(o) for o in cleaned_io_output_list]
                )
        else:
            potential_answers_list = [None] * len(ost_step_list)

        print("Generated OST Step: ", ost_step_list[0])
        return ost_step_list, potential_answers_list

    def generate_query(self,
        user_question: str,
        solution_trace: Dict[int, Dict[str, str]],
        paraphrased: bool,
        parent_is_subquestion: bool
        ):
        
        # The list of queries to be generated using the existing OST steps 
        query_generation_list = []

        if parent_is_subquestion:
            existing_ost_steps, next_ost_step_id = concat_subqs_subas_as_ost_steps(solution_trace) #Case: All nodes before this node are subqs_subas.
        else:
            intermediate_sol, _ = concat_ost_steps_querygeneration(solution_trace) #existing_ost_steps = concat_ost_steps(solution_trace)  #Case: Concat all the ost steps before current step (including retrieval steps). Does not use the subquestion, subanswer or query steps directly. 
        
        #print("Intermediate solution used in query generation action:\n", intermediate_sol)
        
        io_input = self.fewshot_querygen_config["prompt_template"].format(
                examples=self.fewshot_querygen_prompt if not paraphrased else self.fewshot_ost_prompt_rephrased,
                question=user_question,
            )
       
        io_input += f"\nIntermediate solution: {intermediate_sol}\nQuery for Retrieval:"    
        

        print("Input for query generation step:", io_input)
        #ch = input("Pause - Enter a character to continue")

       
        io_output_list = self.io.generate(
            model_input=io_input, max_tokens=1024, num_return=self.qg_nodes, stop_tokens=self.fewshot_querygen_config["stop_tokens"]  
        )
        
        query_generation_list = [io_output.strip() for io_output in io_output_list]

        #! generate potential answer to the user question
        potential_answers_list: List[List[str]] = []  # essentially direct answer list
        if self.enable_potential_score:
            for query in query_generation_list:
                response_prefix = make_response_prefix(solution_trace, Node_Type.QUERY_GENERATION, new_ost_step=query) # Get answer using the query as the new step 
                potential_score_input = "Question: " + user_question + "\nAnswer: " + response_prefix
                print("Generate potential answer to user question:")

                potential_score_output = self.io.generate(
                    potential_score_input,
                    num_return=self.num_votes,
                    max_tokens=128,
                    stop_tokens= self.fewshot_querygen_config["stop_tokens"],
                )
                potential_score_input2 = [
                    "Question: "
                    + user_question
                    + "\nAnswer: "
                    + response_prefix
                    + z
                    + "\nTherefore, the answer (arabic numerals) is"
                    for z in potential_score_output
                ]
                
               
                print("Clean potential answer")
                cleaned_io_output_list = self.io.generate(
                    potential_score_input2,
                    num_return=1,
                    max_tokens=128,
                    stop_tokens=self.fewshot_querygen_config["stop_tokens"],
                )
                
                cleaned_io_output_list = [z[0] for z in cleaned_io_output_list]

                potential_answers_list.append(
                    [self.evaluator.extract_answer_from_model_completion(o) for o in cleaned_io_output_list]
                )
        else:
            potential_answers_list = [None] * len(query_generation_list)
        
        print("Query generated list: ", query_generation_list)
        return query_generation_list, potential_answers_list


    # Function to perform self-summarization on retrieved theorem 
    def summarize_llm(self, theorem):
        prompt = prompt_summarize
        
        prompt += f"\n\nTheorem:{theorem}\nNatural language theorem:"
        output = self.io.generate(
                prompt,
                num_return=1,
                max_tokens=512,
                stop_tokens=["NO_STOPWORDS"],
            )
        print("Input theorem:", theorem)
        print("Processed simple NL format", output[0])    
        return output[0]    

    
    # Function to perform self-reflection on retriebed theorem 
    def selfreason_relevance(self, nl_theorem, solution_trace, paraphrased, user_question):
        
        # Extract the intermediate solution till this current point 
        intermediate_sol, _ = concat_ost_steps_querygeneration(solution_trace) #existing_ost_steps = concat_ost_steps(solution_trace)  #Case: Concat all the ost steps before current step (including retrieval steps). Does not use the subquestion, subanswer or query steps directly. 
        
        # Get the prompt for self reasoning using an LLM
        io_input = self.selfreason_relevance_config["prompt_template"].format(
                examples=self.fewshot_selfreason_relevance_prompt if not paraphrased else self.fewshot_ost_prompt_rephrased,
                question=user_question,
            )
        
        io_input += f"\nIntermediate solution: {intermediate_sol}\nRetrieved Document: {nl_theorem}\nRelevant:"    
        output = self.io.generate(
                io_input,
                num_return=1,
                max_tokens=200,
                stop_tokens= self.selfreason_relevance_config["stop_tokens"],
            )
        
        output = output[0]
        #print("Output from self reasoning LLM: ", output)
        
        # Extract the relevance label provided by LLM 
        label_index = output.find("Relevant:")
        explanation_index = output.find("Reason:")
        nextquestion_index = output.find("Question:")
        
        # Extract the reason of relevance provided by the LLM
        label = output[label_index+len("Relevant:"):explanation_index]
        reason = output[explanation_index:nextquestion_index] 
        
        print("Question: ", user_question)
        print("Intermediate solution: ", intermediate_sol)
        print("Natural language theorem retrieved: ", nl_theorem)
        print("Relevance Label for theorem: ", label)
        print("Reason for Label for theorem: ", reason)
        print("*"*50)
        return label, reason 

    
    def retrieve(self,
        user_question: str,
        solution_trace: Dict[int, Dict[str, str]],
        reasoning_query,
        thread_id = None,
        node_count = 0,
        use_gold_documents = None ,
        theorem_ids = [],
        paraphrased = None,
        task = None
        ):

        retriever_name = None 

        # Get the id of the question
        if thread_id in self.id:
            question_id = self.id[thread_id]
        else:
            question_id = None

        labels_list = [] 
        reasons_list = []
        
        if use_gold_documents is True: # Do not perform retrieval. Instead use gold documents from the dataset (Only BRIGHT now)
            
            retrieved_docs_list = []
            
            # Get the examples
            examples = self.examples
            ids = [examples[j]['id'] for j in range(len(examples))]
            ground_truth_docs = []
            doc_ids = []

            if task == "theoremqa_questions":
                ground_truth_docs_old = self.similarq_gold_docs[question_id]
                print("All Ground truth docs:", ground_truth_docs_old)
                # doc_ids = ['0']*len(ground_truth_docs)
                i = 0
                for d in ground_truth_docs_old:
                    indx = self.documents.index(d) 
                    print("Index of gold doc", indx)
                    if str(indx) in theorem_ids: # Already in theorem_ids 
                        i+=1
                        continue
                    else:
                        doc_ids.append(str(indx))
                        ground_truth_docs.append(ground_truth_docs_old[i])
                        i+=1
                
            else:
                    try:
                        indx = ids.index(question_id) 
                    except ValueError:
                        print("Element not found in the list.") # Element not in the list

                    e = examples[indx]
                    for gid in e["gold_ids"]:

                        if gid in theorem_ids: # If the gold doc id is already in the theorem ids, then continue 
                            continue 
                        else: # If the gold doc id is not in theorem ids 
                            doc_ids.append(gid)
                            ground_truth_docs.append(self.documents[int(gid)]) # Get the content of the gold documents 
                    
                
                
            
            retrieved_docs_list = [self.summarize_llm(clean(doc)) for doc in ground_truth_docs] # The gold documents are the retrieved docs  
            print("Gold theorems list:\n", retrieved_docs_list)
            dict_retrieval_scores = {}  # No scores since no retrieval is performed 
        
        
        
        else: # Perform retrieval using the generated query from parent QG Node  
            
            if isinstance(reasoning_query, list):
                reasoning_query = reasoning_query[0]

            retrieved_docs_list, dict_retrieval_scores, doc_ids, retriever_name = retrieve_theorems(user_question, reasoning_query, question_id, self.documents, self.doc_emb, node_count, self.directory_path, self.topk_rt, self.retriever, self.threshold) 
            new_doc_ids = []
            new_retrieved_docs_list = []
            
            for i in range(len(retrieved_docs_list)):
                d = doc_ids[i]
                if d not in theorem_ids:
                    new_doc_ids.append(d)
                    new_retrieved_docs_list.append(retrieved_docs_list[i])

            doc_ids = [] 
            retrieved_docs_list = [] 
            doc_ids = new_doc_ids   
            retrieved_docs_list = new_retrieved_docs_list
            retrieved_docs_list = [self.summarize_llm(io_output.strip()) for io_output in retrieved_docs_list]
            #print("Retrieved theorems list", retrieved_docs_list)

            # Perform `self-reasoning` on the retrieved docs 
            if self.retrieval_selfreasoning is True:
                for theorem in retrieved_docs_list:
                    relevance_label, reason =  self.selfreason_relevance(theorem, solution_trace, paraphrased, user_question)
                    labels_list.append(relevance_label)
                    reasons_list.append(reason)

        
        #! generate potential answer to the user question
        potential_answers_list: List[List[str]] = []  # essentially direct answer list
        if self.enable_potential_score:
            for doc in retrieved_docs_list:
                response_prefix = make_response_prefix(solution_trace, Node_Type.RETRIEVAL, new_ost_step=doc) # Get answer using the theorem as the new step 
                potential_score_input = "Question: " + user_question + "\nAnswer: " + response_prefix
                print("Generate potential answer to user question:")

                potential_score_output = self.io.generate(
                    potential_score_input,
                    num_return=self.num_votes,
                    max_tokens=128,
                    stop_tokens= self.fewshot_querygen_config["stop_tokens"],
                )

                potential_score_input2 = [
                    "Question: "
                    + user_question
                    + "\nAnswer: "
                    + response_prefix
                    + z
                    + "\nTherefore, the answer (arabic numerals) is"
                    for z in potential_score_output
                ]
                
               
                print("Clean potential answer")
                cleaned_io_output_list = self.io.generate(
                    potential_score_input2,
                    num_return=1,
                    max_tokens=128,
                    stop_tokens=self.fewshot_querygen_config["stop_tokens"],
                )
                
                cleaned_io_output_list = [z[0] for z in cleaned_io_output_list]

                potential_answers_list.append(
                    [self.evaluator.extract_answer_from_model_completion(o) for o in cleaned_io_output_list]
                )
        else:
            potential_answers_list = [None] * len(retrieved_docs_list)

        return retrieved_docs_list, potential_answers_list, dict_retrieval_scores, doc_ids, labels_list, reasons_list, retriever_name  

        
class Reasoning_MCTS_Node(MCTS_Node):
    def __init__(
        self,
        parent: "Reasoning_MCTS_Node",
        depth: int,
        node_type: Node_Type,
        thread_id: int = None,
        verbose: bool = False,
        # --- For instantiating root node ---
        node_value: float = None,
        generator: Generator = None,
        disable_a5: bool = None,
        user_question: str = None,
        question_id: int = -1,
        max_depth_allowed: int = None,
        max_qg_nodes: int = None,
        disable_a1: bool = None,
        disable_a3: bool = None,
        disable_a4: bool = None,
        disable_a6: bool = None,
        task: str = None,
        # -----------------------------------
        # --- For instantiating REPHRASED_USER_QUESTION node ---
        rephrased_user_question: str = None,
        # ------------------------------------------------------
        expected_answer: str = None,
        # --- For instantiating DIRECT_ANSWER node ---
        direct_answer: str = None,
        # --------------------------------------------
        # --- For instantiating SUBQUESTION node ---
        subquestion: str = None,
        subanswer: str = None,
        is_new_subquestion: bool = None,
        # ------------------------------------------
        # --- For instantiating RE_SUBANSWER node ---
        re_subanswer: str = None,
        # -------------------------------------------
        # --- For instantiating QUERY_GENERATION node ---
        query_generation: str = None,
        candidate_query_info: dict = None,
        queryidx_candidate_query: int = None,
        precond_thm: str = None, 
        precond_ques: str = None,
        thm_subject: str = None,  
        # -------------------------------------------
        # --- For instantiating RETRIEVAL node ---
        retrieval: str = None,
        theorem_no: str = None,
        theorem_ids = [],  # List of ids of the retrieved theorems which are added in the solution trace (To prevent same theorems to be repeated)
        dict_retrieval_scores: dict = None,
        use_gold_documents: bool = None, 
        retriever_name: str = None, 
        # -------------------------------------------
        # --- For instantiating OST_STEP node ---
        ost_step: str = None,
        # ---------------------------------------
        # --- For node selection (not in sanity checks yet) ---
        enable_potential_score: bool = None,
        potential_answers: List[str] = None,
    ) -> None:
        """params:
        subquestion: the node is proposing a new subquestion
        subanswer: the answer corresponding to the new subquestion the node proposed
        re_subanswer: the node is proposing a new subanswer to the parent's subquestion
        """
        super().__init__()

        #! sanity checks
        try:
            assert depth is not None
            assert node_type is not None
            # if node_value is not None:    
            #     assert node_value > 0, breakpoint()

            if node_type is Node_Type.USER_QUESTION:
                assert depth == 0
                assert all(
                    attr is None
                    for attr in [
                        parent,
                        node_value,
                        rephrased_user_question,
                        direct_answer,
                        subquestion,
                        subanswer,
                        is_new_subquestion,
                        re_subanswer,
                        ost_step,
                        dict_retrieval_scores
                    ]
                )
                assert all(
                    attr is not None
                    for attr in [generator, disable_a5, user_question, expected_answer, max_depth_allowed, max_qg_nodes, disable_a1, disable_a3, disable_a4]
                )
            elif node_type is Node_Type.REPHRASED_USER_QUESTION:
                assert depth == 1
                assert all(
                    attr is None
                    for attr in [
                        node_value,
                        generator,
                        disable_a5,
                        user_question,
                        expected_answer,
                        direct_answer,
                        subquestion,
                        subanswer,
                        is_new_subquestion,
                        re_subanswer,
                        ost_step,
                        max_depth_allowed,
                        disable_a1,
                        disable_a3, 
                        disable_a4,
                        dict_retrieval_scores
                    ]
                )
                assert all(attr is not None for attr in [parent, rephrased_user_question])
            elif node_type is Node_Type.DIRECT_ANSWER:
                assert depth > 0
                assert all(
                    attr is None
                    for attr in [
                        generator,
                        disable_a5,
                        user_question,
                        expected_answer,
                        subquestion,
                        subanswer,
                        is_new_subquestion,
                        re_subanswer,
                        ost_step,
                        max_depth_allowed,
                        disable_a1,
                        disable_a3, 
                        disable_a4, 
                        dict_retrieval_scores
                    ]
                )
                assert all(attr is not None for attr in [parent, node_value, direct_answer])
            elif node_type is Node_Type.SUBQUESTION:
                assert depth > 0
                assert all(
                    attr is None
                    for attr in [
                        generator,
                        disable_a5,
                        user_question,
                        expected_answer,
                        direct_answer,
                        re_subanswer,
                        ost_step,
                        max_depth_allowed,
                        disable_a1,
                        disable_a3, 
                        disable_a4, 
                        dict_retrieval_scores
                    ]
                )
                assert all(
                    attr is not None for attr in [parent, node_value, subquestion, subanswer, is_new_subquestion]
                )
            elif node_type is Node_Type.RE_SUBANSWER:
                assert depth > 0
                assert all(
                    attr is None
                    for attr in [
                        generator,
                        disable_a5,
                        user_question,
                        expected_answer,
                        direct_answer,
                        subquestion,
                        subanswer,
                        is_new_subquestion,
                        ost_step,
                        max_depth_allowed,
                        disable_a1,
                        disable_a3, 
                        disable_a4,
                        dict_retrieval_scores 
                    ]
                )
                assert all(attr is not None for attr in [parent, node_value, re_subanswer])
            elif node_type is Node_Type.OST_STEP:
                assert depth > 0
                assert all(
                    attr is None
                    for attr in [
                        generator,
                        disable_a5,
                        user_question,
                        rephrased_user_question,
                        expected_answer,
                        direct_answer,
                        subquestion,
                        subanswer,
                        is_new_subquestion,
                        re_subanswer,
                        max_depth_allowed,
                        disable_a1,
                        disable_a3, 
                        disable_a4, 
                        dict_retrieval_scores
                    ]
                )
                assert all(attr is not None for attr in [parent, ost_step])

            elif node_type is Node_Type.QUERY_GENERATION:
                assert depth > 0
                assert all(
                    attr is None
                    for attr in [
                        node_value,
                        generator,
                        disable_a5,
                        user_question,
                        rephrased_user_question,
                        expected_answer,
                        direct_answer,
                        subquestion,
                        subanswer,
                        is_new_subquestion,
                        re_subanswer,
                        max_depth_allowed,
                        disable_a1,
                        ost_step,
                        retrieval,
                        disable_a3, 
                        disable_a4, 
                        dict_retrieval_scores

                    ]
                )
                assert all(attr is not None for attr in [parent,query_generation])

            
            elif node_type is Node_Type.RETRIEVAL:
                assert depth > 0
                assert all(
                    attr is None
                    for attr in [
                        node_value,
                        generator,
                        disable_a5,
                        user_question,
                        rephrased_user_question,
                        expected_answer,
                        direct_answer,
                        subquestion,
                        subanswer,
                        is_new_subquestion,
                        re_subanswer,
                        max_depth_allowed,
                        disable_a1,
                        ost_step,
                        query_generation,
                        disable_a3, 
                        disable_a4, 

                    ]
                )
                #assert all(attr is not None for attr in [parent, retrieval, dict_retrieval_scores, retriever_name])

        except AssertionError:
            print(f"Instantiating node with type {node_type} failed!")
            breakpoint()
            exit()

        #! attributes
        self.parent = parent  # if parent is None, then the node is the root
        self.children: List["Reasoning_MCTS_Node"] = []
        self.depth = depth
        self.node_type = node_type
        self.node_value = node_value
        self.direct_answer = direct_answer
        self.subquestion = subquestion
        self.subanswer = subanswer
        self.is_new_subquestion = is_new_subquestion
        self.re_subanswer = re_subanswer
        self.ost_step = ost_step
        
        self.query_generation = query_generation
        self.candidate_query_info = candidate_query_info
        self.queryidx_candidate_query = queryidx_candidate_query
        self.precond_thm = precond_thm
        self.precond_ques = precond_ques
        self.thm_subject = thm_subject

        self.retrieval = retrieval
        self.theorem_no = theorem_no
        self.retriever_name = retriever_name
        self.theorem_ids = theorem_ids

        if parent is None:  # root
            self.verbose = verbose
            self.user_question = user_question
            self.question_id =  question_id
            self.task = task
            self.expected_answer = expected_answer
            self.generator = generator
            self.disable_a5 = disable_a5
            self.question_index = generator.question_index
            self.max_depth_allowed = max_depth_allowed
            self.max_qg_nodes = max_qg_nodes
            self.disable_a1 = disable_a1
            self.disable_a3 = disable_a3
            self.disable_a4 = disable_a4
            self.disable_a6 = disable_a6
            self.enable_potential_score = enable_potential_score
            self.use_gold_documents = use_gold_documents
            self.thread_id = thread_id
            
        
        else:  # inherit from parent
            self.verbose = parent.verbose
            self.user_question = parent.user_question
            self.question_id = parent.question_id
            self.task = parent.task
            self.expected_answer = parent.expected_answer
            self.generator = parent.generator
            self.disable_a5 = parent.disable_a5
            self.question_index = parent.generator.question_index
            self.max_depth_allowed = parent.max_depth_allowed
            self.max_qg_nodes = parent.max_qg_nodes
            self.disable_a1 = parent.disable_a1
            self.disable_a3 = parent.disable_a3
            self.disable_a4 = parent.disable_a4
            self.disable_a6 = parent.disable_a6
            self.enable_potential_score = parent.enable_potential_score
            self.use_gold_documents = parent.use_gold_documents
            self.thread_id = parent.thread_id
            

        #! keep track of paraphrasing
        if node_type is Node_Type.USER_QUESTION:
            self.paraphrased = False
        elif node_type is Node_Type.REPHRASED_USER_QUESTION:
            self.paraphrased = True
            self.user_question = rephrased_user_question
        else:
            assert parent is not None
            self.paraphrased = parent.paraphrased

        #! record number of subquestions till now
        if parent is None:  # root
            self.subquestion_counter = 0
        else:
            if node_type is Node_Type.SUBQUESTION and is_new_subquestion:
                self.subquestion_counter = parent.subquestion_counter + 1
            else:
                self.subquestion_counter = parent.subquestion_counter

        #! record number of one-step thought steps till now
        if parent is None:  # root
            self.ost_step_counter = 0
        else:
            if node_type is Node_Type.OST_STEP:
                self.ost_step_counter = parent.ost_step_counter + 1
            else:
                self.ost_step_counter = parent.ost_step_counter

        
        #! record number of retrieval steps till now 
        if parent is None:
            self.retrieval_step_counter = 0 
        else:
            if node_type is Node_Type.RETRIEVAL:
                self.retrieval_step_counter = parent.retrieval_step_counter + 1 
            else:
                self.retrieval_step_counter = parent.retrieval_step_counter 
        
        
        #! record number of query generation steps till now 
        if parent is None:
            self.query_step_counter = 0 
        else:
            if node_type is Node_Type.QUERY_GENERATION:
                self.query_step_counter = parent.query_step_counter + 1 
            else:
                self.query_step_counter = parent.query_step_counter


        #! record solution trace from root to the current node. key: subquestion id
        if parent is None:  # root
            assert self.node_type is Node_Type.USER_QUESTION
            self.solution_trace: Dict[int, Dict[str, str]] = {0: {"user_question": user_question, "ost_step": {}}}
        else:
            assert self.node_type is not Node_Type.USER_QUESTION
            self.solution_trace = deepcopy(parent.solution_trace)  # Copy solution trace of the parent 

            if node_type is Node_Type.REPHRASED_USER_QUESTION:
                self.solution_trace[0]["user_question"] = rephrased_user_question
            
            elif node_type is Node_Type.DIRECT_ANSWER:
                assert self.subquestion_counter in self.solution_trace.keys()
                assert self.subquestion_counter == parent.subquestion_counter
                self.solution_trace[self.subquestion_counter]["direct_answer"] = {
                    "text": direct_answer,
                    "value": node_value,
                }
            elif node_type is Node_Type.SUBQUESTION:
                assert is_new_subquestion and self.subquestion_counter == parent.subquestion_counter + 1
                self.solution_trace[self.subquestion_counter] = {
                    "subquestion": subquestion,
                    "subanswer": {"text": subanswer, "value": node_value},
                    "ost_step": {},
                }
            
            elif node_type is Node_Type.RE_SUBANSWER:
                assert parent.subquestion is not None
                assert self.subquestion_counter == parent.subquestion_counter
                assert self.solution_trace[self.subquestion_counter]["subquestion"] == parent.subquestion
                self.solution_trace[self.subquestion_counter]["subanswer"] = {"text": re_subanswer, "value": node_value}
            
            elif node_type is Node_Type.OST_STEP:
                assert "ost_step" in self.solution_trace[self.subquestion_counter].keys()
                if self.node_value is not None:
                    self.solution_trace[self.subquestion_counter]["ost_step"][self.ost_step_counter] = [ost_step, self.node_value] 
                else:
                    self.solution_trace[self.subquestion_counter]["ost_step"][self.ost_step_counter] = ost_step
            
            # Add query generation and retrieval as part of the ost steps with identifier 
            elif node_type is Node_Type.QUERY_GENERATION: 
                assert "ost_step" in self.solution_trace[self.subquestion_counter].keys()
                self.solution_trace[self.subquestion_counter]["ost_step"][str(self.ost_step_counter) + "_querygeneration"] = [query_generation, precond_thm, precond_ques, thm_subject, candidate_query_info, queryidx_candidate_query] 
            
            elif node_type is Node_Type.RETRIEVAL:
                assert "ost_step" in self.solution_trace[self.subquestion_counter].keys()
                self.solution_trace[self.subquestion_counter]["ost_step"][str(self.ost_step_counter) + "_retrieval"] = [retrieval, dict_retrieval_scores, retriever_name, theorem_no]

        #! potential_score for intermediate nodes (only used for node selection)
        if self.enable_potential_score:
            self.potential_answers = potential_answers
            self.potential_score = 0
            if parent is None:  # root
                assert self.node_type is Node_Type.USER_QUESTION
                self.potential_answers_history = {}
            else:
                assert self.node_type is not Node_Type.USER_QUESTION
                self.potential_answers_history = deepcopy(parent.potential_answers_history)
                self.potential_answers_history[self.depth] = potential_answers

    def __str__(self) -> str:
        type2str = {
            Node_Type.USER_QUESTION: "U",
            Node_Type.REPHRASED_USER_QUESTION: "RU",
            Node_Type.DIRECT_ANSWER: "DA",
            Node_Type.SUBQUESTION: "SQ",
            Node_Type.RE_SUBANSWER: "RS",
            Node_Type.OST_STEP: "TS",
            Node_Type.QUERY_GENERATION: "QG",
            Node_Type.RETRIEVAL: "RT"

        }
        return f"{type2str[self.node_type]}-{self.id}"

    def _create_children(self):
        def do_action_generate_direct_answers():
            verbose_print(f"---- Generating direct answers for node {self.id}...", self.verbose)

            #! ACTION: generate direct answer for the user question (w/ or w/o hint)
            if (
                self.node_type is not Node_Type.USER_QUESTION
                and self.node_type is not Node_Type.REPHRASED_USER_QUESTION
            ):
                hint = make_hint(self.solution_trace, self.node_type)  # Use retrieval doc as part of input as well 
            else:
                hint = None

            # Add pdb breakpoint here
            # breakpoint()
            (direct_answer_list, value_list) = self.generator.generate_direct_answers(
                user_question=self.user_question, paraphrased=self.paraphrased, hint=hint, gold_answer=self.expected_answer, question=self.user_question, solution_trace=self.solution_trace
            )
            
            for direct_answer, value in zip(direct_answer_list, value_list):
                if np.isnan(value) or value < 0:
                    print("Value is invalid for direcr answer !")
                self.children.append(
                    Reasoning_MCTS_Node(
                        parent=self,
                        depth=self.depth + 1,
                        node_type=Node_Type.DIRECT_ANSWER,
                        node_value=value,
                        direct_answer=direct_answer,
                        theorem_ids = self.theorem_ids
                        
                    )
                )

        def do_action_generate_subquestions():
            verbose_print(f"---- Generating subquestions for node {self.id}...", self.verbose)

            #! ACTION: generate new subquestions
            (subquestion_list, subanswer_list, value_list, potential_answers_list) = (
                self.generator.generate_subquestions(
                    user_question=self.user_question, solution_trace=self.solution_trace, paraphrased=self.paraphrased
                )
            )
            for subquestion, subanswer, value, potential_answers in zip(
                subquestion_list, subanswer_list, value_list, potential_answers_list
            ):
                if np.isnan(value) or value <= 0:
                    value = 0.01
                    # breakpoint()
                self.children.append(
                    Reasoning_MCTS_Node(
                        parent=self,
                        depth=self.depth + 1,
                        node_type=Node_Type.SUBQUESTION,
                        node_value=value,
                        subquestion=subquestion,
                        subanswer=subanswer,
                        is_new_subquestion=True,
                        potential_answers=deepcopy(potential_answers)
                    )
                )

        def do_action_generate_re_subanswers():
            verbose_print(f"---- Generating re-subanswers for node {self.id}...", self.verbose)

            #! ACTION: re-generate subanswers for the previous subquestion
            (re_subanswer_list, value_list, potential_answers_list) = self.generator.generate_re_subanswers(
                user_question=self.user_question,
                solution_trace=self.solution_trace,
                paraphrased=self.paraphrased,
            )
            for re_subanswer, value, potential_answers in zip(re_subanswer_list, value_list, potential_answers_list):
                if np.isnan(value) or value <= 0:
                    breakpoint()
                self.children.append(
                    Reasoning_MCTS_Node(
                        parent=self,
                        depth=self.depth + 1,
                        node_type=Node_Type.RE_SUBANSWER,
                        node_value=value,
                        re_subanswer=re_subanswer,
                        potential_answers=deepcopy(potential_answers)
                    )
                )

        def do_action_generate_rephrased_user_question():
            verbose_print(f"---- Generating rephrased user question for node {self.id}...", self.verbose)

            #! ACTION: generate paraphrased question for the root question
            rephrased_user_question_list, potential_answers_list = self.generator.generate_rephrased_user_question(
                user_question=self.user_question
            )
            for rephrased_user_question, potential_answers in zip(rephrased_user_question_list, potential_answers_list):
                self.children.append(
                    Reasoning_MCTS_Node(
                        parent=self,
                        depth=self.depth + 1,
                        node_type=Node_Type.REPHRASED_USER_QUESTION,
                        rephrased_user_question=rephrased_user_question,
                        potential_answers=deepcopy(potential_answers),
                    )
                )

        def do_action_generate_ost_step(parent_is_subquestion=False):
            verbose_print(f"---- Generating one-step thought steps for node {self.id}...", self.verbose)

            #! ACTION: generate one-step thought step
            ost_step_list, potential_answers_list = self.generator.generate_ost_step(
                user_question=self.user_question,
                solution_trace=self.solution_trace,
                paraphrased=self.paraphrased,
                parent_is_subquestion=parent_is_subquestion,
            )
            
            count = 0
            for ost_step, potential_answers in zip(ost_step_list, potential_answers_list):
                
                # Check if the OST Step is a terminal node, then add a node value attribute which can be the reward 
                if reach_terminal_ost_step(ost_step) is True :
                    likelihood = self.generator.likelihood_answer(ost_step_list, count, self.expected_answer, self.user_question)
                    self.children.append(
                        Reasoning_MCTS_Node(
                            parent=self,
                            depth=self.depth + 1,
                            node_type=Node_Type.OST_STEP,
                            ost_step=ost_step,
                            node_value=likelihood, 
                            potential_answers=deepcopy(potential_answers),
                            theorem_ids = self.theorem_ids
                        )
                    )
                else:
                    self.children.append(
                        Reasoning_MCTS_Node(
                            parent=self,
                            depth=self.depth + 1,
                            node_type=Node_Type.OST_STEP,
                            ost_step=ost_step,
                            potential_answers=deepcopy(potential_answers),
                            theorem_ids = self.theorem_ids
                        )
                    )
                
                count+=1    
            
        def do_action_generate_query(parent_is_subquestion=False):
            verbose_print(f"---- Generating query generation steps for node {self.id}...", self.verbose)

            #! ACTION: Generate a query for retrieval using the previous OST steps 
            query_generation_list, potential_answers_list = self.generator.generate_query(
                user_question=self.user_question,
                solution_trace=self.solution_trace,
                paraphrased=self.paraphrased,
                parent_is_subquestion=parent_is_subquestion,
            )

            
            if self.generator.LLM_candidate_theorems is False:
                
                d = {}
                count = 0
                for query in query_generation_list:
                    d[f"query{count}"] =  {}
                    d[f"query{count}"]["fullquery"] = query
                    count+=1


                potential_answers = None 
                i = 0 
                for query in query_generation_list:
                    self.children.append(
                            Reasoning_MCTS_Node(
                                parent=self,
                                depth=self.depth + 1,
                                node_type=Node_Type.QUERY_GENERATION,
                                query_generation = d[f"query{i}"]["fullquery"], 
                                potential_answers=deepcopy(potential_answers),
                                theorem_ids = self.theorem_ids,
                                candidate_query_info = d,
                                queryidx_candidate_query = i

                            )
                        )

                    i = i + 1 


            elif self.generator.LLM_candidate_theorems is True:
                d = {}  # Dictionary to store the queries 
                count = 0
                for query in query_generation_list:
                    theorems, preconditions, why_preconditions, subjects = get_candidate_theorems(query)
                    
                    for j in range(len(theorems)):
                        
                        thm = theorems[j]
                        precond_thm = None 
                        precond_ques = None 
                        subject_thm = None 

                        try:
                            precond_thm = preconditions[j]
                        except:
                            pass   

                        try:
                            precond_ques = why_preconditions[j]
                        except:
                            pass   

                        try:
                            subject_thm = subjects[j]
                        except:
                            pass 
                        
                        count += 1
                        d[f"query{count}"] =  {}
                        d[f"query{count}"]["candidate_theorem"] = thm 
                        d[f"query{count}"]["preconditions_theorem"] = precond_thm
                        d[f"query{count}"]["preconditions_question"] = precond_ques
                        d[f"query{count}"]["subject_candidatethm"] = subject_thm 
                        d[f"query{count}"]["fullquery"] = query


                i = 1 
                while i<=count : # Iterate over all the queries generated 
                    
                    potential_answers = None 
                    self.children.append(
                        Reasoning_MCTS_Node(
                            parent=self,
                            depth=self.depth + 1,
                            node_type=Node_Type.QUERY_GENERATION,
                            query_generation = d[f"query{i}"]["candidate_theorem"], 
                            potential_answers=deepcopy(potential_answers),
                            theorem_ids = self.theorem_ids,
                            candidate_query_info = d,
                            queryidx_candidate_query = i

                        )
                    )
                    i += 1

        
        def do_action_retrieve_docs(use_gold_documents = False, theorem_ids = []):
            # Use_gold_documents argument: Whether to use available gold documents in the dataset 
             
            if use_gold_documents is True:
                verbose_print(f"---- Using gold documents for node {self.id}...", self.verbose)
                retrieved_list,potential_answers_list, dict_retrieval_scores, doc_ids, labels_list, reasons_list, retriever_name  = self.generator.retrieve(
                    user_question=self.user_question,
                    solution_trace=self.solution_trace,
                    reasoning_query = None,
                    node_count = self.id, 
                    use_gold_documents = use_gold_documents,
                    theorem_ids = theorem_ids,
                    paraphrased=self.paraphrased,
                    thread_id = self.thread_id,
                    task = self.task
                )

            else:
                verbose_print(f"---- Retrieving docs for node {self.id}...", self.verbose)
                reasoning_query = self.solution_trace[self.subquestion_counter]["ost_step"][str(self.ost_step_counter) + "_querygeneration"]
                
                
                #! ACTION: Retrieve docs using the query generated in the previous step 
                retrieved_list, potential_answers_list, dict_retrieval_scores, doc_ids, labels_list, reasons_list, retriever_name  = self.generator.retrieve(
                    user_question=self.user_question,
                    solution_trace=self.solution_trace,
                    reasoning_query = reasoning_query,
                    node_count = self.id,
                    theorem_ids = theorem_ids,
                    thread_id = self.thread_id,
                )
            
            doc_no = 0     
            for doc, potential_answers in zip(retrieved_list, potential_answers_list):
                
                if self.generator.retrieval_selfreasoning is True:
                    
                    # Check the labels of relevance 
                    label = labels_list[doc_no]
                    explanation = reasons_list[doc_no]
                    
                    
                    if "False" in label or "false" in label:
                        # Store the theorem_id, relevance_label, and the explanation as a negative 
                        d = {"question_id": self.question_id, "theorem_id": doc_ids[doc_no], "LLM_relevance_label": 0, "LLM_explanation": explanation}    
                        
                        hard_negatives_dir = os.path.join(self.generator.directory_path, "hard_negatives")
                        os.makedirs(hard_negatives_dir, exist_ok=True)  # Create the directory if it doesn't exist

                        fpath = os.path.join(hard_negatives_dir, f"{str(self.question_id)}.json")
                        
                        with open(fpath, "a") as f:
                            f.write(json.dumps(d) + "\n")
                        
                        continue  # Do not add that retrieval node as children 

                    if "True" in label or "True" in label:  
                        
                        # Store the theorem_id, relevance_label, and the explanation as a positive
                        d = {"question_id": self.question_id, "theorem_id": doc_ids[doc_no], "LLM_relevance_label": 1, "LLM_explanation": explanation}    
                        
                        hard_negatives_dir = os.path.join(self.generator.directory_path, "hard_negatives")
                        os.makedirs(hard_negatives_dir, exist_ok=True)  # Create the directory if it doesn't exist

                        fpath = os.path.join(hard_negatives_dir, f"{str(self.question_id)}.json")
                        
                        with open(fpath, "a") as f:
                            f.write(json.dumps(d) + "\n")

                
                new_thm_ids = theorem_ids.copy()
                new_thm_ids.append(doc_ids[doc_no])
                self.children.append(
                    Reasoning_MCTS_Node(
                        parent=self,
                        depth=self.depth + 1,
                        node_type=Node_Type.RETRIEVAL,
                        dict_retrieval_scores = dict_retrieval_scores, 
                        retrieval = doc, 
                        theorem_no = doc_ids[doc_no],
                        potential_answers=deepcopy(potential_answers),
                        theorem_ids = new_thm_ids,
                        retriever_name = retriever_name
                    )
                )
                doc_no+=1


        #! create children
        if self.node_type is Node_Type.USER_QUESTION:
            # A1: Propose an one-step thought.
            if not self.disable_a1:
                do_action_generate_ost_step()

            # A2: Propose the remaining thought steps
            do_action_generate_direct_answers()

            # A3: Propose next sub-question along with its answer.
            if not self.disable_a3:
                do_action_generate_subquestions()

            # A5: Rephrase the question/sub-question.
            if not self.disable_a5:
                do_action_generate_rephrased_user_question()

            # A6: Generate query for retrieval 
            if not self.disable_a6 and self.use_gold_documents is False:
                if self.query_step_counter < self.max_qg_nodes:  # If number of QUERY_GENERATION steps is less than max allowed
                    print("Self.disable_a6",self.disable_a6)
                    do_action_generate_query()

            if self.use_gold_documents is True: # You can retrieve docs at the root USER QUESTION Node 
                do_action_retrieve_docs(self.use_gold_documents, self.theorem_ids)


        elif self.node_type is Node_Type.REPHRASED_USER_QUESTION:
            # A1: Propose an one-step thought.
            if not self.disable_a1:
                do_action_generate_ost_step()

            # A2: Propose the remaining thought steps
            do_action_generate_direct_answers()

            # A3: Propose next sub-question along with its answer.
            if not self.disable_a3:
                do_action_generate_subquestions()
            
            if not self.disable_a6:
                if self.query_step_counter < self.max_qg_nodes:
                    print("Self.disable_a6",self.disable_a6)
                    do_action_generate_query()        

        elif self.node_type is Node_Type.DIRECT_ANSWER:
            raise ValueError("DIRECT_ANSWER node cannot create children!!")
        elif self.node_type is Node_Type.SUBQUESTION:
            # A1: Propose an one-step thought.
            if not self.disable_a1:
                do_action_generate_ost_step(parent_is_subquestion=True)

            # A2: Propose the remaining thought steps
            do_action_generate_direct_answers()

            # A3: Propose next sub-question along with its answer.
            if not self.disable_a3:
                do_action_generate_subquestions()

            # A4: Answer the sub-question again.
            if not self.disable_a4:
                do_action_generate_re_subanswers()

            # A6: Generate query for retrieval 
            if not self.disable_a6 and self.use_gold_documents is False:
                    if self.query_step_counter < self.max_qg_nodes:
                        do_action_generate_query(parent_is_subquestion=True)  

        elif self.node_type is Node_Type.RE_SUBANSWER:
            # A1: Propose an one-step thought.
            if not self.disable_a1:
                do_action_generate_ost_step(parent_is_subquestion=True)

            # A2: Propose the remaining thought steps
            do_action_generate_direct_answers()
            
            if not self.disable_a3:
                # A3: Propose next sub-question along with its answer.
                do_action_generate_subquestions()

            # A6: Generate query for retrieval 
            if not self.disable_a6:
                if self.query_step_counter < self.max_qg_nodes:
                    do_action_generate_query(parent_is_subquestion=True)    


        elif self.node_type is Node_Type.OST_STEP:
            
            # # A1: Propose an one-step thought.
            if not self.disable_a1:
                do_action_generate_ost_step()
            
            # # A2: Propose the remaining thought steps
            # do_action_generate_direct_answers()

            # A6: Generate query for retrieval 
            #print("Generate query for retrieval")
            if not self.disable_a6 and self.use_gold_documents is False:
                if self.query_step_counter < self.max_qg_nodes:
                    #print("Self.disable_a6",self.disable_a6)
                    do_action_generate_query()  
            
            if len(self.children) == 0:       #It is possible that no children are added to a QUERY_GENERATION NODE (no new theorem added, not passing generator self reasoning/ not correct format of query)
                # Generate a DIRECT ANSWER from the node 
                do_action_generate_direct_answers()

            
            if self.use_gold_documents is True: # You can add gold docs after an OST Step 
                do_action_retrieve_docs(self.use_gold_documents, self.theorem_ids)

            
     

        
        elif self.node_type is Node_Type.QUERY_GENERATION:
            
            # A7: Retrieve using the given query in the QUERY_GENERATION NODE 
            #print("Retrieving docs using query")
            do_action_retrieve_docs(self.use_gold_documents, self.theorem_ids) 
            
            if len(self.children) == 0:       #It is possible that no children are added to a QUERY_GENERATION NODE (no new theorem added, not passing generator self reasoning)
                
                if not self.disable_a1:
                    do_action_generate_ost_step()

                # Generate a DIRECT ANSWER from the node 
                do_action_generate_direct_answers()

        
        elif self.node_type is Node_Type.RETRIEVAL:
            
            # A1: Propose an one-step thought.
            if not self.disable_a1:
                do_action_generate_ost_step()
            
            # A2: Propose the remaining thought steps
            do_action_generate_direct_answers()
        
        assert self.children
        return self.children

    def is_valid_leaf_node(self):
        #! a valid solution can only be in SUBQUESTION type or DIRECT_ANSWER type
        # return (
        #     self.node_type is Node_Type.SUBQUESTION and reach_terminal_subquestion(self.subquestion, self.user_question)
        # ) or self.node_type is Node_Type.DIRECT_ANSWER
        return (
            (
                self.node_type is Node_Type.SUBQUESTION
                and reach_terminal_subquestion(self.subquestion, self.user_question)
            )
           or self.node_type is Node_Type.DIRECT_ANSWER
           or (self.node_type is Node_Type.OST_STEP and reach_terminal_ost_step(self.ost_step))
             
        )


    
    def is_valid_solution_node(self):
        #! a valid solution can only be in SUBQUESTION type or DIRECT_ANSWER type or OST_STEP type 
        return (
            (
                self.node_type is Node_Type.SUBQUESTION
                and reach_terminal_subquestion(self.subquestion, self.user_question)
            )
            or self.node_type is Node_Type.DIRECT_ANSWER
            or (self.node_type is Node_Type.OST_STEP and reach_terminal_ost_step(self.ost_step))
             
        )

    def set_potential_score(self, score: float):
        self.potential_score = score

    def find_children(self, rollout_id: int):
        self.children = self.children or self._create_children()
        for child in self.children:
            child.set_rollout_id(rollout_id)
        assert self.children
        return self.children

    def is_terminal(self):
        return self.depth >= self.max_depth_allowed or self.is_valid_leaf_node()

    def calculate_reward(self):
        if self.is_valid_leaf_node():
            assert self.node_value is not None, breakpoint()
            return self.node_value
        else:
            return 0

    def skip_backprop(self):
        return self.node_type is Node_Type.USER_QUESTION or self.node_type is Node_Type.REPHRASED_USER_QUESTION


def search_for_answers(args, user_question: str, question_id: int, gt_answer: str, generator: Generator, thread_id: int):
    verbose_print(
        f"********************* Searching for answers to question {question_id} ********************* ", args.verbose
    )

    #! build an MCTS searcher
    mcts_searcher = MCTS_Searcher(
        exploration_weight=args.mcts_exploration_weight,
        weight_scheduler=args.mcts_weight_scheduler,
        num_rollouts=args.num_rollouts,
        discount=args.mcts_discount_factor,
        verbose=args.verbose,
    )

    #! build the MCTS tree
    root_node = Reasoning_MCTS_Node(
        parent=None,
        depth=0,
        node_type=Node_Type.USER_QUESTION,
        verbose=args.verbose,
        generator=generator,
        disable_a5=args.disable_a5,
        user_question=user_question,
        question_id = question_id,
        expected_answer=gt_answer,
        max_depth_allowed=args.max_depth_allowed,
        max_qg_nodes = args.max_qg_nodes,
        disable_a1=args.disable_a1,
        disable_a3=args.disable_a3,
        disable_a4=args.disable_a4,
        disable_a6=args.disable_a6,
        enable_potential_score=args.enable_potential_score,
        use_gold_documents=args.use_gold_documents,
        theorem_ids = [],   # Empty list of ids of documents added for retrieval (for Root Node)
        thread_id = thread_id,
        task = args.task
    )

    model_solutions = []
    model_all_solutions = []
    model_rollout_nodes = []
    for i in (pbar := trange(args.num_rollouts, disable=True, position=0)):
        rollout_node = mcts_searcher.do_rollout(root_node, i)
        model_rollout_nodes.append(rollout_node)

        _, best_solution, _, chosen_node, all_solution_nodes, all_solutions = stochastic_find_best_solution(
            root_node, generator.evaluator, enable_potential_score=args.enable_potential_score
        )
        model_solutions.append(best_solution)
        model_all_solutions.append(all_solutions)

        # TODO: potential race condition here
        if args.save_tree:
            with open(
                os.path.join(
                    args.answer_sheets_dir,
                    f"Question {question_id:04d} - Rollout {i}.tree",
                ),
                "w",
            ) as f:
                print_tree_from_root(
                    mcts_searcher=mcts_searcher,
                    rollout_id=i,
                    root_node=root_node,
                    chosen_node=chosen_node,
                    file=f,
                )

    #! record final traces
    js = [{"trace": node.solution_trace, "rollout_id": node.rollout_id} for node in all_solution_nodes]
    with open(os.path.join(args.answer_sheets_dir, f"Question {question_id:04d} - Final Solutions.json"), "w") as f:
        json.dump(js, f)

    js2 = [{"trace": node.solution_trace, "rollout_id": i} for i, node in enumerate(model_rollout_nodes)]
    with open(os.path.join(args.answer_sheets_dir, f"Question {question_id:04d} - Rollout Solutions.json"), "w") as f:
        json.dump(js2, f)

    if args.enable_potential_score:
        js = [node.potential_answers_history for node in all_solution_nodes]
        with open(os.path.join(args.answer_sheets_dir, f"Question {question_id:04d} - Potentials.json"), "w") as f:
            json.dump(js, f)

    return model_solutions, i, model_all_solutions

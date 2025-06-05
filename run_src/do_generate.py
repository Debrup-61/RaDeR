# Licensed under the MIT license.

import sys
import os, json, time
from tqdm import tqdm

sys.path.append(".")

from common.utils import fix_seeds, setup_model_parallel, read_json
from common.arguments import get_parser, post_process_args, save_args
from run_src.rstar_utils import GeneratorError
from MCTS_for_reasoning import Generator, search_for_answers
from eval_src.Evaluator import *



def load_data(data_root: str, dataset_name: str, test_json_filename: str):
    test_file = os.path.join(args.data_root, args.dataset_name, args.test_json_filename + ".json")
    assert os.path.exists(test_file), f"Test file {test_file} does not exist."
    data_item_list = read_json(test_file)

    return data_item_list


def load_generator(api: str, model_ckpt: str, evaluator: Evaluator, seed: int = None, tensor_parallel_size: int = None, half_precision: bool = None):
    if api == "vllm":
        assert seed is not None, "Seed must be provided for vLLM."
        assert tensor_parallel_size is not None, "Tensor parallel size must be provided for vLLM."
        assert half_precision is not None, "Half precision flag must be provided for vLLM."

        from models.vLLM_API import load_vLLM_model
        tokenizer, model = load_vLLM_model(model_ckpt, seed, tensor_parallel_size, half_precision)
   
    elif api == "huggingface":
        from models.HuggingFace_API import load_HF_model

        tokenizer, model = load_HF_model(model_ckpt)
    
    elif api == "gpt3.5-turbo":
        from models.OpenAI_API import load_OpenAI_model

        tokenizer, model = load_OpenAI_model(model_ckpt)
    
    else:
        raise Exception(f"API {api} not supported.")
    
    return Generator(args, tokenizer, model, evaluator)

def main(args):
    fix_seeds(args.seed)
    if args.model_parallel:
        args.local_rank, args.world_size = setup_model_parallel()
    else:
        args.local_rank, args.world_size = 0, 1

    data_item_list = load_data(args.data_root, args.dataset_name, args.test_json_filename)

    evaluator = eval(f"{args.dataset_name}Evaluator()")

    # TODO: we will want to create the generator only once and use across threads
    generator = load_generator(args.api, args.model_ckpt, evaluator, args.seed, args.tensor_parallel_size, args.half_precision)
    print("Generator initialized!", generator)

    total_correct = 0
    total_correct_limit = 0
    num_tested = 0
    start_time = time.time()

    for i, data_item in enumerate(
        (pbar := tqdm(data_item_list, disable=args.local_rank > 0 or args.verbose, position=1))
    ):
        if i < args.start_idx or i >= args.end_idx:
            continue
        

        #problem_id, problem, gt_solution = data_item["id"], data_item["problem"], data_item["solution"]
        problem_id, problem, gt_solution = str(i), data_item["problem"], data_item["solution"]
    
        gt_answer = evaluator.extract_answer_from_gold_solution(gt_solution)
        #gt_answer = data_item["gold_answer"]

        js = {
            "id": problem_id,
            "problem": problem,
            "model_completion": None,
            "model_answer": None,
            "all_model_completions": {},
            "gold_solution": gt_solution,
            "gold_answer": gt_answer,
        }

        model_solutions, stopping_id, model_all_solutions = [], -1, []

        # try:
        generator.id = problem_id   # Id added for retrieval from theorems candidate pool 
        model_solutions, stopping_id, model_all_solutions = search_for_answers(
            args=args, user_question=problem, question_id=i, gt_answer=gt_answer, generator=generator
        )
        
       

        # except GeneratorError as e:
        #     print(e)
        #     js["generator_error"] = {
        #         "source": e.source,
        #         "io_input": e.io_input,
        #         "io_output_list": e.io_output_list,
        #     }
        # except Exception as e:
        #     print(e)
        #     js["other_error"] = {"text": str(e)}

        num_tested += 1

        with open(os.path.join(args.answer_sheets_dir, f"Question {i:04d} - Answer.json"), "w") as f:
            json.dump(js, f)

        #TODO: will need to be careful of potential race conditions for this file
        with open(os.path.join(args.run_outputs_dir, "intermediate_result.txt"), "w") as f:
            f.write(
                f"Total calls: {generator.io.call_counter}, Avg calls: {generator.io.call_counter/(num_tested):.2f}\n"
            )
            f.write(
                f"Total tokens: {generator.io.token_counter}, Avg tokens: {generator.io.token_counter/(num_tested):.2f}\n"
            )

    end_time = time.time()

    print(f"==> Total calls: {generator.io.call_counter}, Avg calls: {generator.io.call_counter/(num_tested):.2f}")
    print(f"==> Total tokens: {generator.io.token_counter}, Avg tokens: {generator.io.token_counter/(num_tested):.2f}")
    print(f"==> Total time: {end_time-start_time:.2f}s, Avg time: {(end_time-start_time)/(num_tested):.2f}s")

    with open(os.path.join(args.run_outputs_dir, "final_result.txt"), "w") as f:
        f.write(f"Total calls: {generator.io.call_counter}, Avg calls: {generator.io.call_counter/(num_tested):.2f}\n")
        f.write(
            f"Total tokens: {generator.io.token_counter}, Avg tokens: {generator.io.token_counter/(num_tested):.2f}\n"
        )
        f.write(f"Total time: {end_time-start_time:.2f}s, Avg time: {(end_time-start_time)/(num_tested):.2f}s\n")


if __name__ == "__main__":
    #! -------------------------------- Arguments --------------------------------
    parser = get_parser()

    parser.add_argument("--num_rollouts", type=int, default=15)
    parser.add_argument(
        "--num_subquestions", type=int, default=3, help="Number of trials for proposing the next subquestion"
    )
    parser.add_argument("--num_votes", type=int, default=10)
    parser.add_argument("--max_depth_allowed", type=int, default=5)

    # MCTS
    parser.add_argument("--mcts_discount_factor", type=float, default=1.0)
    parser.add_argument("--mcts_exploration_weight", type=float, default=2.0)
    parser.add_argument("--mcts_weight_scheduler", choices=["exp", "lin", "const"], default="const")
    parser.add_argument("--mcts_num_last_votes", type=int, default=None)
    parser.add_argument("--save_tree", action="store_true")

    # Action1: Propose an one-step thought.
    parser.add_argument("--num_a1_steps", type=int, default=None)
    parser.add_argument("--disable_a1", action="store_true")
    parser.add_argument("--disable_a3", action="store_true")
    parser.add_argument("--disable_a4", action="store_true")
    parser.add_argument("--disable_a6", action="store_true")
      
    # Paraphrasing
    parser.add_argument("--modify_prompts_for_rephrasing", action="store_true")
    parser.add_argument("--disable_a5", action="store_true")
    
    #! -------------------------- Used for QUERY GENERATION --------------------------
    parser.add_argument("--qg_nodes", type=int, default=1)      # Number of QUERY GENERATION nodes generated from OST Step 
    
    # Top-k parameter for RETRIEVAL 
    parser.add_argument("--use_gold_documents", action="store_true")
    parser.add_argument("--topk_rt", type=int, default=5)
    parser.add_argument("--LLM_candidate_theorems", action='store_true', help="Whether to use LLM generated candidate theorem as query")

    #! ---------- Use `self-reasoning` with generator LLM (do not explore irrelevant theorems in MCTS) ---------------
    parser.add_argument("--retrieval_selfreasoning", action="store_true")

    #! -------------------------- Used for selecting answer --------------------------
    parser.add_argument("--enable_potential_score", action="store_true")

    
    #! ---------------------------Use gold answer to calculate reward ----------------
    parser.add_argument("--bool_goldanswer_reward", action="store_true")
    
    #! ---------------Whether to also check for if the retrieved theorem is applied ----------------
    parser.add_argument("--bool_retrievalreward", action="store_true")
 
    args = parser.parse_args()
    
    if args.mcts_num_last_votes is None:
        args.mcts_num_last_votes = 4     # Number of generations of DIRECT ANSWER to get the likelihood (V) 

    if not args.disable_a1:
        if args.num_a1_steps is None:
            args.num_a1_steps = 1

    args.max_qg_nodes = int(args.max_depth_allowed/2)   # Maximum 1/2 th of the steps can be QG steps    

    #! ----------------------------------------------------------------------------

    prompts_dir = os.path.join(args.prompts_root, args.dataset_name)
    args.fewshot_cot_prompt_path = os.path.join(prompts_dir, "fewshot_cot", "fewshot_cot_prompt_new.txt")
    args.fewshot_cot_config_path = os.path.join(prompts_dir, "fewshot_cot", "fewshot_cot_config_new.json")
    

    args.fewshot_ost_prompt_path = os.path.join(prompts_dir, "fewshot_ost", "fewshot_ost_prompt_new.txt")
    args.fewshot_ost_config_path = os.path.join(prompts_dir, "fewshot_ost", "fewshot_ost_config_new.json")

    args.decompose_template_path = os.path.join(prompts_dir, "decompose", "decompose_template.json")
    args.decompose_prompt_path = os.path.join(prompts_dir, "decompose", "decompose_prompt.txt")
     
    # args.fewshot_querygen_prompt_path = os.path.join(prompts_dir, "fewshot_ost", "fewshot_query_prompt.txt")
    # args.fewshot_querygen_config_path = os.path.join(prompts_dir, "fewshot_ost", "fewshot_query_gen_config.json")
    
    args.fewshot_selfreason_relevance_prompt_path = os.path.join(prompts_dir, "fewshot_ost", "fewshot_selfreason_prompt.txt")
    args.selfreason_relevance_config_path = os.path.join(prompts_dir, "fewshot_ost", "fewshot_selfreason_config.json")

    args.fewshot_retrievalreward_prompt_path = os.path.join(prompts_dir, "fewshot_ost", "fewshot_retrieval_reward_prompt.txt")
    args.fewshot_retrievalreward_config_path = os.path.join(prompts_dir, "fewshot_ost", "fewshot_retrieval_reward_prompt_config.json")
    
    
    if args.LLM_candidate_theorems is True:
        args.fewshot_querygen_prompt_path = os.path.join(prompts_dir, "fewshot_ost", "fewshot_querygenprompt2.txt")
        args.fewshot_querygen_config_path = os.path.join(prompts_dir, "fewshot_ost", "fewshot_query2_config.json")
    else:
        args.fewshot_querygen_prompt_path = os.path.join(prompts_dir, "fewshot_ost", "fewshot_querygen3.txt")
        args.fewshot_querygen_config_path = os.path.join(prompts_dir, "fewshot_ost", "fewshot_query3_config.json")

    if not args.disable_a5:
        args.rephrasing_prompt_template_path = os.path.join(prompts_dir, "rephrasing_prompt_template.txt")
        if args.modify_prompts_for_rephrasing:
            args.fewshot_cot_prompt_rephrased_path = os.path.join(
                prompts_dir, "fewshot_cot", "fewshot_cot_prompt_rephrased.txt"
            )
            args.fewshot_ost_prompt_rephrased_path = os.path.join(
                prompts_dir, "fewshot_ost", "fewshot_ost_prompt_rephrased.txt"
            )
            args.decompose_prompt_rephrased_path = os.path.join(
                prompts_dir, "decompose", "decompose_prompt_rephrased.txt"
            )
        else:
            args.fewshot_cot_prompt_rephrased_path = os.path.join(prompts_dir, "fewshot_cot", "fewshot_cot_prompt.txt")
            args.fewshot_ost_prompt_rephrased_path = os.path.join(prompts_dir, "fewshot_ost", "fewshot_ost_prompt.txt")
            args.decompose_prompt_rephrased_path = os.path.join(prompts_dir, "decompose", "decompose_prompt.txt")

    args = post_process_args(args)
    print(args)
    #ch = input("Enter a key to continue:")
    save_args(args)
    main(args)

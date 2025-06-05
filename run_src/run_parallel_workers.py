import os, sys, json, threading
import concurrent.futures
from tqdm import tqdm

# Ensure the module paths are set up properly.
sys.path.append(".")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(SCRIPT_DIR, "../.env")
from dotenv import load_dotenv
load_dotenv(ENV_PATH)


from common.utils import fix_seeds, setup_model_parallel, read_json
from common.arguments import get_parser, post_process_args, save_args
from run_src.rstar_utils import GeneratorError
from MCTS_for_reasoning import Generator, search_for_answers
from eval_src.Evaluator import *
import time

from models.repllama_server_API import generate_with_repllama_batched
import numpy as np
from datasets import load_dataset
from multiprocessing import Manager


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
    
    elif api == "vllm-server":
        from models.vLLM_server_API import load_vLLM_server

        tokenizer, model = load_vLLM_server(model_ckpt)
    
    else:
        raise Exception(f"API {api} not supported.")
    
    return Generator(args, tokenizer, model, evaluator)


def process_batch(data_item_list, offset, start: int, end: int, generator: Generator, thread_id: int, file_lock, args):
    total_correct = 0
    total_correct_limit = 0
    num_tested = 0
    start_time = time.time()
    print(f"length of data item list on Thread {thread_id}", len(data_item_list))
    generator.similarq_gold_docs = {}

    for i in tqdm(range(start, end), disable=args.verbose, position=thread_id):
        data_item = data_item_list[i]
        #problem_id, problem, gt_solution = str(data_item["id"]), data_item["problem"], data_item["solution"]
        problem_id, problem, gt_solution = str(i), data_item["problem"], data_item["solution"]
        #gt_answer = evaluator.extract_answer_from_gold_solution(gt_solution)
        gt_answer = generator.evaluator.extract_answer_from_gold_solution(gt_solution)
        #gt_answer = str(data_item["gold_answer"])
        #generator.similarq_gold_docs[problem_id] = data_item["gold_contents"]
        
        

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
        generator.id[thread_id] = problem_id   # Id added for retrieval from theorems candidate pool 
        model_solutions, stopping_id, model_all_solutions = search_for_answers(
            args=args, user_question=problem, question_id=offset+i, gt_answer=gt_answer, generator=generator, thread_id=thread_id
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
        
        with open(os.path.join(args.answer_sheets_dir, f"Question {offset+i} - Answer.json"), "w") as f:
            json.dump(js, f)

        # Fix write to file race condition using threading lock
        file_lock.acquire()
        try:
            with open(os.path.join(args.run_outputs_dir, "intermediate_result.txt"), "w") as f:
                f.write(
                    f"Total calls: {generator.io.call_counter}, Avg calls: {generator.io.call_counter/(num_tested):.2f}\n"
                )
                f.write(
                    f"Total tokens: {generator.io.token_counter}, Avg tokens: {generator.io.token_counter/(num_tested):.2f}\n"
                )
        except Exception as e:
            print(f"Error writing to file: {e}")
        finally:
            file_lock.release()

    end_time = time.time()
    if num_tested == 0:
        print("Num tested 0:", start, end)
        print("Len data item list", len(data_item_list))
        #ch = input("Enter key to continue")
    
    print(f"Thread {thread_id} results:")
    if num_tested == 0:
        print("==> No tests were run.")
    else:
        print(f"==> Total calls: {generator.io.call_counter}, Avg calls: {generator.io.call_counter/(num_tested):.2f}")
        print(f"==> Total tokens: {generator.io.token_counter}, Avg tokens: {generator.io.token_counter/(num_tested):.2f}")
        print(f"==> Total time: {end_time-start_time:.2f}s, Avg time: {(end_time-start_time)/(num_tested):.2f}s")

    file_lock.acquire()
    try:
        with open(os.path.join(args.run_outputs_dir, "final_result.txt"), "a") as f:
            f.write(f"Thread {thread_id} results:\n")
            if num_tested == 0:
                f.write("No tests were run.\n")
            else:
                f.write(f"Total calls: {generator.io.call_counter}, Avg calls: {generator.io.call_counter/(num_tested):.2f}\n")
                f.write(f"Total tokens: {generator.io.token_counter}, Avg tokens: {generator.io.token_counter/(num_tested):.2f}\n")
                f.write(f"Total time: {end_time-start_time:.2f}s, Avg time: {(end_time-start_time)/(num_tested):.2f}s\n")
    except Exception as e:
        print(f"Error writing to file: {e}")
    finally:
        file_lock.release()


    
    # print(f"==> Total calls: {generator.io.call_counter}, Avg calls: {generator.io.call_counter/(num_tested):.2f}")
    # print(f"==> Total tokens: {generator.io.token_counter}, Avg tokens: {generator.io.token_counter/(num_tested):.2f}")
    # print(f"==> Total time: {end_time-start_time:.2f}s, Avg time: {(end_time-start_time)/(num_tested):.2f}s")

    # file_lock.acquire()
    # try:
    #     with open(os.path.join(args.run_outputs_dir, "final_result.txt"), "a") as f:
    #         f.write(f"Thread {thread_id} results:\n")
    #         f.write(f"Total calls: {generator.io.call_counter}, Avg calls: {generator.io.call_counter/(num_tested):.2f}\n")
    #         f.write(
    #             f"Total tokens: {generator.io.token_counter}, Avg tokens: {generator.io.token_counter/(num_tested):.2f}\n"
    #         )
    #         f.write(f"Total time: {end_time-start_time:.2f}s, Avg time: {(end_time-start_time)/(num_tested):.2f}s\n")
    # except Exception as e:
    #     print(f"Error writing to file: {e}")
    # finally:
    #     file_lock.release()


def process_threads(data_item_list, offset, process_id: int, file_lock, args):
    

    evaluator = eval(f"{args.dataset_name}Evaluator()")

    generator = load_generator(
        args.api, 
        args.model_ckpt, 
        evaluator, 
        args.seed, 
        args.tensor_parallel_size, 
        args.half_precision
    )
    print("Generator initialized.", generator)

    # break up dataset into chunks for each thread
    total = len(data_item_list)

    chunk_size = total // args.num_workers
    indices = []
    for i in range(args.num_workers):
        t_start = i * chunk_size
        t_end = total if i == args.num_workers - 1 else (i + 1) * chunk_size
        indices.append((t_start, t_end))
    
    
    print("Starting threads on process", process_id)
    #ch = input("Enter character to continue:")
    if args.num_workers > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = []
            t_id = 1
            for start, end in indices:
                futures.append(executor.submit(process_batch, data_item_list, offset, start, end, generator, t_id, file_lock, args))
                t_id += 1
            for i, future in enumerate(futures):
                future.result()
    
    elif args.num_workers == 1:
        # for debugging with no threads
        start = time.time()
        process_batch(data_item_list, offset, 0, total, generator, 1, file_lock, args)

    print(f"Process {process_id} finished in {time.time() - start: .2f}s")
        


def setup_logging():
    import builtins
    import logging

    print("Setting up logging...")
    # Configure your logger as usual – here we simply log to stdout
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(processName)s - %(message)s')

    # Save the original print function
    original_print = builtins.print

    # Override print so it sends everything to logging.info
    def patched_print(*args, **kwargs):
        message = " ".join(str(arg) for arg in args)
        logging.info(message)

    builtins.print = patched_print
    print("Logging setup complete.")

# NOTE: Only works with repllama and vLLM_server APIs
def wait_for_servers(model_ckpt: str):
    #from models.repllama_server_API import client as repllama_client
    #from models.RaDeR_retriever_server_API import ret_client as RaDeR_client
    #from models.vLLM_server_API import client as vLLM_client
    
    repllama_ready = False
    generator_ready = False

    print("Waiting for servers to be ready...")
    while True:
        if not repllama_ready:
            try:
                response = repllama_client.embeddings.create(input=["Test"], model=os.environ["REPLLAMA_MERGED_HUGGINGFACE_PATH"])
                repllama_ready = True
            except Exception as e:
                pass
        elif not generator_ready:
            try:
                response = vLLM_client.chat.completions.create(messages=[{"role": "user", "content": "Test"}], model=model_ckpt)
                generator_ready = True
            except Exception as e:
                pass
        else:
            break

        time.sleep(5)
    
    print("Servers ready.")


def main_parallel():
    # load dataset
    doc_pairs = load_dataset('xlangai/bright', 'documents',cache_dir=args.cache_dir)[args.task]
    # Get the ids and the content for the documents
    documents = []
    doc_ids = []
    for dp in doc_pairs:
        documents.append(dp['content'])
        doc_ids.append(dp['id'])
    
    # save documents to memmap file
    doc_cache_file = os.path.join(args.cache_dir, 'BRIGHT_documents.npy')
    if not os.path.exists(doc_cache_file):
        print("saving to", doc_cache_file)
        
        # save documents to file
        np.save(doc_cache_file, np.array(documents, dtype=object))

        # save doc ids as well
        doc_id_file = os.path.join(args.cache_dir, 'BRIGHT_doc_ids.npy')
        np.save(doc_id_file, np.array(doc_ids, dtype=int)) 
    
    # create embeddings and store in memmap file
    emb_cache_file = os.path.join(args.cache_dir, f'0.npy')
    if not os.path.exists(emb_cache_file):
        print("Creating document embeddings...")
        doc_emb = generate_with_repllama_batched([f"Document: {document}</s>" for document in documents], max_threads=20)

        print(f"Saving to {emb_cache_file}...")
        np.save(emb_cache_file, doc_emb)
        del doc_emb

    del documents
    del doc_ids

    print("Loading data...")
    test_file = os.path.join(args.data_root, args.dataset_name, args.test_json_filename + ".json")
    assert os.path.exists(test_file), f"Test file {test_file} does not exist."
    data_item_list = read_json(test_file)
    
    #data_item_list = data_item_list[args.start_idx:args.end_idx]
    # break up dataset into chunks for processes
    total = len(data_item_list)
    chunk_size = total // args.num_cpu
    print("Total examples in dataset:", total)
    print("Num cpus:", args.num_cpu)
    print("Chunk size", chunk_size)

    indices = []
    for i in range(args.num_cpu):
        start = i * chunk_size
        end = total if i == args.num_cpu - 1 else (i + 1) * chunk_size
        indices.append((start, end))
    
    
    # wait for servers to be set up
    #wait_for_servers(args.model_ckpt)
    
    setup_logging()
    print("Starting processes...")
    if args.num_cpu > 1:
        with Manager() as manager:
            file_lock = manager.Lock()
            with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_cpu) as p_executor:
                futures = []
                id = 1
                for start, end in indices:
                    # each process will get a specific chunk of the data
                    futures.append(p_executor.submit(process_threads, data_item_list[start: end], start, id, file_lock, args))
                    id += 1
                
                del data_item_list # don't need it in parent process
                for i, future in enumerate(futures):
                    future.result()
    
    elif args.num_cpu == 1:
        # for debugging with no child processes
        file_lock = threading.Lock()
        process_threads(data_item_list, 0, 1, file_lock, args)
    
    print("Parallel MCTS generation complete.")


if __name__ == "__main__":
    parser = get_parser()
    
    parser.add_argument('--task', type=str, default='theoremqa_theorems')


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
    parser.add_argument("--retriever", type=str, default="repllama")
    parser.add_argument("--threshold", type=int, default=0.7)
    parser.add_argument("--LLM_candidate_theorems", action='store_true', help="Whether to use LLM generated candidate theorem prompts for query generation")

    #! ---------- Use `self-reflection/self-reasoning` with generator LLM (do not explore irrelevant theorems in MCTS) ---------------
    parser.add_argument("--retrieval_selfreasoning", action="store_true")

    #! -------------------------- Used for selecting answer --------------------------
    parser.add_argument("--enable_potential_score", action="store_true")

    #! ---------------------------Use gold answer to calculate reward ----------------
    parser.add_argument("--bool_goldanswer_reward", action="store_true")
    
    #! ---------------Whether to also check for if the retrieved theorem is applied ----------------
    parser.add_argument("--bool_retrievalreward", action="store_true")

    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--num_cpu", type=int, default=1)
 
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

    fix_seeds(args.seed)
    
    # start threads
    main_parallel()

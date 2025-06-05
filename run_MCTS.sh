#!/bin/bash

#SBATCH -p superpod-a100 # Submit job to to gpu partition superpod-a100, gpu, gypsum-gpu, l40s
#SBATCH -t 06-00:00:00   # Set max job time for 30 days
#SBATCH --gpus=2       # Request access to GPU
#SBATCH --cpus-per-task=45    # Request 50 CPUs
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --mem=1000G     # Memory allocated
#SBATCH -o slurm-%j.out  # %j = job ID 

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
set -a
source "$SCRIPT_DIR/.env"
set +a


set -x
source ~/.bashrc
source RaDeR_env/bin/activate

huggingface-cli login --token $HUGGING_FACE_HUB_TOKEN

NUM_CPUS=${SLURM_CPUS_PER_TASK}  # Retrieve allocated CPUs
echo "Running with $NUM_CPUS CPUs"

#Start BM25 server in background as retriever in MCTS 
python models/BM25_server_API.py > BM25_server.log 2>&1 &
SERVER3_PID=$!  # Store process ID

# Start RaDer/Repllama server in the background as retriever in MCTS
#./servers/rader_server.sh RaDeR/merged_retriever_Qwen-2.5-7B-Instruct_MATH_questionpartialsol_and_LLMquery_full > servers/RaDeR_server.log 2>&1 &

./servers/repllama_server.sh > servers/RepLLama_server.log 2>&1 &
SERVER1_PID=$!  # Store process ID

# Start generator server in the background
./servers/generator_server.sh Qwen/Qwen2.5-7B-Instruct > servers/generator_server.log 2>&1 &
SERVER2_PID=$!  # Store process ID

# Wait for servers to start (Adjust as needed)
sleep 600

# Check if both servers are running
if ps -p $SERVER1_PID > /dev/null && ps -p $SERVER2_PID > /dev/null; then
    echo "Both servers started successfully. Proceeding with the Python script..."
else
    echo "Error: One or both servers failed to start. Exiting..."
    exit 1
fi



# Data generation command using MCTS rewards (based on gold answer)
python run_src/run_parallel_workers.py \
    --dataset_name MATH \
    --test_json_filename shuffled_train \
    --api vllm-server \
    --model_ckpt Qwen/Qwen2.5-7B-Instruct \
    --note TEST_MCTS_MATH_repllama \
    --retriever repllama \
    --max_depth_allowed 6 \
    --num_a1_steps 2 \
    --num_rollouts 16 \
    --disable_a5 \
    --save_tree \
    --disable_a3 \
    --disable_a4 \
    --bool_goldanswer_reward \
    --LLM_candidate_theorems \
    --num_workers 10 \
    --retrieval_selfreasoning \
    --run_outputs_root outputs_MCTS/run_outputs \
    --cache_dir BRIGHT_cache \
    --num_cpu 16



# # Evaluation command using MCTS with gold docs 
# python run_src/run_parallel_workers.py \
#     --dataset_name BRIGHT \
#     --task theoremqa_questions \
#     --test_json_filename similarqs \
#     --api vllm-server \
#     --model_ckpt Qwen/Qwen2.5-7B-Instruct \
#     --note BRIGHT_evaluation_similarq_RaDeR_golddocuments_final \
#     --max_depth_allowed 6 \
#     --num_a1_steps 2 \
#     --disable_a5 \
#     --save_tree \
#     --num_rollouts 16 \
#     --disable_a3 \
#     --disable_a4 \
#     --use_gold_documents \
#     --num_workers 4 \
#     --run_outputs_root outputs_MCTS/run_outputs \
#     --cache_dir cache/doc_emb/RaDeR_trained_retriever/theoremqa_questions/long_False_1 \
#     --num_cpu 45 \
#     --verbose        


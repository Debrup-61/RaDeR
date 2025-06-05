#!/bin/bash

#SBATCH -p superpod-a100  # Submit job to GPU partition
#SBATCH -t 04-00:00:00    # Set max job time for 3 days
#SBATCH --gpus=1          # Request access to 1 GPU
#SBATCH --constraint=a100
#SBATCH --mem=1000G       # Memory allocated
#SBATCH -o slurm-%j.out   # %j = job ID
#SBATCH --cpus-per-task=50 

MODEL_NAME=$1  # The first argument passed to the script

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
set -a
source "$SCRIPT_DIR/.env"
set +a

set -x
source ~/.bashrc
source 
huggingface-cli login --token $HUGGING_FACE_HUB_TOKEN

if [[ -z "$CONDA_DEFAULT_ENV" ]]; then
    echo "No Conda environment is currently activated."
else
    echo "Conda environment '$CONDA_DEFAULT_ENV' is currently activated."
fi


if [ "$MODEL_NAME" = "RaDeR" ]; then
  ./servers/rader_server.sh "$RaDeR_MERGED_HUGGINGFACE_PATH" \
    > RaDeR_retriever_vllm_server.log 2>&1 &
  sleep 200  
  python get_document_embeddings.py --task theoremqa_theorems --model_name RaDeR_model_name --model_hf_dir $RaDeR_MERGED_HUGGINGFACE_PATH --cache_dir BRIGHT_cache/doc_emb
elif [ "$MODEL_NAME" = "RepLLama" ]; then
  ./servers/repllama_server.sh "$REPLLAMA_MERGED_HUGGINGFACE_PATH" \
    > RepLLama_retriever_vllm_server.log 2>&1 &
  sleep 20 
  python get_document_embeddings.py --task theoremqa_theorems --model_name RepLLama --model_hf_dir $REPLLAMA_MERGED_HUGGINGFACE_PATH --cache_dir BRIGHT_cache/doc_emb
  
else
  echo "Unsupported MODEL_NAME: $MODEL_NAME"
fi



#DATASETS=('theoremqa_theorems' 'theoremqa_questions' 'pony' 'biology' 'earth_science' 'economics' 'psychology' 'robotics' 'stackoverflow' 'sustainable_living' 'leetcode')


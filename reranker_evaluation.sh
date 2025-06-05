#!/bin/bash

#SBATCH -p superpod-a100  # Submit job to GPU partition
#SBATCH -t 03-00:00:00    # Set max job time for 7 days
#SBATCH --gpus=1       # Request access to 1 GPU
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=50 
#SBATCH --mem=1000G       # Memory allocated
#SBATCH -o slurm-%j.out   # %j = job ID

set -a
source .env
set +a

set -x
source ~/.bashrc
source RaDeR_env/bin/activate
huggingface-cli login --token $HUGGING_FACE_HUB_TOKEN

if [[ -z "$CONDA_DEFAULT_ENV" ]]; then
    echo "No Conda environment is currently activated."
else
    echo "Conda environment '$CONDA_DEFAULT_ENV' is currently activated."
fi

DATASETS=('theoremqa_theorems' 'theoremqa_questions' 'aops' 'pony' 'biology' 'earth_science' 'economics' 'psychology' 'robotics' 'stackoverflow' 'sustainable_living' 'leetcode')

for ((i=0; i<${#DATASETS[@]}; i+=1)); do
    dataset1=${DATASETS[i]}
    # Example for BM25 (assuming we have the BM25 retrieval score file)
    python rerank.py --task ${dataset1} \
        --base_model_path <HUGGINGFACE BASE MODEL PATH > \
        --lora_model_path <HUGGINGFACE PATH TO RADER RERANKER > \
        --score_file BRIGHT_eval_outputs/${dataset1}_bm25_long_False/score.json \
        --rerank_score_file  BRIGHT_eval_outputs/${dataset1}_bm25_long_False/reranker_Qwen25-7B-instruct_MATH_allquerytypes_n100_score.json \
        --batch_size 1 \
        --input_k 100 \
        --k 10

done
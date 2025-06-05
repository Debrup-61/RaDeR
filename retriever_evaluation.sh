#!/bin/bash

#SBATCH -p superpod-a100  # Submit job to GPU partition
#SBATCH -t 03-00:00:00    # Set max job time for 7 days
#SBATCH --gpus=1       # Request access to 4 GPUs
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


DATASETS=('theoremqa_theorems' 'theoremqa_questions' 'aops' 'pony' 'biology' 'earth_science' 'economics' 'psychology' 'robotics' 'stackoverflow' 'sustainable_living' 'leetcode')

for ((i=0; i<${#DATASETS[@]}; i+=1)); do
    dataset1=${DATASETS[i]}
    python retriever_evalaution.py --task ${dataset1} --model trained_model --trainedmodel_name <NAME OF FOLDER IN BRIGHT_cache/doc_emb > --trainedmodel_hf <PATH TO HUGGINGFACE PATH OF RADER >  --reasoning gpt4
done     
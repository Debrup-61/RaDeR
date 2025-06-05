#!/bin/bash

#SBATCH -p superpod-a100 # Submit job to to gpu partition superpod-a100, gpu, gypsum-gpu, l40s
#SBATCH -t 2-00:00:00   # Set max job time for 30 days
#SBATCH --gpus=1       # Request access to GPU
#SBATCH --cpus-per-task=30    # Request 50 CPUs
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --mem=1000G     # Memory allocated
#SBATCH -o slurm-%j.out  # %j = job ID 

set -a
source .env
set +a

set -x
source ~/.bashrc
source RaDeR_env/bin/activate
huggingface-cli login --token $HUGGING_FACE_HUB_TOKEN

NUM_CPUS=${SLURM_CPUS_PER_TASK}  # Retrieve allocated CPUs
echo "Running with $NUM_CPUS CPUs"

# Start generator server in the background
./servers/generator_server.sh Qwen/Qwen2.5-7B-Instruct > servers/generator_server.log 2>&1 &
SERVER1_PID=$!  # Store process ID

# Wait for servers to start (Adjust as needed)
sleep 600

python models/vLLM_server_API.py
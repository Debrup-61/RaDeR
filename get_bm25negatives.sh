#!/bin/bash

#SBATCH -p superpod-a100 # Submit job to to gpu partition superpod-a100, gpu, gypsum-gpu, l40s
#SBATCH -t 1-00:00:00    # Set max job time for 30 days
#SBATCH --gpus=1         # Request access to GPU
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --mem=500G       # Memory allocated

set -x
source ~/.bashrc
source RaDeR_env/bin/activate

#Start BM25 server in background 
python models/BM25_server_API.py > BM25_server.log 2>&1 &
SERVER1_PID=$!  # Store process ID

# Start Repllama server in the background 
./servers/repllama_server.sh > servers/repllama_server.log 2>&1 &
SERVER2_PID=$!  # Store process ID

# Wait for servers to start (Adjust as needed)
sleep 300

python models/test_BM25.py 







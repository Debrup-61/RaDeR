#!/bin/bash


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
set -a
source "$SCRIPT_DIR/../.env"
set +a


export CUDA_VISIBLE_DEVICES=0


# Login to Hugging Face
huggingface-cli login --token $HUGGING_FACE_HUB_TOKEN

# Check if model is provided as argument
if [ -z "$1" ]; then
  echo "Error: Model checkpoint is required as first argument"
  exit 1
fi

MODEL_CKPT=$1

# Serve the model with vLLM for generation
vllm serve $MODEL_CKPT \
  --api-key abc \
  --trust-remote-code \
  --gpu-memory-utilization 1 \
  --swap-space 8 \
  --port 8010 \
  --disable-log-requests \
  --max-num-seqs 5000 \

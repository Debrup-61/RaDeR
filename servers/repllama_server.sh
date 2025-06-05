#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

set -a
source "$SCRIPT_DIR/../.env"
set +a

export CUDA_VISIBLE_DEVICES=0


# Login to Hugging Face
huggingface-cli login --token $HUGGING_FACE_HUB_TOKEN

# Serve the repllama model with vLLM
vllm serve $REPLLAMA_MERGED_HUGGINGFACE_PATH \
  --task embed \
  --trust-remote-code \
  --gpu-memory-utilization 0.9 \
  --api-key abc \
  --tokenizer meta-llama/Llama-2-7b-hf \
  --port 8011 \
  --disable-log-requests \
  --max-num-seqs 5000

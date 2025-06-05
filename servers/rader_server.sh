#!/bin/bash

MODEL_NAME=$1  # The first argument passed to the script


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
set -a
source "$SCRIPT_DIR/../.env"
set +a

export CUDA_VISIBLE_DEVICES=0


# Login to Hugging Face
huggingface-cli login --token $HUGGING_FACE_HUB_TOKEN

# Serve the repllama model with vLLM
vllm serve "$MODEL_NAME" \
  --task embed \
  --trust-remote-code \
  --override-pooler-config '{"pooling_type": "LAST", "normalize": true}' \
  --gpu-memory-utilization 0.9 \
  --api-key abc \
  --tokenizer Qwen/Qwen2.5-7B-Instruct \
  --port 8001 \
  --disable-log-requests \
  --max-num-seqs 5000
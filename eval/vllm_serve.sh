#!/usr/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2
export VLLM_USE_V1="1"
export VLLM_WORKER_MULTIPROC_METHOD="spawn"

MODEL_PATH=$1
MODEL_NAME=$2

conda run -n qwen --live-stream vllm serve "$MODEL_PATH" \
    --port 61131 \
    --gpu-memory-utilization 1 \
    --tensor-parallel-size 3 \
    --served-model-name "$MODEL_NAME" \
    --trust-remote-code \
    --disable-log-requests \
    --limit_mm_per_prompt "image=10"

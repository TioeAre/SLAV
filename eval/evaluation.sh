#!/usr/bin/bash

export CUDA_VISIBLE_DEVICES=2

MODEL_NAME=$1
EVAL_MODEL_NAME=$2
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

conda run -n qwen --live-stream python \
    "$SCRIPT_DIR"/test_model.py --model_name "$MODEL_NAME" --eval_model_name "$EVAL_MODEL_NAME"

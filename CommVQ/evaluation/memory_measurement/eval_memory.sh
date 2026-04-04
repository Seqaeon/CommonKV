#!/bin/bash
MODEL_NAME=$1
export CUDA_LAUNCH_BLOCKING=1

CUDA_VISIBLE_DEVICES=0 python -u eval_memory.py \
    --model_name ${MODEL_NAME} \
    --attn_implementation flash_attention_2

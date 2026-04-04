#!/bin/bash
START=32000
END=64000
STEP=16000

MODEL_NAMES=(
  $1
)

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
  python -u run_needle_in_haystack.py --s_len $START --e_len $END \
      --model_name ${MODEL_NAME} \
      --attn_implementation flash_attention_2 \
      --step $STEP
done



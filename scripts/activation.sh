#!/bin/bash

GPU="${GPU:-0,1}"

MODELS=(
    "Qwen/Qwen3-4B"
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
)

for MODEL in "${MODELS[@]}"; do
    echo "Running Activation for $MODEL"
    python src/activation/main.py \
        --model-name "$MODEL" \
        --gpu $GPU
done

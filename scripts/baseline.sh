#!/bin/bash

GPU="${GPU:-0,1}"

MODELS=(
    "Qwen/Qwen3-4B"
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
)

# GSM8K Train (Limit 1000)
for MODEL in "${MODELS[@]}"; do
    echo "Running Baseline for $MODEL on GSM8K Train"
    python src/intervention/main.py \
        --model-name "$MODEL" \
        --dataset gsm8k \
        --subset main \
        --split train \
        --limit 1000 \
        --gpu $GPU
done

# GSM8K Test
for MODEL in "${MODELS[@]}"; do
    echo "Running Baseline for $MODEL on GSM8K Test"
    python src/intervention/main.py \
        --model-name "$MODEL" \
        --dataset gsm8k \
        --subset main \
        --split test \
        --gpu $GPU
done

# Math 500
for MODEL in "${MODELS[@]}"; do
    echo "Running Baseline for $MODEL on Math 500"
    python src/intervention/main.py \
        --model-name "$MODEL" \
        --dataset math_500 \
        --gpu $GPU
done

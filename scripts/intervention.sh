#!/bin/bash

GPU="${GPU:-0,1}"

MODELS=(
    "Qwen/Qwen3-4B"
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
)

# GSM8K Train (Limit from baseline)
for MODEL in "${MODELS[@]}"; do
    echo "Running Intervention for $MODEL on GSM8K Train"
    python src/intervention/main.py \
        --model-name "$MODEL" \
        --dataset gsm8k \
        --subset main \
        --split train \
        --intervention-type local \
        --gpu $GPU
done

# GSM8K Test
for MODEL in "${MODELS[@]}"; do
    echo "Running Intervention for $MODEL on GSM8K Test"
    python src/intervention/main.py \
        --model-name "$MODEL" \
        --dataset gsm8k \
        --subset main \
        --split test \
        --intervention-type local \
        --gpu $GPU
done

# Math 500
for MODEL in "${MODELS[@]}"; do
    echo "Running Intervention for $MODEL on Math 500"
    python src/intervention/main.py \
        --model-name "$MODEL" \
        --dataset math_500 \
        --intervention-type local \
        --gpu $GPU
done

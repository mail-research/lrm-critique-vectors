#!/bin/bash

GPU="${GPU:-0,1}"

MODELS=(
    "Qwen/Qwen3-4B"
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
)

DATASETS=("gsm8k" "math" "olympiadbench" "omnimath")
COEFFS=(-1.0 -0.8 -0.6 -0.4 -0.2 0.0 0.2 0.4 0.6 0.8 1.0)

for MODEL in "${MODELS[@]}"; do
    case $MODEL in
        "Qwen/Qwen3-4B") STEER_LAYERS="21" ;;
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B") STEER_LAYERS="13" ;;
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B") STEER_LAYERS="28" ;;
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B") STEER_LAYERS="44" ;;
        *) echo "Warning: No steer layers for $MODEL"; continue ;;
    esac

    for DATASET in "${DATASETS[@]}"; do
        for COEFF in "${COEFFS[@]}"; do
            echo "=== $MODEL | $DATASET | coeff=$COEFF | layers=$STEER_LAYERS ==="
            python src/steer/main.py \
                --model-name "$MODEL" \
                --error-dataset processbench \
                --error-split "$DATASET" \
                --steer-layers "$STEER_LAYERS" \
                --steer-coeff "$COEFF" \
                --gpu $GPU
        done
    done
done

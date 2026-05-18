#!/bin/bash

GPU="${GPU:-0,1}"
STACKS=5

MODELS=(
    "Qwen/Qwen3-4B"
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
)

DATASETS=("gsm8k" "math_500")
COEFFS=(-1.0 0.0 1.0)

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
            echo "=== $MODEL | $DATASET | coeff=$COEFF | stacks=$STACKS | layers=$STEER_LAYERS ==="
            python src/tts_synthetic/main.py \
                --model-name "$MODEL" \
                --dataset "$DATASET" \
                --gpu "$GPU" \
                --stacks "$STACKS" \
                --steer-coeff "$COEFF" \
                --steer-layers "$STEER_LAYERS"
        done
    done
done

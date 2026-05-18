#!/bin/bash

GPU="${GPU:-0,1}"
STACKS=5

MODELS=(
    "Qwen/Qwen3-4B"
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
)

SPLITS=("logical_deduction" "multistep_arithmetic" "tracking_shuffled_objects" "word_sorting" "dyck_languages")
COEFFS=(-1.0 0.0 1.0)

for MODEL in "${MODELS[@]}"; do
    case $MODEL in
        "Qwen/Qwen3-4B") STEER_LAYERS="21" ;;
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B") STEER_LAYERS="13" ;;
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B") STEER_LAYERS="28" ;;
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B") STEER_LAYERS="44" ;;
        *) echo "Warning: No steer layers for $MODEL"; continue ;;
    esac

    for SPLIT in "${SPLITS[@]}"; do
        for COEFF in "${COEFFS[@]}"; do
            echo "=== $MODEL | $SPLIT | coeff=$COEFF | stacks=$STACKS | layers=$STEER_LAYERS ==="
            python src/tts_bigbench/main.py \
                --model-name "$MODEL" \
                --split "$SPLIT" \
                --gpu "$GPU" \
                --stacks "$STACKS" \
                --steer-coeff "$COEFF" \
                --steer-layers "$STEER_LAYERS"
        done
    done
done

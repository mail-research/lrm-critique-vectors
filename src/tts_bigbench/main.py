import argparse
import copy
import os
import sys
import shutil
from pathlib import Path
import torch
from transformers import AutoConfig
from vllm import LLM

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))
from utils import cleanup_vllm, BASE_TEMPLATE
from steer import load_steering_config, apply_steering_vllm
from steer.big_bench_mistake import BIGBENCH_DATASETS
from activation import get_probe_dir
from tts_bigbench_utils import (
    get_steer_tag,
    run_tts_bigbench_evaluation,
)
from tts_bigbench_inference import (
    run_bigbench_natural,
    run_tts_bigbench_right,
)
from tts_bigbench_data import load_bigbench_data


def main():
    """Main function to run test-time scaling on BIG-Bench Mistake data."""
    parser = argparse.ArgumentParser(description="Run test-time scaling on BIG-Bench Mistake data.")
    
    # Evaluation Arguments
    parser.add_argument('--model-name', type=str, required=True, help='Name of the model to analyze')
    parser.add_argument("--split", required=True, choices=BIGBENCH_DATASETS, help="BIG-Bench Mistake category.")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of samples for evaluation.")
    parser.add_argument("--gpu", type=str, default="0", help="GPU ID to use.")
    parser.add_argument("--xverify-model", type=str, default="xVerify-0.5B-I", choices=list(BASE_TEMPLATE.keys()), help="xVerify model.")
    
    # TTS Arguments
    parser.add_argument("--stacks", type=int, default=3, help="Number of RIGHT scaling steps (stacking 'Wait').")
    
    # Steering Arguments
    parser.add_argument("--steer-coeff", type=float, default=0.0, help="Coefficient for steering.")
    parser.add_argument("--steer-layers", type=str, default=None, help="Indices of layers to apply steering.")
    
    args = parser.parse_args()

    # === Environment Setup ===
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["VLLM_USE_V1"] = "1"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
    
    # === Load steering config ===
    model_config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    num_layers = model_config.num_hidden_layers
    
    # === RoPE scaling for Qwen3 long context: https://huggingface.co/Qwen/Qwen3-4B ===
    rope_scaling = None
    if "Qwen3" in args.model_name:
        rope_scaling = {
            "rope_type": "yarn",
            "factor": 4.0,
            "original_max_position_embeddings": 32768
        }
    
    steer_layers = (
        [int(x) for x in args.steer_layers.split(",")]
        if args.steer_layers
        else list(range(num_layers))
    )
    
    # === Build output directory ===
    steer_tag = get_steer_tag(args.steer_coeff, steer_layers if args.steer_layers else None)
    output_dir = (
        project_root.parent / "results" / "test-time-scaling-bigbench"
        / args.model_name.replace("/", "__") / args.split / steer_tag
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # === Prepare steered model if needed ===
    steered_model_dir = None
    if args.steer_coeff != 0.0:
        print(f"INFO: Preparing steered model with coefficient {args.steer_coeff}...")
        
        # Load steering vectors from gsm8k train
        steer_args = copy.copy(args)
        steer_args.dataset = "gsm8k"
        steer_args.split = "train"
        steer_args.steer_coeff = args.steer_coeff
        steer_args.steer_layers = steer_layers
        steering_vectors = load_steering_config(steer_args)
        if not steering_vectors:
            print("ERROR: Could not load steering vectors.")
            sys.exit(1)
        
        # Build steering config
        steer_config = {}
        for layer_idx in steer_layers:
            if layer_idx not in steering_vectors:
                continue
            sv = steering_vectors[layer_idx]
            steer_config[f"layers.{layer_idx}"] = {
                "steering_vector": sv,
                "steering_coefficient": args.steer_coeff,
                "action": "add",
            }
        
        # Apply steering for vllm model
        steer_config_dir = get_probe_dir(project_root, args) / "steer_configs"
        steer_config_dir.mkdir(parents=True, exist_ok=True)
        steer_config_path = steer_config_dir / f"tts_bigbench_{steer_tag}.pt"
        torch.save(steer_config, steer_config_path)
        
        steered_model_dir = (
            project_root.parent / "results" / "steered_ckpt"
            / args.model_name.replace("/", "__")
            / f"tts_bigbench_{steer_tag}"
        )
        apply_steering_vllm(
            model_name=args.model_name,
            output_dir=steered_model_dir,
            steer_config_path=str(steer_config_path),
        )
        
        del steering_vectors, steer_config
        torch.cuda.empty_cache()

    # === Load vLLM model ===
    llm_kwargs = dict(
        model=str(steered_model_dir) if steered_model_dir else args.model_name,
        tokenizer=args.model_name,
        tensor_parallel_size=len(args.gpu.split(",")),
        trust_remote_code=True,
        gpu_memory_utilization=0.95,
        max_model_len=120000,
        seed=0,
    )
    if rope_scaling:
        llm_kwargs["rope_scaling"] = rope_scaling
    
    llm = LLM(**llm_kwargs)

    # === Load BIG-Bench data ===
    samples = load_bigbench_data(
        split=args.split,
        model_name=args.model_name,
        limit=args.limit,
    )
    print(f"INFO: Loaded {len(samples)} BIG-Bench samples for evaluation")

    # === Run natural (no Wait) and TTS stacks (append Wait) ===
    run_bigbench_natural(args, llm, output_dir, samples)
    
    completed_stacks = 0
    if args.stacks > 0:
        completed_stacks = run_tts_bigbench_right(args, llm, output_dir, args.stacks)

    # === Run xVerify evaluation ===
    cleanup_vllm(llm)
    eval_kwargs = dict(
        model_name=args.model_name,
        split=args.split,
        xverify_model=args.xverify_model,
        steer_coeff=args.steer_coeff,
    )
    
    run_tts_bigbench_evaluation(
        output_dir / "natural",
        stack=0,
        **eval_kwargs
    )
    
    for stack in range(1, completed_stacks + 1):
        run_tts_bigbench_evaluation(
            output_dir / f"stack_{stack}",
            stack=stack,
            **eval_kwargs
        )

    # === Cleanup model checkpoint ===
    if steered_model_dir and steered_model_dir.is_dir():
        shutil.rmtree(steered_model_dir)
        parent = steered_model_dir.parent
        while parent.is_dir() and not any(parent.iterdir()) and parent.name != "results":
            parent.rmdir()
            parent = parent.parent
    
    print(f"\n✅ Experiment complete. Results saved to: {output_dir}")


if __name__ == "__main__":
    main()

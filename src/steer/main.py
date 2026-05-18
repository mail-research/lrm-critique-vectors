import argparse
import os
import sys
import shutil
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoConfig
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))
from activation import get_probe_dir
from utils import get_think_tags, cleanup_vllm
from steer import (
    load_steering_config,
    apply_steering_vllm,
    # ProcessBench
    CRITIQUE_TEMPLATE,
    process_data_processbench, 
    process_metrics_processbench,
    # BIG-Bench Mistake
    BIGBENCH_TEMPLATES,
    BIGBENCH_DATASETS,
    process_data_bigbench,
    process_metrics_bigbench,
)


def main():
    """Main function to run steering evaluation."""
    parser = argparse.ArgumentParser(description="Run steering evaluation with vLLM for error detection.")
    
    # Evaluation Arguments
    parser.add_argument("--model-name", type=str, required=True, help="Name of the Hugging Face model to use.")
    parser.add_argument("--error-dataset", type=str, required=True, choices=["processbench", "bigbench"], help="Error detection dataset to use.")
    parser.add_argument("--error-split", type=str, required=True, help="Dataset split. ProcessBench: gsm8k/math/olympiadbench/omnimath. BIG-Bench: multistep_arithmetic/word_sorting/dyck_languages/logical_deduction/tracking_shuffled_objects.")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of samples for evaluation.")
    parser.add_argument("--gpu", type=str, default="0", help="GPU ID to use (e.g., '0' for single GPU, '0,1,2,3' for multi-GPU).")
    
    # Steering Arguments
    parser.add_argument("--steer-layers", type=str, default=None, help="Indices of layers to apply steering (comma-separated). If not provided, all layers are used.")
    parser.add_argument("--steer-coeff", type=float, required=True, help="Coefficient for steering.")
    
    args = parser.parse_args()

    # === Validate error-split based on error-dataset ===
    if args.error_dataset == "processbench":
        valid_splits = ["gsm8k", "math", "olympiadbench", "omnimath"]
        if args.error_split not in valid_splits:
            parser.error(f"For processbench, --error-split must be one of: {valid_splits}")
    elif args.error_dataset == "bigbench":
        if args.error_split not in BIGBENCH_DATASETS:
            parser.error(f"For bigbench, --error-split must be one of: {BIGBENCH_DATASETS}")

    # === Base Setup ===
    args.dataset = "gsm8k"
    args.split = "train"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["VLLM_USE_V1"] = "1"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model_format_path = project_root.parent / 'configs/model_config.yaml'
    _, think_end_tag = get_think_tags(args.model_name, model_format_path)
    
    num_layers = AutoConfig.from_pretrained(args.model_name).num_hidden_layers
    if args.steer_layers:
        args.steer_layers = [int(x) for x in args.steer_layers.split(",")]
    else:
        args.steer_layers = list(range(num_layers))

    # === Steering Config ID ===
    steer_action = "add"
    layers_id = "_".join(map(str, args.steer_layers))
    steer_tag = f"layers_{layers_id}_coeff_{args.steer_coeff}_{steer_action}"
    
    # === Load Steering Vectors ===
    steering_vectors = load_steering_config(args)
    if not steering_vectors:
        print("Error: Could not load or compute steering vectors.")
        sys.exit(1)

    # === Prepare steering config from steering vectors ===
    steer_config = {}
    print("Preparing steering vectors...")
    for layer_idx in args.steer_layers:
        if layer_idx not in steering_vectors:
            print(f"Error: Activations for layer {layer_idx} not found.")
            sys.exit(1)
        
        sv = steering_vectors[layer_idx]
        steer_config[f"layers.{layer_idx}"] = {
            "steering_vector": sv,
            "steering_coefficient": args.steer_coeff,
            "action": steer_action,
        }
    del steering_vectors
    torch.cuda.empty_cache()

    # === Save steering config ===
    steer_config_dir = get_probe_dir(project_root, args) / "steer_configs"
    steer_config_dir.mkdir(parents=True, exist_ok=True)
    steer_config_path = steer_config_dir / f"{steer_tag}.pt"
    torch.save(steer_config, steer_config_path)

    # === Apply steering for vllm model ===
    steered_model_dir = (
        project_root.parent
        / "results" / "steered_ckpt"
        / args.model_name.replace("/", "__")
        / steer_tag
    )      
    apply_steering_vllm(
        model_name=args.model_name,
        output_dir=steered_model_dir,
        steer_config_path=str(steer_config_path),
    )
    del steer_config
    torch.cuda.empty_cache()

    # === Process Data based on dataset ===
    if args.error_dataset == "processbench":
        input_data, prompt_token_ids = process_data_processbench(
            split=args.error_split,
            tokenizer=tokenizer,
            template=CRITIQUE_TEMPLATE
        )
    else:  # bigbench
        input_data, prompt_token_ids = process_data_bigbench(
            split=args.error_split,
            tokenizer=tokenizer,
            template=BIGBENCH_TEMPLATES[args.error_split]
        )
    
    if args.limit:
        print(f"INFO: Limit to {args.limit} samples.")
        input_data = input_data.select(range(args.limit))
        prompt_token_ids = prompt_token_ids[:args.limit]

    # === Run Evaluation with Steered vLLM Model ===
    llm = LLM(
        model=str(steered_model_dir), 
        tokenizer=args.model_name,
        tensor_parallel_size=len(args.gpu.split(",")),
        trust_remote_code=True,
        gpu_memory_utilization=0.95,
        max_model_len=20000,
        seed=0,
    )
    sampling_params = SamplingParams(
        temperature=0.6,
        max_tokens=16384, 
        seed=0,
    )
    generations = llm.generate(
        [TokensPrompt(prompt_token_ids=p) for p in prompt_token_ids],
        sampling_params=sampling_params,
        use_tqdm=True
    )
    
    # === Cleanup vllm.LLM ===
    cleanup_vllm(llm)

    # === Process Results and Save ===
    output_path = Path(
        project_root.parent / "results",
        f"steer-error-detection-{args.error_dataset}",
        args.model_name.replace("/", "__"),
        steer_tag
    )
    
    if args.error_dataset == "processbench":
        process_metrics_processbench(
            generations=generations,
            input_data=input_data,
            output_dir=output_path,
            config_name=args.error_split,
            args=args,
            think_end_tag=think_end_tag,
            tokenizer=tokenizer,
            model_name=args.model_name
        )
    else:  # bigbench
        process_metrics_bigbench(
            generations=generations,
            input_data=input_data,
            output_dir=output_path,
            config_name=args.error_split,
            args=args,
            think_end_tag=think_end_tag,
            tokenizer=tokenizer,
            model_name=args.model_name
        )
    print(f"INFO: Results saved to {output_path}")

    # === Cleanup model checkpoint ===
    if steered_model_dir.is_dir():
        shutil.rmtree(steered_model_dir)
        parent = steered_model_dir.parent
        while parent.is_dir() and not any(parent.iterdir()) and parent.name != "results":
            parent.rmdir()
            parent = parent.parent
    
    print("✅ Evaluation complete.")


if __name__ == '__main__':
    main()

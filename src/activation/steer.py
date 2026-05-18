import json
import os
import sys
import argparse
from pathlib import Path
import torch
import yaml
from transformers import AutoTokenizer, AutoConfig

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))
from activation import get_probe_dir, load_activations, run_steering_vector_analysis
from intervention import build_global_task_yaml, build_local_task_yaml
from utils import (
    run_lm_evaluation,
    apply_steering_vllm,
    BASE_TEMPLATE,
    BASE_CONFIG,
    get_task_name,
    get_dataset_config,
    find_latest_file,
    get_think_tags
)


def main():
    """Apply steering with vLLM."""
    parser = argparse.ArgumentParser(description="Run steering evaluation with lm-eval-harness.")
    
    # Evaluation Arguments
    parser.add_argument("--model-name", type=str, required=True, help="Name of the Hugging Face model to use.")
    parser.add_argument("--dataset", required=True, choices=["mmlu", "gpqa", "arc", "aime_2024", "aime_2025", "math_500", "gsm8k"], help="Dataset to evaluate.")
    parser.add_argument("--subset", type=str, default=None, help="Dataset subset to evaluate.")
    parser.add_argument("--split", type=str, default=None, help="Dataset split to evaluate. If not provided, uses default from dataset config.")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of samples for evaluation.")
    parser.add_argument("--gpu", type=str, default="0", help="GPU ID to use (e.g., '0' for single GPU, '0,1,2,3' for multi-GPU).")
    parser.add_argument("--xverify-model", type=str, default="xVerify-0.5B-I", choices=list(BASE_TEMPLATE.keys()), help="xVerify model to evaluate.")

    # Activation Source Arguments
    parser.add_argument("--intervention-local", action="store_true", help="Whether to perform local intervention.")
    parser.add_argument("--intervention-source", type=str, choices=["baseline"], default="baseline", help="Source of generated samples for local intervention (used for probe source and local eval task).")

    # Steering Arguments
    parser.add_argument("--steer-layers", type=int, nargs='+', required=True, help="1-based indices of layers to apply steering vectors to.")
    parser.add_argument("--steer-coeffs", type=float, nargs='+', required=True, help="Coefficients for steering. One per layer or a single value for all.")
    parser.add_argument("--steer-action", type=str, default="add", choices=["add", "clamp"], help="Action to apply the steering vector.")
    parser.add_argument("--steer-application", type=str, default="prompt", choices=["prompt", "think"], help="When to apply the steering vector.")
    parser.add_argument("--steer-sample-type", type=str, default="incorrect", choices=["correct", "incorrect", "out_of_token"], help="Use only samples of this type to evaluate.")

    args = parser.parse_args()

    # === Base Setup ===    
    _, think_end = get_think_tags(args.model_name, Path("configs") / "model_config.yaml")
    split = args.split or get_dataset_config(
        dataset=args.dataset, subset=args.subset,
        model_name=args.model_name, config_path=Path("configs")
    ).get("test_split")
    base_task_name = get_task_name(
        args.dataset, project_root.parent / "configs",
        args.subset, split
    )

    # === Directory and Task Name Setup ===
    layers_id = "_".join(map(str, args.steer_layers))
    coeffs = args.steer_coeffs
    coeffs_id = (str(coeffs[0]) if len(set(coeffs)) == 1 else "_".join(map(str, coeffs))).replace(".", "p")
    steer_suffix = f"steer_layers_{layers_id}_coeffs_{coeffs_id}_{args.steer_action}_at_{args.steer_application}"
    intervention_type_name = "local" if args.intervention_local else "global"
    steer_tag = f"{intervention_type_name}_{steer_suffix}"
    
    # === Create task YAML in lm-evaluation-harness ===
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    sample_ids = None

    if args.steer_application == "think":
        prefix = "{{doc['full_intervened_prompt']}}" if args.intervention_local else "{{arguments.gen_args_0.arg_0}}"
        custom_doc_to_text = f"{prefix}{{{{resps[0][0].split('{think_end}')[0]}}}}{think_end}"
    else:
        custom_doc_to_text = None

    # === Local intervention ===
    if args.intervention_local:
        base_dir = project_root.parent / "results" / base_task_name / args.intervention_source / args.model_name.replace("/", "__")
        
        intervened_run_dir = (
            base_dir.parent
            / "intervened_local"
            / args.model_name.replace("/", "__")
        )
        
        if args.steer_application == "think":
            search_dir = intervened_run_dir
            file_pattern = "samples*.jsonl"
        else:
            search_dir = intervened_run_dir / "data"
            file_pattern = "*.jsonl"

        intervened_file = find_latest_file(search_dir, file_pattern)
        print(f"INFO: Found latest intervened data file: {intervened_file}")

        task_name, split = build_local_task_yaml(
            dataset=args.dataset,
            subset=args.subset,
            split=split,
            model_name=args.model_name,
            intervened_file_path=intervened_file,
            task_name_suffix=steer_suffix,
            custom_doc_to_text=custom_doc_to_text,
        )
    
    # === Global intervention ===
    else:
        full_task_name = f"{base_task_name}_{steer_tag}"
        sample_ids = None
        intervened_file = None
        
        # Find the latest baseline log file.
        if args.steer_sample_type or args.steer_application == "think":
            baseline_results_dir = project_root.parent / "results" / base_task_name / "baseline" / args.model_name.replace("/", "__")
            try:
                latest_log_file = max(baseline_results_dir.glob("samples*.jsonl"), key=os.path.getctime)
            except ValueError:
                print(f"ERROR: No baseline result found in {baseline_results_dir}")
                sys.exit(1)
            print(f"INFO: Using baseline result: {latest_log_file}")

            if args.steer_application == "think":
                intervened_file = latest_log_file

            if args.steer_sample_type:
                with open(latest_log_file, 'r', encoding='utf-8') as f:
                    samples = [json.loads(line) for line in f]
                
                sample_ids = [s["doc_id"] for s in samples if s.get("judgment").get("final") == args.steer_sample_type]
                print(f"INFO: Found {len(sample_ids)} {args.steer_sample_type} samples.")
                full_task_name += f"_{args.steer_sample_type}_only"
                steer_tag += f"_{args.steer_sample_type}_only"

        if args.steer_application == "think":
            
            task_name = full_task_name
            task_config = {
                **BASE_CONFIG,
                "task": task_name,
                "dataset_path": "json",
                "dataset_kwargs": {"data_files": {split: str(intervened_file)}},
                "doc_to_text": custom_doc_to_text or "{{full_intervened_prompt}}",
                "doc_to_target": "",
                "test_split": split,
            }

            # Save the task config to a yaml file in lm-eval's directory
            model_prefix = args.model_name.split("/")[0]
            tasks_dir_base = (
                project_root / "lm-eval" / "lm_eval" / "tasks" / "intervention_tasks" / model_prefix
            )
            tasks_dir = tasks_dir_base / "steer" / intervention_type_name
            tasks_dir.mkdir(parents=True, exist_ok=True)
            
            # Handle sample_ids for steer sample type
            process_docs_function = None
            if sample_ids is not None:
                utils_path = tasks_dir / "utils.py"
                with open(utils_path, "w") as f:
                    f.write("from datasets import Dataset\n\n")
                    f.write("def process_docs(docs):\n")
                    f.write(f"    sample_ids_to_keep = {set(sample_ids)}\n")
                    f.write("    docs = docs.add_column('original_doc_id', range(len(docs)))\n")
                    f.write("    filtered_docs = docs.filter(lambda x: x['original_doc_id'] in sample_ids_to_keep)\n")
                    f.write("    return filtered_docs\n")
                process_docs_function = "utils.process_docs"
            
            task_yaml_path = tasks_dir / f"{task_name}.yaml"
            with open(task_yaml_path, "w", encoding="utf-8") as f:
                yaml.dump(task_config, f)
                if process_docs_function:
                    f.write(f"\nprocess_docs: !function {process_docs_function}\n")
        else:
            task_name, split = build_global_task_yaml(
                dataset=args.dataset,
                subset=args.subset,
                split=split,
                model_name=args.model_name,
                intervention_type=None,
                intervention_content=None,
                intervention_position=None,
                tokenizer=tokenizer,
                thinking_end=False,
                immediate_answer=False,
                task_name=full_task_name,
                sample_ids=sample_ids,
                custom_doc_to_text=custom_doc_to_text,
            )
    
    # === Load activations ===
    probe_dir = get_probe_dir(project_root, args)
    original_acts, intervened_acts = load_activations(probe_dir)

    if len(args.steer_coeffs) == 1:
        coeffs = [args.steer_coeffs[0]] * len(args.steer_layers)
    elif len(args.steer_coeffs) == len(args.steer_layers):
        coeffs = args.steer_coeffs
    else:
        print("ERROR: Number of coefficients must match number of layers, or be a single value.")
        sys.exit(1)

    # === Get model config ===
    model_config = AutoConfig.from_pretrained(args.model_name)
    num_layers = model_config.num_hidden_layers

    # === Prepare steering vectors and run logit lens analysis ===
    steering_vectors = run_steering_vector_analysis(
        args=args,
        probe_dir=probe_dir,
        original_acts=original_acts,
        intervened_acts=intervened_acts,
        num_layers=num_layers,
        tokenizer=tokenizer,
    )
    del original_acts, intervened_acts
    torch.cuda.empty_cache()

    # === Prepare steering config from steering vectors ===
    steer_config = {}
    for layer_idx in args.steer_layers:
        if layer_idx not in steering_vectors:
            print(f"Error: Activations for layer {layer_idx} not found.")
            sys.exit(1)
        
        sv = steering_vectors[layer_idx]
        layer_pos = args.steer_layers.index(layer_idx)
        coeff = coeffs[layer_pos]
        layer_name = f"layers.{layer_idx - 1}"
        steer_config[layer_name] = {
            "steering_vector": sv,
            "steering_coefficient": coeff,
            "action": args.steer_action,
        }
    del steering_vectors
    torch.cuda.empty_cache()
    
    # === Save the steering config ===
    steer_config_dir = probe_dir / "steer_configs"
    steer_config_dir.mkdir(parents=True, exist_ok=True)
    steer_config_path = steer_config_dir / f"{steer_tag}.pt"
    torch.save(steer_config, steer_config_path)
    print(f"INFO: Saved steering config to {steer_config_path}")

    # === Create specific steered model for this configuration ===
    layers_id = "_".join(map(str, args.steer_layers))
    coeffs = args.steer_coeffs
    coeffs_id = (str(coeffs[0]) if len(set(coeffs)) == 1 else "_".join(map(str, coeffs))).replace(".", "p")
    steer_config_id = f"layers_{layers_id}_coeffs_{coeffs_id}_{args.steer_action}_at_{args.steer_application}"
    
    base_output_dir_parts = [
        project_root.parent / "results",
        "steered_vllm_model",
        args.model_name.replace("/", "__"),
        steer_config_id
    ]
    steered_model_dir = Path(*base_output_dir_parts)
    
    # === Apply steering for vllm model ===
    steered_model_dir.mkdir(parents=True, exist_ok=True)
    apply_steering_vllm(
        model_name=args.model_name,
        output_dir=steered_model_dir,
        steer_config_path=str(steer_config_path),
    )

    # === Output directory setup (for results) ===
    output_dir_parts = [
        project_root.parent / "results",
        get_task_name(args.dataset, project_root.parent / "configs", args.subset, split),
        "steered",
        args.model_name.replace("/", "__"),
        args.steer_application,
        steer_tag
    ]
    output_dir = Path(*[p for p in output_dir_parts if p is not None])
    
    # === Run Evaluation ===  
    run_lm_evaluation(
        model_name=str(steered_model_dir),
        task_name=task_name,
        limit=args.limit,
        gpu=args.gpu,
        dataset=args.dataset,
        subset=args.subset,
        split=split,
        xverify_model=args.xverify_model,
        output_dir=output_dir,
        intervention_type=("local" if args.intervention_local else "global") + (
            f"_{args.steer_application}" if args.steer_application == "think" else ""
        ),
        steer_config_path=str(steer_config_path),  # Pass the saved config path
        steer_layers=args.steer_layers,
        steer_coeffs=coeffs,
        steer_action=args.steer_action,
        cleanup_model=True,  # Clean up the model checkpoint after evaluation
    )

if __name__ == "__main__":
    main()
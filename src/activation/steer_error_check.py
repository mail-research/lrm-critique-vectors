import json
import sys
import argparse
from pathlib import Path
import torch
import yaml
from transformers import AutoTokenizer, AutoConfig

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))
from activation import get_probe_dir, load_activations, run_steering_vector_analysis
from utils import (
    run_lm_evaluation,
    apply_steering_vllm,
    BASE_TEMPLATE,
    BASE_CONFIG,
    get_dataset_config,
    get_think_tags,
    get_task_name,
    find_latest_file,
)

# === Prompt for the error checking task ===
ERROR_CHECK_PROMPT_TEMPLATE = """Given a question and a thinking process, determine whether the thinking process 
has correctly solved the question in order to derive the correct answer. 

- Answer with boxed{yes} if it has correctly solved the question.  
- Answer with boxed{no} otherwise.  
- Do NOT solve the question yourself. Only evaluate the provided reasoning.  

Question:  
{question}  

Thinking Process:  
{thinking_process}"""


def build_error_check_task_yaml(
    input_file_path: str,
    task_name: str,
    model_name: str,
    think_start: str,
    think_end: str,
) -> str:
    """Pre-processes data and builds a YAML configuration file for the error checking task."""
    
    project_root = Path(__file__).resolve().parents[1]

    # === Process data ===
    processed_data = []
    with open(input_file_path, "r", encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line)

            # Extract thinking process and question
            full_response = sample["arguments"]["gen_args_0"]["arg_0"] + sample["resps"][0][0]
            start_index = full_response.find(think_start)
            end_index = full_response.find(think_end)
            thinking_process = full_response[start_index + len(think_start):end_index].strip()
            question = sample["doc"]["question"]

            # Set ground truth
            judgment = sample.get("judgment").get("thinking")
            if judgment == "correct":
                target = r"boxed{yes}"
            elif judgment == "incorrect":
                target = r"boxed{no}"
            else:
                continue

            processed_data.append({
                "question": question,
                "thinking_process": thinking_process,
                "target": target,
            })

    # === Save processed data ===
    model_prefix = model_name.split("/")[0]
    tasks_dir = (
        project_root.parent / "src" / "lm-eval" / "lm_eval" / "tasks" / "intervention_tasks" / model_prefix
    )

    processed_file_path = tasks_dir / f"{task_name}_data.jsonl"
    with open(processed_file_path, "w", encoding="utf-8") as f:
        for item in processed_data:
            f.write(json.dumps(item) + "\n")

    # === Build task config ===
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prompt_content = ERROR_CHECK_PROMPT_TEMPLATE.replace("{question}", "{{question}}").replace(
        "{thinking_process}", "{{thinking_process}}"
    )
    messages = [{"role": "user", "content": prompt_content}]
    doc_to_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    task_config = {
        **BASE_CONFIG,
        "task": task_name,
        "dataset_path": "json",
        "dataset_kwargs": {"data_files": {"test": str(processed_file_path.resolve())}},
        "doc_to_text": doc_to_text,
        "doc_to_target": "target",
        "test_split": "test",
    }

    # === Save YAML config ===
    tasks_dir.mkdir(parents=True, exist_ok=True)
    task_yaml_path = tasks_dir / f"{task_name}.yaml"
    with open(task_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(task_config, f)

    return task_name


def main():
    parser = argparse.ArgumentParser(description="Run steering error check evaluation with lm-eval-harness.")
    
    # Evaluation Arguments
    parser.add_argument("--model-name", type=str, required=True, help="Name of the Hugging Face model to use.")
    parser.add_argument("--dataset", required=True, choices=["mmlu", "gpqa", "arc", "aime_2024", "aime_2025", "math_500", "gsm8k"], help="Dataset to evaluate.")
    parser.add_argument("--subset", type=str, default=None, help="Dataset subset to evaluate.")
    parser.add_argument("--split", type=str, default=None, help="Dataset split to evaluate. If not provided, uses default from dataset config.")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of samples for evaluation.")
    parser.add_argument("--gpu", type=str, default="0", help="GPU ID to use (e.g., '0' for single GPU, '0,1,2,3' for multi-GPU).")
    parser.add_argument("--main-process-port", type=int, default=29500, help="Main process port for accelerate.")
    parser.add_argument("--xverify-model", type=str, default="xVerify-0.5B-I", choices=list(BASE_TEMPLATE.keys()), help="xVerify model to evaluate.")

    # Activation Source Arguments
    parser.add_argument("--intervention-source", type=str, choices=["baseline"], default="baseline", help="Source of generated samples for local intervention.")
    
    # Steering Arguments
    parser.add_argument("--steer-layers", type=int, nargs='+', required=True, help="1-based indices of layers to apply steering vectors to.")
    parser.add_argument("--steer-coeffs", type=float, nargs='+', required=True, help="Coefficients for steering. One per layer or a single value for all.")
    parser.add_argument("--steer-action", type=str, default="add", choices=["add", "clamp"], help="Action to apply the steering vector.")

    args = parser.parse_args()

    # === Base Setup ===
    think_start, think_end = get_think_tags(args.model_name, Path("configs") / "model_config.yaml")
    split = args.split or get_dataset_config(
        dataset=args.dataset, subset=args.subset,
        model_name=args.model_name, config_path=Path("configs")
    ).get("test_split")
    dataset_name = get_task_name(
        dataset=args.dataset, config_path=Path("configs"),
        subset=args.subset, split=split
    )

    # === Load intervened data with generated responses ===
    base_results_dir = project_root.parent / "results" / dataset_name / args.intervention_source
    input_dir = base_results_dir / "intervened_local"
    input_dir /= args.model_name.replace("/", "__")
    input_file_path = find_latest_file(input_dir, "samples_*.jsonl")
    if not input_file_path:
        raise FileNotFoundError(f"No samples file found in {input_dir}")
    print(f"Found latest input file: {input_file_path}")
    
    # === Directory and Task Name Setup ===
    layers_id = "_".join(map(str, args.steer_layers))
    coeffs = args.steer_coeffs
    coeffs_id = (str(coeffs[0]) if len(set(coeffs)) == 1 else "_".join(map(str, coeffs))).replace(".", "p")
    steer_suffix = f"steer_layers_{layers_id}_coeffs_{coeffs_id}_{args.steer_action}"
    input_file_name = Path(input_file_path).stem
    base_task_name = f"error_check_{input_file_name}"
    task_name = f"{base_task_name}_{steer_suffix}"

    # === Create task YAML in lm-evaluation-harness ===
    task_name = build_error_check_task_yaml(
        input_file_path=input_file_path,
        task_name=task_name,
        model_name=args.model_name,
        think_start=think_start,
        think_end=think_end,
    )
    
    # === Load activations ===
    probe_dir = get_probe_dir(project_root, args)
    original_acts, intervened_acts = load_activations(probe_dir)
    if len(args.steer_coeffs) == 1:
        coeffs = [args.steer_coeffs[0]] * len(args.steer_layers)
    elif len(args.steer_coeffs) != len(args.steer_layers):
        print("Error: Number of coefficients must match number of layers, or be a single value.")
        sys.exit(1)
    
    # === Get model config ===
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model_config = AutoConfig.from_pretrained(args.model_name)
    num_layers = model_config.num_hidden_layers

    # === Prepare steering vectors ===
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
    print("Preparing steering vectors...")
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
    steer_config_path = steer_config_dir / f"error_check_{steer_suffix}.pt"
    torch.save(steer_config, steer_config_path)
    print(f"Saved steering config to {steer_config_path}")

    # === Create specific steered model for this configuration ===
    layers_id = "_".join(map(str, args.steer_layers))
    coeffs = args.steer_coeffs
    coeffs_id = (str(coeffs[0]) if len(set(coeffs)) == 1 else "_".join(map(str, coeffs))).replace(".", "p")
    steer_config_id = f"layers_{layers_id}_coeffs_{coeffs_id}_{args.steer_action}_error_check"

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

    # === Output directory setup ===
    output_dir_parts = [
        project_root.parent / "results",
        get_task_name(args.dataset, project_root.parent / "configs", args.subset, split),
        "steered",
        args.model_name.replace("/", "__"),
        "error_check",
        steer_suffix,
    ]
    output_dir = Path(*[p for p in output_dir_parts if p is not None])
    
    # === Run Evaluation ===
    run_lm_evaluation(
        model_name=str(steered_model_dir),
        task_name=task_name,
        limit=args.limit,
        gpu=args.gpu,
        xverify_model=args.xverify_model,
        output_dir=output_dir,
        steer_config_path=str(steer_config_path), # Pass the saved config path
        steer_layers=args.steer_layers,
        steer_coeffs=coeffs,
        steer_action=args.steer_action,
        is_error_check_task=True,
        cleanup_model=True,  # Clean up the model checkpoint after evaluation
    )


if __name__ == "__main__":
    main()
import argparse
import sys
import os
import torch
import json
import gc
import shutil
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))
from activation import get_probe_dir
from utils import get_task_name, get_think_tags, find_latest_file, Model, run_xverify, cleanup_vllm
from steer import load_steering_config, apply_steering_vllm


def generate_outputs(
    model: LLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    model_name: str
) -> tuple[List[str], List[bool], List[str]]:
    """Generate completions from the given model."""
    
    # === Generate complete responses ===
    prompt_token_ids = [tokenizer.encode(text, add_special_tokens=False) for text in prompts]
    sampling_params = SamplingParams(temperature=0.0, max_tokens=16384, seed=0)
    generations = model.generate(
        [TokensPrompt(prompt_token_ids=p) for p in prompt_token_ids],
        sampling_params=sampling_params,
        use_tqdm=True
    )

    # === Cleanup vLLM ===
    cleanup_vllm(model)
    
    # === Get think end tag and check for valid samples ===
    think_end_tag = get_think_tags(model_name, project_root.parent / "configs" / "model_config.yaml")[1]    
    final_outputs = []
    validity_flags = []
    full_responses = []
    
    for i, gen in enumerate(generations):
        full_text = gen.outputs[0].text
        full_responses.append(full_text)
        
        # Valid samples: must contain think_end_tag
        if think_end_tag in full_text:
            final_answer = full_text.split(think_end_tag)[1].strip()
            final_outputs.append(final_answer)
            validity_flags.append(True)
        else: # Out of token - no think_end_tag
            final_outputs.append("")
            validity_flags.append(False)
    
    return final_outputs, validity_flags, full_responses


def generate_and_evaluate(
    final_outputs: List[str],
    questions: List[str],
    ground_truths: List[str],
    prompts: List[str],
    full_responses: List[str],
    validity_flags: List[bool],
    model_name: str,
    xverify_model: Model,
    output_dir: Path,
    save_name: str
) -> float:
    """Run xVerify evaluation and save combined log with generation results."""
    
    # === Only run xVerify on valid samples ===
    samples = [
        {'question': q, 'llm_output': output, 'correct_answer': gt}
        for output, q, gt, valid in zip(final_outputs, questions, ground_truths, validity_flags)
        if valid and output.strip()
    ]
    
    # === Run xVerify evaluation (using pre-loaded model) ===
    judgments = run_xverify(samples, xverify_model, batch_size=128)  
    correct = sum(1 for judgment in judgments if judgment == "correct")
    total_samples = len(validity_flags)
    accuracy = correct / total_samples if total_samples > 0 else 0.0
    
    # === Save combined log with generation results and evaluation ===
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / f"{save_name}.log"
    
    with open(log_file, 'w') as f:
        f.write(f"=== Results: {save_name} ===\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Overall Accuracy: {accuracy:.4f} ({correct}/{total_samples})\n")
        f.write(f"Valid (not Out-of-token) samples: {sum(validity_flags)}/{len(prompts)}\n\n")
        
        sample_idx = 0
        for i, (prompt, full_resp, final_out, valid, q, gt) in enumerate(zip(
            prompts, full_responses, final_outputs, validity_flags, questions, ground_truths
        )):
            f.write(f"------------ Sample {i+1} ------------\n")
            
            if valid and final_out.strip():
                f.write(f"Judgment: {judgments[sample_idx]}\n")
                sample_idx += 1
            else:
                f.write(f"Judgment: N/A (invalid or empty output)\n")
            
            f.write(f"Question: {q}\n")
            f.write(f"Ground Truth: {gt}\n")
            f.write(f"Prompt: {prompt}\n")
            f.write(f"Full Response:\n{full_resp}\n")
            f.write(f"Final Output: {final_out}\n\n")
    
    print(f"Saved results to {log_file}")
    return accuracy


def analyze_layer_effects(
    args: argparse.Namespace,
    tokenizer: AutoTokenizer,
    intervened_samples: List[dict],
    steering_vectors: Dict[int, torch.Tensor],
    num_samples: int,
    num_layers: int,
    output_dir: Path
) -> Dict[str, List[float]]:
    """Analyze layer effects by comparing accuracy with positive and negative steering."""   
    
    # === Extract prompts, questions, and ground truths ===
    prompts, questions, ground_truths = [], [], []
    samples = intervened_samples[:num_samples]
    for sample in samples:
        prompt = sample['arguments']['gen_args_0']['arg_0']
        prompts.append(prompt)
        questions.append(str(sample['doc']['question']))
        ground_truths.append(str(sample['doc']['ground_truth']))

    # === Test with positive and negative steering coefficients ===
    coeff_scale = args.coeff_scale
    coefficients = [coeff_scale, -coeff_scale]
    layer_effects = {"0.0": [], str(coeff_scale): [], str(-coeff_scale): []}
    
    # === Setup baseline model (no steering) ===
    baseline_model = LLM(
        model=args.model_name,
        tokenizer=args.model_name,
        tensor_parallel_size=len(args.gpu.split(",")),
        trust_remote_code=True,
        gpu_memory_utilization=0.95,
        max_model_len=20000,
        seed=0,
    )
    final_outputs_baseline, baseline_validity, full_responses_baseline = generate_outputs(
        baseline_model, tokenizer, prompts, args.model_name
    )
    
    # === Evaluate baseline model (no steering) ===
    xverify_model = Model(model_name=f"IAAR-Shanghai/{args.xverify_model}")
    baseline_accuracy = generate_and_evaluate(
        final_outputs_baseline, questions, ground_truths,
        prompts, full_responses_baseline, baseline_validity,
        args.model_name, xverify_model, output_dir, "baseline"
    )
    layer_effects["0.0"] = [baseline_accuracy] * num_layers
    del xverify_model
    torch.cuda.empty_cache(); gc.collect()
    
    # === Evaluate each layer with steering ===
    for layer_idx in tqdm(range(num_layers), desc="Processing layers"):
        if layer_idx not in args.steer_layers or layer_idx not in steering_vectors:
            # No steering for this layer
            layer_effects[str(coeff_scale)].append(0.0)
            layer_effects[str(-coeff_scale)].append(0.0)
            continue
            
        # Test each coefficient (-1.0 and 1.0) for this layer
        for coeff in coefficients:
            
            # Prepare steering config
            steer_action = "add"
            coeff_id = str(coeff)
            steer_tag = f"layer_{layer_idx}_coeff_{coeff_id}_{steer_action}"
            steer_config = {}
            sv = steering_vectors[layer_idx]
            steer_config[f"layers.{layer_idx}"] = {
                "steering_vector": sv,
                "steering_coefficient": coeff,
                "action": steer_action,
            }
            
            # Save steering config
            probe_dir = get_probe_dir(project_root, args)
            steer_config_dir = probe_dir / "steer_configs"
            steer_config_dir.mkdir(parents=True, exist_ok=True)
            steer_config_path = steer_config_dir / f"{steer_tag}.pt"
            torch.save(steer_config, steer_config_path)
            
            # Apply steering for vllm model
            temp_model_dir = (
                project_root.parent
                / "results" / "steered_ckpt"
                / args.model_name.replace("/", "__")
                / steer_tag
            )       
            apply_steering_vllm(
                model_name=args.model_name,
                output_dir=temp_model_dir,
                steer_config_path=str(steer_config_path),
            )
            del steer_config
            torch.cuda.empty_cache()    

            # Generate steered outputs
            steered_model = LLM(
                model=str(temp_model_dir),
                tokenizer=args.model_name,
                tensor_parallel_size=len(args.gpu.split(",")),
                trust_remote_code=True,
                gpu_memory_utilization=0.95,
                max_model_len=20000,
                seed=0,
            )          
            final_outputs_steered, steered_validity, full_responses_steered = generate_outputs(
                steered_model, tokenizer, prompts, args.model_name
            )
            
            # Evaluate steered model
            xverify_model = Model(model_name=f"IAAR-Shanghai/{args.xverify_model}")
            accuracy = generate_and_evaluate(
                final_outputs_steered, questions, ground_truths,
                prompts, full_responses_steered, steered_validity,
                args.model_name, xverify_model, output_dir, 
                f"layer_{layer_idx}_coeff_{coeff}"
            )                        
            layer_effects[str(coeff)].append(accuracy)            
            del xverify_model
            torch.cuda.empty_cache(); gc.collect()

            # Cleanup model checkpoint
            if steer_config_path.exists():
                steer_config_path.unlink()
            if temp_model_dir.exists():
                shutil.rmtree(temp_model_dir, ignore_errors=True)
    
    return layer_effects


def save_layer_effects_log(
    layer_effects: Dict[str, List[float]],
    output_dir: Path,
    model_name: str,
    num_samples: int,
    log_filename: str
) -> None:
    """Save detailed layer effects log and show top 5 layers with highest differences."""
    
    log_path = output_dir / log_filename
    with open(log_path, 'w') as f:
        f.write(f"Layer Effects Analysis Log\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Number of samples: {num_samples}\n")
        f.write(f"Number of layers: {len(layer_effects['0.0'])}\n")
        f.write("=" * 50 + "\n\n")
        
        # Get coefficient values
        keys = list(layer_effects.keys())
        pos_coeff = [k for k in keys if float(k) > 0 and k != "0.0"][0]
        neg_coeff = [k for k in keys if float(k) < 0][0]
        f.write("{:<6} {:<9} {:<9} {:<9} {:<11}\n".format("Layer", "Baseline", f"Pos({pos_coeff})", f"Neg({neg_coeff})", "Difference"))
        f.write("{:-<6} {:-<9} {:-<9} {:-<9} {:-<11}\n".format("", "", "", "", ""))
        
        # Calculate differences
        differences = []
        for i, (baseline, positive, negative) in enumerate(zip(
            layer_effects["0.0"], 
            layer_effects[pos_coeff], 
            layer_effects[neg_coeff]
        )):
            diff = positive - negative
            differences.append((i, diff))
            f.write("{:<6} {:<9.4f} {:<9.4f} {:<9.4f} {:<11.4f}\n".format(i, baseline, positive, negative, diff))
        
        f.write("\n" + "=" * 50 + "\n")
        f.write("Top 5 layers with highest positive-negative differences:\n")
        f.write("=" * 50 + "\n")
        
        # Sort to log differences
        differences.sort(key=lambda x: x[1], reverse=True)
        for i, (layer, diff) in enumerate(differences[:5], 1):
            baseline = layer_effects["0.0"][layer]
            positive = layer_effects[pos_coeff][layer]
            negative = layer_effects[neg_coeff][layer]
            f.write(f"{i}. Layer {layer}: diff={diff:.4f} (baseline={baseline:.4f}, positive={positive:.4f}, negative={negative:.4f})\n")


def main():
    """Main function to run layer effect analysis based on accuracy differences."""
    parser = argparse.ArgumentParser(description="Analyze layer effects of steering vectors based on accuracy.")
    
    # Evaluation Arguments
    parser.add_argument('--model-name', type=str, required=True, help='Name of the model to analyze')
    parser.add_argument("--gpu", type=str, default="0", help="GPU ID to use (e.g., '0' for single GPU, '0,1,2,3' for multi-GPU).")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of samples for evaluation.")
    parser.add_argument("--xverify-model", type=str, default="xVerify-0.5B-I", help="xVerify model to use for evaluation.")

    # Activation Arguments
    parser.add_argument("--layers", type=str, default=None, help="List of layer indices to evaluate. If None, evaluate all layers.")
    parser.add_argument("--coeff-scale", type=float, default=1.0, help="Coefficient scale for steering (e.g., 1.0 evaluates -1.0 and 1.0)")
    
    args = parser.parse_args()
    
    # === Base Setup ===
    args.dataset = "gsm8k"
    args.subset = None
    args.split = "train"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["VLLM_USE_V1"] = "1"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # === Load steering layers ===    
    num_layers = AutoConfig.from_pretrained(args.model_name).num_hidden_layers
    if args.layers:
        args.steer_layers = [int(x) for x in args.layers.split(",")]
    else:
        args.steer_layers = list(range(num_layers))
    
    steering_vectors = load_steering_config(args)
    if not steering_vectors:
        print("Error: Could not load or compute steering vectors.")
        sys.exit(1)
    
    # === Load intervened samples and filter them based on thinking judgment ===
    dataset_name = get_task_name(args.dataset, Path("configs"), args.subset, args.split)
    model_path = args.model_name.replace("/", "__")
    samples_dir = project_root.parent / "results" / dataset_name / "intervened_local" / model_path
    latest_jsonl = find_latest_file(samples_dir, "samples_*.jsonl")
    all_samples = [json.loads(line) for line in open(latest_jsonl, 'r', encoding='utf-8')]
    intervened_samples = [
        s for s in all_samples
        if s.get('judgment').get('thinking') == 'incorrect'
    ]
    intervened_samples = intervened_samples[:args.limit or None]
    
    # === Output directories ===
    output_dir = get_probe_dir(project_root, args)
    layer_effect_dir = output_dir / "layer-effect"
    layer_effect_dir.mkdir(parents=True, exist_ok=True)
        
    # === Set unified naming structure ===
    save_name = "layer_effects"
    if args.layers:
        save_name += f"_{args.layers.replace(',', '_')}"
    save_name += f"_coeff_{str(args.coeff_scale).replace('.', '_')}"
    log_filename = f"{save_name}_summary.log"
    
    # === Analyze layer effects and save results ===
    layer_effects = analyze_layer_effects(
        args, 
        tokenizer, 
        intervened_samples, 
        steering_vectors,
        args.limit,
        num_layers,
        layer_effect_dir  
    )    
    save_layer_effects_log(layer_effects, layer_effect_dir, args.model_name, len(intervened_samples), log_filename)
    

if __name__ == "__main__":
    main()

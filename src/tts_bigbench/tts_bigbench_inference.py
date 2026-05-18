import json
import sys
from pathlib import Path
from datetime import datetime
from vllm import SamplingParams

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))
from utils import find_latest_file, get_think_tags
from tts_bigbench.tts_bigbench_data import load_stacked_data

# Token budgets
MAX_TOKENS_FIRST_PASS = 16384   # Reasoning budget per stack
MAX_TOKENS_SECOND_PASS = 8192   # Final-answer budget after truncation
MAX_MODEL_LEN = 100000          # Global max model length


def save_generations(samples: list, responses: list, output_dir: Path, mode: str):
    """Save generation results."""
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S.%f")
    samples_file = output_dir / f"samples_{timestamp}.jsonl"
    
    with open(samples_file, "w", encoding="utf-8") as f:
        for sample, response in zip(samples, responses):
            prompt = sample["full_intervened_prompt"]
            result = {
                "doc_id": sample["doc_id"],
                "doc": {
                    "id": sample["doc_id"],
                    "question": sample["question"],
                    "ground_truth": sample["ground_truth"],
                    "mistake_index": sample.get("mistake_index"),
                    "is_correct": sample.get("is_correct"),
                },
                "arguments": {"gen_args_0": {"arg_0": prompt}},
                "resps": [[response]],
                "target": sample["ground_truth"],
            }
            if mode == "right":
                result["doc"]["search"] = sample.get("search")
                result["doc"]["stack"] = sample.get("stack")
            if sample.get("truncated"):
                result["truncated"] = True
            if "judgment" in sample:
                result["judgment"] = sample["judgment"]
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    print(f"INFO: Saved {len(samples)} samples to {samples_file}")


def generate_with_two_pass(llm, samples, prompts, think_end_tag):
    """Generate with a first pass for reasoning and a second pass for final answer if truncated."""
    # First pass: reasoning
    sampling_params = SamplingParams(temperature=0.6, max_tokens=MAX_TOKENS_FIRST_PASS, seed=0)
    generations = llm.generate(prompts, sampling_params)
    
    responses = []
    truncated_indices, truncated_prompts = [], []
    
    for i, (sample, gen) in enumerate(zip(samples, generations)):
        response = gen.outputs[0].text
        
        # If hit max tokens and thinking block is not closed
        if gen.outputs[0].finish_reason == "length" and think_end_tag not in response:
            sample["truncated"] = True
            truncated_indices.append(i)
            
            # Close thinking and prepare for final answer
            full_resp = response.rstrip() + think_end_tag
            truncated_prompts.append(prompts[i] + full_resp)
            responses.append(full_resp)
        else:
            responses.append(response)
            
    # Second pass: final answer
    if truncated_prompts:
        print(f"INFO: {len(truncated_prompts)} samples truncated, running second pass for final answers...")
        sampling_params_2 = SamplingParams(temperature=0.6, max_tokens=MAX_TOKENS_SECOND_PASS, seed=0)
        final_gens = llm.generate(truncated_prompts, sampling_params_2)
        
        for idx, final_gen in zip(truncated_indices, final_gens):
            responses[idx] += final_gen.outputs[0].text
            
    return responses


def run_bigbench_natural(args, llm, output_dir: Path, samples: list):
    """Run natural generation on BIG-Bench data (no Wait appended)."""
    
    print("INFO: Running natural generation (Stack 0)...")
    natural_dir = output_dir / "natural"
    natural_dir.mkdir(parents=True, exist_ok=True)
    
    _, think_end_tag = get_think_tags(args.model_name, project_root.parent / "configs" / "model_config.yaml")
    
    # === Generate using the intervened prompt ===
    prompts = [s["full_intervened_prompt"] for s in samples]
    responses = generate_with_two_pass(llm, samples, prompts, think_end_tag)
    save_generations(samples, responses, natural_dir, mode="natural")


def run_tts_bigbench_right(args, llm, output_dir: Path, stacks: int) -> int:
    """Run RIGHT mode: extend thinking by stacking 'Wait' interventions."""
    
    _, think_end_tag = get_think_tags(args.model_name, project_root.parent / "configs" / "model_config.yaml")
    
    for stack in range(1, stacks + 1):
        print(f"\n--- RIGHT TTS BIGBENCH: Stack {stack} ---")
        stack_dir = output_dir / f"stack_{stack}"
        stack_dir.mkdir(parents=True, exist_ok=True)
        
        # Load previous stack results (natural for stack 1)
        base_dir = output_dir / ("natural" if stack == 1 else f"stack_{stack - 1}")
        latest_file = find_latest_file(base_dir, "samples_*.jsonl")
        
        # Load and prepare: extracts thinking, removes think end tag, appends "Wait"
        stacked_samples = load_stacked_data(
            generated_path=latest_file,
            model_name=args.model_name,
            stack=stack,
            limit=args.limit,
        )
        
        # Generate with two-pass logic and save results
        prompts = [s["full_intervened_prompt"] for s in stacked_samples]
        try:
            responses = generate_with_two_pass(llm, stacked_samples, prompts, think_end_tag)
        except ValueError as e:
            if "longer than the maximum model length" in str(e):
                print(f"WARNING: Stack {stack} exceeded max model length. Stop at stack {stack - 1}.")
                stack_dir.rmdir()
                return stack - 1
            raise
        save_generations(stacked_samples, responses, stack_dir, mode="right")
    
    return stacks

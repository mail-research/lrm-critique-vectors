import json
import sys
from pathlib import Path
from typing import List, Optional
from transformers import AutoTokenizer

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))
from utils import get_think_tags

# Simple question templates for TTS (no error detection instruction)
TTS_BIGBENCH_TEMPLATES = {
    "multistep_arithmetic": "Solve the arithmetic expression.\n\nProblem: {input}",
    "word_sorting": "Sort the following words alphabetically.\n\nProblem: {input}",
    "dyck_languages": "Complete the rest of the sequence, making sure that the parentheses are closed properly.\n\nProblem: {input}",
    "logical_deduction": "Solve the following logical deduction problem.\n\nProblem:\n{input}",
    "tracking_shuffled_objects": "Track the shuffled objects.\n\nProblem:\n{input}",
}


def load_bigbench_data(
    split: str,
    model_name: str,
    limit: Optional[int] = None,
) -> List[dict]:
    """
    Load BIG-Bench Mistake data with full CoT pre-filled.
    
    For all samples, we pre-fill the complete CoT trace (all steps).
    - Correct samples (mistake_index=null): Full correct reasoning
    - Mistake samples (mistake_index>=0): Full reasoning including errors
    
    This tests if LRMs can self-correct through extended thinking ("Wait").
    """
    
    template = TTS_BIGBENCH_TEMPLATES.get(split)
    if template is None:
        raise ValueError(f"Unknown split: {split}. Available: {list(TTS_BIGBENCH_TEMPLATES.keys())}")
    
    # === Load data file ===
    data_dir = project_root.parent / "data"
    file_path = data_dir / f"{split}.jsonl"
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    print(f"INFO: Loading BIG-Bench Mistake data from {file_path}")
    
    # === Get think tags and tokenizer ===
    config_path = project_root.parent / "configs"
    start_tag, end_tag = get_think_tags(model_name, config_path / "model_config.yaml")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # === Process each sample ===
    processed = []
    with open(file_path, "r") as f:
        for idx, line in enumerate(f):
            if limit and idx >= limit:
                break
                
            item = json.loads(line.strip())
            mistake_index = item.get("mistake_index")
            steps = item["steps"]
            
            # Determine if sample has mistakes (0-indexed)
            is_correct = (mistake_index is None)
    
            # Format chain-of-thought steps
            cot_formatted = "\n\n".join(steps)
            
            # Build prompt with template
            prompt = template.format(input=item["input"])
            messages = [{"role": "user", "content": prompt}]
            templated_prompt = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            
            # Add think start tag + pre-filled CoT
            if start_tag not in templated_prompt:
                templated_prompt += start_tag
            full_intervened_prompt = templated_prompt + cot_formatted
            
            processed.append({
                "doc_id": idx,
                "question": item["input"],
                "ground_truth": item.get("target"),
                "answer": item.get("answer"),
                "steps": steps,
                "mistake_index": mistake_index,
                "is_correct": is_correct,
                "full_intervened_prompt": full_intervened_prompt,
                "stack": 0,
            })
    
    correct_count = sum(1 for s in processed if s["is_correct"])
    incorrect_count = len(processed) - correct_count
    print(f"INFO: Loaded {len(processed)} samples (correct: {correct_count}, mistake: {incorrect_count})")
    
    return processed


def load_stacked_data(
    generated_path: Path,
    model_name: str,
    stack: int = 1,
    limit: Optional[int] = None,
) -> List[dict]:
    """Load previous stack results and append 'Wait' for continued thinking."""
    
    # === Load generated samples and setup tags ===
    config_path = project_root.parent / "configs"
    start_tag, end_tag = get_think_tags(model_name, config_path / "model_config.yaml")
    
    with open(generated_path, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f]
    if limit:
        samples = samples[:limit]
    
    # === Process each sample ===
    processed = []
    for sample in samples:
        doc_id = sample.get("doc_id")
        doc = sample.get("doc", {})
        prompt = sample["arguments"]["gen_args_0"]["arg_0"]
        resp = sample["resps"][0][0] if sample.get("resps") else ""
        full = prompt + resp
        
        # Extract thinking (between start and end tags)
        if start_tag in full and end_tag in full:
            start_idx = full.find(start_tag) + len(start_tag)
            end_idx = full.find(end_tag)
            thinking = full[start_idx:end_idx].strip()
        elif start_tag in full:
            start_idx = full.find(start_tag) + len(start_tag)
            thinking = full[start_idx:].strip()
        else:
            print(f"WARNING: Sample {doc_id} has no thinking, skipping.")
            continue
        
        # Reconstruct prompt with "Wait" appended
        new_thinking = thinking + "\n\nWait"
        base_prompt_end = full.find(start_tag) + len(start_tag)
        base_prompt = full[:base_prompt_end]
        full_intervened = base_prompt + new_thinking
        
        processed.append({
            "doc_id": doc_id,
            "question": doc.get("question"),
            "ground_truth": doc.get("ground_truth", sample.get("target")),
            "mistake_index": doc.get("mistake_index"),
            "is_correct": doc.get("is_correct"),
            "search": new_thinking,
            "full_intervened_prompt": full_intervened,
            "stack": stack,
        })
    
    return processed

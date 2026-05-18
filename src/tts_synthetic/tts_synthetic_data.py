import json
import sys
from pathlib import Path
from typing import List, Optional

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))
from utils import get_think_tags, get_task_name, find_latest_file


def load_synthetic_data(
    dataset: str,
    subset: Optional[str],
    split: str,
    model_name: str,
    limit: Optional[int] = None,
) -> List[dict]:
    """Load synthetic data from intervened_local results."""
    
    # === Build paths ===
    config_path = project_root.parent / "configs"
    dataset_name = get_task_name(dataset, config_path, subset, split)
    model_path = model_name.replace("/", "__")
    samples_dir = project_root.parent / "results" / dataset_name / "intervened_local" / model_path
    
    # === Load latest samples file ===
    latest_jsonl = find_latest_file(samples_dir, "samples_*.jsonl")
    if not latest_jsonl or not latest_jsonl.exists():
        raise FileNotFoundError(f"No samples file found in {samples_dir}")
    
    print(f"INFO: Loading synthetic data from {latest_jsonl}")
    with open(latest_jsonl, 'r', encoding='utf-8') as f:
        all_samples = [json.loads(line) for line in f]
    
    print(f"INFO: Loaded {len(all_samples)} samples")
    
    if limit:
        all_samples = all_samples[:limit]
    
    # === Process each sample ===
    processed = []
    for sample in all_samples:
        doc = sample.get("doc", {})
        doc_id = doc.get("id", sample.get("doc_id"))
        
        # Get the prompt and response
        prompt = sample["arguments"]["gen_args_0"]["arg_0"]
        processed.append({
            "doc_id": doc_id,
            "question": doc.get("question"),
            "ground_truth": doc.get("ground_truth"),
            "full_intervened_prompt": prompt,
            "stack": 0,
            "doc": doc,
        })
    
    return processed


def load_stacked_data(
    generated_path: Path,
    model_name: str,
    stack: int = 1,
    limit: Optional[int] = None,
) -> List[dict]:
    """
    Load previous stack results and append 'Wait' for continued thinking.
    """
    
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
            "search": new_thinking,
            "full_intervened_prompt": full_intervened,
            "stack": stack,
        })
    
    return processed

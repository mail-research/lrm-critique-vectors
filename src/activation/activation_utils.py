import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))
from utils import get_task_name


# ###################################################
#                     DIRECTORIES
# ###################################################

def get_probe_dir(project_root: Path, args: argparse.Namespace) -> Path:
    gsm8k_task = get_task_name("gsm8k", project_root.parent / "configs", "main", "train")

    segments = [
        project_root.parent / "results" / gsm8k_task, "intervened_local",
        args.model_name.replace("/", "__"), "probe",
    ]

    return Path(*segments)


def load_conflict_samples(
    args: argparse.Namespace,
    dataset_name: str
) -> Tuple[Optional[List[dict]], Optional[List[dict]]]:    
    """Load original and intervened conflict samples."""
    
    # === Load original samples ===
    samples_dir = project_root.parent / "results" / dataset_name
    original_dir = samples_dir / "baseline" / args.model_name.replace("/", "__")
    try:
        original_file = max(original_dir.glob("samples_*.jsonl"), key=os.path.getmtime)
        with open(original_file, "r", encoding="utf-8") as f:
            original_samples = [json.loads(line) for line in f]
        print(f"Loaded {len(original_samples)} original samples from {original_file}")
    except ValueError as e:
        print(f"Error: {e}")
        return None, None
    
    # === Load intervened samples ===
    intervened_dir = (
        project_root.parent / "results" / dataset_name /
        "intervened_local" / args.model_name.replace("/", "__")
    )
    try:
        intervened_file = max(intervened_dir.glob("samples_*.jsonl"), key=os.path.getmtime)
        with open(intervened_file, encoding="utf-8") as f:
            intervened_samples = [json.loads(line) for line in f]
        print(f"Loaded {len(intervened_samples)} intervened samples from {intervened_file}")
    except ValueError:
        print("No matching intervened samples found.")
        return None, None

    # === Find conflict samples in intervened data ===
    conflict_samples = [s for s in intervened_samples if s["judgment"]["thinking"] == "incorrect" and s["judgment"]["final"] == "correct"]
    conflict_ids = {s["doc"]["id"] for s in conflict_samples}
    print(f"Found {len(conflict_samples)} initial conflict samples with {len(conflict_ids)} unique IDs")

    # === Collect original samples corresponding to conflict IDs, filtering for correct ones ===
    correct_original_ids = {
        s["doc_id"] for s in original_samples
        if s["doc_id"] in conflict_ids and s["judgment"]["thinking"] == s["judgment"]["final"] == "correct"
    }
    original_conflict_samples = [s for s in original_samples if s["doc_id"] in correct_original_ids]
        
    # === Collect original correct samples with conflict IDs ===
    conflict_samples = [s for s in conflict_samples if s["doc"]["id"] in correct_original_ids]
    print(f"Found {len(conflict_samples)} pairs of (correct original, conflict intervened).")
    if not conflict_samples:
        raise ValueError("No conflict pairs found where the original sample is correct.")

    return original_conflict_samples, conflict_samples

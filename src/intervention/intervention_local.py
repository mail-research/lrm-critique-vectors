import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple
import yaml

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))
from intervention import GPTIntervener
from utils import (
    get_think_tags,
    extract_reasoning_trace,
    get_task_name,
    BASE_CONFIG,
    find_latest_file,
)


def apply_llm_intervention(
    generated_path: Path,
    model_name: str,
    api_key: str,
    gpt_model: str,
    output_path: Path,
    ids: Optional[List[int]] = None,
    limit: Optional[int] = None,
) -> None:
    """Generate reasoning traces for specified samples using an LLM."""
    
    # === Load JSONL generated sample ===
    latest_samples_file = find_latest_file(generated_path / model_name.replace("/", "__"), "samples_*.jsonl")
    if not latest_samples_file or not latest_samples_file.exists():
        raise FileNotFoundError(f"No samples file (samples_*.jsonl) found in {generated_path}")
    print(f"INFO: Loading samples from {latest_samples_file}")
    with open(latest_samples_file, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f]
    
    # === Setup basic formats ===
    dataset_name = generated_path.parent.name
    data_processor = dataset_name.split("-")[0]
    model_formats_path = Path(__file__).resolve().parents[2] / "configs/model_config.yaml"
    start_tag, end_tag = get_think_tags(model_name, model_formats_path)

    # === Prepare processing ===
    samples_by_id = {s["doc_id"]: s for s in samples}
    target_ids = set(ids) if ids is not None else set(samples_by_id.keys())
    if limit is not None:
        target_ids = set(list(target_ids)[:limit])
    print(f"LLM intervention: {len(target_ids)} samples will be processed to generate traces.")

    # === Initialize LLM client to intervene ===
    intervener = GPTIntervener(api_key=api_key, model_name=gpt_model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for sample_id in target_ids:
            if sample_id not in samples_by_id:
                continue
                    
            # === Extract question and ground truth ===
            print(f"Generating traces for sample ID: {sample_id}")
            sample = samples_by_id[sample_id]
            question, ground_truth, _, _ = extract_reasoning_trace(
                sample=sample,
                data_processor=data_processor or dataset_name,
                start_tag=start_tag,
                end_tag=end_tag,
            )
        
            # === Generate intervened traces using LLM ===
            trace = intervener.intervene_reasoning_chunks(
                question=question, ground_truth=ground_truth
            )
            if trace is None:
                print(f"Could not generate trace for sample {sample_id}")
                continue

            content = trace.get("content", "")
            thinking_trace = content + "\n\n"
            processed_sample = {
                "id": sample_id,
                "question": question,
                "ground_truth": ground_truth,
                "search": thinking_trace,
            }
            f.write(json.dumps(processed_sample) + "\n")


def build_local_task_yaml(
    dataset: str,
    subset: Optional[str],
    split: Optional[str],
    model_name: str,
    intervened_file_path: Path,
    task_name_suffix: Optional[str] = None,
    custom_doc_to_text: Optional[str] = None,
) -> Tuple[str, str]:
    """Build and save a YAML configuration file for a local intervention task."""
    
    project_root = Path(__file__).resolve().parents[1]
    dataset_name = get_task_name(
        dataset=dataset,
        config_path=project_root.parent / "configs",
        subset=subset,
        split=split,
    )

    # === Setup task name and config ===
    task_name_parts = [
        dataset_name, model_name.split("/")[0], "intervened", "local", 
    ]
    base_task_name = "_".join(task_name_parts)
    full_task_name = f"{base_task_name}_{task_name_suffix}" if task_name_suffix else base_task_name
    task_config = {
        **BASE_CONFIG,
        "task": full_task_name,
        "dataset_path": "json",
        "dataset_kwargs": {"data_files": {split: str(intervened_file_path)}},
        "doc_to_text": custom_doc_to_text or "{{full_intervened_prompt}}",
        "doc_to_target": "",
        "test_split": split,
    }

    # === Save the task config to a yaml file in lm-eval's directory ===
    tasks_dir_base = (
        project_root / "lm-eval" / "lm_eval" / "tasks" / "intervention_tasks" / model_name.split("/")[0]
    )

    if "steer_local" in full_task_name:
        tasks_dir = tasks_dir_base / "steer" / "local"
    else:
        tasks_dir = tasks_dir_base / "intervened_local"
    
    tasks_dir.mkdir(parents=True, exist_ok=True)
    task_yaml_path = tasks_dir / f"{full_task_name}.yaml"
    with open(task_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(task_config, f)
        
    return full_task_name, split

import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple
import yaml

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from utils import (
    get_think_tags,
    extract_reasoning_trace,
    get_task_name,
    BASE_CONFIG,
)

def apply_tts_intervention(
    generated_path: Path,
    model_name: str,
    dataset_name: str,
    output_path: Path,
    stack: int = 1,
    intervention_source: str = "baseline",
    ids: Optional[List[int]] = None,
    limit: Optional[int] = None,
):
    """Append "Wait" to the reasoning trace of generated responses."""
    
    # === Load JSONL generated sample ===
    with open(generated_path, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f]

    # === Setup basic formats ===
    model_formats_path = Path(__file__).resolve().parents[2] / "configs/model_config.yaml"
    start_tag, end_tag = get_think_tags(model_name, model_formats_path)
    
    if stack > 1 or intervention_source == "intervened_local":
        data_processor = "tts"
    else:
        if not dataset_name:
            raise ValueError("dataset_name must be provided for stack 1")
        data_processor = dataset_name.split("-")[0]

    # === Prepare processing ===
    samples_by_id = {s["doc_id"]: s for s in samples}
    target_ids = set(ids) if ids is not None else set(samples_by_id.keys())
    if limit is not None:
        target_ids = set(list(target_ids)[:limit])
    print(f"TTS intervention: {len(target_ids)} samples will be processed.")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for sample_id in target_ids:
            if sample_id not in samples_by_id:
                continue
            
            sample = samples_by_id[sample_id]
            
            question, ground_truth, full_template_prompt, thinking_trace = extract_reasoning_trace(
                sample=sample,
                data_processor=data_processor,
                start_tag=start_tag,
                end_tag=end_tag,
            )

            if thinking_trace is None:
                thinking_trace = ""

            # Append "Wait"
            new_thinking_trace = thinking_trace.strip() + "\n\n" + "Wait"
            processed_data = {
                "id": sample_id,
                "question": question,
                "ground_truth": ground_truth,
                "search": new_thinking_trace,
                "full_intervened_prompt": full_template_prompt + new_thinking_trace,
            }
            f.write(json.dumps(processed_data) + "\n")


def build_tts_task_yaml(
    dataset: str,
    subset: Optional[str],
    split: Optional[str],
    model_name: str,
    intervened_file_path: Path,
    stack: int,
) -> Tuple[str, str]:
    """Build and save a YAML configuration file for a TTS intervention task."""
    
    project_root = Path(__file__).resolve().parents[1]
    dataset_name = get_task_name(
        dataset=dataset,
        config_path=project_root.parent / "configs",
        subset=subset,
        split=split,
    )

    # === Setup task name and config ===
    task_name = f"{dataset_name}_intervened_tts_stack_{stack}"
    task_config = {
        **BASE_CONFIG,
        "task": task_name,
        "dataset_path": "json",
        "dataset_kwargs": {"data_files": {split: str(intervened_file_path)}},
        "doc_to_text": "{{full_intervened_prompt}}",
        "doc_to_target": "",
        "test_split": split,
    }

    # Save the task config to a yaml file in lm-eval's directory
    model_prefix = model_name.split("/")[0]
    tasks_dir = (
        project_root / "lm-eval" / "lm_eval" / "tasks" / "intervention_tasks" / model_prefix / "tts"
    )
    
    tasks_dir.mkdir(parents=True, exist_ok=True)
    task_yaml_path = tasks_dir / f"{task_name}.yaml"
    with open(task_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(task_config, f)
        
    return task_name, split

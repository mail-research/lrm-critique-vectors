import re
from pathlib import Path
import yaml
import sys
import os
import torch
import gc
import ray
from typing import Dict, Any, Optional
from vllm import LLM
from .data_process import DATASET_PROCESSORS
from vllm.distributed.parallel_state import (
    destroy_model_parallel,
    destroy_distributed_environment,
)


with open(Path(__file__).resolve().parents[2] / "configs" / "generation_config.yaml", "r") as f:
    BASE_CONFIG = yaml.safe_load(f)


def get_task_name(dataset: str, config_path: Path, subset: str = None, split: str = None) -> str:
    """Get the task name based on dataset, subset, and split."""
    dataset_file = config_path / f"{dataset}.yaml"
    with open(dataset_file, "r") as f:
        dataset_config = yaml.safe_load(f)
    
    task_subject = subset or dataset_config.get("dataset_name")
    task_name = f"{dataset}-{task_subject}" if task_subject else dataset
    return f"{task_name}-{split}" if split else task_name


def get_think_tags(model_name, formats):
    """Get the thinking tags for a specific model."""
    if isinstance(formats, (str, Path)): 
        with open(formats, "r") as f:
            model_formats = yaml.safe_load(f)
    else:
        model_formats = formats
        
    for fmt in model_formats:
        if any(name.lower() in model_name.lower() for name in fmt["names"]):
            return fmt["format"]["think_start"], fmt["format"]["think_end"]
    return "<think>\n", "\n</think>\n\n"


def get_dataset_config(
    dataset: str,
    model_name: str,
    subset: str = None,
    split: str = None,
    config_path: Path = Path("configs"),
    intervention_content: Optional[str] = None,
    intervention_position: Optional[str] = None,
) -> Dict[str, Any]:
    """Load the dataset configuration from a YAML file."""
    
    # === Load dataset YAML ===
    dataset_file = config_path / f"{dataset}.yaml"
    with open(dataset_file, "r") as f:
        dataset_config = yaml.safe_load(f)

    # === Build task name ===
    prefix  = model_name.split("/")[0].replace('/', '_')
    task = get_task_name(dataset, config_path, subset, split)
    task += (
        f"_{prefix}_{intervention_position}"
        if intervention_content else
        f"_{prefix}_baseline"
    )

    # === Merge configs ===
    config = {**BASE_CONFIG, **dataset_config, "task": task}
    if "generation_kwargs" in dataset_config:
        config["generation_kwargs"] = {
            **BASE_CONFIG.get("generation_kwargs", {}),
            **dataset_config["generation_kwargs"]
        }
    
    # === Override dataset name if subset provided ===
    if subset and "dataset_name" in config:
        config["dataset_name"] = subset

    return config


def extract_timestamp_from_filename(filename: str) -> str:
    """Extract timestamp from filename using regex."""
    match = re.search(r'(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}\.\d+)', filename)
    return match.group(1) if match else "unknown_timestamp"


def split_reasoning_into_chunks(reasoning_trace: str) -> list[str]:
    """Split a reasoning trace into chunks based on \n\n."""
    return [chunk.strip() for chunk in reasoning_trace.split("\n\n") if chunk.strip()]


def extract_reasoning_trace(sample, data_processor, start_tag, end_tag):
    """Extract the question, ground truth, and reasoning trace from lm-eval results."""
    doc = sample["doc"]
    processor = DATASET_PROCESSORS.get(data_processor)
    question, ground_truth = processor(doc)

    # === Full question with model generated response appended ===
    full_template_prompt = sample["arguments"]["gen_args_0"]["arg_0"]
    full_text = full_template_prompt + sample["resps"][0][0]

    # === Extract the thinking trace only ===
    pattern = re.compile(f"{re.escape(start_tag)}(.*?){re.escape(end_tag)}", re.DOTALL)
    match = pattern.search(full_text)

    if match:
        return question, ground_truth, full_template_prompt, match.group(1).strip()
    elif start_tag in full_text:
        return question, ground_truth, full_template_prompt, "out_of_token"
    else:
        raise ValueError("Invalid thinking trace in the response.")


def find_latest_file(directory: Path, pattern: str) -> Path:
    """Find the latest file in a directory matching a pattern."""
    try:
        matching_files = list(directory.glob(pattern))
        if not matching_files:
            raise ValueError(f"No file matching '{pattern}' found in {directory}")
        return max(matching_files, key=os.path.getmtime)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)


def cleanup_vllm(llm: LLM):
    """Cleanup vLLM resources."""
    destroy_model_parallel()
    destroy_distributed_environment()
    llm.llm_engine.engine_core.shutdown()
    del llm.llm_engine.model_executor
    del llm
    torch.cuda.empty_cache()
    gc.collect()
    ray.shutdown()
import yaml
import sys
import shutil
from pathlib import Path
from typing import Optional, List, Tuple

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))
from utils import get_think_tags, get_dataset_config, get_task_name


def build_thinking_prompt(
    text,
    model_name,
    tokenizer,
    model_formats,
    thinking=None,
    thinking_end=True,
    immediate_answer=False
):
    """Format user input and assistant thinking into a prompt."""
    
    # === User part ===
    chat = [{"role": "user", "content": text}]
    user_prompt = tokenizer.apply_chat_template(chat, tokenize=False)

    # === Model-specific prefix & thinking tags ===
    think_start_tag, think_end_tag = get_think_tags(model_name, model_formats)
    prefix = next(fmt["format"]["prefix"] for fmt in model_formats if any(
        name.lower() in model_name.lower() for name in fmt["names"]
    ))
    
    # === Assistant part ===
    if thinking is None:
        assistant_part = prefix + think_start_tag
    else:
        assistant_part = prefix
        if think_start_tag not in thinking:
            assistant_part += think_start_tag
        assistant_part += thinking
        
        for end_tag in ["</think>", "</thought>"]:
            assistant_part = assistant_part.replace(end_tag, "")
    
    # === Directly end the thinking ===
    if thinking_end:
        assistant_part += think_end_tag

    # === Directly add the answer ===
    if immediate_answer:
        assistant_part += " The answer is:"

    return user_prompt + assistant_part


def build_intervention_prompt(
    text: str,
    model_name: str,
    model_formats: list,
    tokenizer,
    intervention_content: str,
    intervention_position: str,
    thinking_end: bool,
    immediate_answer: bool
) -> str:
    """
    Construct the prompt with intervention at the specified position.
    Returns the full prompt string for doc_to_text in YAML file.
    """
    
    if intervention_content is None:
        return build_thinking_prompt(
            text=text,
            model_name=model_name,
            tokenizer=tokenizer,
            model_formats=model_formats,
            thinking=None,
            thinking_end=thinking_end,
            immediate_answer=immediate_answer
        )

    if intervention_position == "after_prompt":
        # === Insert intervention before assistant's thinking ===
        return build_thinking_prompt(
            text=text + " " + intervention_content,
            model_name=model_name,
            tokenizer=tokenizer,
            model_formats=model_formats,
            thinking=None,
            thinking_end=thinking_end,
            immediate_answer=immediate_answer
        )
    
    elif intervention_position == "after_think":
        # === Insert intervention as the assistant's thinking content ===
        return build_thinking_prompt(
            text=text,
            model_name=model_name,
            tokenizer=tokenizer,
            model_formats=model_formats,
            thinking=intervention_content,
            thinking_end=thinking_end,
            immediate_answer=immediate_answer
        )
    
    raise ValueError("Position must be 'after_prompt' or 'after_think'")


def build_global_task_yaml(
    *,
    dataset: str,
    subset: Optional[str] = None,
    split: Optional[str] = None,
    model_name: str,
    tokenizer,
    intervention_type: Optional[str] = None,
    intervention_content: Optional[str] = None,
    intervention_position: Optional[str] = None,
    thinking_end: bool = False,
    immediate_answer: bool = False,
    task_name: Optional[str] = None,
    sample_ids: Optional[List[int]] = None,
    custom_doc_to_text: Optional[str] = None,
    lm_eval_path: Path = Path("src/lm-eval"),
    config_path: Path = Path("configs"),
) -> Tuple[str, str]:
    """Build and save a YAML configuration file for a global intervention task."""
    
    # === Load dataset configuration ===
    task_config = get_dataset_config(
        dataset=dataset,
        subset=subset,
        split=None,
        model_name=model_name,
        config_path=config_path,
        intervention_content=intervention_content,
        intervention_position=intervention_position,
    )
    split = split or task_config.get("test_split")
    task_config["test_split"] = split

    # === Regenerate task name ===
    if task_name is None:
        base_task_name = get_task_name(dataset, config_path, subset, split)
        prefix = model_name.split("/")[0].replace('/', '_')
        intervention_suffix = (
            f"_{prefix}_{intervention_position}"
            if intervention_content else
            f"_{prefix}_baseline"
        )
        task_name = base_task_name + intervention_suffix
    task_config["task"] = task_name

    # === Update intervention prompts ===
    if custom_doc_to_text:
        task_config["doc_to_text"] = custom_doc_to_text
    else:
        formats_file = config_path / "model_config.yaml"
        with open(formats_file, "r") as f:
            model_formats = yaml.safe_load(f)
        
        base_doc_to_text = task_config["doc_to_text"]
        intervention_prompt = build_intervention_prompt(
            text=base_doc_to_text,
            model_name=model_name,
            model_formats=model_formats,
            tokenizer=tokenizer,
            intervention_content=intervention_content,
            intervention_position=intervention_position,
            thinking_end=thinking_end,
            immediate_answer=immediate_answer,
        )
        task_config["doc_to_text"] = intervention_prompt

    # === Directory of yaml files ===
    tasks_dir = lm_eval_path / "lm_eval" / "tasks" / "intervention_tasks"
    model_prefix = model_name.split("/")[0]
    
    if "steer_global" in task_name:
        task_dir = tasks_dir / model_prefix / "steer" / "global"
    elif intervention_type == "global" and intervention_content is not None:
        task_dir = tasks_dir / model_prefix / "intervened_global" / intervention_position
    elif intervention_type is None and intervention_content is None:
        task_dir = tasks_dir / model_prefix / "baseline"
    else:
        raise ValueError(
            f"Unsupported combination: intervention_type={intervention_type}, "
            f"intervention_content={'not None' if intervention_content is not None else 'None'}"
        )
    task_dir.mkdir(parents=True, exist_ok=True)

    # === For gpqa dataset, copy utils.py file from lm-eval repo ===
    process_docs_function = None
    if dataset == "gpqa":
        utils_src = lm_eval_path / "lm_eval/tasks/gpqa/zeroshot/utils.py"
        utils_dest = task_dir / "utils.py"
        shutil.copy2(utils_src, utils_dest)
        process_docs_function = "utils.process_docs"

    # === If sample_ids is provided, create utils.py to filter samples ===
    if sample_ids is not None:
        utils_path = task_dir / "utils.py"
        with open(utils_path, "w") as f:
            f.write("from datasets import Dataset\n\n")
            f.write("def process_docs(docs):\n")
            f.write(f"    sample_ids_to_keep = {set(sample_ids)}\n")
            f.write("    docs = docs.add_column('original_doc_id', range(len(docs)))\n")
            f.write("    filtered_docs = docs.filter(lambda x: x['original_doc_id'] in sample_ids_to_keep)\n")
            f.write("    return filtered_docs\n")
        process_docs_function = "utils.process_docs"
    
    # === Save YAML file ===
    yaml_file = task_dir / f"{task_name}.yaml"
    config_to_dump = {k: v for k, v in task_config.items() if k != "task"}
    with open(yaml_file, "w", encoding="utf-8") as f:
        yaml.dump({"task": task_name, **config_to_dump}, f, allow_unicode=True, sort_keys=False)
        if process_docs_function:
            f.write(f"\nprocess_docs: !function {process_docs_function}\n")

    return task_name, split
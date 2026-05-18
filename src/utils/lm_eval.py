import subprocess
import os
from pathlib import Path
import shutil
from .utils import get_think_tags, get_task_name
from .logger import save_responses
project_root = Path(__file__).resolve().parents[1]


def run_lm_evaluation(
    model_name: str, 
    task_name: str, 
    limit: int = None,
    gpu: str = None,
    dataset: str = None,
    subset: str = None,
    split: str = None,
    xverify_model: str = None,
    output_dir: Path = None,
    intervention_type: str = None,
    # Global intervention args
    intervention_content: str = None,
    intervention_position: str = None,
    thinking_end: bool = False,
    immediate_answer: bool = False,
    # Local intervention args
    intervene_sample: str = None,
    config_path: Path = Path("configs"),
    # Steering arg
    steer_config_path: str = None,
    steer_layers: list = None,
    steer_coeff: float = None,
    steer_action: str = None,
    cleanup_model: bool = False,
):
    """Run lm-evaluation-harness evaluation."""

    # === GPU handling ===
    gpu_list = [g.strip() for g in (gpu or "0").split(",") if g.strip()]
    gpu, num_gpus = ",".join(gpu_list), len(gpu_list)
    
    # === Model arguments ===
    model_args = [
        f"pretrained={model_name}",
        "trust_remote_code=True",
        f"tensor_parallel_size={num_gpus}",
        "gpu_memory_utilization=0.95",
        f"max_model_len=20000",
        "seed=0",
    ]   
    model_args_str = ",".join(model_args)
    
    # === Output directory ===
    if output_dir is None:
        task_prefix = get_task_name(dataset, config_path, subset, split)
        subdir = (
            Path("intervened_global") / intervention_position
            if intervention_content
            else Path("baseline")
        )
        output_dir = Path.cwd() / "results" / task_prefix / subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # === lm-eval command ===
    cmd = [
        "python", "-m", "lm_eval",
        "--model", "vllm",
        "--model_args", model_args_str,
        "--tasks", task_name,
        "--batch_size", "auto",
        "--output_path", str(output_dir),
        "--seed", "0",
        "--log_samples",
        "--predict_only",
    ]
    if limit:
        cmd += ["--limit", str(limit)]

    # === Environment setup ===
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["VLLM_USE_V1"] = "1"
    env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    print(f"INFO: Using GPUs: {gpu}")
    
    # === Print command ===
    sep = "=" * 80
    print(f"\n{sep}\n\033[1;33m{'Evaluation Run':^80}\033[0m\n{sep}")
    print(f"\033[1mModel:\033[0m {model_name}\n\033[1mTask:\033[0m {task_name}\n\033[1mOutput dir:\033[0m {output_dir}\n\033[1mlm-eval Command:\033[0m")
    cyan, reset, i = "\033[1;36m", "\033[0m", 0
    while i < len(cmd):
        args = f"{cmd[i]} {cmd[i+1]}" if i + 1 < len(cmd) and not cmd[i+1].startswith("--") else cmd[i]
        print(f"  {cyan}{args}{reset} \\ ")
        i += 2 if i + 1 < len(cmd) and not cmd[i+1].startswith("--") else 1
    print(f"{sep}\n")
    
    # === Run command ===
    subprocess.run(cmd, check=True, cwd=project_root, env=env)
    
    # === Save log file ===
    save_responses(
        results_dir=output_dir / model_name.replace("/", "__"),
        model_name=model_name,
        task_name=task_name,
        limit=limit,
        gpu=gpu,
        dataset=dataset,
        subset=subset,
        split=split,
        xverify_model=xverify_model,
        config_path=config_path,
        think_start_tag=get_think_tags(model_name, config_path/"model_config.yaml")[0],
        think_end_tag=get_think_tags(model_name, config_path/"model_config.yaml")[1],
        intervention_type=intervention_type,
        intervention_content=intervention_content,
        intervention_position=intervention_position,
        thinking_end=thinking_end,
        immediate_answer=immediate_answer,
        intervene_sample=intervene_sample,
        steer_config_path=steer_config_path,
        steer_layers=steer_layers,
        steer_coeff=steer_coeff,
        steer_action=steer_action,
    )
    
    # === Cleanup model checkpoint if requested ===
    if cleanup_model and "steered_vllm_model" in str(model_name):
        model_path = Path(model_name).resolve()
        if model_path.exists() and model_path.is_dir():
            shutil.rmtree(model_path)
            parent = model_path.parent
            while "steered_vllm_model" in str(parent) and parent.exists() and not any(parent.iterdir()):
                print(f"INFO: Removing empty parent directory: {parent}")
                parent.rmdir()
                parent = parent.parent
    
    print("✅ Evaluation completed.")
import argparse
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import json
import yaml
from transformers import AutoTokenizer
from intervention_global import build_global_task_yaml
from intervention_local import apply_llm_intervention, build_local_task_yaml
from intervention_tts import apply_tts_intervention, build_tts_task_yaml

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))
from utils import (
    run_lm_evaluation,
    get_task_name, 
    extract_timestamp_from_filename, 
    find_latest_file, 
    get_dataset_config,
    get_think_tags,
    BASE_TEMPLATE
)


def main():
    """Main function to run intervention and evaluation."""
    parser = argparse.ArgumentParser(description="Run intervention and evaluation.")

    # Core arguments
    parser.add_argument("--model-name", type=str, required=True, help="Name of the model to use.")
    parser.add_argument("--dataset", required=True, choices=["mmlu", "gpqa", "arc", "aime_2024", "aime_2025", "math_500", "gsm8k"], help="Dataset to evaluate.")
    parser.add_argument("--subset", type=str, default=None, help="Dataset subset to evaluate.")
    parser.add_argument("--split", type=str, default=None, help="Dataset split to evaluate. If not provided, uses default from dataset config.")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of samples to process.")
    parser.add_argument("--gpu", type=str, default="0", help="GPU ID to use (e.g., '0' for single GPU, '0,1,2,3' for multi-GPU).")
    parser.add_argument("--xverify-model", type=str, default="xVerify-0.5B-I", choices=list(BASE_TEMPLATE.keys()), help="xVerify model to evaluate.")

    # Intervention arguments
    parser.add_argument("--intervention-type", type=str, choices=["global", "local", "tts"], default=None, help="Type of intervention.")
    parser.add_argument("--gpt-model", type=str, default="gpt-5", help="GPT model for intervention.")

    # Global intervention arguments
    parser.add_argument("--intervention-content", type=str, default=None, help="Text for global intervention.")
    parser.add_argument("--intervention-position", type=str, choices=["after_prompt", "after_think"], default="after_prompt", help="Position for global intervention.")
    parser.add_argument("--thinking-end", action="store_true", help="Include a closing thinking tag (e.g., ) for global intervention.")
    parser.add_argument("--immediate-answer", action="store_true", help="Add 'The answer is:' after the thinking process for global intervention.")

    # Test-time intervention arguments
    parser.add_argument("--intervention-source", type=str, choices=["baseline", "intervened_global", "intervened_local", "steered"], default="baseline", help="Source of generated samples for intervention.")
    parser.add_argument("--stack", type=int, default=1, help="Number of times to stack the 'Wait' intervention for tts intervention.")

    args = parser.parse_args()


    # ###################################################
    #                  GLOBAL INTERVENTION          
    # ###################################################
    if args.intervention_type == "global" or args.intervention_type is None:
        
        # === Setup global intervention parameters ===
        if args.intervention_type is None:
            args.intervention_content = None
            args.intervention_position = None
            args.thinking_end = False
            args.immediate_answer = False
        
        # === Build task YAML in lm-evaluation-harness ===
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        task_name, split = build_global_task_yaml(
            dataset=args.dataset, subset=args.subset, split=args.split,
            model_name=args.model_name, intervention_type=args.intervention_type,
            intervention_content=args.intervention_content,
            intervention_position=args.intervention_position,
            tokenizer=tokenizer, thinking_end=args.thinking_end,
            immediate_answer=args.immediate_answer,
        )
        
        # === Run lm-eval evaluation ===
        run_lm_evaluation(
            model_name=args.model_name, task_name=task_name, limit=args.limit, gpu=args.gpu,
            dataset=args.dataset, subset=args.subset, split=split,
            xverify_model=args.xverify_model, output_dir=None,
            intervention_type=args.intervention_type, intervention_content=args.intervention_content,
            intervention_position=args.intervention_position, thinking_end=args.thinking_end,
            immediate_answer=args.immediate_answer,
        )
    
    
    # ###################################################
    #                  LOCAL INTERVENTION          
    # ###################################################
    elif args.intervention_type == "local":
        
        # === Setup local intervention parameters ===
        split = args.split or get_dataset_config(
            model_name=args.model_name,
            dataset=args.dataset,
            subset=args.subset,
            config_path=Path("configs"),
        ).get("test_split")
        dataset_name = get_task_name(args.dataset, Path("configs"), args.subset, split)

        # === Base directories ===
        base_dir = project_root.parent / "results" / dataset_name
        model_path = args.model_name.replace("/", "__")
        samples_dir = base_dir / "baseline" 

        # === Load API key ===
        load_dotenv(dotenv_path=Path(".env.local"))
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in .env.local file.")

        # === Error data path: data/{dataset}_{split}_error.jsonl ===
        error_name = f"{args.dataset}_{split}_error.jsonl" if split else f"{args.dataset}_error.jsonl"
        error_data_path = project_root.parent / "data" / error_name

        # === Generate error data if it doesn't exist ===
        if not error_data_path.exists():
            print(f"INFO: Generating error data to {error_data_path}")
            error_data_path.parent.mkdir(parents=True, exist_ok=True)
            apply_llm_intervention(
                generated_path=samples_dir, model_name=args.model_name,
                api_key=api_key,
                gpt_model=args.gpt_model,
                ids=None, limit=args.limit, output_path=error_data_path
            )
        print(f"INFO: Loaded error data from {error_data_path}")
        raw_samples = [json.loads(l) for l in open(error_data_path, encoding='utf-8')]

        # === Process raw samples for the evaluation model ===
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model_formats = yaml.safe_load(open(Path(__file__).resolve().parents[2] / "configs/model_config.yaml"))
        think_start_tag, _ = get_think_tags(args.model_name, model_formats)
        
        processed_samples = []
        for sample in raw_samples:
            question = sample["question"]
            search = sample["search"]
            templated_question = tokenizer.apply_chat_template(
                [{"role": "user", "content": question}],
                tokenize=False,
                add_generation_prompt=True
            )   
            if think_start_tag not in templated_question:
                templated_question += think_start_tag
            
            sample["full_intervened_prompt"] = templated_question + search
            processed_samples.append(sample)

        # === Save processed samples to a temp file ===       
        model_safe_name = args.model_name.replace("/", "__")
        intervened_data_dir = base_dir / "intervened_local" / "data"
        intervened_data_dir.mkdir(parents=True, exist_ok=True)
        intervened_file = intervened_data_dir / f"{model_safe_name}_intervened_local_data.jsonl"
        with open(intervened_file, "w", encoding="utf-8") as f:
            for sample in processed_samples:
                f.write(json.dumps(sample) + "\n")

        # === Build task YAML and run evaluation ===
        task_name, split = build_local_task_yaml(
            dataset=args.dataset, subset=args.subset,
            split=split, model_name=args.model_name,
            intervened_file_path=intervened_file
        )
        
        run_lm_evaluation(
            model_name=args.model_name, task_name=task_name, limit=None,
            gpu=args.gpu, dataset=args.dataset, subset=args.subset,
            split=split, xverify_model=args.xverify_model, 
            output_dir=base_dir / "intervened_local",
            intervention_type=args.intervention_type,
        )
        intervened_file.unlink()

    # ###################################################
    #             TEST-TIME SCALING INTERVENTION          
    # ###################################################
    elif args.intervention_type == "tts":
        
        # === Setup TTS intervention parameters ===
        split = args.split or get_dataset_config(
            args.dataset, args.subset, args.model_name, Path("configs")
        ).get("test_split")
        dataset_name = get_task_name(args.dataset, Path("configs"), args.subset, split)
        base_dir = project_root.parent / "results" / dataset_name
        model_path = args.model_name.replace("/", "__")

        # === Determine the source directory for the TTS intervention ===
        if args.intervention_source == 'baseline':
            tts_src = base_dir / 'baseline'
        elif args.intervention_source == 'intervened_global':
            tts_src = base_dir / 'intervened_global' / args.intervention_position
        elif args.intervention_source == 'intervened_local':
            tts_src = base_dir / 'intervened_local'
        else:
            raise ValueError(f"Unsupported intervention source: {args.intervention_source}")

        # === Find the latest existing stack to continue from ===
        latest_stack = 0
        tts_root = tts_src / "tts"
        if tts_root.exists():
            for d in tts_root.glob("stack_*"):
                try:
                    n = int(d.name.split('_')[-1])
                    if (d / model_path).exists() and list((d / model_path).glob("samples_*.jsonl")):
                        latest_stack = max(latest_stack, n)
                except Exception:
                    continue
        
        start_stack = latest_stack + 1
        if start_stack > args.stack:
            print(f"INFO: All TTS stacks up to {args.stack} are already generated.")
            return
        
        # === Loop through each TTS stack to apply interventions iteratively ===
        for stack in range(start_stack, args.stack + 1):
            print(f"\n--- Running TTS Intervention: Stack {stack} ---")
            data_dir = tts_root / f"stack_{stack}" / "data" / args.model_name.replace('/', '_')
            stack_dir = tts_root / f"stack_{stack}"

            # Check if intervened data already exists for this stack
            if list(data_dir.glob("*.jsonl")):
                intervened_file_path = find_latest_file(data_dir, "*.jsonl")
                print(f"INFO: Loading existing intervened data from {intervened_file_path}")
            else:
                if stack == 1:
                    src_dir = tts_src / model_path
                else:
                    src_dir = tts_root / f"stack_{stack - 1}" / model_path

                latest_generated = find_latest_file(src_dir, "samples_*.jsonl")
                if not latest_generated or not latest_generated.exists():
                    raise FileNotFoundError(f"Missing input file for stack {stack} in {src_dir}")

                print(f"INFO: Using input file for stack {stack}: {latest_generated}")
                timestamp = extract_timestamp_from_filename(latest_generated.name)
                intervened_file_path = data_dir / f"tts_stack_{stack}_data_{timestamp}.jsonl"

                apply_tts_intervention(
                    generated_path=latest_generated, model_name=args.model_name,
                    dataset_name=dataset_name, intervention_source=args.intervention_source,
                    output_path=intervened_file_path,
                    limit=args.limit, stack=stack,
                )

            # === Build task YAML for the new TTS dataset ===
            task_name, split = build_tts_task_yaml(
                dataset=args.dataset, subset=args.subset,
                split=split, model_name=args.model_name,
                intervened_file_path=intervened_file_path,
                stack=stack
            )

            # === Run lm-eval evaluation on the new TTS data ===
            run_lm_evaluation(
                model_name=args.model_name, task_name=task_name, limit=args.limit,
                gpu=args.gpu, dataset=args.dataset, subset=args.subset,
                split=split, xverify_model=args.xverify_model, output_dir=stack_dir,
                intervention_type=args.intervention_type,
            )


if __name__ == "__main__":
    main()

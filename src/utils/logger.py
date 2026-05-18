import json
import yaml
from pathlib import Path
from transformers import AutoTokenizer
from .utils import extract_timestamp_from_filename
from .xverify_eval import run_xverify_evaluation


def get_status(j):
    return '✅ CORRECT' if j == 'correct' else '❌ INCORRECT' if j == 'incorrect' else '⚠️ OUT OF TOKEN'


def save_responses(
    results_dir: Path,
    model_name: str,
    task_name: str,
    limit: int = None,
    gpu: str = None,
    dataset: str = None,
    subset: str = None,
    split: str = None,
    config_path: Path = None,
    think_start_tag: str = None,
    think_end_tag: str = None,
    intervention_type: str = None,
    xverify_model: str = None,
    **kwargs,
):
    """Generate log file from evaluation results."""
    
    # === If subset is not provided, try to get it from the dataset config ===
    if not subset and dataset and config_path and (dataset_file := config_path / f"{dataset}.yaml").exists():
        with open(dataset_file, "r") as f:
            subset = yaml.safe_load(f).get("dataset_name")

    # === Log all configuration parameters ===
    config_data = {
        "model_name": model_name, "task_name": task_name,"limit": limit,
        "gpu": gpu, "dataset": dataset, "subset": subset, "split": split,
        "think_start_tag": think_start_tag, "think_end_tag": think_end_tag,
        "intervention_type": intervention_type, "xverify_model": xverify_model,
        **kwargs,
    }
    
    # === Find the latest jsonl file and load samples ===
    latest_samples_file = max(results_dir.glob("samples*.jsonl"), key=lambda p: p.stat().st_mtime)
    with open(latest_samples_file, 'r', encoding='utf-8') as f:
        samples = [json.loads(line) for line in f if line.strip()]

    # === Sort samples by ID to ensure consistent ordering ===
    if config_data.get('intervention_type') in ('local', 'tts'):
        samples.sort(key=lambda x: x['doc']['id'])
    if config_data.get('intervention_type') == 'local_think':
        samples.sort(key=lambda x: x['doc']['doc']['id'])
    elif samples and 'original_doc_id' in samples[0].get('doc'):
        samples.sort(key=lambda x: x['doc']['original_doc_id'])
    else:
        samples.sort(key=lambda x: x['doc_id'])

    # === Run xVerify evaluation and update samples ===
    if config_data.get('xverify_model'):
        judgments = run_xverify_evaluation(samples, config_data)
        for sample, judgment in zip(samples, judgments):
            sample['metrics'] = ['xverify']
            sample['judgment'] = judgment
            has_conflict = isinstance(judgment, dict) and judgment.get('thinking') != judgment.get('final')
            sample['conflict'] = 'yes' if has_conflict else 'no'
        
        # Update the samples file
        with open(latest_samples_file, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    # === Generate log file ===
    tokenizer = AutoTokenizer.from_pretrained(config_data["model_name"])
    timestamp = extract_timestamp_from_filename(latest_samples_file.name)
    log_file = results_dir / f"generate_results_{timestamp}.log"
    with open(log_file, 'w', encoding='utf-8') as f:
        write_log(f, samples, config_data, tokenizer)


def write_log(f, samples: list, config_data: dict, tokenizer):
    """Write the formatted log content to the file."""
    width = 80

    # === Calculate Token Stats ===
    total_tokens = num_samples = 0
    think_start_tag = config_data.get('think_start_tag')
    for sample in samples:
        resps = sample.get('resps', [])
        if not resps or not resps[0]:
            continue

        resp = resps[0][0] if isinstance(resps[0], list) else resps[0]
        full_text = sample['arguments']['gen_args_0']['arg_0'] + resp
        text = full_text.split(think_start_tag, 1)[1]
        total_tokens += len(tokenizer.encode(text))
        num_samples += 1

    avg_tokens = total_tokens / num_samples if num_samples else 0
    
    # === Write Config Section ===
    f.write(f'{"-" * width}\nCONFIG\n{"-" * width}\n')
    config_items = [
        ('Dataset', config_data.get('dataset')), ('Subset', config_data.get('subset')), 
        ('Split', config_data.get('split')), ('Task Name', config_data.get('task_name')),
        ('Model Name', config_data.get('model_name')), ('xVerify Model', config_data.get('xverify_model')),
        ('Limit', config_data.get('limit')), ('GPU', config_data.get('gpu')),
    ]
    
    # === Add intervention parameters ===
    if config_data.get('intervention_type') == 'global':
        config_items.extend([
            ('Intervention Type', 'Global'),
            ('Intervention Content', config_data.get('intervention_content')),
            ('Intervention Position', config_data.get('intervention_position')),
            ('Thinking End', config_data.get('thinking_end')),
            ('Immediate Answer', config_data.get('immediate_answer')),
        ])
    elif config_data.get('intervention_type') == 'local':
        config_items.extend([
            ('Intervention Type', 'Local'),
            ('Intervene Sample Type', config_data.get('intervene_sample')),
        ])
    elif config_data.get('intervention_type') == 'tts':
        stack_num = config_data.get('task_name').split('tts_stack_')[-1]
        config_items.extend([
            ('Intervention Type', 'TTS'),
            ('Stack Size', stack_num),
        ])
    else:
        config_items.append(('Intervention Type', 'None'))

    # === Add steering parameters ===
    if config_data.get('steer_config_path'):
        config_items.extend([
            ('Steering Type', 'Steered'),
            ('Steering Config Path', config_data.get('steer_config_path')),
            ('Steering Layers', config_data.get('steer_layers')),
            ('Steering Coeff', config_data.get('steer_coeff')),
            ('Steering Action', config_data.get('steer_action')),
        ])

    # === Write config items ===
    for key, value in config_items:
        if value is not None:
            f.write(f"{key:<25} : {value}\n")
    f.write("\n")

    # === Write Stats Section ===
    judgments = [sample.get('judgment') for sample in samples]
    if not judgments: 
        return
    total = len(judgments)
    
    final_counts = {'correct': 0, 'incorrect': 0, 'out_of_token': 0}
    transition_counts = {}
    for j in judgments:
        if isinstance(j, dict):
            final = j.get('final')
            if final in final_counts:
                final_counts[final] += 1
            transition = (j.get('thinking'), final)
            transition_counts[transition] = transition_counts.get(transition, 0) + 1

    f.write(f'{"-" * width}\nSTATS\n{"-" * width}\n')
    stats_data = {
        'Total Samples': total,
        'Average Token Count': f"{avg_tokens:.2f}",
        'Final Correct Rate': final_counts.get('correct', 0) / total if total else 0,
        'Final Incorrect Rate': final_counts.get('incorrect', 0) / total if total else 0,
        'Final Out of Token Rate': final_counts.get('out_of_token', 0) / total if total else 0,
        'Thinking ❌, Final ❌': transition_counts.get(('incorrect', 'incorrect'), 0) / total if total else 0,
        'Thinking ✅, Final ✅': transition_counts.get(('correct', 'correct'), 0) / total if total else 0,
        'Thinking ✅, Final ❌': transition_counts.get(('correct', 'incorrect'), 0) / total if total else 0,
        'Thinking ❌, Final ✅': transition_counts.get(('incorrect', 'correct'), 0) / total if total else 0,
    }
    
    for k, v in stats_data.items():
        pad = 23 if "✅" in k or "❌" in k else 25
        if k == 'Total Samples' or k == 'Average Token Count':
            fmt = f"{v}"
        else:
            fmt = f"{v:.2%} ({int(v*total)} samples)"
        f.write(f"{k:<{pad}} : {fmt}\n")
    f.write("\n")

    # === Write Samples ===
    f.write('\n'.join(['*'*width]*3) + '\n\nSAMPLES\n')
    think_end_tag = config_data.get('think_end_tag')

    for i, sample in enumerate(samples):
        is_local_or_tts = config_data.get('intervention_type') in ['local', 'tts']
        if is_local_or_tts:
            sample_id = sample['doc']['id']
        elif 'original_doc_id' in sample.get('doc', {}):
            sample_id = sample['doc']['original_doc_id']
        else:
            sample_id = sample['doc_id']
        
        f.write(f"\n\n{'-'*width}\nSample #{sample_id}\n")
        f.write('-' * width + '\n')

        if 'arguments' in sample and 'gen_args_0' in sample['arguments']:
            f.write(f"\n📝 Prompt:\n{sample['arguments']['gen_args_0']['arg_0']}\n")

        if 'resps' in sample and sample['resps']:
            response = sample['resps'][0][0] if isinstance(sample['resps'][0], list) else sample['resps'][0]
            num_tokens = len(tokenizer.encode(response))

            if think_end_tag and think_end_tag in response:
                thinking_part, final_part = response.split(think_end_tag, 1)
                f.write(f'🤔 Continue to think:\n{thinking_part}{think_end_tag}\n')
                f.write(f'💬 Final Answer:\n{final_part.strip()}\n\n')
            else:
                f.write(f'🤔 Continue to think:\n{response}\n\n')
        
        f.write('📊 Evaluation:\n')
        if num_tokens is not None:
            f.write(f"Token Count: {num_tokens}\n")
        judgment = sample.get('judgment')

        if isinstance(judgment, dict):
            f.write(f"Thinking: {get_status(judgment.get('thinking'))}\n")
            f.write(f"Final: {get_status(judgment.get('final'))}\n")
            f.write(f"xVerify Judgment: {judgment.get('final')}\n")
            if sample.get('conflict')=='yes': f.write('❗CONFLICT!\n')
        elif judgment:
            f.write(f'xVerify Judgment: {judgment}\n{get_status(judgment)}\n')
        
        if is_local_or_tts:
            gt = sample['doc'].get('ground_truth')
        elif "think" in str(config_data.get("intervention_type", "")):
            d = sample['doc']['doc']
            gt = d.get('answer') or d.get('ground_truth')
        else:
            gt = sample.get('target')
        
        if gt is not None: f.write(f'Ground Truth: {gt}\n')
        f.write('\n')

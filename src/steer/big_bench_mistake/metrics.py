"""Metrics and evaluation for BIG-Bench Mistake datasets."""

import json
import os
import re
import numpy as np
from datetime import datetime
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))


def extract_final_answer(generated_text: str, think_end_tag: str) -> str | None:
    """Extract the final answer after the think end tag."""
    if think_end_tag.strip() not in generated_text:
        return None
    parts = generated_text.split(think_end_tag.strip(), 1)
    return parts[1].strip()


def extract_boxed_answer(text: str) -> int | None:
    """Extract answer from \\boxed{N} format. Returns integer or None."""
    
    # Match \boxed{...} pattern - handle both single backslash and escaped
    pattern = r'\\boxed\{([^}]*)\}'
    matches = re.findall(pattern, text)
    if not matches:
        return None
    
    # Take the last boxed answer (in case model outputs multiple)
    answer_str = matches[-1].strip()
    try:
        return int(answer_str)
    except ValueError:
        return None


def write_bigbench_log(output_dir, config_name, res_data, args, metrics, timestamp):
    """Write a formatted log file for BIG-Bench Mistake evaluation."""
    log_file_path = os.path.join(output_dir, f'{config_name}_summary_{timestamp}.log')
    width = 80

    with open(log_file_path, 'w', encoding='utf-8') as f:
        
        # === Write Config Section ===
        f.write(f'{"-" * width}\nCONFIG\n{"-" * width}\n')
        config_items = [
            ('Model Name', args.model_name),
            ('Dataset', args.error_dataset),
            ('Split', args.error_split),
            ('Limit', args.limit),
        ]
        if hasattr(args, 'steer_layers') and args.steer_layers:
            config_items.extend([
                ('Steering Layers', args.steer_layers),
                ('Steering Coeffs', args.steer_coeff),
            ])
        
        for key, value in config_items:
            if value is not None:
                f.write(f"{key:<25} : {value}\n")
        f.write("\n")

        # === Write Stats Section ===
        f.write(f'{"-" * width}\nSTATS\n{"-" * width}\n')
        total_samples = len(res_data)
        correct_predictions = sum(1 for d in res_data if d['match'])
        out_of_token_predictions = sum(1 for d in res_data if d.get('status') == 'out_of_token')
        incorrect_predictions = total_samples - correct_predictions - out_of_token_predictions
        
        stats_data = {
            'Total Samples': total_samples,
            'Accuracy': f"{(correct_predictions / total_samples) * 100:.2f}%" if total_samples > 0 else "N/A",
            'Error Detection Accuracy': f"{metrics['error_acc']:.2f}%",
            'Correct Solution Accuracy': f"{metrics['correct_acc']:.2f}%",
            'F1 Score': f"{metrics['f1']:.2f}",
            '✅ Correct Predictions': correct_predictions,
            '❌ Incorrect Predictions': incorrect_predictions,
            '⚠️ Out of Token': out_of_token_predictions,
        }
        for k, v in stats_data.items():
            f.write(f"{k:<25} : {v}\n")
        f.write("\n")

        # === Write Samples Section ===
        f.write('\n'.join(['*'*width]*3) + '\n\nSAMPLES\n')
        for i, sample in enumerate(res_data):
            f.write(f"\n\n{'-'*width}\nSample #{i}\n{'-'*width}\n")
            
            if 'full_prompt' in sample:
                f.write(f"\n📝 Full Prompt:\n{sample['full_prompt']}\n")
            else:
                f.write(f"\n📝 Input:\n{sample['input']}\n")
                tagged_response = ''
                for sdx, step in enumerate(sample['steps']):
                    tagged_response += f'''Thought {sdx}: {step}\n'''
                f.write(f"\n🧩 Steps:\n{tagged_response.strip()}\n")
            
            f.write(f"\n🤖 Model Response:\n{sample['generated_critique']}\n")
            
            f.write('\n\n📊 Evaluation:\n')
            f.write(f"Prediction: {sample['prediction']}\n")
            f.write(f"Ground Truth: {sample['label']}\n")
            
            if sample.get('status') == 'out_of_token':
                status = '⚠️ OUT OF TOKEN'
            else:
                status = '✅ CORRECT' if sample['match'] else '❌ INCORRECT'
            f.write(f"Status: {status}\n")


def process_metrics_bigbench(
    generations, 
    input_data, 
    output_dir, 
    config_name, 
    args=None, 
    think_end_tag=None, 
    tokenizer=None, 
    model_name=None
):
    """Process generations, calculate metrics, and save results for BIG-Bench Mistake."""

    # === Process Generations ===
    res_data = []

    for i in range(len(input_data)):
        d = dict(input_data[i])
        generated_critique = generations[i].outputs[0].text
        d['generated_critique'] = generated_critique

        if think_end_tag.strip() not in generated_critique:
            d['status'] = 'out_of_token'
            d['prediction'] = None
        else:
            final_answer = extract_final_answer(generated_critique, think_end_tag)
            d['prediction'] = extract_boxed_answer(final_answer) if final_answer else None
        
        # Match: prediction equals label
        d['match'] = (d.get('prediction') == d['label'])
        res_data.append(d)

    # === Calculate Metrics ===
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f'{config_name}_results_{timestamp}.jsonl'), 'w') as f:
        for e in res_data:
            f.write(json.dumps(e) + '\n')

    error_data = [e for e in res_data if e['label'] != -1]
    correct_data = [e for e in res_data if e['label'] == -1]
    acc1 = np.mean([e['match'] for e in error_data]) * 100 if error_data else 0
    acc2 = np.mean([e['match'] for e in correct_data]) * 100 if correct_data else 0
    f1 = 2 * acc1 * acc2 / (acc1 + acc2) if (acc1 + acc2) > 0 else 0
    metrics = {"error_acc": acc1, "correct_acc": acc2, "f1": f1}

    write_bigbench_log(output_dir, config_name, res_data, args, metrics, timestamp)

    return metrics

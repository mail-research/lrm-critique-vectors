# Adapted from https://github.com/QwenLM/ProcessBench/blob/main/code/run_eval.py

import json
import os
import re
import numpy as np
from datetime import datetime

def extract_answer(solution_text: str):
    """Extract the answer from a \boxed{} environment."""
    boxed_pattern = r'\\boxed\{([^}]*)\}'
    matches = re.findall(boxed_pattern, solution_text)
    if matches:
        return matches[-1].strip()
    return None


def write_processbench_log(output_dir, config_name, res_data, args, metrics, timestamp):
    """Write a formatted log file for ProcessBench evaluation."""
    log_file_path = os.path.join(output_dir, f'{config_name}_summary_{timestamp}.log')
    width = 80

    with open(log_file_path, 'w', encoding='utf-8') as f:
        
        # === Write Config Section ===
        f.write(f'{"-" * width}\nCONFIG\n{"-" * width}\n')
        config_items = [
            ('Model Name', args.model_name),
            ('Dataset', args.dataset),
            ('Split', args.split),
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
            f.write(f"\n\n{'-'*width}\nSample #{sample['id']}\n{'-'*width}\n")
            f.write(f"\n📝 Problem:\n{sample['problem']}\n")
            
            tagged_response = ''
            for sdx, step in enumerate(sample['steps']):
                tagged_response += f'''<paragraph_{sdx}>
{step}
</paragraph_{sdx}>
\n'''
            f.write(f"\n🧩 Solution:\n{tagged_response.strip()}\n")
            
            f.write(f"\n🤖 Model Response:\n{sample['generated_critique']}\n")
            
            f.write('\n\n📊 Evaluation:\n')
            f.write(f"Prediction: {sample['prediction']}\n")
            f.write(f"Ground Truth: {sample['label']}\n")
            
            if sample.get('status') == 'out_of_token':
                status = '⚠️ OUT OF TOKEN'
            else:
                status = '✅ CORRECT' if sample['match'] else '❌ INCORRECT'
            f.write(f"Status: {status}\n")


def process_metrics_processbench(
    generations, 
    input_data, 
    output_dir, 
    config_name, 
    args=None, 
    think_end_tag=None, 
    tokenizer=None, 
    model_name=None
):
    """Process generations, calculate metrics, and save results for ProcessBench."""

    # === Process Generations ===
    res_data = []
    for i in range(len(input_data)):
        d = input_data[i].copy()
        generated_critique = generations[i].outputs[0].text
        d['generated_critique'] = generated_critique

        if think_end_tag.strip() not in generated_critique:
            d['status'] = 'out_of_token'
            pred = None
        else:
            pred = extract_answer(generated_critique)
            try:
                pred = int(pred)
            except (ValueError, TypeError):
                pred = None

        d['prediction'] = pred
        d['match'] = (pred == d['label'])
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

    write_processbench_log(output_dir, config_name, res_data, args, metrics, timestamp)

    return metrics

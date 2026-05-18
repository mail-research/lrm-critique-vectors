import json
import sys
from pathlib import Path
from typing import Dict

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))
from utils import get_think_tags, run_xverify_evaluation
from transformers import AutoTokenizer


def get_steer_tag(steer_coeff: float, steer_layers: list = None) -> str:
    """Generate tag for steering configuration."""
    if steer_coeff == 0.0:
        return "baseline"
    
    sign = "positive" if steer_coeff > 0 else "negative"
    coeff_str = f"{steer_coeff:.1f}".replace("-", "")
    
    if steer_layers:
        layers_str = "_".join(map(str, steer_layers))
        return f"{sign}_layers_{layers_str}_coeff_{coeff_str}"
    else:
        return f"{sign}_coeff_{coeff_str}"


def extract_thinking_content(text: str, start_tag: str, end_tag: str) -> str:
    """Extract content between think tags."""
    if start_tag in text and end_tag in text:
        start = text.find(start_tag) + len(start_tag)
        end = text.find(end_tag)
        return text[start:end].strip()
    elif start_tag in text:
        start = text.find(start_tag) + len(start_tag)
        return text[start:].strip()
    return ""


def get_status(j):
    """Get status emoji for judgment."""
    return '✅' if j == 'correct' else '❌' if j == 'incorrect' else '⚠️'


def load_natural_judgments(output_dir: Path) -> Dict[int, dict]:
    """Load judgments from natural (no Wait) run for comparison."""
    natural_dir = output_dir / "natural"
    files = list(natural_dir.glob("samples_*.jsonl"))
    if not files:
        return {}
    
    samples_file = max(files, key=lambda p: p.stat().st_mtime)
    judgments = {}
    with open(samples_file, "r", encoding="utf-8") as f:
        for line in f:
            s = json.loads(line)
            doc_id = s.get("doc_id")
            j = s.get("judgment", {})
            if isinstance(j, dict):
                judgments[doc_id] = {
                    "thinking": j.get("thinking"),
                    "final": j.get("final"),
                }
    return judgments


def run_tts_synthetic_evaluation(
    results_dir: Path, 
    model_name: str, 
    dataset: str,
    subset: str = None, 
    split: str = "test", 
    xverify_model: str = "xVerify-0.5B-I",
    steer_coeff: float = 0.0, 
    stack: int = None,
):
    """Run xVerify evaluation and save logs."""
    
    # === Load results ===
    files = list(results_dir.glob("samples_*.jsonl"))
    if not files:
        print(f"WARNING: No samples found in {results_dir}")
        return
    
    samples_file = max(files, key=lambda p: p.stat().st_mtime)
    with open(samples_file, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f]
    
    # === Run xVerify ===
    config_path = project_root.parent / "configs"
    start_tag, end_tag = get_think_tags(model_name, config_path / "model_config.yaml")
    config_data = {
        "xverify_model": xverify_model,
        "think_start_tag": start_tag,
        "think_end_tag": end_tag,
        "dataset": dataset,
        "intervention_type": "tts",
    }
    
    truncated_count = sum(1 for s in samples if s.get("truncated"))
    print(f"INFO: Running xVerify for {len(samples)} samples ({truncated_count} truncated)")
    
    judgments = run_xverify_evaluation(samples, config_data)
    for sample, judgment in zip(samples, judgments):
        sample["judgment"] = judgment
    
    # === Calculate stats ===
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    total = len(samples)
    final_counts = {'correct': 0, 'incorrect': 0, 'out_of_token': 0}
    thinking_counts = {'correct': 0, 'incorrect': 0, 'out_of_token': 0}
    
    total_think_tokens = 0
    for s in samples:
        j = s.get("judgment", {})
        if isinstance(j, dict):
            final = j.get('final')
            thinking = j.get('thinking')
            if final in final_counts:
                final_counts[final] += 1
            if thinking in thinking_counts:
                thinking_counts[thinking] += 1
        
        # Count thinking tokens
        resp = s["resps"][0][0] if s.get("resps") else ""
        prompt = s["arguments"]["gen_args_0"]["arg_0"]
        thinking_text = extract_thinking_content(prompt + resp, start_tag, end_tag)
        total_think_tokens += len(tokenizer.encode(thinking_text, add_special_tokens=False))
    
    avg_think_tokens = total_think_tokens / total if total > 0 else 0.0
    
    # === Save log ===
    timestamp = samples_file.stem.split("samples_")[-1]
    log_file = results_dir / f"generate_results_{timestamp}.log"
    width = 80
    
    with open(log_file, "w", encoding="utf-8") as f:
        title = "NATURAL (no Wait)" if stack == 0 else f"STACK {stack}"
        f.write(f"{'='*width}\n{title}\n{'='*width}\n\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Dataset: {dataset} ({subset}, {split})\n")
        f.write(f"Steer Coeff: {steer_coeff}\n\n")
        
        f.write(f'{"-" * width}\nSTATS\n{"-" * width}\n')
        f.write(f"Total: {total}, Truncated: {truncated_count}\n")
        f.write(f"Avg Thinking Tokens: {avg_think_tokens:.1f}\n\n")
        
        f.write(f"{'Metric':<20} {'Count':<10} {'Rate':<10}\n")
        f.write(f"{'-'*40}\n")
        f.write(f"{'Final Correct':<20} {final_counts['correct']:<10} {final_counts['correct']/total:.2%}\n")
        f.write(f"{'Final Incorrect':<20} {final_counts['incorrect']:<10} {final_counts['incorrect']/total:.2%}\n")
        f.write(f"{'Think Correct':<20} {thinking_counts['correct']:<10} {thinking_counts['correct']/total:.2%}\n")
        f.write(f"{'Think Incorrect':<20} {thinking_counts['incorrect']:<10} {thinking_counts['incorrect']/total:.2%}\n")
        
        # === Samples ===
        f.write(f"\n{'*'*width}\nSAMPLES\n{'*'*width}\n")
        for s in samples:
            sample_id = s['doc']['id']
            j = s.get('judgment', {})
            think_j = j.get('thinking', 'N/A') if isinstance(j, dict) else 'N/A'
            final_j = j.get('final', 'N/A') if isinstance(j, dict) else 'N/A'
            
            f.write(f"\n{'-'*width}\nSample #{sample_id}\n")
            f.write(f"Think={get_status(think_j)} Final={get_status(final_j)}\n")
            
            # Prompt and response
            prompt = s['arguments']['gen_args_0']['arg_0']
            resp = s['resps'][0][0] if s.get('resps') else ''
            
            f.write(f"\n📝 Prompt:\n{prompt}\n")
            
            if end_tag in resp:
                think_part, final_part = resp.split(end_tag, 1)
                f.write(f"\n🤔 Thinking:\n{think_part}{end_tag}\n")
                f.write(f"\n💬 Final:\n{final_part.strip()}\n")
            else:
                f.write(f"\n🤔 Response:\n{resp}\n")
            
            f.write(f"\nGround Truth: {s['doc']['ground_truth']}\n")
    
    print(f"INFO: Saved log to {log_file}")
    
    # === Update samples file with judgments ===
    with open(samples_file, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    
    return {
        "total": total,
        "final_correct": final_counts.get('correct', 0),
        "thinking_correct": thinking_counts.get('correct', 0),
        "avg_think_tokens": avg_think_tokens,
    }

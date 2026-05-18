import json
from pathlib import Path
from datasets import Dataset
from .template_prompt import BIGBENCH_TEMPLATES


MAX_PROMPT_TOKENS = 4000  # Filter out samples exceeding this token limit


def process_data_bigbench(split: str, tokenizer, template: str = None):
    """Load BIG-Bench Mistake dataset and prepare tokenized prompts."""
    print(f"INFO: Processing data for BIG-Bench-Mistake/{split}...")
    
    if template is None:
        raise ValueError(f"Unknown split: {split}. Available: {list(BIGBENCH_TEMPLATES.keys())}")

    data_dir = Path(__file__).resolve().parents[3] / "data"
    file_path = data_dir / f"{split}.jsonl"
    data = []
    with open(file_path, "r") as f:
        for line in f:
            item = json.loads(line.strip())
            label = item.get("mistake_index")
            # Convert null to -1 for "no mistake" cases
            if label is None:
                label = -1
            data.append({
                "input": item["input"],
                "steps": item["steps"],
                "target": item.get("target"),
                "answer": item.get("answer"),
                "label": label,
            })
    input_data = Dataset.from_list(data)
    prompt_token_ids = []
    processed_data = []
    
    for e in input_data:
        steps_formatted = "\n".join([f"Thought {i}: {s}" for i, s in enumerate(e["steps"])])
        prompt = template.format(input=e["input"], steps=steps_formatted)

        messages = [{"role": "user", "content": prompt}]
        token_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, add_special_tokens=False
        )
        
        # Skip samples that exceed the token limit
        MAX_PROMPT_TOKENS = 4000
        if len(token_ids) > MAX_PROMPT_TOKENS:
            continue

        e_dict = dict(e)
        e_dict["full_prompt"] = prompt
        processed_data.append(e_dict)
        prompt_token_ids.append(token_ids)
    
    return Dataset.from_list(processed_data), prompt_token_ids

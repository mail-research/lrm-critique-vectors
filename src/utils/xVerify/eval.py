# Adapted from https://github.com/IAAR-Shanghai/xVerify/blob/main/src/xVerify/eval.py

from tqdm import tqdm
from typing import List, Dict
from .model import Model
from .prompts import JUDGE_PROMPT
import torch
import gc


def run_xverify(samples: List[Dict], model: Model, batch_size: int = 1) -> List[str]:
    """Runs xVerify evaluation on a list of samples."""
    
    prompts = [
        JUDGE_PROMPT.format(
            question=item.get('question'),
            output=item.get('llm_output'),
            answer=item.get('correct_answer')
        ) for item in samples
    ]
    judgments = []
        
    # === Generate responses in batches ===
    for i in tqdm(range(0, len(prompts), batch_size), desc=f"Verifying with '{model.model_name}'"):
        if torch.cuda.is_available():
            print(f"  Processing batch {i//batch_size + 1}/{(len(prompts)-1)//batch_size + 1}")
        
        batch_prompts = prompts[i:i + batch_size]
        responses = model.batch_generate(batch_prompts)
        judgments.extend([resp.lower() if resp else "out_of_token" for resp in responses])
        
        del batch_prompts, responses
        torch.cuda.empty_cache()
        gc.collect()
    
    return judgments
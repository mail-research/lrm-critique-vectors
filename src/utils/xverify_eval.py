from .xVerify import Model, run_xverify
from .data_process import DATASET_PROCESSORS
import torch
import gc
import os


def run_xverify_evaluation(samples: list, config_data: dict) -> list[dict]:
    """Run xVerify evaluation on the generated samples."""
    
    # === Base configs ===
    xverify_model_name = config_data['xverify_model']
    if config_data.get('gpu'):
        os.environ['CUDA_VISIBLE_DEVICES'] = config_data['gpu']
    model = Model(model_name=f"IAAR-Shanghai/{xverify_model_name}")
    think_start_tag = config_data.get('think_start_tag')
    think_end_tag = config_data.get('think_end_tag')
    dataset_name = config_data.get('dataset')
    intervention_type = config_data.get("intervention_type") or ""
    
    # === Set processor ===
    if intervention_type == 'local':
        processor_key = 'intervened_local'
    elif intervention_type == 'tts':
        processor_key = 'tts'
    elif intervention_type and 'think' in intervention_type:
        processor_key = 'steered'
    else:
        processor_key = dataset_name
    processor = DATASET_PROCESSORS.get(processor_key)
    
    # === Initialize params ===
    data_to_eval_thinking, data_to_eval_final, indices_to_eval = [], [], []
    all_judgments = [None] * len(samples)

    # === Extract answer for each sample ===
    for i, sample in enumerate(samples):
        resp = sample.get('resps')[0][0]
        prompt = sample.get('arguments').get('gen_args_0').get('arg_0')
        full_text = prompt + resp

        # Extract thinking and final parts
        if think_end_tag and think_end_tag in full_text:
            question, answer = processor(sample['doc'])
            thinking_text, final_text = full_text.split(think_end_tag, 1)
            thinking_text = thinking_text.split(think_start_tag, 1)[-1]
            paragraphs = thinking_text.split('\n\n')
            trimmed_thinking = '\n\n'.join(paragraphs[-2:]) if len(paragraphs) > 2 else thinking_text  # Only keep last 2 paragraphs in thinking
            data_to_eval_thinking.append({'question': question, 'llm_output': trimmed_thinking, 'correct_answer': answer})
            data_to_eval_final.append({'question': question, 'llm_output': final_text, 'correct_answer': answer})
            indices_to_eval.append(i)
        else:
            all_judgments[i] = {'thinking': 'out_of_token', 'final': 'out_of_token'}
    
    # === Run xVerify for thinking and final parts ===
    with torch.inference_mode():
        
        # Thinking
        thinking_judgments = run_xverify(data_to_eval_thinking, model, batch_size=128)  
        torch.cuda.empty_cache(); gc.collect()
        
        # Final
        final_judgments = run_xverify(data_to_eval_final, model, batch_size=128)  
        torch.cuda.empty_cache(); gc.collect()

    for i, idx in enumerate(indices_to_eval):
        all_judgments[idx] = {'thinking': thinking_judgments[i], 'final': final_judgments[i]}

    return all_judgments
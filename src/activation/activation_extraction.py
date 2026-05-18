import gc
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))
from utils import get_task_name, get_think_tags
from activation import load_conflict_samples


def calculate_spans(
    texts: List[str],
    tokenizer: PreTrainedTokenizer,
    end_tag: str,
    start_tag: str,
    log_file: Optional[Path] = None,
) -> List[Tuple[int, int]]:
    """Find (start, end) token indices to collect activations for the final answer."""
    
    spans = []
    end_tag_len = len(tokenizer(end_tag, add_special_tokens=False).input_ids)
    
    for i, text in enumerate(texts):
        encoding = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        tokens = encoding.input_ids[0]

        # Find start tag token index
        start_tag_char_idx = text.find(start_tag)
        start_tag_token_idx = encoding.char_to_token(start_tag_char_idx)
        start_tag_len = len(tokenizer(start_tag, add_special_tokens=False).input_ids)

        # Find end tag token index
        end_tag_char_idx = text.rfind(end_tag)
        end_tag_token_idx = encoding.char_to_token(end_tag_char_idx)
        
        # Collect activations from final answer
        start_idx = end_tag_token_idx + end_tag_len
        end_idx = len(tokens)
        spans.append((start_idx, end_idx))
        
        # Log extracted text
        if log_file and i < 10:
            with open(log_file, "a") as f:
                extracted = tokens[start_idx:end_idx]
                f.write(f"--- Sample {i} ---\n")
                f.write(f"Span: ({start_idx}, {end_idx})\n")
                f.write(f"Tokens extracted: {end_idx - start_idx}\n")
                f.write(f"{ '=' * 80}\n")
                f.write(f"Original response: \n{repr(text)}\n")
                f.write(f"{ '=' * 80}\n")
                f.write(f"Decoded tokens: \n{repr(tokenizer.decode(extracted))}\n")
                f.write(f"{ '=' * 80}\n\n")
    return spans


def get_activations_for_split(args, split: str, model, tokenizer, probe_dir: Path):
    """Helper to load data and extract activations for a given split."""
    
    # === Define cache paths for activations ===
    activations_dir = probe_dir / "activations"
    activations_dir.mkdir(parents=True, exist_ok=True)

    first_activations_path = activations_dir / f"{args.dataset}_{split}_original_activations.pt"
    second_activations_path = activations_dir / f"{args.dataset}_{split}_intervened_activations.pt"

    # === If activations are cached, load them ===
    if first_activations_path.exists() and second_activations_path.exists():
        first_activations = torch.load(first_activations_path)
        second_activations = torch.load(second_activations_path)
        return first_activations, second_activations

    # === Load samples ===
    dataset_name = get_task_name(
        dataset=args.dataset,
        config_path=Path("configs"),
        split=split,
    )

    first_samples, second_samples = load_conflict_samples(args, dataset_name)
    
    if not first_samples or not second_samples:
        print(f"Could not load samples for dataset: {args.dataset}, split: {split}.")
        sys.exit(1)

    # === Filter long samples by pairs ===
    valid_indices = []
    for i, (first_sample, second_sample) in enumerate(zip(first_samples, second_samples)):
        first_text = first_sample["arguments"]["gen_args_0"]["arg_0"] + first_sample["resps"][0][0]
        second_text = second_sample["arguments"]["gen_args_0"]["arg_0"] + second_sample["resps"][0][0]
        first_len = len(tokenizer(first_text, return_tensors="pt", add_special_tokens=False).input_ids[0])
        second_len = len(tokenizer(second_text, return_tensors="pt", add_special_tokens=False).input_ids[0])
        
        if first_len <= 6000 and second_len <= 6000:
            valid_indices.append(i)
        else:
            print(f"Skipping pair {i} - too long: first={first_len}, second={second_len} tokens")
    
    first_texts = [first_samples[i]["arguments"]["gen_args_0"]["arg_0"] + first_samples[i]["resps"][0][0] for i in valid_indices]
    second_texts = [second_samples[i]["arguments"]["gen_args_0"]["arg_0"] + second_samples[i]["resps"][0][0] for i in valid_indices]
    
    # === Initialize and clear extraction logs ===
    log_dir = probe_dir / "extraction_example" / f"{args.dataset}_{split}"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    first_log = log_dir / "original_token_extraction.log"
    second_log = log_dir / "intervened_token_extraction.log"
    
    first_log.unlink(missing_ok=True)
    second_log.unlink(missing_ok=True)

    # === Calculate spans and layer indices ===
    start_tag, end_tag = get_think_tags(args.model_name, project_root.parent / "configs" / "model_config.yaml")
    first_spans = calculate_spans(first_texts, tokenizer, end_tag, start_tag, log_file=first_log)
    second_spans = calculate_spans(second_texts, tokenizer, end_tag, start_tag, log_file=second_log)
    layers = args.layers or list(range(model.config.num_hidden_layers))

    # === Collect activations ===
    first_activations = get_activations(
        model, tokenizer, first_texts, layers, first_spans
    )
    second_activations = get_activations(
        model, tokenizer, second_texts, layers, second_spans
    )

    # === Save the computed activations to reuse ===
    torch.save(first_activations, first_activations_path)
    torch.save(second_activations, second_activations_path)

    return first_activations, second_activations


def get_activations(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    texts: List[str],
    layers: Optional[List[int]] = None,
    spans: Optional[List[Tuple[int, int]]] = None,
) -> Dict[int, torch.Tensor]:
    """
    Collects activations from texts.

    Args:
        model: The pretrained transformer model.
        tokenizer: The tokenizer for the model.
        texts: A list of input strings.
        layers: A list of layer indices to get representation from. If None, collects from all layers.
        spans: A list of (start_idx, end_idx) tuples for activation collection, one for each input.
    """
    
    # === Validate inputs ===
    if spans is None:
        raise ValueError("`spans` must be specified for activation collection.")
    if len(spans) != len(texts):
        raise ValueError("Length of `spans` must match the batch size.")

    # === Prepare model and device ===
    device = next(model.parameters()).device
    if layers is None:
        layers = list[int](range(model.config.num_hidden_layers))
    
    # === Process each sample ===
    all_activations = {i: [] for i in layers}
    for text, (start_idx, end_idx) in zip(texts, spans):
        
        # Tokenize and forward pass
        enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)        
        input_ids = enc["input_ids"].to(device, non_blocking=True)
        
        with torch.inference_mode():
            outputs = model(
                input_ids,
                output_hidden_states=True,
                use_cache=False,
            )
        
        # Collect activations
        for layer_idx in layers:
            activation = outputs.hidden_states[layer_idx+1][0, start_idx:end_idx].mean(dim=0)
            all_activations[layer_idx].append(activation.cpu())
        
        del outputs, input_ids, enc
        torch.cuda.empty_cache(); gc.collect()
    
    # === Stack activations for each layer ===
    for layer_idx in layers:
        all_activations[layer_idx] = torch.stack(all_activations[layer_idx])
    
    return all_activations

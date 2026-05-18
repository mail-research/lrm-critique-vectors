import argparse
import sys
import os
import torch
import jinja2
import torch
import re
import json
from pathlib import Path
from typing import Dict, Optional
from transformers import AutoModelForCausalLM, PreTrainedTokenizer, AutoTokenizer, AutoConfig

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))
from activation import get_probe_dir


def load_activations(args:argparse.Namespace, probe_dir: Path) -> tuple[dict, dict]:
    """Load activations.

    For intervened_local: original vs intervened

    Returns (first_acts, second_acts) where:
        - first_acts: label=0 (original)
        - second_acts: label=1 (intervened)
    """
    
    activations_dir = probe_dir / "activations"
    first_path = activations_dir / f"{args.dataset}_{args.split}_original_activations.pt"
    second_path = activations_dir / f"{args.dataset}_{args.split}_intervened_activations.pt"
    
    try:
        print(f"Loading first activations from: {first_path}")
        first_acts = torch.load(first_path, map_location='cpu')
        print(f"Loading second activations from: {second_path}")
        second_acts = torch.load(second_path, map_location='cpu')
    except FileNotFoundError as e:
        print(f"Error: Activation file not found. {e}")
        sys.exit(1)

    return first_acts, second_acts


def run_steering_vector_analysis(
    args: argparse.Namespace,
    probe_dir: Path,
    first_acts: Dict[int, torch.Tensor],
    second_acts: Dict[int, torch.Tensor],
    num_layers: int,
    tokenizer: PreTrainedTokenizer,
) -> Dict[int, torch.Tensor]:
    """
    Compute steering vectors for each layer and run logit lens analysis.
    
    Steering vector = mean(second_acts) - mean(first_acts)
    For intervened_local: intervened - original

    Returns:
        A dictionary mapping each layer index to its steering vector.    
    """
    
    # === Setup logit lens directory ===
    logit_lens_dir = probe_dir / "logit_lens"
    logit_lens_dir.mkdir(parents=True, exist_ok=True)
    all_layers = range(num_layers) 
    missing_files = [
        i for i in all_layers 
        if i in first_acts and i in second_acts and not (logit_lens_dir / f"layer_{i}_top_tokens.log").exists()
    ]
    
    # === Find unembeddding matrix ===
    unembedding_matrix = None 
    if missing_files:
        model = AutoModelForCausalLM.from_pretrained(args.model_name, dtype="auto")
        unembedding_matrix = model.get_output_embeddings().weight.detach().clone().to(
            f"cuda:0", torch.float32
        )
        del model
        torch.cuda.empty_cache()

    # === Find steering vectors for all layers ===
    steering_vectors = {}
    for layer_idx in all_layers:
        if layer_idx not in first_acts or layer_idx not in second_acts:
            continue

        # Calculate steering vector
        sv = torch.mean(second_acts[layer_idx], dim=0) - torch.mean(first_acts[layer_idx], dim=0)
        if sv.ndim == 1: sv = sv.unsqueeze(0)
        steering_vectors[layer_idx] = sv

        # Logit Lens Analysis
        logit_lens_log_path = logit_lens_dir / f"layer_{layer_idx}_top_tokens.log"
        if not logit_lens_log_path.exists():
            if unembedding_matrix is None:
                print(f"Error: Logit lens analysis for layer {layer_idx} is required, but the unembedding matrix was not loaded.")
                sys.exit(1)
            
            logits = sv.to(unembedding_matrix.device, torch.float32) @ unembedding_matrix.T
            top_tokens = torch.topk(logits.squeeze(), 10)
            
            log_lines = [
                "Top 10 tokens from steering vector unembedding:",
                "Rank | Token ID | Logit Value | Token",
                "--- | --- | --- | ---"
            ]
            for rank, (logit_val, token_id) in enumerate(zip(*top_tokens), 1):
                token_id = token_id.item()
                log_lines.append(
                    f"{rank} | {token_id} | {logit_val.item():.4f} | {repr(tokenizer.decode(token_id))}"
                )

            logit_lens_log_path.write_text("\n".join(log_lines))

    # === Free GPU memory ===
    if unembedding_matrix is not None:
        del unembedding_matrix
        torch.cuda.empty_cache()
        
    return steering_vectors


def load_steering_config(args: argparse.Namespace) -> Optional[Dict[int, torch.Tensor]]:
    """Load steering configuration."""
    
    # === Check if steering config file exists ===
    if hasattr(args, 'steer_layers') and hasattr(args, 'steer_coeff'):
        steer_action = "add"
        layers_id = "_".join(map(str, args.steer_layers))
        steer_tag = f"layers_{layers_id}_coeff_{args.steer_coeff}_{steer_action}"
        probe_dir = get_probe_dir(project_root, args)
        steer_config_path = probe_dir / "steer_configs" / f"{steer_tag}.pt"
        if steer_config_path.exists():
            print(f"Loading steering config from: {steer_config_path}")
            steer_config = torch.load(steer_config_path, map_location="cpu")
            return {
                int(m.group(1)): cfg["steering_vector"]
                for name, cfg in steer_config.items()
                if (m := re.match(r"layers\.(\d+)", name))
            }
    
    # === Load Activations ===
    print("No steering config found. Computing from activations...") 
    probe_dir = get_probe_dir(project_root, args)
    first_acts, second_acts = load_activations(args, probe_dir)    
    
    # === Prepare Steering Vectors ===
    model_config = AutoConfig.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    steering_vectors = run_steering_vector_analysis(
        args=args,
        probe_dir=probe_dir,
        first_acts=first_acts,
        second_acts=second_acts,
        num_layers=model_config.num_hidden_layers,
        tokenizer=tokenizer,
    )    
    del first_acts, second_acts
    torch.cuda.empty_cache()
    
    return steering_vectors


def apply_steering_vllm(
    model_name: str,
    output_dir: Path,
    steer_config_path: str = None,
):
    """
    Applies steering vectors to model biases and saves the steered model.
    This creates a specific steered model checkpoint for a given configuration.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # === Load base model ===
    print(f"INFO: Creating new steered model at: {output_dir}")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True,
        device_map="cpu", dtype='auto',
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # === Locate transformer block layers ===
    for path in [
        ('layers', None),
        ('h', None),
        ('model', 'layers'),
        ('transformer', 'h'),
        ('model', 'h'),
        ('transformer', 'layers'),
    ]:
        target = getattr(base_model, path[0], None)
        block_modules = getattr(target, path[1], None) if path[1] else target
        if block_modules is not None:
            break
    else:
        raise AttributeError("Could not locate transformer block layers in this model.")
    
    # === Ensure all MLP down_proj layers have biases ===
    for layer in block_modules:
        mlp = getattr(layer, 'mlp', None)
        if not mlp or not hasattr(mlp, 'down_proj'):
            continue
        proj = mlp.down_proj
        if isinstance(proj, torch.nn.Linear):
            if not hasattr(proj, 'bias') or proj.bias is None:
                proj.bias = torch.nn.Parameter(torch.zeros(
                    proj.out_features, device=proj.weight.device, dtype=proj.weight.dtype
                ))

    # === Apply steering vectors if available ===
    applied_count = 0
    if steer_config_path and os.path.exists(steer_config_path):
        print(f"INFO: Applying steering vectors from: {steer_config_path}")
        steer_config = torch.load(steer_config_path, map_location="cpu")

        for name, cfg in steer_config.items():
            
            # Get layer index and coefficient
            match = re.match(r"layers\.(\d+)", name)
            idx = int(match.group(1))
            mlp = getattr(block_modules[idx], "mlp", None)
            layer = mlp.down_proj

            coef = cfg.get("steering_coefficient", 0)
            if coef == 0: continue

            # Add steering vector alpha * v to layer bias
            vec = coef * cfg["steering_vector"].to(layer.weight.device, dtype=layer.weight.dtype)
            layer.bias.data.add_(vec.squeeze(0) if vec.dim() > 1 else vec)
            applied_count += 1

        print(f"✅ Applied steering to {applied_count} layers.")
    else:
        print(f"INFO: No steering config found. Using zero biases.")

    # === Prepare class info for custom model ===
    causal_cls = base_model.__class__.__name__
    causal_mod = base_model.__class__.__module__
    prefix, base_cls, base_mod = "", None, None
    for attr in ("model", "transformer"):
        sub = getattr(base_model, attr, None)
        if sub:
            prefix, base_cls, base_mod = attr, sub.__class__.__name__, sub.__class__.__module__
            break

    base_model.tie_weights()
    custom_causal_cls = f"Steered{causal_cls}"
    custom_base_cls = f"Steered{base_cls}" if base_cls else None
    
    # === Render custom model from Jinja template ===
    template_path = project_root.parent / "configs" / "steer_template.py.jinja"
    template = jinja2.Template(template_path.read_text(encoding='utf-8'))
    rendered = template.render(
        causal_lm_module_name=causal_mod,
        causal_lm_class_name=causal_cls,
        custom_causal_lm_class_name=custom_causal_cls,
        base_model_prefix=prefix,
        base_model_module_name=base_mod,
        base_model_class_name=base_cls,
        custom_base_model_class_name=custom_base_cls,
    )
    custom_model_path = output_dir / "steered_model.py"
    custom_model_path.write_text(rendered)

    # === Save the model and tokenizer with the applied steering ===
    base_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # === Update config.json ===
    config_path = output_dir / "config.json"
    config = json.load(open(config_path))
    config.update({
        "trust_remote_code": True,
        "architectures": [c for c in [custom_causal_cls, custom_base_cls] if c],
        "auto_map": {
            "AutoModelForCausalLM": f"steered_model.{custom_causal_cls}",
            "AutoModel": f"steered_model.{custom_base_cls or custom_causal_cls}",
        },
    })
    json.dump(config, open(config_path, "w"), indent=2)

    print(f"✅ Steered model ready at: {output_dir}")

    # === Cleanup ===
    del base_model, tokenizer, block_modules, target
    torch.cuda.empty_cache()

import argparse
import os
from pathlib import Path
import sys
import torch
import random
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
from torchmetrics.classification import BinaryCalibrationError
import pandas as pd
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))
from activation import get_probe_dir, get_activations_for_split
from utils import get_dataset_config


class LinearProbe(nn.Module):
    def __init__(self, input_dim):
        super(LinearProbe, self).__init__()
        self.linear = nn.Linear(input_dim, 2)

    def forward(self, x):
        return self.linear(x)


def evaluate_probe(
    probe: nn.Module,
    activations: tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
) -> dict:
    """Evaluate a trained probe on a given dataset."""
    
    # === Data Preparation ===
    X_data = torch.cat(activations, dim=0).to(torch.float32)
    y_data = torch.cat([torch.zeros(activations[0].size(0)), torch.ones(activations[1].size(0))]).long()    
    data_loader = DataLoader(TensorDataset(X_data, y_data), batch_size=32, shuffle=False)
        
    # === Evaluation ===
    probe.eval()
    outputs_list = []
    with torch.no_grad():
        for batch_X, _ in data_loader:
            outputs_list.append(probe(batch_X.to(device)))
    outputs = torch.cat(outputs_list)
    probs = F.softmax(outputs, dim=1)
    _, preds = torch.max(outputs.data, 1)

    # === Conversion to numpy arrays ===
    y_cpu = y_data.cpu().numpy()
    preds_cpu = preds.cpu().numpy()
    probs_pos_cpu = probs.detach().cpu().numpy()[:, 1]

    return {
        "acc": accuracy_score(y_cpu, preds_cpu),
        "roc_auc": roc_auc_score(y_cpu, probs_pos_cpu) if len(np.unique(y_cpu)) > 1 else 0.5,
        "brier_score": brier_score_loss(y_cpu, probs_pos_cpu),
        "ece": BinaryCalibrationError().to(device)(probs[:, 1], y_data.to(device)).item(),
    }


def train_probe(
    train_activations: tuple[torch.Tensor, torch.Tensor],
    val_activations: dict[str, tuple[torch.Tensor, torch.Tensor]],
    layer_idx: int,
    epochs: int = 200,
    lr: float = 1e-3,
    device: torch.device = None,
    seed: int = 0,
) -> dict:
    """
    Train a linear probe, tracking best epoch for each dataset independently.
    """
    
    # === Seed ===
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    # === Data Preparation ===
    X_train = torch.cat(train_activations, dim=0).to(torch.float32)
    y_train = torch.cat([torch.zeros(train_activations[0].size(0)), torch.ones(train_activations[1].size(0))]).long()
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    
    val_loaders = {}
    val_labels = {}
    for name, (orig_acts, int_acts) in val_activations.items():
        X_val = torch.cat([orig_acts, int_acts], dim=0).to(torch.float32)
        y_val = torch.cat([torch.zeros(orig_acts.size(0)), torch.ones(int_acts.size(0))]).long()
        val_loaders[name] = DataLoader(TensorDataset(X_val, y_val), batch_size=32, shuffle=False)
        val_labels[name] = y_val.cpu().numpy()

    # === Model & Training Setup ===
    probe = LinearProbe(X_train.shape[-1]).to(device)
    optimizer = optim.AdamW(probe.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()    

    # === Training Loop ===
    best_epoch = {name: {"acc": 0.0, "epoch": 0, "model_state": None} for name in val_activations.keys()}    
    best_train_acc = 0.0
    best_train_epoch = 0
    best_train_state = None
    
    print(f"\n--- Training Probe for Layer {layer_idx} ---")
    for epoch in range(epochs):
        probe.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = probe(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        
        # Calculate training accuracy
        probe.eval()
        train_preds = []
        with torch.no_grad():
            for batch_X, _ in train_loader:
                outputs = probe(batch_X.to(device))
                _, predicted = torch.max(outputs.data, 1)
                train_preds.extend(predicted.cpu().numpy())
        train_acc = accuracy_score(y_train.cpu().numpy(), train_preds)
        
        # Track best training accuracy during training
        if train_acc > best_train_acc:
            best_train_acc = train_acc
            best_train_epoch = epoch
            best_train_state = probe.state_dict().copy()
            best_train_metrics = {"acc": train_acc}
        
        # Validation Step
        for name, loader in val_loaders.items():
            preds = []
            with torch.no_grad():
                for batch_X, _ in loader:
                    outputs = probe(batch_X.to(device))
                    _, predicted = torch.max(outputs.data, 1)
                    preds.extend(predicted.cpu().numpy())
            val_acc = accuracy_score(val_labels[name], preds)
            
            # Update best epoch for this dataset
            if val_acc > best_epoch[name]["acc"]:
                best_epoch[name]["acc"] = val_acc
                best_epoch[name]["epoch"] = epoch
                best_epoch[name]["model_state"] = probe.state_dict().copy()

    # === Evaluate metrics at best epoch for each dataset ===
    results = {"layer": layer_idx}
    
    probe.load_state_dict(best_train_state)
    best_train_metrics = evaluate_probe(probe, train_activations, device)
    results["gsm8k_best_train_acc"] = best_train_metrics["acc"]
    results["gsm8k_best_train_epoch"] = best_train_epoch
    results["gsm8k_train_acc"] = best_train_metrics["acc"]  
    print(f"[Layer {layer_idx}] gsm8k-train: best epoch {best_train_epoch}, acc {best_train_metrics['acc']:.4f}")

    for name, info in best_epoch.items():
        probe.load_state_dict(info["model_state"])
        metrics = evaluate_probe(probe, val_activations[name], device)
        for key, value in metrics.items():
            results[f"{name}_{key}"] = value
        results[f"{name}_best_epoch"] = info["epoch"]
        print(f"[Layer {layer_idx}] {name}: best epoch {info['epoch']}, acc {info['acc']:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run activation generation and probe training for gsm8k.")

    # Base arguments
    parser.add_argument("--model-name", type=str, required=True, help="Name of the model to use.")
    parser.add_argument("--gpu", type=str, default="0", help="GPU ID to use (e.g., '0' for single GPU, '0,1,2,3' for multi-GPU).")

    # Activation collection arguments
    parser.add_argument("--layers", type=int, nargs='+', default=None, help="List of layer indices to collect activations from.")

    # Training probe arguments
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs for probe training.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for probe training.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
        
    args = parser.parse_args()
    
    # === Environment and Model Setup ===
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="auto", dtype="auto")    
    model.gradient_checkpointing_enable()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # === Directory Setup ===
    probe_dir = get_probe_dir(project_root, args)
    probe_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving probe results to: {probe_dir}")

    # === Activation Collection ===
    val_activations = {}

    # ID data
    args.dataset = "gsm8k"
    print(f"--- Collecting activations for dataset: {args.dataset}, split: train ---")
    train_orig_acts, train_int_acts = get_activations_for_split(args, "train", model, tokenizer, probe_dir)
    print(f"--- Collecting activations for dataset: {args.dataset}, split: test ---")
    id_test_orig, id_test_int = get_activations_for_split(args, "test", model, tokenizer, probe_dir)
    val_activations["gsm8k"] = (id_test_orig, id_test_int)

    # OOD data
    OOD_DATASETS = ["math_500"]
    for ood_name in OOD_DATASETS:
        ood_args = argparse.Namespace(**vars(args))
        ood_args.dataset = ood_name
        ood_split = get_dataset_config(
            model_name=args.model_name, 
            dataset=ood_name, 
            config_path=project_root.parent / "configs"
        ).get("test_split")
        print(f"--- Collecting activations for dataset: {ood_name}, split: {ood_split} ---")
        orig_acts, int_acts = get_activations_for_split(ood_args, ood_split, model, tokenizer, probe_dir)
        val_activations[ood_name] = (orig_acts, int_acts)
    
    # === Probe Training for each layer ===
    device = torch.device("cuda:0")
    results = []
    layers = args.layers or list(range(model.config.num_hidden_layers))
    for layer_idx in layers:
        val_activations_for_layer = {}
        for name, (orig_acts, int_acts) in val_activations.items():
            if layer_idx in orig_acts and layer_idx in int_acts:
                val_activations_for_layer[name] = (orig_acts[layer_idx], int_acts[layer_idx])

        layer_metrics = train_probe(
            train_activations=(train_orig_acts[layer_idx], train_int_acts[layer_idx]),
            val_activations=val_activations_for_layer,
            layer_idx=layer_idx,
            epochs=args.epochs,
            lr=args.lr,
            device=device,
            seed=args.seed,
        )
        results.append(layer_metrics)

    # === Save results to CSV ===
    df = pd.DataFrame(results)
    csv_path = probe_dir / "probe_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Probe results saved to {csv_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Train the learned halting head on top of frozen Coconut hidden states.

The halting head is a small MLP that takes the hidden state at each latent
position and predicts whether to halt (1) or continue (0).

Usage:
    python train_halt_head.py \
        --labels halt_labels/halt_labels.json \
        --output-dir halt_head/ \
        [--val-split 0.1] \
        [--lambda-sparse 0.05] \
        [--hidden-size 128] \
        [--epochs 20] \
        [--lr 1e-3] \
        [--batch-size 256] \
        [--seed 42]
"""

import json
import argparse
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, roc_auc_score

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_LATENT         = 6
HIDDEN_SIZE_GPT2 = 768
MIN_LATENT_STEPS = 2
POSITIONS        = list(range(MIN_LATENT_STEPS, N_LATENT))  # [2, 3, 4, 5]


# ---------------------------------------------------------------------------
# Halting Head Model
# ---------------------------------------------------------------------------

class HaltingHead(nn.Module):
    def __init__(self, input_size: int = HIDDEN_SIZE_GPT2, inner_size: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, inner_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(inner_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (batch, hidden_size) → (batch, 1) halt probability"""
        return self.net(h)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class HaltDataset(Dataset):
    def __init__(self, items: list[dict]):
        """
        Each item in items is a (hidden_state, label, pos, n_steps) tuple.
        """
        self.features = []
        self.labels   = []
        self.positions = []
        self.n_steps  = []

        for entry in items:
            optimal_halt = entry["optimal_halt"]
            n_steps      = entry["n_steps"]
            hidden_states = entry["hidden_states"]  # list of N_LATENT lists

            for i, pos in enumerate(POSITIONS):
                h_idx = i + 1  # hidden state AFTER the last completed pass
                if h_idx >= len(hidden_states):
                    break
                h     = torch.tensor(hidden_states[h_idx], dtype=torch.float32)
                label = 1 if pos >= optimal_halt else 0
                self.features.append(h)
                self.labels.append(label)
                self.positions.append(pos)
                self.n_steps.append(n_steps)

        self.features  = torch.stack(self.features)   # (N, hidden_size)
        self.labels    = torch.tensor(self.labels, dtype=torch.float32)
        self.positions = torch.tensor(self.positions, dtype=torch.float32)
        self.n_steps   = torch.tensor(self.n_steps,  dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.features[idx],
            self.labels[idx],
            self.positions[idx],
            self.n_steps[idx],
        )

    def class_weights(self) -> torch.Tensor:
        """Inverse frequency weights for BCE loss to handle class imbalance."""
        n_pos = self.labels.sum().item()
        n_neg = len(self.labels) - n_pos
        w_pos = len(self.labels) / (2 * n_pos) if n_pos > 0 else 1.0
        w_neg = len(self.labels) / (2 * n_neg) if n_neg > 0 else 1.0
        return torch.tensor([w_neg, w_pos])


# ---------------------------------------------------------------------------
# Data loading and splitting
# ---------------------------------------------------------------------------

def load_labels(path: str) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} labeled samples from {path}")
    return data


def train_val_split(data: list, val_split: float, seed: int) -> tuple:
    rng = random.Random(seed)
    shuffled = data.copy()
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_split))
    return shuffled[n_val:], shuffled[:n_val]


def print_dataset_stats(data: list, name: str) -> None:
    optimal_halts = [d["optimal_halt"] for d in data]
    n_steps_list  = [d["n_steps"] for d in data]

    halt_dist = defaultdict(int)
    for h in optimal_halts:
        halt_dist[h] += 1

    # Count label distribution
    n_halt = sum(1 for d in data for pos in POSITIONS if pos >= d["optimal_halt"])
    n_cont = sum(1 for d in data for pos in POSITIONS if pos < d["optimal_halt"])

    print(f"\n{name} ({len(data)} samples, {n_halt+n_cont} training pairs):")
    print(f"  Label distribution: halt={n_halt} ({n_halt/(n_halt+n_cont)*100:.1f}%), "
          f"continue={n_cont} ({n_cont/(n_halt+n_cont)*100:.1f}%)")
    print(f"  Optimal halt distribution:")
    for k in sorted(halt_dist.keys()):
        print(f"    halt@{k}: {halt_dist[k]} ({halt_dist[k]/len(data)*100:.1f}%)")
    print(f"  Step count: mean={np.mean(n_steps_list):.1f}, "
          f"min={min(n_steps_list)}, max={max(n_steps_list)}")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_epoch(
    model: HaltingHead,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    lambda_sparse: float,
    pos_weight: torch.Tensor,
    device: str,
) -> dict:
    model.train()
    total_loss = 0.0
    total_bce  = 0.0
    total_sparse = 0.0
    correct = 0
    total   = 0

    bce_fn = nn.BCELoss(reduction="none")

    for features, labels, positions, _ in loader:
        features  = features.to(device)
        labels    = labels.to(device)
        positions = positions.to(device)

        optimizer.zero_grad()
        p_halt = model(features).squeeze(1)  # (batch,)

        # Weighted BCE: upweight the minority class
        weights = torch.where(labels == 1, pos_weight[1].to(device),
                              pos_weight[0].to(device))
        bce  = (bce_fn(p_halt, labels) * weights).mean()

        # Sparsity penalty: encourage halting at earlier positions
        # normalized position in [0,1] range
        pos_norm = (positions - MIN_LATENT_STEPS) / (N_LATENT - MIN_LATENT_STEPS)
        sparse = (p_halt * pos_norm).mean()

        loss = bce + lambda_sparse * sparse
        loss.backward()
        optimizer.step()

        total_loss   += loss.item()
        total_bce    += bce.item()
        total_sparse += sparse.item()

        preds   = (p_halt > 0.5).float()
        correct += (preds == labels).sum().item()
        total   += len(labels)

    return {
        "loss":   total_loss / len(loader),
        "bce":    total_bce  / len(loader),
        "sparse": total_sparse / len(loader),
        "acc":    correct / total,
    }


@torch.no_grad()
def evaluate(
    model: HaltingHead,
    loader: DataLoader,
    lambda_sparse: float,
    pos_weight: torch.Tensor,
    device: str,
) -> dict:
    model.eval()
    all_preds  = []
    all_labels = []
    all_probs  = []
    all_pos    = []
    all_steps  = []

    bce_fn = nn.BCELoss(reduction="none")
    total_loss = 0.0

    for features, labels, positions, n_steps in loader:
        features  = features.to(device)
        labels    = labels.to(device)
        positions = positions.to(device)

        p_halt = model(features).squeeze(1)

        weights = torch.where(labels == 1, pos_weight[1].to(device),
                              pos_weight[0].to(device))
        bce  = (bce_fn(p_halt, labels) * weights).mean()
        pos_norm = (positions - MIN_LATENT_STEPS) / (N_LATENT - MIN_LATENT_STEPS)
        loss = bce + lambda_sparse * (p_halt * pos_norm).mean()
        total_loss += loss.item()

        preds = (p_halt > 0.5).float()
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        all_probs.extend(p_halt.cpu().tolist())
        all_pos.extend(positions.cpu().tolist())
        all_steps.extend(n_steps.tolist())

    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0

    # Per-position accuracy
    pos_acc = defaultdict(lambda: [0, 0])  # [correct, total]
    for pred, label, pos in zip(all_preds, all_labels, all_pos):
        pos_acc[int(pos)][1] += 1
        pos_acc[int(pos)][0] += int(pred == label)

    return {
        "loss": total_loss / len(loader),
        "acc":  acc,
        "auc":  auc,
        "pos_acc": {
            k: v[0] / v[1] for k, v in sorted(pos_acc.items())
        },
    }


# ---------------------------------------------------------------------------
# Simulated inference: estimate avg latent steps on val set
# ---------------------------------------------------------------------------

@torch.no_grad()
def simulate_halting(
    model: HaltingHead,
    val_data: list,
    device: str,
) -> dict:
    """
    Simulate the halting head's decisions on the val set.
    Reports avg latent steps used and estimated accuracy preservation.
    """
    model.eval()
    halt_positions = []
    correct_at_halt = []

    for entry in val_data:
        optimal_halt = entry["optimal_halt"]
        hidden_states = entry["hidden_states"]
        correct_by_pos = entry["correct_by_pos"]

        halted_at = N_LATENT  # default: use all steps
        for i, pos in enumerate(POSITIONS):
            h_idx = i + 1  # hidden state AFTER the last completed pass
            if h_idx >= len(hidden_states):
                break
            h = torch.tensor(hidden_states[h_idx], dtype=torch.float32).unsqueeze(0).to(device)
            p = model(h).item()
            if p > 0.5:
                halted_at = pos
                break

        halt_positions.append(halted_at)
        correct_at_halt.append(correct_by_pos.get(str(halted_at), False))

    avg_halt   = np.mean(halt_positions)
    compute_saved = (N_LATENT - avg_halt) / N_LATENT * 100
    accuracy_preserved = np.mean(correct_at_halt) * 100

    return {
        "avg_halt_pos":    avg_halt,
        "compute_saved":   compute_saved,
        "acc_at_halt_pos": accuracy_preserved,
        "halt_dist": dict(zip(*np.unique(halt_positions, return_counts=True))),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels",        required=True)
    parser.add_argument("--output-dir",    default="halt_head")
    parser.add_argument("--val-split",     type=float, default=0.1)
    parser.add_argument("--lambda-sparse", type=float, default=0.05)
    parser.add_argument("--hidden-size",   type=int,   default=128)
    parser.add_argument("--epochs",        type=int,   default=20)
    parser.add_argument("--lr",            type=float, default=1e-3)
    parser.add_argument("--batch-size",    type=int,   default=256)
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--device",        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {args.device}")
    print(f"Lambda sparse: {args.lambda_sparse}")
    print(f"MLP inner size: {args.hidden_size}")

    # Load and split
    data = load_labels(args.labels)
    train_data, val_data = train_val_split(data, args.val_split, args.seed)
    print_dataset_stats(train_data, "Train")
    print_dataset_stats(val_data,   "Val")

    train_ds = HaltDataset(train_data)
    val_ds   = HaltDataset(val_data)

    pos_weight = train_ds.class_weights()
    print(f"\nClass weights: continue={pos_weight[0]:.3f}, halt={pos_weight[1]:.3f}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0)

    # Model
    model = HaltingHead(
        input_size=HIDDEN_SIZE_GPT2,
        inner_size=args.hidden_size,
    ).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nHalting head parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    # Training loop
    best_val_auc = 0.0
    best_epoch   = 0
    history      = []

    print(f"\n{'='*70}")
    print(f"{'Epoch':>6}  {'TrLoss':>8}  {'TrAcc':>7}  "
          f"{'VaLoss':>8}  {'VaAcc':>7}  {'AUC':>7}")
    print(f"{'-'*70}")

    for epoch in range(1, args.epochs + 1):
        tr = train_epoch(model, train_loader, optimizer,
                         args.lambda_sparse, pos_weight, args.device)
        va = evaluate(model, val_loader, args.lambda_sparse,
                      pos_weight, args.device)
        scheduler.step()

        history.append({"epoch": epoch, "train": tr, "val": va})

        print(f"{epoch:>6}  {tr['loss']:>8.4f}  {tr['acc']:>7.3f}  "
              f"{va['loss']:>8.4f}  {va['acc']:>7.3f}  {va['auc']:>7.3f}")

        if va["auc"] > best_val_auc:
            best_val_auc = va["auc"]
            best_epoch   = epoch
            torch.save(model.state_dict(), out_dir / "halt_head_best.pt")

    print(f"\nBest epoch: {best_epoch} (val AUC={best_val_auc:.4f})")

    # Load best and evaluate
    model.load_state_dict(torch.load(out_dir / "halt_head_best.pt"))
    va_final = evaluate(model, val_loader, args.lambda_sparse,
                        pos_weight, args.device)

    print(f"\nPer-position val accuracy (best model):")
    for pos, acc in va_final["pos_acc"].items():
        print(f"  Position {pos}: {acc*100:.1f}%")

    # Simulate halting on val set
    sim = simulate_halting(model, val_data, args.device)
    print(f"\nSimulated halting on val set:")
    print(f"  Avg halt position:  {sim['avg_halt_pos']:.2f}/{N_LATENT}")
    print(f"  Compute saved:      {sim['compute_saved']:.1f}%")
    print(f"  Acc at halt pos:    {sim['acc_at_halt_pos']:.1f}%")
    print(f"  Halt distribution:  {dict(sim['halt_dist'])}")

    # Save everything
    torch.save(model.state_dict(), out_dir / "halt_head_final.pt")
    with open(out_dir / "training_history.json", "w") as f:
        json.dump({
            "args": vars(args),
            "best_epoch": best_epoch,
            "best_val_auc": best_val_auc,
            "history": history,
            "final_val": va_final,
            "simulation": {
                k: ({str(kk): int(vv) for kk, vv in v.items()}
                    if isinstance(v, dict)
                    else float(v) if isinstance(v, (np.floating, np.integer))
                    else v)
                for k, v in sim.items()
            },
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2, default=str)

    print(f"\nSaved to {out_dir}/")
    print(f"  halt_head_best.pt    — best checkpoint (by val AUC)")
    print(f"  halt_head_final.pt   — final epoch checkpoint")
    print(f"  training_history.json")


if __name__ == "__main__":
    main()
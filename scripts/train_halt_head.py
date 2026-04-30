#!/usr/bin/env python3
"""
Train the learned halting head on top of frozen Coconut hidden states.

The halting head is a small MLP that takes the hidden state at each latent
position and predicts whether to halt (1) or continue (0).

Usage:
    python train_halt_head.py \
        --labels halt_labels/halt_labels.h5 \
        --output-dir halt_head/ \
        [--val-split 0.1] \
        [--hidden-size 128] \
        [--epochs 20] \
        [--lr 1e-3] \
        [--batch-size 256] \
        [--seed 67]
"""

import json
import argparse
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, roc_auc_score

from coconut import HaltingHead

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HIDDEN_SIZE_GPT2      = 768
N_LATENT              = 6
MIN_LATENT_STEPS      = 2
N_CANDIDATE_POS = N_LATENT - MIN_LATENT_STEPS # [2, 3, 4, 5] -> 4


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class HaltDataset(Dataset):
    """Dataset loaded directly from flat HDF5 arrays."""
    def __init__(self,
                 features:  torch.Tensor,
                 labels:    torch.Tensor,
                 positions: torch.Tensor,
                 n_steps:   torch.Tensor):
        self.features  = features
        self.labels    = labels
        self.positions = positions
        self.n_steps   = n_steps

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

def load_h5(path: str, val_split: float, seed: int) -> tuple:
    """
    Load flat HDF5 label file and split into train/val datasets.
    Returns (train_dataset, val_dataset, meta) where meta is a dict of stats.
    """
    import h5py
    print(f"Loading {path}...")
    with h5py.File(path, "r") as f:
        features  = torch.tensor(f["hidden_states"][:], dtype=torch.float32)
        labels    = torch.tensor(f["labels"][:],        dtype=torch.float32)
        positions = torch.tensor(f["positions"][:],     dtype=torch.float32)
        n_steps   = torch.tensor(f["n_steps"][:],       dtype=torch.long)
        n_pairs   = int(f.attrs.get("n_pairs",   len(labels)))
        n_samples = int(f.attrs.get("n_samples", n_pairs // N_CANDIDATE_POS))

    print(f"  {n_pairs} training pairs from {n_samples} labeled samples")

    # Shuffle pair indices (not sample-aware, but fine for MLP training)
    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_pairs, generator=rng)
    features  = features[perm]
    labels    = labels[perm]
    positions = positions[perm]
    n_steps   = n_steps[perm]

    n_val = max(1, int(n_pairs * val_split))
    def split(t): return t[n_val:], t[:n_val]

    tr_f,  va_f  = split(features)
    tr_l,  va_l  = split(labels)
    tr_p,  va_p  = split(positions)
    tr_ns, va_ns = split(n_steps)

    train_ds = HaltDataset(tr_f,  tr_l,  tr_p,  tr_ns)
    val_ds   = HaltDataset(va_f,  va_l,  va_p,  va_ns)

    # Print stats
    for ds, name in [(train_ds, "Train"), (val_ds, "Val")]:
        n_h = ds.labels.sum().int().item()
        n_c = len(ds.labels) - n_h
        print(f"\n{name} ({len(ds.labels)} pairs):")
        print(f"  halt={n_h} ({n_h/len(ds.labels)*100:.1f}%)  "
              f"continue={n_c} ({n_c/len(ds.labels)*100:.1f}%)")

    return train_ds, val_ds


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_epoch(
    model: HaltingHead,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    pos_weight: torch.Tensor,
    device: str,
) -> dict:
    model.train()
    total_loss = 0.0
    correct = 0
    total   = 0

    bce_fn = nn.BCELoss(reduction="none")

    for features, labels, positions, _ in loader:
        features  = features.to(device)
        labels    = labels.to(device)

        optimizer.zero_grad()
        p_halt = model(features).squeeze(1)

        weights = torch.where(labels == 1, pos_weight[1].to(device),
                              pos_weight[0].to(device))
        loss = (bce_fn(p_halt, labels) * weights).mean()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds   = (p_halt > 0.5).float()
        correct += (preds == labels).sum().item()
        total   += len(labels)

    return {
        "loss": total_loss / len(loader),
        "acc":  correct / total,
    }


@torch.no_grad()
def evaluate(
    model: HaltingHead,
    loader: DataLoader,
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
        loss = (bce_fn(p_halt, labels) * weights).mean()
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
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels",      required=True,
                        help="Path to halt_labels.h5 (HDF5 format)")
    parser.add_argument("--output-dir",  default="halt_head")
    parser.add_argument("--val-split",   type=float, default=0.1)
    parser.add_argument("--hidden-size", type=int,   default=128)
    parser.add_argument("--epochs",      type=int,   default=20)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--batch-size",  type=int,   default=256)
    parser.add_argument("--seed",        type=int,   default=67)
    parser.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {args.device}")
    print(f"MLP inner size: {args.hidden_size}")

    # Load data
    train_ds, val_ds = load_h5(args.labels, args.val_split, args.seed)

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

    print(f"\n{'='*60}")
    print(f"{'Epoch':>6}  {'TrLoss':>8}  {'TrAcc':>7}  "
          f"{'VaLoss':>8}  {'VaAcc':>7}  {'AUC':>7}")
    print(f"{'-'*60}")

    for epoch in range(1, args.epochs + 1):
        tr = train_epoch(model, train_loader, optimizer, pos_weight, args.device)
        va = evaluate(model, val_loader, pos_weight, args.device)
        scheduler.step()

        history.append({"epoch": epoch, "train": tr, "val": va})

        print(f"{epoch:>6}  {tr['loss']:>8.4f}  {tr['acc']:>7.3f}  "
              f"{va['loss']:>8.4f}  {va['acc']:>7.3f}  {va['auc']:>7.3f}")

        if va["auc"] > best_val_auc:
            best_val_auc = va["auc"]
            best_epoch   = epoch
            torch.save(model.state_dict(), out_dir / "halt_head_best.pt")

    print(f"\nBest epoch: {best_epoch} (val AUC={best_val_auc:.4f})")

    # Load best and report per-position accuracy
    model.load_state_dict(torch.load(out_dir / "halt_head_best.pt",
                                     map_location=args.device))
    va_final = evaluate(model, val_loader, pos_weight, args.device)

    print(f"\nPer-position val accuracy (best model):")
    for pos, acc in va_final["pos_acc"].items():
        print(f"  Position {int(pos)}: {acc*100:.1f}%")

    # Save
    torch.save(model.state_dict(), out_dir / "halt_head_final.pt")
    with open(out_dir / "training_history.json", "w") as f:
        json.dump({
            "args":          vars(args),
            "best_epoch":    best_epoch,
            "best_val_auc":  best_val_auc,
            "history":       history,
            "final_val":     va_final,
            "timestamp":     datetime.now().isoformat(),
        }, f, indent=2, default=str)

    print(f"\nSaved to {out_dir}/")
    print(f"  halt_head_best.pt     — best checkpoint (by val AUC)")
    print(f"  halt_head_final.pt    — final epoch checkpoint")
    print(f"  training_history.json")
    print(f"\nNote: run analyze_halting.py --halt-head to evaluate on the full val set.")


if __name__ == "__main__":
    main()
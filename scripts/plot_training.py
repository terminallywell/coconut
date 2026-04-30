"""
Training dynamics visualization for the learned halting head.

Produces a two-panel figure:
  Left:  Train and val loss over epochs
  Right: Val AUC over epochs with best epoch marked

Usage:
    python scripts/plot_training.py \
        --history checkpoints/head/training_history.json \
        --output-dir figures/
"""

import json
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid", font_scale=1.2)

palette = sns.color_palette('Set2')

# Paired color theme
TRAIN_LOSS_COLOR = palette[0]
VAL_LOSS_COLOR   = palette[1]
AUC_COLOR        = palette[2]
BEST_COLOR       = palette[3]


def load_history(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def plot_training_dynamics(history: dict, output_path: Path, dpi: int = 150):
    epochs     = [e["epoch"]          for e in history["history"]]
    tr_loss    = [e["train"]["loss"]   for e in history["history"]]
    va_loss    = [e["val"]["loss"]     for e in history["history"]]
    va_auc     = [e["val"]["auc"]      for e in history["history"]]
    best_epoch = history["best_epoch"]
    best_auc   = history["best_val_auc"]

    fig, axes = plt.subplots(2, 1, figsize=(6, 8), constrained_layout=True)
    fig.suptitle("Halting Head Training Dynamics", fontsize=13, fontweight="bold")

    # --- Panel (a): Loss ---
    ax = axes[0]
    ax.plot(epochs, tr_loss, "-", color=TRAIN_LOSS_COLOR,
            linewidth=2.0, label="Train loss")
    ax.plot(epochs, va_loss, "-", color=VAL_LOSS_COLOR,
            linewidth=2.0, label="Val loss")
    ax.axvline(best_epoch, color=BEST_COLOR, linestyle="--",
               linewidth=1.2, alpha=0.7, label=f"Best epoch ({best_epoch})")
    # ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("(a) Train and Validation Loss")
    ax.legend(fontsize=10)

    # --- Panel (b): Val AUC ---
    ax = axes[1]
    ax.plot(epochs, va_auc, "-", color=AUC_COLOR,
            linewidth=2.0, label="Val AUC")
    ax.scatter([best_epoch], [best_auc], color=BEST_COLOR,
               zorder=5, s=50, label=f"Best AUC = {best_auc:.3f}")
    ax.axvline(best_epoch, color=BEST_COLOR, linestyle="--",
               linewidth=1.2, alpha=0.7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AUC")
    ax.set_title("(b) Validation AUC")
    ax.margins(y=0.15)
    ax.legend(fontsize=10)

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--history",    required=True,
                        help="Path to training_history.json")
    parser.add_argument("--output-dir", default="figures")
    parser.add_argument("--dpi",        type=int, default=150)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    history = load_history(args.history)
    print(f"Loaded {len(history['history'])} epochs")
    print(f"Best epoch: {history['best_epoch']} "
          f"(val AUC={history['best_val_auc']:.4f})")

    plot_training_dynamics(
        history,
        output_path=out_dir / "training_dynamics.png",
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()

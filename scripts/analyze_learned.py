#!/usr/bin/env python3
"""
Experiment C (evaluation): Learned halting head threshold sweep for Coconut GSM8K.

Sweeps the halt decision threshold of a trained halting head MLP and measures
the accuracy vs. latent steps saved tradeoff. Results are stratified by number
of gold reasoning steps.

Usage (run from repo root):
    python scripts/analyze_learned.py \
        --checkpoint checkpoints/gsm/jiviteshjn_s1r_ck13 \
        --halting-head checkpoints/head/halt_head_best.pt \
        --val-path data/gsm_valid.json \
        --output results/halting_results_learned.json \
        [--min-latent-steps 2] \
        [--device cuda]
"""

import json
import argparse
import random
import sys
from datetime import datetime
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from halting_utils import N_LATENT, load_model, load_val_data, evaluate_with_threshold

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HEAD_THRESHOLDS  = np.arange(0.05, 1.0, 0.05).tolist()
HIDDEN_SIZE_GPT2 = 768


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",       required=True)
    parser.add_argument("--halting-head",        required=True,
                        help="Path to trained halting head .pt checkpoint")
    parser.add_argument("--val-path",         default="data/gsm_valid.json")
    parser.add_argument("--output",           default="results/halting_results_learned.json")
    parser.add_argument("--min-latent-steps", type=int, default=2)
    parser.add_argument("--device",           default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed",             type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Device: {args.device}")
    print(f"Min latent steps: {args.min_latent_steps}")
    print(f"Decision thresholds: {HEAD_THRESHOLDS}")

    model, tokenizer = load_model(args.checkpoint, args.device)
    data             = load_val_data(args.val_path)

    # Load halting head
    from coconut import HaltingHead
    halting_head = HaltingHead(input_size=HIDDEN_SIZE_GPT2).to(args.device)
    halting_head.load_state_dict(
        torch.load(args.halting_head, map_location=args.device)
    )
    halting_head.eval()
    print(f"Loaded halting head from {args.halting_head}")

    results = {
        "metadata": {
            "checkpoint":       args.checkpoint,
            "halting_head":        args.halting_head,
            "val_path":         args.val_path,
            "n_latent":         N_LATENT,
            "min_latent_steps": args.min_latent_steps,
            "thresholds":       HEAD_THRESHOLDS,
            "timestamp":        datetime.now().isoformat(),
        },
        "thresholds": {},
    }

    for ht in HEAD_THRESHOLDS:
        label = f"halting_head_t{ht:.2f}"
        desc  = f"Learned head (threshold={ht:.2f})"

        print(f"\n--- {desc} ---")
        result = evaluate_with_threshold(
            model, tokenizer, data, args.device,
            halt_threshold=None,
            min_latent_steps=args.min_latent_steps,
            halting_head=halting_head,
            halting_head_threshold=ht,
            desc=desc,
        )
        results["thresholds"][label] = {
            "threshold": ht,
            "halt_mode": "learned",
            **result,
        }

        acc   = result["accuracy"]
        avg   = result["avg_latent_used"]
        saved = (N_LATENT - avg) / N_LATENT * 100
        print(f"  Accuracy: {acc*100:.1f}%  |  "
              f"Avg latent: {avg:.2f}/{N_LATENT}  |  "
              f"Steps saved: {saved:.1f}%")

        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)

    # Summary
    print("\n" + "=" * 65)
    print("SUMMARY: Accuracy vs. Latent Steps Saved")
    print("=" * 65)
    print(f"{'Threshold':>12}  {'Accuracy':>10}  {'Avg Latent':>11}  {'Saved':>7}")
    print("-" * 65)
    for label, r in results["thresholds"].items():
        saved = (N_LATENT - r["avg_latent_used"]) / N_LATENT * 100
        print(f"{r['threshold']:>12.2f}  {r['accuracy']*100:>9.1f}%  "
              f"{r['avg_latent_used']:>9.2f}/{N_LATENT}  "
              f"{saved:>6.1f}%")

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Experiment A: Corruption analysis for Coconut GSM8K checkpoint.

Measures accuracy under progressive, single-position, and intra-step
corruption of latent thought positions. Results are stratified by
problem difficulty (number of gold reasoning steps).

Usage (run from repo root with coconut env active):
    python analyze_corruption.py \
        --checkpoint checkpoints/eval/jiviteshjn_s1r_ck13 \
        --val-path data/gsm_valid.json \
        --output corruption_results.json \
        [--n-calibration 50] \
        [--device cuda]

Requirements:
    - coconut.py must be the modified version with corrupt_positions support
    - Checkpoint must be a stage-3 GSM8K Coconut checkpoint (6 latent positions)
"""

import os
import re
import sys
import json
import argparse
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_LATENT = 6          # c_thought=2, max_latent_stage=3 → 3*2 = 6
MODEL_ID = "openai-community/gpt2"
MAX_NEW_TOKENS = 100  # matches run.py default for GSM8K

# DIFFICULTY_BUCKETS = {
#     "easy":   (1, 2),
#     "medium": (3, 4),
#     "hard":   (5, 99),
# }


# ---------------------------------------------------------------------------
# Model loading (mirrors run.py setup)
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, device: str) -> tuple:
    """Load GPT-2 + Coconut wrapper from checkpoint."""
    # import here so script works from repo root
    from coconut import Coconut

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens("<|start-latent|>")
    tokenizer.add_tokens("<|end-latent|>")
    tokenizer.add_tokens("<|latent|>")

    latent_id  = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id   = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id     = tokenizer.convert_tokens_to_ids("<|end-latent|>")

    base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    base_model.resize_token_embeddings(len(tokenizer))

    model = Coconut(base_model, latent_id, start_id, end_id, tokenizer.eos_token_id)

    print(f"Loading checkpoint from {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Missing keys: {missing[:5]}{'...' if len(missing)>5 else ''}")
    if unexpected:
        print(f"  Unexpected keys: {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")

    model = model.to(device)
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_val_data(val_path: str) -> list[dict]:
    with open(val_path) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} validation samples from {val_path}")
    return data


def build_prompt(sample: dict, tokenizer, device: str, latent_id: int,
                 start_id: int, end_id: int) -> torch.Tensor:
    """
    Build the input_ids tensor for a GSM8K sample at stage 3
    (all 3 reasoning steps replaced by latent tokens).
    Format mirrors run.py's dataset construction.
    """
    question = sample["question"].strip()

    # Stage 3: all steps replaced → 3 groups of 2 latent tokens
    latent_block = (
        [start_id]
        + [latent_id] * N_LATENT
        + [end_id]
    )

    # Tokenize question
    q_ids = tokenizer.encode(question, add_special_tokens=False)

    # Full sequence: question + latent block (no answer — we'll generate)
    input_ids = q_ids + latent_block
    return torch.tensor(input_ids, device=device).unsqueeze(0)


def extract_answer(text: str) -> str:
    """Match run.py's extraction: last token after '#', strip commas."""
    return text.split("#")[-1].replace(",", "").strip()


# ---------------------------------------------------------------------------
# Calibration: collect per-position hidden state statistics
# ---------------------------------------------------------------------------

@torch.no_grad()
def calibrate(model, tokenizer, data: list[dict], device: str,
              n_samples: int = 50) -> list[tuple]:
    """
    Run n_samples clean forward passes and collect hidden states at each
    latent position. Returns list of (mean, std) tensors, one per position.
    """
    print(f"\nCalibrating noise stats on {n_samples} samples...")
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id  = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id    = tokenizer.convert_tokens_to_ids("<|end-latent|>")

    collected = defaultdict(list)  # position → list of hidden state tensors

    samples = random.sample(data, min(n_samples, len(data)))

    for sample in tqdm(samples, desc="Calibration"):
        input_ids = build_prompt(sample, tokenizer, device,
                                 latent_id, start_id, end_id)
        labels    = input_ids.clone()
        attn_mask = torch.ones_like(input_ids)
        pos_ids   = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)

        outputs = model.forward(
            input_ids, attn_mask, labels, pos_ids,
            collect_hidden_states=True,
        )

        for pos_idx, h in enumerate(outputs.collected_hidden_states):
            collected[pos_idx].append(h)

    noise_stats = []
    for pos_idx in range(N_LATENT):
        if pos_idx not in collected or len(collected[pos_idx]) == 0:
            print(f"  Warning: no hidden states collected for position {pos_idx}")
            noise_stats.append((torch.zeros(1), torch.ones(1)))
            continue
        stacked = torch.stack(collected[pos_idx])  # (n_samples, hidden_size)
        mean = stacked.mean(dim=0)
        std  = stacked.std(dim=0).clamp(min=1e-6)
        l2   = stacked.norm(dim=1).mean().item()
        print(f"  Position {pos_idx}: mean_L2={l2:.2f}, "
              f"std_mean={std.mean().item():.4f}")
        noise_stats.append((mean, std))

    return noise_stats


# ---------------------------------------------------------------------------
# Single evaluation run
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model,
    tokenizer,
    data: list[dict],
    device: str,
    corrupt_positions: set | None = None,
    noise_stats: list | None = None,
    desc: str = "Eval",
) -> dict:
    """
    Run generation on all samples with optional corruption.
    Returns dict with overall accuracy and per-difficulty accuracy.
    """
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id  = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id    = tokenizer.convert_tokens_to_ids("<|end-latent|>")

    correct_by_diff = defaultdict(int)
    total_by_diff   = defaultdict(int)
    correct_total   = 0

    for sample in tqdm(data, desc=desc, leave=False):
        n_steps  = len(sample["steps"])
        answer   = sample["answer"].replace(",", "").strip()

        input_ids = build_prompt(sample, tokenizer, device,
                                 latent_id, start_id, end_id)
        attn_mask = torch.ones_like(input_ids)

        outputs = model.generate(
            input_ids,
            attn_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            corrupt_positions=corrupt_positions,
            noise_stats=noise_stats,
            synced_gpus=False,
        )

        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        pred = extract_answer(text)
        correct = int(pred == answer)
        correct_total += correct

        # group by # golden steps
        k = n_steps
        correct_by_diff[k] += correct
        total_by_diff[k]   += 1

    n = len(data)
    result = {
        "accuracy":    correct_total / n,
        "n_correct":   correct_total,
        "n_total":     n,
        "by_difficulty": {
            str(k): {
                "accuracy": correct_by_diff[k] / total_by_diff[k]
                            if total_by_diff[k] > 0 else None,
                "n_correct": correct_by_diff[k],
                "n_total":   total_by_diff[k],
            }
            for k in sorted(total_by_diff.keys())
        },
    }
    return result


# def difficulty_bucket(n_steps: int) -> str:
#     for bkt, (lo, hi) in DIFFICULTY_BUCKETS.items():
#         if lo <= n_steps <= hi:
#             return bkt
#     return "hard"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",     required=True)
    parser.add_argument("--val-path",       default="data/gsm_valid.json")
    parser.add_argument("--output",         default="corruption_results.json")
    parser.add_argument("--n-calibration",  type=int, default=50)
    parser.add_argument("--device",         default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed",           type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Device: {args.device}")
    print(f"N latent positions: {N_LATENT}")

    model, tokenizer = load_model(args.checkpoint, args.device)
    data = load_val_data(args.val_path)
    noise_stats = calibrate(model, tokenizer, data, args.device, args.n_calibration)

    results = {
        "metadata": {
            "checkpoint":    args.checkpoint,
            "val_path":      args.val_path,
            "n_latent":      N_LATENT,
            "n_calibration": args.n_calibration,
            "timestamp":     datetime.now().isoformat(),
            "noise_l2": [
                float(noise_stats[i][0].norm().item()) if i < len(noise_stats) else None
                for i in range(N_LATENT)
            ],
        },
        "experiments": {},
    }

    def run(label, corrupt_positions, desc=None):
        desc = desc or label
        print(f"\n--- {label} ---")
        r = evaluate(model, tokenizer, data, args.device,
                     corrupt_positions=corrupt_positions,
                     noise_stats=noise_stats,
                     desc=desc)
        results["experiments"][label] = r
        acc = r["accuracy"]
        print(f"  Overall: {acc*100:.1f}%  ({r['n_correct']}/{r['n_total']})")
        for bkt, v in r["by_difficulty"].items():
            if v["n_total"] > 0:
                print(f"  {bkt:6s}: {v['accuracy']*100:.1f}%  (n={v['n_total']})")
        # save incrementally
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)

    # 1. Clean baseline
    run("clean", corrupt_positions=None, desc="Clean baseline")

    # 2. Progressive forward corruption: corrupt positions 0..k
    for k in range(1, N_LATENT + 1):
        run(f"progressive_fwd_{k}",
            corrupt_positions=set(range(k)),
            desc=f"Fwd corrupt 0..{k-1}")

    # 3. Progressive reverse corruption: corrupt positions k..5
    for k in range(N_LATENT - 1, -1, -1):
        run(f"progressive_rev_{k}",
            corrupt_positions=set(range(k, N_LATENT)),
            desc=f"Rev corrupt {k}..{N_LATENT-1}")

    # 4. Single-position corruption
    for k in range(N_LATENT):
        run(f"single_{k}",
            corrupt_positions={k},
            desc=f"Single corrupt pos {k}")

    # 5. Intra-step: first thoughts only (positions 0, 2, 4)
    run("intra_first_thoughts",
        corrupt_positions={0, 2, 4},
        desc="First thoughts corrupted")

    # 6. Intra-step: second thoughts only (positions 1, 3, 5)
    run("intra_second_thoughts",
        corrupt_positions={1, 3, 5},
        desc="Second thoughts corrupted")

    # Final summary
    print("\n" + "=" * 55)
    print("SUMMARY")
    print("=" * 55)
    print(f"{'Experiment':<30} {'Accuracy':>10}")
    print("-" * 55)
    for label, r in results["experiments"].items():
        print(f"{label:<30} {r['accuracy']*100:>9.1f}%")

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Experiment B: Entropy-based early halting for Coconut GSM8K.

Sweeps halt_threshold over a range of entropy values (in nats) and
measures the accuracy vs. average latent positions used tradeoff.
Results are stratified by number of gold reasoning steps.

Usage (run from repo root with coconut env active):
    python analyze_halting.py \
        --checkpoint checkpoints/eval/jiviteshjn_s1r_ck13 \
        --val-path data/gsm_valid.json \
        --output halting_results.json \
        [--min-latent-steps 2] \
        [--device cuda]
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

N_LATENT    = 6
MODEL_ID    = "openai-community/gpt2"
MAX_NEW_TOKENS = 100

# Threshold sweep: log-spaced between 0.1 and 8.0 nats
# GPT-2 vocab = 50,257 → max entropy = ln(50257) ≈ 10.8 nats
# In practice clean model entropy at latent positions is typically 2-7 nats
THRESHOLDS = [None] + list(np.round(np.logspace(-1, 0.9, 16), 3))
# None = no halting (clean baseline)
# Values: ~0.1 (very aggressive) to ~8.0 (very conservative)


# ---------------------------------------------------------------------------
# Model loading (identical to analyze_corruption.py)
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, device: str) -> tuple:
    from coconut import Coconut

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens("<|start-latent|>")
    tokenizer.add_tokens("<|end-latent|>")
    tokenizer.add_tokens("<|latent|>")

    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id  = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id    = tokenizer.convert_tokens_to_ids("<|end-latent|>")

    base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    base_model.resize_token_embeddings(len(tokenizer))

    model = Coconut(base_model, latent_id, start_id, end_id, tokenizer.eos_token_id)

    print(f"Loading checkpoint from {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Missing keys: {missing[:5]}{'...' if len(missing)>5 else ''}")

    model = model.to(device)
    model.eval()
    return model, tokenizer


def load_val_data(val_path: str) -> list:
    with open(val_path) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} validation samples")
    return data


def build_prompt(sample, tokenizer, device, latent_id, start_id, end_id):
    question = sample["question"].strip()
    latent_block = [start_id] + [latent_id] * N_LATENT + [end_id]
    q_ids = tokenizer.encode(question, add_special_tokens=False)
    input_ids = q_ids + latent_block
    return torch.tensor(input_ids, device=device).unsqueeze(0)


def extract_answer(text: str) -> str:
    return text.split("#")[-1].replace(",", "").strip()


# ---------------------------------------------------------------------------
# Calibration: measure entropy distribution under clean inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def measure_entropy_distribution(model, tokenizer, data, device, n_samples=100):
    """
    Measure the distribution of per-position next-token entropy under clean
    inference. Used to choose sensible threshold values.
    """
    print(f"\nMeasuring entropy distribution on {n_samples} samples...")
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id  = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id    = tokenizer.convert_tokens_to_ids("<|end-latent|>")

    entropies_by_pos = defaultdict(list)

    samples = random.sample(data, min(n_samples, len(data)))

    for sample in tqdm(samples, desc="Entropy calibration"):
        input_ids = build_prompt(sample, tokenizer, device,
                                 latent_id, start_id, end_id)
        labels    = input_ids.clone()
        attn_mask = torch.ones_like(input_ids)
        pos_ids   = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)

        # Run forward with per-pass logit collection
        # We re-implement a simplified version here to capture per-pass entropy
        from coconut import Coconut
        # Access the underlying forward but intercept logits per pass
        # Simpler: run with halt_threshold=inf (never halts) and collect
        outputs = model.forward(
            input_ids, attn_mask, labels, pos_ids,
            halt_threshold=float('inf'),  # never halts, but runs halt check
            min_latent_steps=0,
            collect_hidden_states=False,
        )
        # outputs.logits is concatenated across passes
        # We need per-pass entropy — use a dedicated collection pass instead
        _collect_per_pass_entropy(model, input_ids, attn_mask, labels, pos_ids,
                                   entropies_by_pos)

    print("\nEntropy distribution by latent position (nats):")
    print(f"  {'Pos':>4}  {'Mean':>7}  {'P10':>7}  {'P50':>7}  {'P90':>7}")
    for pos in range(N_LATENT):
        vals = entropies_by_pos[pos]
        if vals:
            arr = np.array(vals)
            print(f"  {pos:>4}  {arr.mean():>7.3f}  "
                  f"{np.percentile(arr,10):>7.3f}  "
                  f"{np.percentile(arr,50):>7.3f}  "
                  f"{np.percentile(arr,90):>7.3f}")

    return entropies_by_pos


@torch.no_grad()
def _collect_per_pass_entropy(model, input_ids, attn_mask, labels, pos_ids,
                               entropies_by_pos):
    """Run forward pass manually to collect per-pass entropy."""
    from coconut import Coconut

    latent_indices = (input_ids == model.latent_token_id).nonzero()
    latent_lists = [
        [idx[1].item() for idx in latent_indices if idx[0] == i]
        for i in range(input_ids.shape[0])
    ]
    max_n_latents = max(len(l) for l in latent_lists)
    inputs_embeds = model.embedding(input_ids)
    next_compute_range = (0, latent_indices[:, 1].min().item())
    kv_cache = None

    for pass_idx in range(max_n_latents):
        if kv_cache is None:
            outputs = model.base_causallm(
                inputs_embeds=inputs_embeds[:, next_compute_range[0]:next_compute_range[1], :],
                attention_mask=attn_mask[:, next_compute_range[0]:next_compute_range[1]],
                position_ids=pos_ids[:, next_compute_range[0]:next_compute_range[1]],
                output_hidden_states=True,
            )
            hidden_states_offset = 0
        else:
            past_key_values = [
                (k[:, :, :next_compute_range[0], :], v[:, :, :next_compute_range[0], :])
                for k, v in kv_cache
            ]
            outputs = model.base_causallm(
                inputs_embeds=inputs_embeds[:, next_compute_range[0]:next_compute_range[1], :],
                attention_mask=attn_mask[:, :next_compute_range[1]],
                position_ids=pos_ids[:, next_compute_range[0]:next_compute_range[1]],
                past_key_values=past_key_values,
                output_hidden_states=True,
            )
            hidden_states_offset = next_compute_range[0]

        # Compute entropy at this pass
        last_logit = outputs.logits[0, -1, :]
        probs = torch.softmax(last_logit, dim=-1)
        entropy = -(probs * probs.log().clamp(min=-1e9)).sum().item()
        entropies_by_pos[pass_idx].append(entropy)

        next_compute_range = (
            next_compute_range[1],
            input_ids.shape[1] if pass_idx + 1 >= max_n_latents
            else next_compute_range[1] + 1,
        )
        hidden_states = outputs.hidden_states[-1]
        kv_cache = outputs.past_key_values

        filling_indices = [
            (i, mask_list[pass_idx])
            for i, mask_list in enumerate(latent_lists)
            if len(mask_list) > pass_idx
        ]
        tensor_list = [
            [inputs_embeds[b, p, :] for p in range(inputs_embeds.shape[1])]
            for b in range(inputs_embeds.shape[0])
        ]
        for batch_idx, token_idx in filling_indices:
            tensor_list[batch_idx][token_idx] = hidden_states[
                batch_idx, token_idx - 1 - hidden_states_offset, :
            ]
        inputs_embeds = torch.stack([
            torch.stack(tensor_list[b]) for b in range(inputs_embeds.shape[0])
        ])


# ---------------------------------------------------------------------------
# Evaluation with halting
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_with_threshold(
    model, tokenizer, data, device,
    halt_threshold, min_latent_steps,
    halting_head=None, halt_head_threshold=0.5, desc="Eval"
):
    """
    Evaluate with a given halt_threshold or halting_head. Returns accuracy,
    average latent steps used, and per-step-count breakdown.
    """
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id  = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id    = tokenizer.convert_tokens_to_ids("<|end-latent|>")

    correct_by_steps  = defaultdict(int)
    total_by_steps    = defaultdict(int)
    latent_by_steps   = defaultdict(list)
    correct_total     = 0
    latent_used_total = []

    for sample in tqdm(data, desc=desc, leave=False):
        n_steps = len(sample["steps"])
        answer  = sample["answer"].replace(",", "").strip()

        input_ids = build_prompt(sample, tokenizer, device,
                                 latent_id, start_id, end_id)
        attn_mask = torch.ones_like(input_ids)

        out_tokens, n_used = model.generate(
            input_ids,
            attn_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            halt_threshold=halt_threshold,
            min_latent_steps=min_latent_steps,
            halting_head=halting_head,
            halt_head_threshold=halt_head_threshold,
            return_n_latent=True,
            synced_gpus=False,
        )

        text = tokenizer.decode(out_tokens[0], skip_special_tokens=True)
        pred = extract_answer(text)
        correct = int(pred == answer)

        correct_total += correct
        latent_used_total.append(n_used)
        correct_by_steps[n_steps]  += correct
        total_by_steps[n_steps]    += 1
        latent_by_steps[n_steps].append(n_used)

    n = len(data)
    result = {
        "accuracy":          correct_total / n,
        "n_correct":         correct_total,
        "n_total":           n,
        "avg_latent_used":   float(np.mean(latent_used_total)),
        "latent_used_dist":  {
            str(k): int(v)
            for k, v in sorted(
                zip(*np.unique(latent_used_total, return_counts=True))
            )
        },
        "by_steps": {
            str(k): {
                "accuracy":        correct_by_steps[k] / total_by_steps[k]
                                   if total_by_steps[k] > 0 else None,
                "n_correct":       correct_by_steps[k],
                "n_total":         total_by_steps[k],
                "avg_latent_used": float(np.mean(latent_by_steps[k]))
                                   if latent_by_steps[k] else None,
            }
            for k in sorted(total_by_steps.keys())
        },
    }
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",       required=True)
    parser.add_argument("--val-path",         default="data/gsm_valid.json")
    parser.add_argument("--output",           default="halting_results.json")
    parser.add_argument("--min-latent-steps", type=int, default=2)
    parser.add_argument("--halt-head",        default=None,
                        help="Path to trained halting head checkpoint (.pt). "
                             "If provided, runs a single eval with the learned head "
                             "instead of the entropy threshold sweep.")
    parser.add_argument("--device",           default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed",             type=int, default=42)
    parser.add_argument("--skip-entropy-cal", action="store_true",
                        help="Skip entropy distribution measurement")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Device: {args.device}")
    print(f"Min latent steps before halting: {args.min_latent_steps}")

    model, tokenizer = load_model(args.checkpoint, args.device)
    data = load_val_data(args.val_path)

    # --- Load halting head if provided ---
    halting_head = None
    if args.halt_head is not None:
        from scripts.train_halt_head import HaltingHead
        halting_head = HaltingHead().to(args.device)
        halting_head.load_state_dict(torch.load(args.halt_head, map_location=args.device))
        halting_head.eval()
        print(f"Loaded halting head from {args.halt_head}")

    results = {
        "metadata": {
            "checkpoint":       args.checkpoint,
            "val_path":         args.val_path,
            "n_latent":         N_LATENT,
            "halt_head":        args.halt_head,
            "min_latent_steps": args.min_latent_steps,
            "timestamp":        datetime.now().isoformat(),
        },
        "entropy_distribution": {},
        "thresholds": {},
    }

    if halting_head is not None:
        # --- Sweep halt head decision thresholds ---
        head_thresholds = np.arange(0.05, 1, 0.05)
        print(f"Halt head probability threshold sweep")

        for ht in head_thresholds:
            label = f"halt_head_t{ht:.2f}"
            desc  = f"Learned head (threshold={ht:.2f})"
            print(f"\n--- {desc} ---")
            result = evaluate_with_threshold(
                model, tokenizer, data, args.device,
                halt_threshold=None,
                min_latent_steps=args.min_latent_steps,
                halting_head=halting_head,
                halt_head_threshold=ht,
                desc=desc,
            )
            results["thresholds"][label] = {
                "threshold":           ht,
                "halt_mode":           "learned",
                **result,
            }
            acc     = result["accuracy"]
            avg_lat = result["avg_latent_used"]
            saved   = (N_LATENT - avg_lat) / N_LATENT * 100
            print(f"  Accuracy: {acc*100:.1f}%  |  "
                  f"Avg latent used: {avg_lat:.2f}/{N_LATENT}  |  "
                  f"Steps saved: {saved:.1f}%")
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)

    else:
        # --- Entropy threshold sweep ---
        print(f"Thresholds to sweep: {[t for t in THRESHOLDS if t is not None]}")

        if not args.skip_entropy_cal:
            ent_dist = measure_entropy_distribution(model, tokenizer, data, args.device)
            results["entropy_distribution"] = {
                str(pos): {
                    "mean": float(np.mean(ent_dist[pos])),
                    "p10":  float(np.percentile(ent_dist[pos], 10)),
                    "p50":  float(np.percentile(ent_dist[pos], 50)),
                    "p90":  float(np.percentile(ent_dist[pos], 90)),
                }
                for pos in range(N_LATENT) if ent_dist[pos]
            }

        for threshold in THRESHOLDS:
            label = "no_halt" if threshold is None else f"thresh_{threshold}"
            desc  = "No halting (baseline)" if threshold is None \
                    else f"Threshold {threshold:.3f} nats"

            print(f"\n--- {desc} ---")
            result = evaluate_with_threshold(
                model, tokenizer, data, args.device,
                halt_threshold=threshold,
                min_latent_steps=args.min_latent_steps,
                desc=desc,
            )
            results["thresholds"][label] = {
                "threshold": threshold,
                **result,
            }

            acc     = result["accuracy"]
            avg_lat = result["avg_latent_used"]
            saved   = (N_LATENT - avg_lat) / N_LATENT * 100
            print(f"  Accuracy: {acc*100:.1f}%  |  "
                  f"Avg latent used: {avg_lat:.2f}/{N_LATENT}  |  "
                  f"Compute saved: {saved:.1f}%")

            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)

    # Final save and summary
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 65)
    print("SUMMARY: Accuracy vs. Compute Tradeoff")
    print("=" * 65)
    print(f"{'Label':>20}  {'Accuracy':>10}  {'Avg Latent':>11}  {'Saved':>7}")
    print("-" * 65)
    for label, r in results["thresholds"].items():
        saved = (N_LATENT - r["avg_latent_used"]) / N_LATENT * 100
        print(f"{label:>20}  {r['accuracy']*100:>9.1f}%  "
              f"{r['avg_latent_used']:>9.2f}/{N_LATENT}  "
              f"{saved:>6.1f}%")

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
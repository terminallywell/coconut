#!/usr/bin/env python3
"""
Experiment B: Entropy-based early halting for Coconut GSM8K.

Sweeps halt_threshold over a range of entropy values (in nats) and measures
the accuracy vs. latent steps saved tradeoff. Results are stratified by
number of gold reasoning steps.

Usage (run from repo root):
    python scripts/analyze_entropy.py \
        --checkpoint checkpoints/gsm/jiviteshjn_s1r_ck13 \
        --val-path data/gsm_valid.json \
        --output results/halting_results_entropy.json \
        [--min-latent-steps 2] \
        [--skip-entropy-cal] \
        [--device cuda]
"""

import json
import argparse
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from halting_utils import N_LATENT, load_model, load_val_data, build_prompt, evaluate_with_threshold

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Log-spaced thresholds from 0.1 to ~8.0 nats
# GPT-2 vocab = 50,257 → max entropy ≈ 10.8 nats
THRESHOLDS = [None] + list(np.round(np.logspace(-1, 0.9, 16), 3))


# ---------------------------------------------------------------------------
# Entropy distribution measurement
# ---------------------------------------------------------------------------

@torch.no_grad()
def measure_entropy_distribution(model, tokenizer, data, device,
                                  n_samples: int = 100) -> dict:
    """
    Measure the distribution of per-position next-token entropy under clean
    inference. Used to verify that chosen thresholds span a useful range.
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
        attn_mask = torch.ones_like(input_ids)
        pos_ids   = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
        _collect_per_pass_entropy(model, input_ids, attn_mask, pos_ids,
                                   entropies_by_pos)

    print("\nEntropy distribution by latent position (nats):")
    print(f"  {'Pos':>4}  {'Mean':>7}  {'P10':>7}  {'P50':>7}  {'P90':>7}")
    for pos in range(N_LATENT):
        vals = entropies_by_pos[pos]
        if vals:
            arr = np.array(vals)
            print(f"  {pos+1:>4}  {arr.mean():>7.3f}  "
                  f"{np.percentile(arr,10):>7.3f}  "
                  f"{np.percentile(arr,50):>7.3f}  "
                  f"{np.percentile(arr,90):>7.3f}")

    return entropies_by_pos


@torch.no_grad()
def _collect_per_pass_entropy(model, input_ids, attn_mask, pos_ids,
                               entropies_by_pos: dict):
    """Run the latent loop manually to collect per-pass next-token entropy."""
    latent_indices = (input_ids == model.latent_token_id).nonzero()
    latent_lists   = [
        [idx[1].item() for idx in latent_indices if idx[0] == i]
        for i in range(input_ids.shape[0])
    ]
    max_n_latents  = max(len(l) for l in latent_lists)
    inputs_embeds  = model.embedding(input_ids)
    next_compute_range = (0, latent_indices[:, 1].min().item())
    kv_cache = None

    for pass_idx in range(max_n_latents):
        if kv_cache is None:
            outputs = model.base_causallm(
                inputs_embeds=inputs_embeds[
                    :, next_compute_range[0]:next_compute_range[1], :],
                attention_mask=attn_mask[
                    :, next_compute_range[0]:next_compute_range[1]],
                position_ids=pos_ids[
                    :, next_compute_range[0]:next_compute_range[1]],
                output_hidden_states=True,
            )
            hidden_states_offset = 0
        else:
            past_key_values = [
                (k[:, :, :next_compute_range[0], :],
                 v[:, :, :next_compute_range[0], :])
                for k, v in kv_cache
            ]
            outputs = model.base_causallm(
                inputs_embeds=inputs_embeds[
                    :, next_compute_range[0]:next_compute_range[1], :],
                attention_mask=attn_mask[:, :next_compute_range[1]],
                position_ids=pos_ids[
                    :, next_compute_range[0]:next_compute_range[1]],
                past_key_values=past_key_values,
                output_hidden_states=True,
            )
            hidden_states_offset = next_compute_range[0]

        last_logit = outputs.logits[0, -1, :]
        probs      = torch.softmax(last_logit, dim=-1)
        entropy    = -(probs * probs.log().clamp(min=-1e9)).sum().item()
        entropies_by_pos[pass_idx].append(entropy)

        next_compute_range = (
            next_compute_range[1],
            input_ids.shape[1] if pass_idx + 1 >= max_n_latents
            else next_compute_range[1] + 1,
        )

        hidden_states = outputs.hidden_states[-1]
        kv_cache      = outputs.past_key_values

        filling_indices = [
            (i, mask_list[pass_idx])
            for i, mask_list in enumerate(latent_lists)
            if len(mask_list) > pass_idx
        ]
        tensor_list = [
            [inputs_embeds[b, p, :] for p in range(inputs_embeds.shape[1])]
            for b in range(inputs_embeds.shape[0])
        ]
        for b, tok_idx in filling_indices:
            tensor_list[b][tok_idx] = hidden_states[
                b, tok_idx - 1 - hidden_states_offset, :
            ]
        inputs_embeds = torch.stack([
            torch.stack(tensor_list[b])
            for b in range(inputs_embeds.shape[0])
        ])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",       required=True)
    parser.add_argument("--val-path",         default="data/gsm_valid.json")
    parser.add_argument("--output",           default="results/halting_results_entropy.json")
    parser.add_argument("--min-latent-steps", type=int, default=2)
    parser.add_argument("--skip-entropy-cal", action="store_true")
    parser.add_argument("--device",           default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed",             type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Device: {args.device}")
    print(f"Min latent steps: {args.min_latent_steps}")
    print(f"Thresholds: {[t for t in THRESHOLDS if t is not None]}")

    model, tokenizer = load_model(args.checkpoint, args.device)
    data             = load_val_data(args.val_path)

    results = {
        "metadata": {
            "checkpoint":       args.checkpoint,
            "val_path":         args.val_path,
            "n_latent":         N_LATENT,
            "min_latent_steps": args.min_latent_steps,
            "thresholds":       [t for t in THRESHOLDS if t is not None],
            "timestamp":        datetime.now().isoformat(),
        },
        "entropy_distribution": {},
        "thresholds": {},
    }

    if not args.skip_entropy_cal:
        ent_dist = measure_entropy_distribution(model, tokenizer, data,
                                                args.device)
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
        results["thresholds"][label] = {"threshold": threshold, **result}

        acc    = result["accuracy"]
        avg    = result["avg_latent_used"]
        saved  = (N_LATENT - avg) / N_LATENT * 100
        print(f"  Accuracy: {acc*100:.1f}%  |  "
              f"Avg latent: {avg:.2f}/{N_LATENT}  |  "
              f"Steps saved: {saved:.1f}%")

        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)

    # Summary
    print("\n" + "=" * 65)
    print("SUMMARY: Accuracy vs. Latent Steps Saved")
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

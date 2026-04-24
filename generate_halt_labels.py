#!/usr/bin/env python3
"""
Generate training labels for the learned halting head.

For each training sample, runs inference halting at each latent position
k in {min_latent_steps, ..., N_LATENT} and records whether the answer is
correct. The optimal halt point is the smallest k that gives a correct answer.

Outputs a JSON file of labeled examples:
    [
        {
            "question": "...",
            "answer": "...",
            "steps": [...],
            "n_steps": 3,
            "optimal_halt": 4,       # smallest k giving correct answer
            "correct_by_pos": {      # accuracy at each halt position
                "2": true,
                "3": true,
                "4": true,
                "5": false
            },
            "hidden_states": {       # per-position hidden states (saved separately)
                "2": "hs_idx_0_pos_2",
                ...
            }
        },
        ...
    ]

Hidden states are saved as a separate .pt file for memory efficiency.

Usage:
    python generate_halt_labels.py \
        --checkpoint checkpoints/eval/jiviteshjn_s1r_ck13 \
        --train-path data/gsm_train.json \
        --output-dir halt_labels/ \
        [--n-samples 20000] \
        [--min-latent-steps 2] \
        [--seed 42] \
        [--device cuda]
"""

import os
import json
import random
import argparse
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

N_LATENT   = 6
MODEL_ID   = "openai-community/gpt2"
MAX_NEW_TOKENS = 100


# ---------------------------------------------------------------------------
# Model loading
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
        print(f"  Missing keys: {missing[:3]}{'...' if len(missing)>3 else ''}")

    model = model.to(device)
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Data loading and stratified sampling
# ---------------------------------------------------------------------------

def load_and_sample(train_path: str, n_samples: int, seed: int) -> list:
    with open(train_path) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} training samples")

    if n_samples >= len(data):
        print(f"Requested {n_samples} >= dataset size, using all samples")
        return data

    # Stratify by step count
    by_steps = defaultdict(list)
    for item in data:
        by_steps[len(item["steps"])].append(item)

    print("Step count distribution in full dataset:")
    for k in sorted(by_steps.keys()):
        print(f"  {k} steps: {len(by_steps[k])} samples")

    # Sample proportionally, preserving step count distribution
    rng = random.Random(seed)
    sampled = []
    for k, items in by_steps.items():
        n = max(1, int(len(items) / len(data) * n_samples))
        sampled.extend(rng.sample(items, min(n, len(items))))

    # Trim or top up to exactly n_samples
    rng.shuffle(sampled)
    sampled = sampled[:n_samples]

    print(f"\nSampled {len(sampled)} items (stratified by step count)")
    sampled_by_steps = defaultdict(int)
    for item in sampled:
        sampled_by_steps[len(item["steps"])] += 1
    for k in sorted(sampled_by_steps.keys()):
        print(f"  {k} steps: {sampled_by_steps[k]} samples")

    return sampled


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_prompt(sample: dict, tokenizer, device: str,
                 latent_id: int, start_id: int, end_id: int) -> torch.Tensor:
    question = sample["question"].strip()
    latent_block = [start_id] + [latent_id] * N_LATENT + [end_id]
    q_ids = tokenizer.encode(question, add_special_tokens=False)
    input_ids = q_ids + latent_block
    return torch.tensor(input_ids, device=device).unsqueeze(0)


def extract_answer(text: str) -> str:
    return text.split("#")[-1].replace(",", "").strip()


# ---------------------------------------------------------------------------
# Single sample labeling
# ---------------------------------------------------------------------------

@torch.no_grad()
def label_sample(
    model,
    tokenizer,
    sample: dict,
    device: str,
    min_latent_steps: int,
    latent_id: int,
    start_id: int,
    end_id: int,
) -> dict | None:
    """
    Run inference halting at each position k in [min_latent_steps, N_LATENT].
    Also collect hidden states from a clean (no-halt) forward pass.

    Returns None if the model gets the answer wrong at ALL halt positions
    (unsolvable sample — discard from training).
    """
    answer = sample["answer"].replace(",", "").strip()
    input_ids = build_prompt(sample, tokenizer, device, latent_id, start_id, end_id)
    attn_mask = torch.ones_like(input_ids)

    # --- 1. Collect hidden states from clean forward pass ---
    labels  = input_ids.clone()
    pos_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
    fwd_out = model.forward(
        input_ids, attn_mask, labels, pos_ids,
        collect_hidden_states=True,
    )
    # collected_hidden_states: list of tensors, one per latent position
    hidden_states = fwd_out.collected_hidden_states  # list of (hidden_size,) tensors

    # --- 2. Run inference at each halt position ---
    correct_by_pos = {}
    for halt_at in range(min_latent_steps, N_LATENT + 1):
        # halt_threshold=inf forces halt at exactly halt_at steps
        # by setting min_latent_steps = halt_at and threshold very low
        # Simpler: use a fixed halt position by running with no threshold
        # and collecting only halt_at passes manually
        out_tokens = _run_with_fixed_halt(
            model, tokenizer, input_ids, attn_mask, device,
            halt_at=halt_at,
        )
        text = tokenizer.decode(out_tokens[0], skip_special_tokens=True)
        pred = extract_answer(text)
        correct_by_pos[halt_at] = (pred == answer)

    # --- 3. Find optimal halt point ---
    optimal_halt = None
    for k in range(min_latent_steps, N_LATENT + 1):
        if correct_by_pos[k]:
            optimal_halt = k
            break

    # Discard if unsolvable at any halt position
    if optimal_halt is None:
        return None

    return {
        "question":      sample["question"],
        "answer":        sample["answer"],
        "steps":         sample["steps"],
        "n_steps":       len(sample["steps"]),
        "optimal_halt":  optimal_halt,
        "correct_by_pos": {str(k): v for k, v in correct_by_pos.items()},
        "hidden_states": [h.tolist() for h in hidden_states],
        # ^ stored inline for simplicity; use separate .pt for large runs
    }


@torch.no_grad()
def _run_with_fixed_halt(
    model, tokenizer, input_ids, attn_mask, device, halt_at: int
) -> torch.Tensor:
    """
    Run generation stopping after exactly halt_at latent passes.
    Uses a very low entropy threshold that always triggers at min_latent_steps=halt_at.
    """
    # Set threshold=inf so it never triggers on entropy,
    # instead we use a custom fixed-halt forward pass
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

    for pass_idx in range(min(halt_at, max_n_latents)):
        if kv_cache is None:
            outputs = model.base_causallm(
                inputs_embeds=inputs_embeds[:, next_compute_range[0]:next_compute_range[1], :],
                attention_mask=attn_mask[:, next_compute_range[0]:next_compute_range[1]],
                position_ids=torch.arange(
                    next_compute_range[0], next_compute_range[1], device=device
                ).unsqueeze(0),
                output_hidden_states=True,
            )
            hidden_states_offset = 0
        else:
            past_kv = [
                (k[:, :, :next_compute_range[0], :], v[:, :, :next_compute_range[0], :])
                for k, v in kv_cache
            ]
            outputs = model.base_causallm(
                inputs_embeds=inputs_embeds[:, next_compute_range[0]:next_compute_range[1], :],
                attention_mask=attn_mask[:, :next_compute_range[1]],
                position_ids=torch.arange(
                    next_compute_range[0], next_compute_range[1], device=device
                ).unsqueeze(0),
                past_key_values=past_kv,
                output_hidden_states=True,
            )
            hidden_states_offset = next_compute_range[0]

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
        for b, tok_idx in filling_indices:
            tensor_list[b][tok_idx] = hidden_states[
                b, tok_idx - 1 - hidden_states_offset, :
            ]
        inputs_embeds = torch.stack([
            torch.stack(tensor_list[b]) for b in range(inputs_embeds.shape[0])
        ])

    # Final pass covering rest of sequence (PAD mode)
    past_kv = [
        (k[:, :, :next_compute_range[0], :], v[:, :, :next_compute_range[0], :])
        for k, v in kv_cache
    ] if kv_cache else None

    outputs = model.base_causallm(
        inputs_embeds=inputs_embeds[:, next_compute_range[0]:, :],
        attention_mask=attn_mask[:, :input_ids.shape[1]],
        position_ids=torch.arange(
            next_compute_range[0], input_ids.shape[1], device=device
        ).unsqueeze(0),
        past_key_values=past_kv,
        output_hidden_states=True,
    )

    # Autoregressive generation
    tokens = input_ids[0].tolist()
    next_token = torch.argmax(outputs.logits[0, -1]).item()
    tokens.append(next_token)
    new_embeds = torch.cat([
        inputs_embeds,
        model.embedding(torch.tensor(next_token, device=device)).view(1, 1, -1)
    ], dim=1)

    for _ in range(MAX_NEW_TOKENS - 1):
        out = model.base_causallm(inputs_embeds=new_embeds)
        next_token = torch.argmax(out.logits[0, -1]).item()
        if next_token == model.eos_token_id:
            break
        tokens.append(next_token)
        new_embeds = torch.cat([
            new_embeds,
            model.embedding(torch.tensor(next_token, device=device)).view(1, 1, -1)
        ], dim=1)

    return torch.tensor(tokens).view(1, -1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",        required=True)
    parser.add_argument("--train-path",        default="data/gsm_train.json")
    parser.add_argument("--output-dir",        default="halt_labels")
    parser.add_argument("--n-samples",         type=int, default=20000)
    parser.add_argument("--min-latent-steps",  type=int, default=2)
    parser.add_argument("--device",            default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed",              type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "halt_labels.json"

    print(f"Device: {args.device}")
    print(f"N samples: {args.n_samples}")
    print(f"Min latent steps: {args.min_latent_steps}")
    print(f"Halt positions: {list(range(args.min_latent_steps, N_LATENT + 1))}")

    model, tokenizer = load_model(args.checkpoint, args.device)

    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id  = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id    = tokenizer.convert_tokens_to_ids("<|end-latent|>")

    data = load_and_sample(args.train_path, args.n_samples, args.seed)

    results = []
    discarded = 0

    for sample in tqdm(data, desc="Generating labels"):
        labeled = label_sample(
            model, tokenizer, sample, args.device,
            args.min_latent_steps, latent_id, start_id, end_id,
        )
        if labeled is None:
            discarded += 1
        else:
            results.append(labeled)

        # Save incrementally every 100 samples
        if len(results) % 100 == 0 and len(results) > 0:
            with open(out_path, "w") as f:
                json.dump(results, f)

    # Final save
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    total = len(results) + discarded
    print(f"\n{'='*55}")
    print(f"SUMMARY")
    print(f"{'='*55}")
    print(f"Total processed:  {total}")
    print(f"Labeled:          {len(results)} ({len(results)/total*100:.1f}%)")
    print(f"Discarded:        {discarded} ({discarded/total*100:.1f}%)")

    # Optimal halt distribution
    halt_dist = defaultdict(int)
    for r in results:
        halt_dist[r["optimal_halt"]] += 1
    print(f"\nOptimal halt point distribution:")
    for k in sorted(halt_dist.keys()):
        print(f"  halt at {k}: {halt_dist[k]} ({halt_dist[k]/len(results)*100:.1f}%)")

    print(f"\nLabels saved to {out_path}")
    print(f"Hidden states stored inline (use --n-samples <=5000 for memory safety)")


if __name__ == "__main__":
    main()

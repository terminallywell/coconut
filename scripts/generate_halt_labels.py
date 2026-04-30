#!/usr/bin/env python3
"""
Generate training labels for the learned halting head.

For each training sample, runs inference halting at each latent position
k in {min_latent_steps, ..., N_LATENT-1} and records whether the answer is
correct. The optimal halt point k* is the smallest k that gives a correct answer.

Outputs two files:
  halt_labels.h5    — flat HDF5 arrays, one row per training pair:
      hidden_states  (N, 768)  float32   hidden state after pass k
      labels         (N,)      bool      1 if k >= k*, else 0
      positions      (N,)      int8      which halt position (2-5)
      n_steps        (N,)      int8      gold step count

  halt_labels_meta.json — per-sample metadata (question, answer, steps,
      optimal_halt, correct_by_pos) indexed by sample order. Pair index
      i corresponds to sample i // N_POSITIONS.

Usage:
    python generate_halt_labels.py \
        --checkpoint checkpoints/eval/jiviteshjn_s1r_ck13 \
        --train-path data/gsm_train.json \
        --output-dir results/ \
        [--n-samples 12000] \
        [--min-latent-steps 2] \
        [--seed 67] \
        [--device cuda]
"""

import json
import random
import argparse
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_LATENT         = 6
MODEL_ID         = "openai-community/gpt2"
MAX_NEW_TOKENS   = 100
HIDDEN_SIZE      = 768
MIN_LATENT_STEPS = 2
# positions where halt decision is made: after passes 2,3,4,5
HALT_POSITIONS   = list(range(MIN_LATENT_STEPS, N_LATENT))  # [2,3,4,5]
N_POSITIONS      = len(HALT_POSITIONS)


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

    by_steps = defaultdict(list)
    for item in data:
        by_steps[len(item["steps"])].append(item)

    print("Step count distribution in full dataset:")
    for k in sorted(by_steps.keys()):
        print(f"  {k} steps: {len(by_steps[k])} samples")

    rng = random.Random(seed)
    sampled = []
    for k, items in by_steps.items():
        n = max(1, int(len(items) / len(data) * n_samples))
        sampled.extend(rng.sample(items, min(n, len(items))))

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

def build_prompt(sample, tokenizer, device, latent_id, start_id, end_id):
    question = sample["question"].strip()
    latent_block = [start_id] + [latent_id] * N_LATENT + [end_id]
    q_ids = tokenizer.encode(question, add_special_tokens=False)
    return torch.tensor(q_ids + latent_block, device=device).unsqueeze(0)


def extract_answer(text: str) -> str:
    return text.split("#")[-1].replace(",", "").strip()


# ---------------------------------------------------------------------------
# Single sample labeling
# ---------------------------------------------------------------------------

@torch.no_grad()
def label_sample(model, tokenizer, sample, device, latent_id, start_id, end_id):
    """
    Returns (hidden_states, correct_by_pos, optimal_halt) or None if unsolvable.

    hidden_states: list of N_LATENT (768,) tensors from clean forward pass
    correct_by_pos: dict {pos: bool} for pos in HALT_POSITIONS
    optimal_halt: smallest pos giving correct answer, or None
    """
    answer = sample["answer"].replace(",", "").strip()
    input_ids = build_prompt(sample, tokenizer, device, latent_id, start_id, end_id)
    attn_mask = torch.ones_like(input_ids)

    # 1. Collect hidden states from clean forward pass
    labels  = input_ids.clone()
    pos_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
    fwd_out = model.forward(
        input_ids, attn_mask, labels, pos_ids,
        collect_hidden_states=True,
    )
    hidden_states = [h.cpu() for h in fwd_out.collected_hidden_states]

    # 2. Run inference halting at each position
    correct_by_pos = {}
    for halt_at in HALT_POSITIONS:
        out_tokens = _run_with_fixed_halt(
            model, tokenizer, input_ids, attn_mask, device, halt_at=halt_at,
        )
        text = tokenizer.decode(out_tokens[0], skip_special_tokens=True)
        correct_by_pos[halt_at] = (extract_answer(text) == answer)

    # 3. Find optimal halt point
    optimal_halt = next(
        (k for k in HALT_POSITIONS if correct_by_pos[k]), None
    )

    if optimal_halt is None:
        return None

    return hidden_states, correct_by_pos, optimal_halt


@torch.no_grad()
def _run_with_fixed_halt(model, tokenizer, input_ids, attn_mask, device, halt_at):
    """Run generation stopping after exactly halt_at latent passes."""
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
                position_ids=torch.arange(next_compute_range[0], next_compute_range[1],
                                          device=device).unsqueeze(0),
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
                position_ids=torch.arange(next_compute_range[0], next_compute_range[1],
                                          device=device).unsqueeze(0),
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

    # Final pass
    past_kv = [
        (k[:, :, :next_compute_range[0], :], v[:, :, :next_compute_range[0], :])
        for k, v in kv_cache
    ] if kv_cache else None
    outputs = model.base_causallm(
        inputs_embeds=inputs_embeds[:, next_compute_range[0]:, :],
        attention_mask=attn_mask[:, :input_ids.shape[1]],
        position_ids=torch.arange(next_compute_range[0], input_ids.shape[1],
                                  device=device).unsqueeze(0),
        past_key_values=past_kv,
    )

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
    parser.add_argument("--output-dir",        default="results")
    parser.add_argument("--n-samples",         type=int, default=12000)
    parser.add_argument("--min-latent-steps",  type=int, default=2)
    parser.add_argument("--device",            default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed",              type=int, default=67)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    h5_path   = out_dir / "halt_labels.h5"
    meta_path = out_dir / "halt_labels_meta.json"

    print(f"Device: {args.device}")
    print(f"N samples: {args.n_samples}")
    print(f"Halt positions: {HALT_POSITIONS}")
    print(f"Training pairs per sample: {N_POSITIONS}")

    model, tokenizer = load_model(args.checkpoint, args.device)
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id  = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id    = tokenizer.convert_tokens_to_ids("<|end-latent|>")

    data = load_and_sample(args.train_path, args.n_samples, args.seed)

    # Accumulate in memory — ~167MB for 12k samples, well within RAM
    all_hidden  = []   # (N_pairs, 768) float32
    all_labels  = []   # (N_pairs,) bool
    all_positions = [] # (N_pairs,) int8
    all_nsteps  = []   # (N_pairs,) int8
    meta        = []   # per-sample metadata
    discarded   = 0

    for sample in tqdm(data, desc="Generating labels"):
        result = label_sample(
            model, tokenizer, sample, args.device,
            latent_id, start_id, end_id,
        )
        if result is None:
            discarded += 1
            continue

        hidden_states, correct_by_pos, optimal_halt = result
        n_steps = len(sample["steps"])

        # Build N_POSITIONS training pairs for this sample
        for i, pos in enumerate(HALT_POSITIONS):
            h_idx = i + 1  # hidden state AFTER last completed pass
            if h_idx >= len(hidden_states):
                continue
            label = int(pos >= optimal_halt)
            all_hidden.append(hidden_states[h_idx].numpy())
            all_labels.append(label)
            all_positions.append(pos)
            all_nsteps.append(n_steps)

        meta.append({
            "question":      sample["question"],
            "answer":        sample["answer"],
            "steps":         sample["steps"],
            "n_steps":       n_steps,
            "optimal_halt":  optimal_halt,
            "correct_by_pos": {str(k): v for k, v in correct_by_pos.items()},
        })

    # Write HDF5
    print(f"\nWriting {len(all_hidden)} pairs to {h5_path}...")
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("hidden_states", data=np.array(all_hidden,    dtype=np.float32))
        f.create_dataset("labels",        data=np.array(all_labels,    dtype=bool))
        f.create_dataset("positions",     data=np.array(all_positions, dtype=np.int8))
        f.create_dataset("n_steps",       data=np.array(all_nsteps,    dtype=np.int8))
        f.attrs["n_pairs"]       = len(all_hidden)
        f.attrs["n_samples"]     = len(meta)
        f.attrs["hidden_size"]   = HIDDEN_SIZE
        f.attrs["halt_positions"] = HALT_POSITIONS
        f.attrs["timestamp"]     = datetime.now().isoformat()

    # Write metadata sidecar
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # Summary
    total = len(meta) + discarded
    print(f"\n{'='*55}")
    print(f"SUMMARY")
    print(f"{'='*55}")
    print(f"Total processed:  {total}")
    print(f"Labeled:          {len(meta)} ({len(meta)/total*100:.1f}%)")
    print(f"Discarded:        {discarded} ({discarded/total*100:.1f}%)")
    print(f"Training pairs:   {len(all_hidden)}")

    halt_dist = defaultdict(int)
    for m in meta:
        halt_dist[m["optimal_halt"]] += 1
    print(f"\nOptimal halt point distribution:")
    for k in sorted(halt_dist.keys()):
        print(f"  halt at {k}: {halt_dist[k]} ({halt_dist[k]/len(meta)*100:.1f}%)")

    size_mb = h5_path.stat().st_size / 1e6
    print(f"\nHDF5 file size: {size_mb:.1f} MB  (vs ~1.1GB JSON)")
    print(f"Saved: {h5_path}")
    print(f"Saved: {meta_path}")


if __name__ == "__main__":
    main()

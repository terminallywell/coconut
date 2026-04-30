"""
Shared utilities for entropy and learned halting analysis scripts.

Provides model loading, data loading, prompt construction, answer extraction,
and the core evaluate_with_threshold function used by both analyze_entropy.py
and analyze_learned.py.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_LATENT       = 6
MODEL_ID       = "openai-community/gpt2"
MAX_NEW_TOKENS = 100


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, device: str) -> tuple:
    sys.path.insert(0, str(Path(__file__).parent.parent))
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
    if unexpected:
        print(f"  Unexpected keys: {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")

    model = model.to(device)
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Data loading and prompt construction
# ---------------------------------------------------------------------------

def load_val_data(val_path: str) -> list:
    with open(val_path) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} validation samples")
    return data


def build_prompt(sample: dict, tokenizer, device: str,
                 latent_id: int, start_id: int, end_id: int) -> torch.Tensor:
    question = sample["question"].strip()
    latent_block = [start_id] + [latent_id] * N_LATENT + [end_id]
    q_ids = tokenizer.encode(question, add_special_tokens=False)
    return torch.tensor(q_ids + latent_block, device=device).unsqueeze(0)


def extract_answer(text: str) -> str:
    return text.split("#")[-1].replace(",", "").strip()


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_with_threshold(
    model,
    tokenizer,
    data: list,
    device: str,
    halt_threshold=None,
    min_latent_steps: int = 2,
    halting_head=None,
    halt_head_threshold: float = 0.5,
    desc: str = "Eval",
) -> dict:
    """
    Run generation on all samples with the given halting configuration.
    Returns accuracy, avg latent steps used, and per-step-count breakdown.

    Either halt_threshold (entropy) or halting_head (learned MLP) may be set,
    but not both. If neither is set, runs clean inference with no halting.
    """
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id  = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id    = tokenizer.convert_tokens_to_ids("<|end-latent|>")

    correct_by_steps = defaultdict(int)
    total_by_steps   = defaultdict(int)
    latent_by_steps  = defaultdict(list)
    correct_total    = 0
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

        text    = tokenizer.decode(out_tokens[0], skip_special_tokens=True)
        pred    = extract_answer(text)
        correct = int(pred == answer)

        correct_total            += correct
        latent_used_total.append(n_used)
        correct_by_steps[n_steps] += correct
        total_by_steps[n_steps]   += 1
        latent_by_steps[n_steps].append(n_used)

    n = len(data)
    return {
        "accuracy":         correct_total / n,
        "n_correct":        correct_total,
        "n_total":          n,
        "avg_latent_used":  float(np.mean(latent_used_total)),
        "latent_used_dist": {
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

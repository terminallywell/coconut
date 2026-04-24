#!/usr/bin/env python3
"""
Checkpoint evaluation script for Coconut GSM8K reproductions.
Downloads checkpoints from HuggingFace, runs eval, and stores results.

Usage:
    python eval_checkpoints.py [--checkpoints CKPT [CKPT ...]] [--skip-download]

Examples:
    # Run all configured checkpoints
    python eval_checkpoints.py

    # Run specific checkpoints only
    python eval_checkpoints.py --checkpoints esther22_ck5 jiviteshjn_ck13

    # Skip download (checkpoints already on disk)
    python eval_checkpoints.py --skip-download
"""

import os
import re
import sys
import json
import shutil
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.resolve()
CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "eval"
RESULTS_FILE = REPO_ROOT / "eval_results.json"
EVAL_YAML = REPO_ROOT / "args" / "gsm_coconut_eval.yaml"
EVAL_YAML_TMP = REPO_ROOT / "args" / "_eval_tmp.yaml"

# Checkpoints to evaluate.
# Add or remove entries here to control what gets downloaded and evaluated.
CHECKPOINTS = {
    # --- Esther22 (reference) ---
    "esther22_ck5": {
        "url": "https://huggingface.co/Esther22/coconut_Reproduction/resolve/main/stage_1_training_ck/checkpoint_5",
        "notes": "Esther22 stage1 ck5 — best documented (35.2% reported)",
    },
    "esther22_ck12": {
        "url": "https://huggingface.co/Esther22/coconut_Reproduction/resolve/main/stage_1_training_ck/checkpoint_12",
        "notes": "Esther22 stage1 ck12 — stage3, 32.8% reported",
    },

    # --- jiviteshjn stage1 ---
    "jiviteshjn_s1_ck5": {
        "url": "https://huggingface.co/jiviteshjn/coconut-checkpoints/resolve/main/gpt2/stage1/gsm-coconut/checkpoint_5",
        "notes": "jiviteshjn stage1 ck5",
    },
    "jiviteshjn_s1_ck10": {
        "url": "https://huggingface.co/jiviteshjn/coconut-checkpoints/resolve/main/gpt2/stage1/gsm-coconut/checkpoint_10",
        "notes": "jiviteshjn stage1 ck10 (latest in stage1 folder)",
    },

    # --- jiviteshjn stage1_resume ---
    "jiviteshjn_s1r_ck9": {
        "url": "https://huggingface.co/jiviteshjn/coconut-checkpoints/resolve/main/gpt2/stage1_resume/gsm-coconut-gpt2-stage1-resume/checkpoint_9",
        "notes": "jiviteshjn stage1_resume ck9 (earliest resume ckpt)",
    },
    "jiviteshjn_s1r_ck13": {
        "url": "https://huggingface.co/jiviteshjn/coconut-checkpoints/resolve/main/gpt2/stage1_resume/gsm-coconut-gpt2-stage1-resume/checkpoint_13",
        "notes": "jiviteshjn stage1_resume ck13 (latest)",
    },
}

# Eval YAML overrides applied to every run.
# Any key here overwrites the value from the base gsm_coconut_eval.yaml.
YAML_OVERRIDES = {
    "only_eval": True,
    "bf16": True,
    "val_path": "data/gsm_valid.json",
    "debug": False,
}

# torchrun settings
NPROC = 1  # change to 4 if running on g5.12xlarge


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def load_results() -> dict:
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return {}


def save_results(results: dict) -> None:
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    log(f"Results saved to {RESULTS_FILE}")


def download_checkpoint(name: str, url: str, dest: Path) -> bool:
    """Download a checkpoint file using wget. Returns True on success."""
    if dest.exists():
        log(f"  [skip] {name} already exists at {dest}")
        return True

    dest.parent.mkdir(parents=True, exist_ok=True)
    log(f"  Downloading {name} from {url}")
    result = subprocess.run(
        ["wget", "-q", "--show-progress", "-O", str(dest), url],
        check=False,
    )
    if result.returncode != 0:
        log(f"  [ERROR] Download failed for {name} (exit {result.returncode})")
        if dest.exists():
            dest.unlink()
        return False
    log(f"  Downloaded {name} ({dest.stat().st_size / 1e6:.1f} MB)")
    return True


def build_eval_yaml(checkpoint_path: Path) -> Path:
    """Load base eval yaml, apply overrides + checkpoint path, write tmp file."""
    with open(EVAL_YAML) as f:
        config = yaml.safe_load(f)

    config.update(YAML_OVERRIDES)
    config["load_model_path"] = str(checkpoint_path)

    with open(EVAL_YAML_TMP, "w") as f:
        yaml.dump(config, f)

    return EVAL_YAML_TMP


def parse_accuracy(output: str) -> float | None:
    """Extract val accuracy from torchrun stdout."""
    # Matches: "Accuracy on validation set: 389 / 1320 = 0.2946..."
    match = re.search(
        r"Accuracy on validation set:\s*\d+\s*/\s*\d+\s*=\s*([0-9.]+)",
        output,
    )
    if match:
        return float(match.group(1))
    return None


def run_eval(checkpoint_path: Path) -> tuple[float | None, str]:
    """Run torchrun eval, return (accuracy, raw_output)."""
    yaml_path = build_eval_yaml(checkpoint_path)

    cmd = [
        "torchrun",
        "--nnodes", "1",
        "--nproc_per_node", str(NPROC),
        "run.py",
        str(yaml_path),
    ]

    log(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )

    combined = result.stdout + result.stderr
    acc = parse_accuracy(combined)

    if result.returncode != 0 and acc is None:
        log(f"  [ERROR] torchrun exited with code {result.returncode}")
        # Print last 20 lines for debugging
        for line in combined.splitlines()[-20:]:
            print(f"    {line}")

    return acc, combined


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Coconut checkpoints")
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=list(CHECKPOINTS.keys()),
        help="Checkpoint names to evaluate (default: all configured)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download step (assume checkpoints already on disk)",
    )
    args = parser.parse_args()

    # Validate requested checkpoints
    unknown = [c for c in args.checkpoints if c not in CHECKPOINTS]
    if unknown:
        print(f"Unknown checkpoints: {unknown}")
        print(f"Available: {list(CHECKPOINTS.keys())}")
        sys.exit(1)

    results = load_results()

    print("=" * 60)
    print(f"Evaluating {len(args.checkpoints)} checkpoint(s)")
    print(f"Checkpoint dir: {CHECKPOINT_DIR}")
    print(f"Results file:   {RESULTS_FILE}")
    print("=" * 60)

    for name in args.checkpoints:
        cfg = CHECKPOINTS[name]
        dest = CHECKPOINT_DIR / name

        print()
        log(f"=== {name} ===")
        log(f"  {cfg['notes']}")

        # Skip if already evaluated
        if name in results:
            log(f"  [skip] Already evaluated: acc={results[name]['accuracy']:.4f}")
            continue

        # Download
        if not args.skip_download:
            ok = download_checkpoint(name, cfg["url"], dest)
            if not ok:
                results[name] = {
                    "accuracy": None,
                    "notes": cfg["notes"],
                    "error": "download_failed",
                    "timestamp": datetime.now().isoformat(),
                }
                save_results(results)
                continue
        else:
            if not dest.exists():
                log(f"  [ERROR] Checkpoint not found at {dest} (--skip-download set)")
                continue

        # Eval
        log(f"  Running eval...")
        acc, output = run_eval(dest)

        results[name] = {
            "accuracy": acc,
            "notes": cfg["notes"],
            "error": None if acc is not None else "eval_failed",
            "timestamp": datetime.now().isoformat(),
        }
        save_results(results)

        if acc is not None:
            log(f"  ✓ Accuracy: {acc:.4f} ({acc*100:.1f}%)")
        else:
            log(f"  ✗ Could not parse accuracy from output")

    # Summary table
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Checkpoint':<30} {'Accuracy':>10}  Notes")
    print("-" * 60)

    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1]["accuracy"] or -1,
        reverse=True,
    )
    for name, r in sorted_results:
        acc_str = f"{r['accuracy']*100:.1f}%" if r["accuracy"] is not None else "FAILED"
        print(f"{name:<30} {acc_str:>10}  {r['notes']}")

    # Clean up tmp yaml
    if EVAL_YAML_TMP.exists():
        EVAL_YAML_TMP.unlink()


if __name__ == "__main__":
    main()

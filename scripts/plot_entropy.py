"""
Experiment B visualization: Entropy-based early halting results.

Produces three plots:
  1. Accuracy vs. Latent Steps Saved (Pareto frontier) — overall + per step count
  2. Avg latent positions used vs. threshold
  3. Accuracy vs. Avg latent positions used, stratified by step count

Run from repo root:
    python plot_halting.py --input halting_results.json --output-dir figures/
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

sns.set_theme(style="darkgrid", palette='Set2', font_scale=1.1)
PALETTE   = sns.color_palette("plasma")
N_LATENT  = 6
BASELINE  = "no_halt"

STEP_COLORS = {
    1: PALETTE[0],
    2: PALETTE[1],
    3: PALETTE[2],
    4: PALETTE[3],
    5: PALETTE[4],
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def parse_thresholds(results: dict) -> list[dict]:
    """
    Returns list of dicts sorted by threshold value (None = baseline first).
    Each dict: threshold, accuracy, avg_latent_used, compute_saved, by_steps
    """
    rows = []
    for label, r in results["thresholds"].items():
        t = r["threshold"]
        rows.append({
            "label":          label,
            "threshold":      t,
            "accuracy":       r["accuracy"],
            "avg_latent":     r["avg_latent_used"],
            "compute_saved":  (N_LATENT - r["avg_latent_used"]) / N_LATENT * 100,
            "by_steps":       r.get("by_steps", {}),
        })
    # Sort: baseline first, then ascending threshold
    rows.sort(key=lambda x: (-1 if x["threshold"] is None else x["threshold"]))
    return rows


# ---------------------------------------------------------------------------
# Plot 1: Overall Pareto frontier (Accuracy vs. Latent Steps Saved)
# ---------------------------------------------------------------------------

def plot_pareto(rows: list[dict], ax: plt.Axes) -> None:
    baseline = next(r for r in rows if r["threshold"] is None)
    sweep    = [r for r in rows if r["threshold"] is not None]

    xs = [r["compute_saved"] for r in sweep]
    ys = [r["accuracy"] * 100 for r in sweep]

    ax.plot(xs, ys, "o-", color=PALETTE[0], linewidth=2,
            markersize=5, label="Entropy halting sweep")

    # Baseline
    ax.axhline(baseline["accuracy"] * 100, color="gray", linestyle="--",
               linewidth=1.5, label=f"No halting ({baseline['accuracy']*100:.1f}%)")

    # Annotate a few interesting points
    for r in sweep:
        if r["compute_saved"] > 10 and r["accuracy"] > baseline["accuracy"] * 0.95:
            ax.annotate(
                f"{r['compute_saved']:.0f}% saved",
                xy=(r["compute_saved"], r["accuracy"] * 100),
                xytext=(5, 5), textcoords="offset points",
                fontsize=8, color=PALETTE[0],
            )
            break  # just annotate the best efficiency point

    ax.set_xlabel("Latent Steps Saved (%)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy–Compute Tradeoff (Pareto Frontier)")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter())


# ---------------------------------------------------------------------------
# Plot 2: Avg latent positions used vs. threshold
# ---------------------------------------------------------------------------

def plot_latent_vs_threshold(rows: list[dict], ax: plt.Axes) -> None:
    sweep = [r for r in rows if r["threshold"] is not None]

    xs = [r["threshold"] for r in sweep]
    ys = [r["avg_latent"] for r in sweep]

    ax.plot(xs, ys, "s-", color=PALETTE[1], linewidth=2, markersize=5)
    ax.axhline(N_LATENT, color="gray", linestyle="--", linewidth=1.5,
               label=f"Max ({N_LATENT} positions)")

    ax.set_xlabel("Entropy Threshold (nats)")
    ax.set_ylabel("Avg Latent Positions Used")
    ax.set_title("Latent Positions Used vs. Threshold")
    ax.set_xscale("log")
    ax.set_ylim(0, N_LATENT + 0.5)
    ax.legend(fontsize=9)


# ---------------------------------------------------------------------------
# Plot 3: Per-step-count Pareto frontier
# ---------------------------------------------------------------------------

def plot_pareto_by_steps(rows: list[dict], ax: plt.Axes) -> None:
    baseline = next(r for r in rows if r["threshold"] is None)
    sweep    = [r for r in rows if r["threshold"] is not None]

    # Collect all step counts present
    step_counts = sorted(
        {int(k) for r in rows for k in r["by_steps"].keys()},
    )
    step_counts = [s for s in step_counts if s in STEP_COLORS]

    for step in step_counts:
        xs, ys = [], []
        for r in sweep:
            entry = r["by_steps"].get(str(step))
            if entry and entry["n_total"] >= 10 and entry["accuracy"] is not None:
                saved = (N_LATENT - entry["avg_latent_used"]) / N_LATENT * 100
                xs.append(saved)
                ys.append(entry["accuracy"] * 100)

        if xs:
            ax.plot(xs, ys, "o-", color=STEP_COLORS[step], linewidth=1.8,
                    markersize=4, label=f"{step} step{'s' if step > 1 else ''}")

            # Baseline point for this step count
            base_entry = baseline["by_steps"].get(str(step))
            if base_entry and base_entry["accuracy"] is not None:
                ax.scatter([0], [base_entry["accuracy"] * 100],
                           color=STEP_COLORS[step], marker="*", s=120, zorder=5)

    ax.set_xlabel("Latent Steps Saved (%)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Per-Step-Count Accuracy–Compute Tradeoff\n(★ = no-halt baseline)")
    ax.legend(title="Gold steps", fontsize=8, title_fontsize=8)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter())


# ---------------------------------------------------------------------------
# Plot 4: Latent positions used vs. gold step count (selected thresholds)
# ---------------------------------------------------------------------------

def plot_latent_by_steps(rows: list[dict], ax: plt.Axes) -> None:
    """
    For a few representative thresholds, show avg latent positions used
    per gold step count. Reveals whether halting is step-count-adaptive.
    """
    baseline = next(r for r in rows if r["threshold"] is None)
    sweep    = [r for r in rows if r["threshold"] is not None]

    # Pick ~4 representative thresholds spread across the sweep
    n = len(sweep)
    indices = [n // 4, n // 2, 3 * n // 4, n - 1] if n >= 4 else list(range(n))
    selected = [sweep[i] for i in sorted(set(indices))]

    step_counts = sorted(
        {int(k) for r in rows for k in r["by_steps"].keys()}
    )

    # Baseline (all steps used)
    base_xs = []
    base_ys = []
    for step in step_counts:
        entry = baseline["by_steps"].get(str(step))
        if entry and entry["n_total"] >= 5:
            base_xs.append(step)
            base_ys.append(entry.get("avg_latent_used") or N_LATENT)
    ax.plot(base_xs, base_ys, "o--", color="gray", linewidth=1.5,
            markersize=5, label="No halting")

    for i, r in enumerate(selected):
        xs, ys = [], []
        for step in step_counts:
            entry = r["by_steps"].get(str(step))
            if entry and entry["n_total"] >= 5 and entry["avg_latent_used"] is not None:
                xs.append(step)
                ys.append(entry["avg_latent_used"])
        if xs:
            t = r["threshold"]
            ax.plot(xs, ys, "o-", color=PALETTE[i + 2], linewidth=1.8,
                    markersize=5, label=f"τ={t:.2f}")

    ax.set_xlabel("Gold Reasoning Steps")
    ax.set_ylabel("Avg Latent Positions Used")
    ax.set_title("Adaptive Compute Allocation by Problem Difficulty")
    ax.set_xticks(step_counts)
    ax.set_ylim(0, N_LATENT + 0.5)
    ax.axhline(N_LATENT, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax.legend(fontsize=8)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",      default="halting_results.json")
    parser.add_argument("--output-dir", default="figures")
    parser.add_argument("--dpi",        type=int, default=150)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = load(args.input)
    rows    = parse_thresholds(results)

    min_steps = results["metadata"].get("min_latent_steps", 2)
    print(f"Loaded {len(rows)} threshold entries "
          f"(min_latent_steps={min_steps})")

    # --- Figure 1: 2×2 overview ---
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    fig.suptitle("Experiment B: Entropy-Based Early Halting — GSM8K",
                 fontsize=13, fontweight="bold")

    plot_pareto(rows, axes[0, 0])
    plot_latent_vs_threshold(rows, axes[0, 1])
    plot_pareto_by_steps(rows, axes[1, 0])
    plot_latent_by_steps(rows, axes[1, 1])

    path = out_dir / "halting_overview.png"
    fig.savefig(path, dpi=args.dpi)
    print(f"Saved: {path}")
    plt.close(fig)

    # --- Figure 2: Large Pareto frontier for paper ---
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    plot_pareto(rows, ax)
    path = out_dir / "halting_pareto.png"
    fig.savefig(path, dpi=args.dpi)
    print(f"Saved: {path}")
    plt.close(fig)

    # --- Figure 3: Per-step-count Pareto for paper ---
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    plot_pareto_by_steps(rows, ax)
    path = out_dir / "halting_pareto_by_steps.png"
    fig.savefig(path, dpi=args.dpi)
    print(f"Saved: {path}")
    plt.close(fig)

    # --- Figure 4: Adaptive allocation for paper ---
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    plot_latent_by_steps(rows, ax)
    path = out_dir / "halting_adaptive_allocation.png"
    fig.savefig(path, dpi=args.dpi)
    print(f"Saved: {path}")
    plt.close(fig)

    # Print summary table
    print("\n" + "=" * 65)
    print(f"{'Threshold':>12}  {'Accuracy':>10}  {'Avg Latent':>11}  {'Saved':>7}")
    print("-" * 65)
    for r in rows:
        t_str = "None" if r["threshold"] is None else f"{r['threshold']:.3f}"
        print(f"{t_str:>12}  {r['accuracy']*100:>9.1f}%  "
              f"{r['avg_latent']:>9.2f}/{N_LATENT}  "
              f"{r['compute_saved']:>6.1f}%")


if __name__ == "__main__":
    main()
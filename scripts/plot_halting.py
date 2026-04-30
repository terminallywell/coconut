"""
Halting experiment visualization.

Produces two figures:
  halting_pareto.png      — overall accuracy vs. latent steps saved
                            (entropy sweep + learned head sweep)
  halting_bysteps.png     — per-step-count Pareto curves
                            (entropy sweep, no-halt baseline squares,
                             overall gray curve)

Usage:
    python scripts/plot_halting.py \
        --entropy  results/halting_results_entropy.json \
        --learned  results/halting_results_learned.json \
        --output-dir figures/
"""

import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid", palette="Set2", font_scale=1.2)

N_LATENT    = 6
STEP_COUNTS = [1, 2, 3, 4, 5]
STEP_N      = {1: 32, 2: 155, 3: 140, 4: 91, 5: 53}

_cmap = plt.colormaps["plasma"].resampled(len(STEP_COUNTS) + 2)
STEP_COLORS = {s: _cmap(i + 1) for i, s in enumerate(STEP_COUNTS)}


def lw(step):
    n = STEP_N.get(step, 1)
    n_max, n_min = STEP_N[2], STEP_N[5]
    lw_min, lw_max = 1.6, 3.2
    return lw_min + (lw_max - lw_min) * (
        (np.log(n) - np.log(n_min)) / (np.log(n_max) - np.log(n_min))
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_entropy(path: str) -> tuple:
    """Returns (entries, no_halt) where entries is sorted by threshold."""
    with open(path) as f:
        d = json.load(f)
    entries = []
    for label, r in d["thresholds"].items():
        if r["threshold"] is None:
            continue
        entries.append({
            "threshold":   r["threshold"],
            "accuracy":    r["accuracy"] * 100,
            "steps_saved": (N_LATENT - r["avg_latent_used"]) / N_LATENT * 100,
            "avg_latent":  r["avg_latent_used"],
            "by_steps":    r.get("by_steps", {}),
        })
    entries.sort(key=lambda x: x["threshold"])
    baseline = d["thresholds"].get("no_halt", {})
    no_halt = {
        "accuracy":   baseline.get("accuracy", 0) * 100,
        "avg_latent": baseline.get("avg_latent_used", N_LATENT),
        "by_steps":   baseline.get("by_steps", {}),
    }
    return entries, no_halt


def load_learned(path: str) -> list:
    """Returns list of entries sorted by threshold."""
    with open(path) as f:
        d = json.load(f)
    entries = []
    for label, r in d["thresholds"].items():
        if r.get("halt_mode") != "learned":
            continue
        t = r.get("threshold")
        if t is None:
            continue
        entries.append({
            "threshold":   t,
            "accuracy":    r["accuracy"] * 100,
            "steps_saved": (N_LATENT - r["avg_latent_used"]) / N_LATENT * 100,
            "avg_latent":  r["avg_latent_used"],
            "by_steps":    r.get("by_steps", {}),
        })
    entries.sort(key=lambda x: x["threshold"])
    return entries


# ---------------------------------------------------------------------------
# Figure 1: Overall Pareto (entropy + learned sweep)
# ---------------------------------------------------------------------------

def plot_pareto(ax, entries, no_halt, learned):
    xs_e = [e["steps_saved"] for e in entries]
    ys_e = [e["accuracy"]    for e in entries]
    ax.plot(xs_e, ys_e, "o-", linewidth=2.2,
            markersize=5, label="Entropy threshold sweep", zorder=2)

    if learned:
        xs_l = [e["steps_saved"] for e in learned]
        ys_l = [e["accuracy"]    for e in learned]
        ax.plot(xs_l, ys_l, "s-", linewidth=2.0,
                markersize=5, label="Learned halting head", zorder=3)

    ax.axhline(no_halt["accuracy"], color="gray", linestyle="--",
               linewidth=1.5, label=f"No halting ({no_halt['accuracy']:.1f}%)", zorder=1)

    ax.set_xlabel("Latent Steps Saved")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy vs. Latent Steps Saved")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.legend(fontsize=10)


# ---------------------------------------------------------------------------
# Figure 2: Per-step Pareto (entropy only, with overall and no-halt squares)
# ---------------------------------------------------------------------------

def plot_by_steps(ax, entries, no_halt):
    legend_handles = []
    legend_labels  = []

    # Overall gray curve
    overall_xs, overall_ys = [], []
    for e in entries:
        overall_xs.append(e["steps_saved"])
        overall_ys.append(e["accuracy"])
    overall_line, = ax.plot(overall_xs, overall_ys, "o-",
                            color="gray", linewidth=1.4,
                            markersize=4, alpha=0.6, zorder=2)

    # No-halt overall baseline square
    ax.scatter([0], [no_halt["accuracy"]], marker="s", s=30,
               color="gray", zorder=5, alpha=0.8)

    # Dotted connector: overall no-halt → first sweep point
    ax.plot([0, overall_xs[0]], [no_halt["accuracy"], overall_ys[0]],
            color="gray", linestyle=":", linewidth=1.2, alpha=0.6, zorder=3)

    legend_handles.append(overall_line)
    legend_labels.append("Overall")

    # Per-step curves
    for step in STEP_COUNTS:
        xs, ys = [], []
        for e in entries:
            by_s = e["by_steps"].get(str(step), {})
            a    = by_s.get("accuracy")
            if a is None or by_s.get("n_total", 0) < 5:
                continue
            saved = (N_LATENT - by_s["avg_latent_used"]) / N_LATENT * 100
            xs.append(saved)
            ys.append(a * 100)

        if not xs:
            continue

        b_by_s = no_halt["by_steps"].get(str(step), {})
        b_acc  = b_by_s.get("accuracy")
        if b_acc is None:
            continue
        b_pct = b_acc * 100

        # No-halt baseline square
        ax.scatter([0], [b_pct], marker="s", s=40,
                   color=STEP_COLORS[step], zorder=5)

        # Dotted connector to first sweep point
        ax.plot([0, xs[0]], [b_pct, ys[0]],
                color=STEP_COLORS[step], linestyle=":",
                linewidth=1.4, zorder=3)

        # Solid sweep line
        line, = ax.plot(xs, ys, "o-",
                        color=STEP_COLORS[step],
                        linewidth=2, markersize=4, zorder=4)
        legend_handles.append(line)
        legend_labels.append(f"{step} step{'s' if step > 1 else ''}")

    ax.set_xlabel("Latent Steps Saved")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy vs. Latent Steps Saved, by Step Count\n"
                 "($\\blacksquare$ = no-halt baseline)")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.legend(handles=legend_handles, labels=legend_labels,
              fontsize=9, title="Gold steps", title_fontsize=9)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entropy",     required=True,
                        help="JSON from entropy threshold sweep")
    parser.add_argument("--learned",     default=None,
                        help="JSON from learned head threshold sweep")
    parser.add_argument("--output-dir",  default="figures")
    parser.add_argument("--dpi",         type=int, default=150)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    entries, no_halt = load_entropy(args.entropy)
    learned          = load_learned(args.learned) if args.learned else []

    print(f"Entropy sweep:    {len(entries)} thresholds")
    print(f"Learned sweep:    {len(learned)} thresholds")
    print(f"No-halt baseline: {no_halt['accuracy']:.1f}%")

    # Figure 1: Overall Pareto
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    plot_pareto(ax, entries, no_halt, learned)
    path = out_dir / "halting_pareto.png"
    fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)

    # Figure 2: Per-step breakdown
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    plot_by_steps(ax, entries, no_halt)
    path = out_dir / "halting_bysteps.png"
    fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()

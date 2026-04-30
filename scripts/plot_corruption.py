"""
Corruption analysis visualization for final report.

Produces four figures:
  1. Single-position accuracy drop by step count (line thickness proportional to frequency)
  2. Progressive forward corruption by step count
  3. Progressive reverse corruption by step count
  4. Intra-step comparison (first vs second thoughts) - bar chart

Run from repo root:
    python plot_corruption.py \
        --input corruption_results_2.json \
        --output-dir figures/
"""

import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid", font_scale=1.15)

STEP_COUNTS = [1, 2, 3, 4, 5]
STEP_N = {1: 32, 2: 155, 3: 140, 4: 91, 5: 53}

# Hue gradient: light to dark across 1 to 5 steps
_cmap = plt.cm.get_cmap("plasma", len(STEP_COUNTS) + 1)
STEP_COLORS = {s: _cmap(i / len(STEP_COUNTS)) for i, s in enumerate(STEP_COUNTS)}

N_LATENT = 6


def load(path):
    with open(path) as f:
        return json.load(f)["experiments"]


def acc(exps, key, step=None):
    e = exps[key]
    if step is None:
        return e["accuracy"]
    entry = e["by_difficulty"].get(str(step))
    if entry is None or entry["n_total"] == 0:
        return None
    return entry["accuracy"]


def lw(step):
    """Line width proportional to log frequency."""
    n = STEP_N.get(step, 1)
    n_max = STEP_N[2]
    n_min = STEP_N[5]
    lw_min, lw_max = 1.6, 3.2
    return lw_min + (lw_max - lw_min) * (np.log(n) - np.log(n_min)) / (np.log(n_max) - np.log(n_min))


def plot_single_position(exps, ax):
    baseline = {s: acc(exps, "clean", s) for s in STEP_COUNTS}
    positions = list(range(N_LATENT))

    for step in STEP_COUNTS:
        drops = []
        for pos in positions:
            a = acc(exps, f"single_{pos}", step)
            b = baseline[step]
            if a is None or b is None or b == 0:
                drops.append(np.nan)
            else:
                drops.append((b - a) * 100)

        ax.plot(positions, drops, "o-",
                color=STEP_COLORS[step],
                linewidth=lw(step),
                markersize=5,
                label=f"{step} step{'s' if step > 1 else ''} ($n$={STEP_N[step]})")

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Corrupted Position")
    ax.set_ylabel("Accuracy Drop from Baseline (pp)")
    ax.set_title("Single-Position Corruption: Accuracy Drop by Step Count")
    ax.set_xticks(positions)
    ax.set_xticklabels([f"pos {p}" for p in positions])
    ax.legend(title="Gold steps\n(width $\\propto$ frequency)",
              fontsize=9, title_fontsize=8, loc="upper left")


def plot_progressive_fwd(exps, ax):
    xs = list(range(N_LATENT + 1))

    for step in STEP_COUNTS:
        ys = [acc(exps, "clean", step) * 100]
        for k in range(1, N_LATENT + 1):
            a = acc(exps, f"progressive_fwd_{k}", step)
            ys.append(a * 100 if a is not None else np.nan)

        ax.plot(xs, ys, "o-",
                color=STEP_COLORS[step],
                linewidth=lw(step),
                markersize=5,
                label=f"{step} step{'s' if step > 1 else ''} ($n$={STEP_N[step]})")

    # Overall dashed curve
    ys_overall = [acc(exps, "clean") * 100]
    for k in range(1, N_LATENT + 1):
        ys_overall.append(acc(exps, f"progressive_fwd_{k}") * 100)
    ax.plot(xs, ys_overall, "k--", linewidth=1.5, alpha=0.5, label="Overall")

    ax.set_xlabel("Number of Positions Corrupted (forward, 0$\\rightarrow k$)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Progressive Forward Corruption by Step Count")
    ax.set_xticks(xs)
    ax.legend(title="Gold steps\n(width $\\propto$ frequency)",
              fontsize=9, title_fontsize=8)


def plot_progressive_rev(exps, ax):
    # progressive_rev_k corrupts positions k..5
    rev_keys = [5, 4, 3, 2, 1, 0]
    xs = list(range(1, len(rev_keys) + 1))
    xlabels = [rf"pos {k}–5" for k in rev_keys]

    for step in STEP_COUNTS:
        ys = []
        for start_pos in rev_keys:
            a = acc(exps, f"progressive_rev_{start_pos}", step)
            ys.append(a * 100 if a is not None else np.nan)

        ax.plot(xs, ys, "o-",
                color=STEP_COLORS[step],
                linewidth=lw(step),
                markersize=5,
                label=f"{step} step{'s' if step > 1 else ''} ($n$={STEP_N[step]})")

    # Dotted baselines
    for step in STEP_COUNTS:
        b = acc(exps, "clean", step)
        if b is not None:
            ax.axhline(b * 100, color=STEP_COLORS[step],
                       linestyle=":", linewidth=0.8, alpha=0.35)

    ax.set_xlabel("Positions Corrupted (reverse, $k\\rightarrow$5)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Progressive Reverse Corruption by Step Count")
    ax.set_xticks(xs)
    ax.set_xticklabels(xlabels, rotation=20, ha="right")
    ax.legend(title="Gold steps\n(width $\\propto$ frequency)",
              fontsize=9, title_fontsize=8)


def plot_intra_step(exps, ax):
    x = np.arange(len(STEP_COUNTS))
    width = 0.25

    def safe(key, step):
        a = acc(exps, key, step)
        return a * 100 if a is not None else 0.0

    clean_accs  = [acc(exps, "clean", s) * 100 for s in STEP_COUNTS]
    first_accs  = [safe("intra_first_thoughts", s)  for s in STEP_COUNTS]
    second_accs = [safe("intra_second_thoughts", s) for s in STEP_COUNTS]

    ax.bar(x - width, clean_accs,  width,
           label="Clean baseline", color="#4C72B0", alpha=0.85)
    ax.bar(x,         first_accs,  width,
           label="First thoughts corrupted (pos 0,2,4)", color="#DD8452", alpha=0.85)
    ax.bar(x + width, second_accs, width,
           label="Second thoughts corrupted (pos 1,3,5)", color="#55A868", alpha=0.85)

    ax.set_xlabel("Gold Reasoning Steps")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Intra-Step Corruption: First vs.\ Second Thoughts")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s} step{'s' if s > 1 else ''}" for s in STEP_COUNTS])
    ax.legend(fontsize=9)


def plot_combined_superplot(exps, output_path, dpi=150):
    """Three-panel superplot with shared y-axis and single shared legend."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True, constrained_layout=True)

    legend_handles = []
    legend_labels  = []

    # Panel (a): Single-position accuracy drop
    ax = axes[0]
    baseline = {s: acc(exps, "clean", s) for s in STEP_COUNTS}
    positions = list(range(N_LATENT))
    for step in STEP_COUNTS:
        drops = []
        for pos in positions:
            a = acc(exps, f"single_{pos}", step)
            b = baseline[step]
            drops.append((b - a) * 100 if (a is not None and b) else np.nan)
        line, = ax.plot(positions, drops, "o-",
                        color=STEP_COLORS[step], linewidth=lw(step), markersize=5)
        legend_handles.append(line)
        legend_labels.append(f"{step} step{'s' if step > 1 else ''} ($n$={STEP_N[step]})")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Corrupted Position")
    ax.set_ylabel("Accuracy Drop from Baseline (pp)")
    ax.set_title("(a) Single-Position")
    ax.set_xticks(positions)
    ax.set_xticklabels([f"pos {p}" for p in positions])

    # Panel (b): Progressive forward
    ax = axes[1]
    xs = list(range(N_LATENT + 1))
    for step in STEP_COUNTS:
        b = acc(exps, "clean", step) * 100
        ys = [b]
        for k in range(1, N_LATENT + 1):
            a = acc(exps, f"progressive_fwd_{k}", step)
            ys.append(a * 100 if a is not None else np.nan)
        drops = [0.0] + [b - y for y in ys[1:]]
        ax.plot(xs, drops, "o-",
                color=STEP_COLORS[step], linewidth=lw(step), markersize=5)
    b0 = acc(exps, "clean") * 100
    ys_all = [b0] + [acc(exps, f"progressive_fwd_{k}") * 100 for k in range(1, N_LATENT + 1)]
    drops_all = [0.0] + [b0 - y for y in ys_all[1:]]
    overall_line, = ax.plot(xs, drops_all, "k--", linewidth=1.5, alpha=0.6)
    legend_handles.append(overall_line)
    legend_labels.append("Overall")
    ax.set_xlabel("Positions Corrupted (forward, 0 to $k$)")
    ax.set_title("(b) Progressive Forward")
    ax.set_xticks(xs)

    # Panel (c): Progressive reverse
    ax = axes[2]
    rev_keys = [5, 4, 3, 2, 1, 0]
    xs_rev = list(range(1, len(rev_keys) + 1))
    for step in STEP_COUNTS:
        b = acc(exps, "clean", step) * 100
        drops = []
        for start_pos in rev_keys:
            a = acc(exps, f"progressive_rev_{start_pos}", step)
            drops.append(b - a * 100 if a is not None else np.nan)
        ax.plot(xs_rev, drops, "o-",
                color=STEP_COLORS[step], linewidth=lw(step), markersize=5)
    ax.set_xlabel("Positions Corrupted (reverse, $k$ to 5)")
    ax.set_title("(c) Progressive Reverse")
    ax.set_xticks(xs_rev)
    ax.set_xticklabels([f"pos {k}–5" for k in rev_keys], rotation=20, ha="right")

    fig.legend(legend_handles, legend_labels,
               loc="lower center", bbox_to_anchor=(0.5, -0.14),
               ncol=len(STEP_COUNTS) + 1,
               title="Gold steps (line width proportional to frequency in dataset)",
               title_fontsize=14, fontsize=14, frameon=True)
    fig.suptitle("Corruption Analysis: Accuracy Drop by Latent Position",
                 fontsize=13, fontweight="bold")
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",      default="corruption_results_2.json")
    parser.add_argument("--output-dir", default="figures")
    parser.add_argument("--dpi",        type=int, default=150)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    exps = load(args.input)

    # Main figure: combined superplot
    plot_combined_superplot(exps, out_dir / "corruption_superplot.png", dpi=args.dpi)

    # Supplementary: individual figures
    for fn, fname in [
        (plot_single_position,  "corruption_single_pos.png"),
        (plot_progressive_fwd,  "corruption_progressive_fwd.png"),
        (plot_progressive_rev,  "corruption_progressive_rev.png"),
        (plot_intra_step,       "corruption_intra_step.png"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
        fn(exps, ax)
        fig.savefig(out_dir / fname, dpi=args.dpi, bbox_inches="tight")
        print(f"Saved: {out_dir / fname}")
        plt.close(fig)

    # 2x2 overview
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), constrained_layout=True)
    fig.suptitle("Corruption Analysis -- GSM8K Latent Positions",
                 fontsize=14, fontweight="bold")
    plot_single_position(exps, axes[0, 0])
    plot_progressive_fwd(exps, axes[0, 1])
    plot_progressive_rev(exps, axes[1, 0])
    plot_intra_step(exps,      axes[1, 1])
    fig.savefig(out_dir / "corruption_overview.png", dpi=args.dpi, bbox_inches="tight")
    print(f"Saved: {out_dir / 'corruption_overview.png'}")
    plt.close(fig)


if __name__ == "__main__":
    main()

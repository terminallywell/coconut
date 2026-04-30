"""
Corruption analysis visualization for final report.

Produces a two-panel superplot:
  (a) Single-position accuracy drop by step count
  (b) Cumulative reverse corruption by step count

Run from repo root:
    python scripts/plot_corruption.py \
        --input results/corruption_results.json \
        --output-dir figures/
"""

import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid", font_scale=1.2)

STEP_COUNTS = [1, 2, 3, 4, 5]
STEP_N      = {1: 32, 2: 155, 3: 140, 4: 91, 5: 53}

_cmap = plt.colormaps["plasma"].resampled(len(STEP_COUNTS) + 2)
STEP_COLORS = {s: _cmap(i + 1) for i, s in enumerate(STEP_COUNTS)}

N_LATENT = 6


def load(path):
    with open(path) as f:
        return json.load(f)["experiments"]


def acc(exps, key, step=None):
    e = exps.get(key)
    if e is None:
        return None
    if step is None:
        return e["accuracy"]
    entry = e["by_difficulty"].get(str(step))
    if entry is None or entry["n_total"] == 0:
        return None
    return entry["accuracy"]


def lw(step):
    """Line width proportional to log frequency."""
    n = STEP_N.get(step, 1)
    n_max, n_min = STEP_N[2], STEP_N[5]
    lw_min, lw_max = 1.6, 3.2
    return lw_min + (lw_max - lw_min) * (
        (np.log(n) - np.log(n_min)) / (np.log(n_max) - np.log(n_min))
    )


def plot_single_position(exps, ax):
    """
    Panel (a): accuracy drop when corrupting each position individually.
    X-axis: position (1-indexed). Y-axis: accuracy drop from baseline (pp).
    """
    baseline  = {s: acc(exps, "baseline", s) for s in STEP_COUNTS}
    positions = list(range(N_LATENT))           # 0-indexed internally
    xlabels   = [str(p + 1) for p in positions]  # 1-indexed for display

    legend_handles = []
    legend_labels  = []

    for step in STEP_COUNTS:
        drops = []
        for pos in positions:
            a = acc(exps, f"single_{pos}", step)
            b = baseline[step]
            drops.append((b - a) * 100 if (a is not None and b) else np.nan)
        line, = ax.plot(positions, drops, "o-",
                        color=STEP_COLORS[step],
                        linewidth=lw(step), markersize=5)
        legend_handles.append(line)
        legend_labels.append(f"{step} step{'s' if step > 1 else ''} ($n$={STEP_N[step]})")

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Corrupted Position", fontsize=12)
    ax.set_ylabel("Accuracy Drop from Baseline (pp)", fontsize=12)
    ax.set_title("(a) Single-Position Corruption")
    ax.set_xticks(positions)
    ax.set_xticklabels(xlabels)
    return legend_handles, legend_labels


def plot_cumulative_rev(exps, ax):
    """
    Panel (b): accuracy drop when cumulatively corrupting from the right.
    Each point corrupts positions k..6 (1-indexed), increasing coverage leftward.
    X-axis: starting position of corruption (1-indexed). Y-axis: accuracy drop (pp).
    """
    # rev_keys are 0-indexed start positions, ordered right-to-left
    rev_keys = list(range(N_LATENT - 1, -1, -1))  # [5,4,3,2,1,0]
    xs       = list(range(len(rev_keys)))
    xlabels  = [f"{k+1}–{N_LATENT}" for k in rev_keys]

    for step in STEP_COUNTS:
        b = acc(exps, "baseline", step)
        if b is None:
            continue
        drops = []
        for start_pos in rev_keys:
            a = acc(exps, f"cumulative_rev_{start_pos}", step)
            drops.append((b - a) * 100 if a is not None else np.nan)
        ax.plot(xs, drops, "o-",
                color=STEP_COLORS[step],
                linewidth=lw(step), markersize=5)

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Corrupted Positions (start–end)", fontsize=12)
    ax.set_ylabel("Accuracy Drop from Baseline (pp)", fontsize=12)
    ax.set_title("(b) Reverse Cumulative Corruption")
    ax.set_xticks(xs)
    ax.set_xticklabels(xlabels, rotation=20, ha="right")


def plot_superplot(exps, output_path, dpi=150):
    """Two-panel superplot with shared y-axis and single shared legend."""
    fig, axes = plt.subplots(2, 1, figsize=(7, 10),
                            sharey=True, constrained_layout=True)

    legend_handles, legend_labels = plot_single_position(exps, axes[0])
    plot_cumulative_rev(exps, axes[1])

    for ax in axes:
        ax.tick_params(left=True, labelleft=True)

    fig.legend(legend_handles, legend_labels,
            loc="lower center", bbox_to_anchor=(0.5, -0.08),
            ncol=3,  # fewer columns fit better below a narrow figure
            title="Gold steps (line width proportional to frequency in dataset)",
            title_fontsize=9, fontsize=9, frameon=True)
    fig.suptitle("Corruption Analysis: Accuracy Drop by Latent Position",
                 fontsize=13, fontweight="bold")
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",      default="results/corruption_results.json")
    parser.add_argument("--output-dir", default="figures")
    parser.add_argument("--dpi",        type=int, default=150)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    exps = load(args.input)

    # Main figure: two-panel superplot
    plot_superplot(exps, out_dir / "corruption_superplot.png", dpi=args.dpi)

    # Individual figures
    for fn, fname in [
        (plot_single_position, "corruption_single_pos.png"),
        (plot_cumulative_rev,  "corruption_cumulative_rev.png"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
        if fn == plot_single_position:
            handles, labels = fn(exps, ax)
            ax.legend(handles, labels,
                      title="Gold steps (width $\\propto$ frequency)",
                      fontsize=9, title_fontsize=9)
        else:
            fn(exps, ax)
            # add legend for standalone figure
            handles = [plt.Line2D([0], [0], color=STEP_COLORS[s],
                                  linewidth=lw(s), marker="o", markersize=5)
                       for s in STEP_COUNTS]
            labels  = [f"{s} step{'s' if s > 1 else ''} ($n$={STEP_N[s]})"
                       for s in STEP_COUNTS]
            ax.legend(handles, labels,
                      title="Gold steps (width $\\propto$ frequency)",
                      fontsize=9, title_fontsize=9)
        fig.savefig(out_dir / fname, dpi=args.dpi, bbox_inches="tight")
        print(f"Saved: {out_dir / fname}")
        plt.close(fig)


if __name__ == "__main__":
    main()

"""
Halting experiment visualization for final report.

Two-panel figure:
  Left:  Overall accuracy vs. latent steps saved (entropy sweep + learned head star)
  Right: Avg latent steps used by gold step count (selected thresholds + learned head)

Usage:
    python plot_halting.py \
        --entropy halting_results.json \
        --learned halting_results_learned.json \
        --output-dir figures/
"""

import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns

sns.set_theme(style="darkgrid", font_scale=1.2)

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


def load_entropy(path):
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


def load_learned_sweep(path):
    """Load a learned head threshold sweep JSON — returns list of (steps_saved, accuracy) entries."""
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


def plot_overall(ax, entries, no_halt, learned_sp, learned_no_sp):
    xs = [e["steps_saved"] for e in entries]
    ys = [e["accuracy"]    for e in entries]

    # Entropy curve
    ax.plot(xs, ys, "o-", color="#5B8DB8", linewidth=2.2,
            markersize=5, label="Entropy threshold sweep", zorder=2)

    # No-halt baseline
    ax.axhline(no_halt["accuracy"], color="gray", linestyle="--",
               linewidth=1.5, label=f"No halting ({no_halt['accuracy']:.1f}%)", zorder=1)

    # Learned head with sparsity penalty
    if learned_sp:
        xs_sp = [e["steps_saved"] for e in learned_sp]
        ys_sp = [e["accuracy"]    for e in learned_sp]
        ax.plot(xs_sp, ys_sp, "s-", color="#E8433A", linewidth=2.0,
                markersize=5, label="Learned head (with sparsity penalty)", zorder=4)

    # Learned head without sparsity penalty
    if learned_no_sp:
        xs_no = [e["steps_saved"] for e in learned_no_sp]
        ys_no = [e["accuracy"]    for e in learned_no_sp]
        ax.plot(xs_no, ys_no, "^-", color="#2BAE8E", linewidth=2.0,
                markersize=5, label="Learned head (no sparsity penalty)", zorder=3)

    ax.set_xlabel("Latent Steps Saved (%)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("(a) Accuracy vs. Latent Steps Saved")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.legend(fontsize=9)


def plot_by_steps(ax, entries, no_halt, learned_sp, learned_no_sp):
    """
    Per-step Pareto curves: accuracy vs. latent steps saved, one line per
    gold step count. No-halt baseline plotted as a star at 0% saved,
    connected to the first entropy threshold point with a dotted segment.
    Learned head shown as a diamond at its actual steps-saved value.
    """
    legend_handles = []
    legend_labels  = []

    for step in STEP_COUNTS:
        # --- collect entropy sweep points for this step count ---
        xs, ys = [], []
        for e in entries:
            by_s = e["by_steps"].get(str(step), {})
            acc  = by_s.get("accuracy")
            if acc is None or by_s.get("n_total", 0) < 5:
                continue
            saved = (N_LATENT - by_s["avg_latent_used"]) / N_LATENT * 100
            xs.append(saved)
            ys.append(acc * 100)

        if not xs:
            continue

        # no-halt baseline at 0% saved
        b_by_s  = no_halt["by_steps"].get(str(step), {})
        b_acc   = b_by_s.get("accuracy")
        if b_acc is None:
            continue
        b_acc_pct = b_acc * 100

        # star at (0, baseline_acc)
        ax.scatter([0], [b_acc_pct], marker="*", s=150,
                   color=STEP_COLORS[step], zorder=5)

        # dotted segment: baseline star → first sweep point
        ax.plot([0, xs[0]], [b_acc_pct, ys[0]],
                color=STEP_COLORS[step], linestyle=":",
                linewidth=lw(step) * 0.8, alpha=0.7, zorder=3)

        # solid sweep line
        line, = ax.plot(xs, ys, "o-",
                        color=STEP_COLORS[step],
                        linewidth=lw(step), markersize=4, zorder=2)

        legend_handles.append(line)
        legend_labels.append(f"{step} step{'s' if step > 1 else ''}")

    # marker legend entries
    legend_handles += []
    legend_labels  += []

    ax.set_xlabel("Latent Steps Saved (%)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("(b) Per-Step-Count Accuracy--Latent Steps Tradeoff\n"
                 "($\\bigstar$ = no-halt baseline, \u25c6 = learned head)")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.legend(handles=legend_handles, labels=legend_labels,
              fontsize=9, title="Gold steps",
              title_fontsize=9)


def plot_adaptive_allocation(ax, entries, no_halt, learned_sp, learned_no_sp):
    """
    Panel (c): Avg latent steps used vs. gold step count.
    Shows learned head vs. two representative entropy thresholds vs. no-halt ceiling.
    """
    steps = STEP_COUNTS

    # No-halt ceiling
    ax.axhline(N_LATENT, color="gray", linestyle=":", linewidth=1.0,
               alpha=0.6, label="No halting (6 steps)")

    # Two representative entropy thresholds
    thresh_styles = [
        ("thresh_0.1",  "#5B8DB8", "--", r"Entropy $\tau{=}0.10$"),
        ("thresh_0.24", "#85C1E9", "-.", r"Entropy $\tau{=}0.24$"),
    ]
    for key, color, ls, label in thresh_styles:
        entry = next((e for e in entries if abs(e["threshold"] - float(key.split("_")[1])) < 0.001), None)
        if entry is None:
            continue
        ys = []
        for s in steps:
            by_s = entry["by_steps"].get(str(s), {})
            avg  = by_s.get("avg_latent_used")
            ys.append(avg if avg is not None else np.nan)
        ax.plot(steps, ys, ls, color=color, linewidth=2.0,
                markersize=5, marker="o", label=label)

    # Learned head — pick midpoint threshold from each sweep for comparison
    for sweep, color, marker, label in [
        (learned_sp,    "#E8433A", "D", "Learned head (SP)"),
        (learned_no_sp, "#2BAE8E", "^", "Learned head (no SP)"),
    ]:
        if not sweep:
            continue
        # use midpoint threshold
        mid = sweep[len(sweep) // 2]
        ys = []
        for s in steps:
            by_s = mid["by_steps"].get(str(s), {})
            avg  = by_s.get("avg_latent_used")
            ys.append(avg if avg is not None else np.nan)
        ax.plot(steps, ys, "-", color=color, linewidth=2.5,
                markersize=7, marker=marker, label=f"{label} (τ={mid['threshold']:.2f})",
                markeredgecolor="black", markeredgewidth=0.5, zorder=5)

    ax.set_xlabel("Gold Reasoning Steps")
    ax.set_ylabel("Avg Latent Steps Used")
    ax.set_title("(c) Adaptive Allocation by Problem Difficulty")
    ax.set_xticks(steps)
    ax.set_ylim(0, N_LATENT + 0.5)
    ax.legend(fontsize=9)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entropy",       required=True,
                        help="JSON from entropy threshold sweep")
    parser.add_argument("--learned-sp",    default=None,
                        help="JSON from learned head sweep (with sparsity penalty)")
    parser.add_argument("--learned-no-sp", default=None,
                        help="JSON from learned head sweep (no sparsity penalty)")
    parser.add_argument("--output-dir",    default="figures")
    parser.add_argument("--dpi",           type=int, default=150)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    entries, no_halt = load_entropy(args.entropy)
    learned_sp    = load_learned_sweep(args.learned_sp)    if args.learned_sp    else []
    learned_no_sp = load_learned_sweep(args.learned_no_sp) if args.learned_no_sp else []

    print(f"Entropy sweep:         {len(entries)} thresholds")
    print(f"Learned (SP) sweep:    {len(learned_sp)} thresholds")
    print(f"Learned (no SP) sweep: {len(learned_no_sp)} thresholds")
    print(f"No-halt baseline:      {no_halt['accuracy']:.1f}%")

    # Two-panel figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    fig.suptitle("Entropy-Based and Learned Halting: Accuracy--Efficiency Tradeoff",
                 fontsize=13, fontweight="bold")
    plot_overall(axes[0], entries, no_halt, learned_sp, learned_no_sp)
    plot_by_steps(axes[1], entries, no_halt, learned_sp, learned_no_sp)
    path = out_dir / "halting_twopanel.png"
    fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)

    # Individual figures
    for fn, fname in [
        (lambda ax: plot_overall(ax, entries, no_halt, learned_sp, learned_no_sp),
         "halting_overall.png"),
        (lambda ax: plot_by_steps(ax, entries, no_halt, learned_sp, learned_no_sp),
         "halting_by_steps.png"),
        (lambda ax: plot_adaptive_allocation(ax, entries, no_halt, learned_sp, learned_no_sp),
         "halting_adaptive.png"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
        fn(ax)
        fig.savefig(out_dir / fname, dpi=args.dpi, bbox_inches="tight")
        print(f"Saved: {out_dir / fname}")
        plt.close(fig)


if __name__ == "__main__":
    main()
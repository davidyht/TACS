"""Camera-ready plots for cv_cifar10_noisy results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Color/marker scheme; TACS gets the highlight.
STYLE = {
    "Random":         dict(color="#888888", marker="o", linestyle=":",  zorder=1),
    "EmbedRetrieval": dict(color="#a6cee3", marker="s", linestyle="--", zorder=2),
    "EL2N":           dict(color="#1f78b4", marker="v", linestyle="--", zorder=2),
    "GraNd":          dict(color="#33a02c", marker="^", linestyle="--", zorder=2),
    "LESS":           dict(color="#e31a1c", marker="D", linestyle="-",  zorder=3),
    "TACS":           dict(color="#000000", marker="*", linestyle="-",  zorder=4, linewidth=2.4, markersize=10),
}


def plot_acc_vs_budget(summary, out_path):
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    methods = [m for m in STYLE if m in summary]
    for m in methods:
        per_k = summary[m]["k"]
        ks = sorted(int(k) for k in per_k)
        means = [per_k[str(k)]["acc_mean"] for k in ks]
        stds = [per_k[str(k)]["acc_std"] for k in ks]
        s = STYLE[m]
        ax.errorbar(ks, means, yerr=stds, label=m, capsize=3, **s)
    ax.set_xlabel("Selected subset size (k)")
    ax.set_ylabel("Target binary test accuracy (%)")
    ax.set_title("CIFAR-10 noisy-pool selection (cats vs dogs, 40% label noise)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9, ncol=2, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"wrote {out_path} (+ .pdf)")


def plot_subset_diagnostic(summary, out_path, target_k=None):
    """Side-by-side bars: selected-subset clean-label fraction and target-class fraction.
    Defaults to the smallest k available (where method differences are sharpest)."""
    methods = [m for m in STYLE if m in summary]
    ks = sorted(int(k) for k in summary[methods[0]]["k"])
    chosen_k = target_k if target_k is not None else ks[1] if len(ks) >= 2 else ks[0]
    largest_k = chosen_k
    cleans = [summary[m]["k"][str(largest_k)]["clean_frac_mean"] for m in methods]
    tgts = [summary[m]["k"][str(largest_k)]["target_frac_mean"] for m in methods]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
    x = np.arange(len(methods))
    colors = [STYLE[m]["color"] for m in methods]
    edge = ["black" if m == "TACS" else "none" for m in methods]
    lw = [2 if m == "TACS" else 0 for m in methods]

    axes[0].bar(x, cleans, color=colors, edgecolor=edge, linewidth=lw)
    axes[0].axhline(0.6, color="gray", linestyle=":", linewidth=1, label="pool clean rate")
    axes[0].set_xticks(x); axes[0].set_xticklabels(methods, rotation=30, ha="right")
    axes[0].set_ylabel(f"Clean-label fraction (k={largest_k})")
    axes[0].set_title("Subset purity")
    axes[0].set_ylim(0, 1.05); axes[0].legend(fontsize=8)

    axes[1].bar(x, tgts, color=colors, edgecolor=edge, linewidth=lw)
    axes[1].axhline(0.2, color="gray", linestyle=":", linewidth=1, label="pool target rate")
    axes[1].set_xticks(x); axes[1].set_xticklabels(methods, rotation=30, ha="right")
    axes[1].set_ylabel(f"Target-class fraction (k={largest_k})")
    axes[1].set_title("Target alignment")
    axes[1].set_ylim(0, 1.05); axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"wrote {out_path} (+ .pdf)")


def plot_improvement_over_random(summary, out_path):
    if "Random" not in summary: return
    methods = [m for m in STYLE if m in summary and m != "Random"]
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    rand = summary["Random"]["k"]
    for m in methods:
        per_k = summary[m]["k"]
        ks = sorted(int(k) for k in per_k if k in rand)
        diffs = [per_k[str(k)]["acc_mean"] - rand[str(k)]["acc_mean"] for k in ks]
        ax.plot(ks, diffs, label=m, **{k: v for k, v in STYLE[m].items() if k != "zorder"})
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xlabel("Selected subset size (k)")
    ax.set_ylabel("Δ accuracy over Random (pp)")
    ax.set_title("Improvement vs. Random baseline")
    ax.grid(alpha=0.3); ax.legend(fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"wrote {out_path} (+ .pdf)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    data = json.loads(Path(args.results).read_text())
    summary = data["summary"]
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)

    plot_acc_vs_budget(summary, out / "acc_vs_budget.png")
    plot_subset_diagnostic(summary, out / "subset_purity_target.png")
    plot_improvement_over_random(summary, out / "improvement_over_random.png")

    # text summary table
    lines = ["# CIFAR-10 noisy-pool selection — summary\n"]
    methods = [m for m in STYLE if m in summary]
    ks = sorted(int(k) for k in summary[methods[0]]["k"])
    header = "| method | " + " | ".join(f"k={k}" for k in ks) + " |"
    sep = "|---" * (len(ks) + 1) + "|"
    lines += [header, sep]
    for m in methods:
        cells = []
        for k in ks:
            r = summary[m]["k"][str(k)]
            cells.append(f"{r['acc_mean']:.2f}±{r['acc_std']:.2f}")
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out/'summary.md'}")


if __name__ == "__main__":
    main()

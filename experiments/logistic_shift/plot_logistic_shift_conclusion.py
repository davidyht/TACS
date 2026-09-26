#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METHOD_ORDER = ["Random", "GradNorm", "ToV_1step", "LESS_tracein", "TACS"]
METHOD_STYLE = {
    "TACS": {"label": "TACS", "color": "#c62828"},
    "LESS_tracein": {"label": "LESS / TraceIn", "color": "#1f77b4"},
    "ToV_1step": {"label": "ToV (1-step)", "color": "#2ca02c"},
    "GradNorm": {"label": "GradNorm", "color": "#7f7f7f"},
    "Random": {"label": "Random", "color": "#bdbdbd"},
}
PANEL_LABELS = ["A", "B", "C", "D"]


def style_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_results(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_curve(summary: dict, method: str, ks: list[int], metric_group: str, metric: str) -> tuple[np.ndarray, np.ndarray]:
    means = []
    stds = []
    for k in ks:
        row = summary[method][metric_group]["k"][str(k)]
        means.append(float(row[f"{metric}_mean"]))
        stds.append(float(row[f"{metric}_std"]))
    return np.asarray(means, dtype=np.float64), np.asarray(stds, dtype=np.float64)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.18,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
        ha="left",
    )


def finalize_axis(ax: plt.Axes) -> None:
    ax.grid(True, axis="y", linestyle="--", linewidth=0.8, alpha=0.25)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot paper-style logistic-shift conclusion figures.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().with_name("logistic_shift_framework_full.json"),
        help="Full logistic-shift result JSON.",
    )
    parser.add_argument(
        "--out-prefix",
        type=Path,
        default=Path(__file__).resolve().with_name("plots").joinpath("logistic_shift_conclusion_full"),
        help="Output path prefix without extension.",
    )
    args = parser.parse_args()

    style_matplotlib()
    data = load_results(args.input.resolve())
    summary = data["summary"]
    ks = [int(k) for k in data["config"]["k_values"]]
    main_k = ks[-1]

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.4))
    ax_perf, ax_purity, ax_auc, ax_rho = axes.reshape(-1)

    legend_handles = []
    legend_labels = []

    for method in METHOD_ORDER:
        style = METHOD_STYLE[method]
        color = style["color"]
        label = style["label"]

        acc_mean, acc_std = extract_curve(summary, method, ks, "performance", "target_test_acc")
        purity_mean, purity_std = extract_curve(summary, method, ks, "mechanism", "purity")

        line_perf = ax_perf.plot(
            ks,
            acc_mean,
            color=color,
            linewidth=2.6 if method == "TACS" else 2.0,
            marker="o",
            markersize=6.2 if method == "TACS" else 5.0,
            markerfacecolor="white" if method != "TACS" else color,
            markeredgewidth=1.4,
            label=label,
        )[0]
        ax_perf.fill_between(ks, acc_mean - acc_std, acc_mean + acc_std, color=color, alpha=0.12)

        ax_purity.plot(
            ks,
            purity_mean,
            color=color,
            linewidth=2.6 if method == "TACS" else 2.0,
            marker="o",
            markersize=6.2 if method == "TACS" else 5.0,
            markerfacecolor="white" if method != "TACS" else color,
            markeredgewidth=1.4,
        )
        ax_purity.fill_between(ks, purity_mean - purity_std, purity_mean + purity_std, color=color, alpha=0.12)

        legend_handles.append(line_perf)
        legend_labels.append(label)

    ax_perf.set_title("Target Test Accuracy vs Selection Budget")
    ax_perf.set_xlabel("Selected subset size k")
    ax_perf.set_ylabel("Target test accuracy")
    ax_perf.set_xticks(ks)
    ax_perf.set_ylim(0.1, 0.98)
    finalize_axis(ax_perf)
    add_panel_label(ax_perf, PANEL_LABELS[0])

    ax_purity.set_title("Target-Source Precision vs Selection Budget")
    ax_purity.set_xlabel("Selected subset size k")
    ax_purity.set_ylabel("Precision of target-pool samples")
    ax_purity.set_xticks(ks)
    ax_purity.set_ylim(0.0, 0.5)
    finalize_axis(ax_purity)
    add_panel_label(ax_purity, PANEL_LABELS[1])

    methods = METHOD_ORDER
    x = np.arange(len(methods))
    auc_means = [float(summary[m]["mechanism"]["auc_target_vs_other_mean"]) for m in methods]
    auc_stds = [float(summary[m]["mechanism"]["auc_target_vs_other_std"]) for m in methods]
    rho_means = [float(summary[m]["mechanism"]["spearman_oracle_horizon_mean"]) for m in methods]
    rho_stds = [float(summary[m]["mechanism"]["spearman_oracle_horizon_std"]) for m in methods]
    colors = [METHOD_STYLE[m]["color"] for m in methods]
    labels = [METHOD_STYLE[m]["label"] for m in methods]

    ax_auc.bar(
        x,
        auc_means,
        yerr=auc_stds,
        color=colors,
        edgecolor="black",
        linewidth=0.7,
        capsize=3.0,
        alpha=0.95,
    )
    ax_auc.axhline(0.5, color="#444444", linestyle="--", linewidth=1.0, alpha=0.6)
    ax_auc.set_title("Structural Recovery (Target-vs-Other AUC)")
    ax_auc.set_ylabel("AUC")
    ax_auc.set_xticks(x)
    ax_auc.set_xticklabels(labels, rotation=18, ha="right")
    ax_auc.set_ylim(0.4, 0.84)
    finalize_axis(ax_auc)
    add_panel_label(ax_auc, PANEL_LABELS[2])

    ax_rho.bar(
        x,
        rho_means,
        yerr=rho_stds,
        color=colors,
        edgecolor="black",
        linewidth=0.7,
        capsize=3.0,
        alpha=0.95,
    )
    ax_rho.axhline(0.0, color="#444444", linestyle="--", linewidth=1.0, alpha=0.6)
    ax_rho.set_title("Attribution Fidelity (Spearman vs Horizon Oracle)")
    ax_rho.set_ylabel("Spearman rank correlation")
    ax_rho.set_xticks(x)
    ax_rho.set_xticklabels(labels, rotation=18, ha="right")
    ax_rho.set_ylim(-0.45, 0.92)
    finalize_axis(ax_rho)
    add_panel_label(ax_rho, PANEL_LABELS[3])

    fig.suptitle(
        f"Logistic Shift ({len(data['config']['seeds'])} seeds): TACS wins downstream and isolates target-support data",
        fontsize=14,
        fontweight="bold",
        y=0.985,
    )
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=5,
        frameon=False,
        columnspacing=1.2,
        handletextpad=0.6,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))

    out_prefix = args.out_prefix.resolve()
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = out_prefix.with_suffix(".png")
    pdf_path = out_prefix.with_suffix(".pdf")
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"main_k={main_k}")
    print(f"png={png_path}")
    print(f"pdf={pdf_path}")


if __name__ == "__main__":
    main()

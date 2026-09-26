#!/usr/bin/env python3
"""Render the clean rank, depth, and anchor ablations from T17/T5/T4 receipts.

The legacy Figure 4 scripts contain recovered or visually reconstructed values.
This replacement accepts only completed, receipt-backed analyses and writes the
same three panel filenames used by the paper, plus a compact JSON plotting
receipt that hashes every source.  No result is embedded in this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/less-clean-ablation-mpl")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


TASKS = ("tydiqa", "mmlu", "bbh")
TASK_LABEL = {"tydiqa": "TyDiQA", "mmlu": "MMLU", "bbh": "BBH"}
COLORS = {"tydiqa": "#0072B2", "mmlu": "#D55E00", "bbh": "#6E6E6E"}
TACS_COLOR = "#0072B2"
FIRST_COLOR = "#D55E00"
GRID = "#DDDDDD"
SIZE = (2.3, 2.45)


class PlotError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_receipt(path: Path, expected_id: str, rule: str) -> dict:
    if not path.is_file():
        raise PlotError(f"missing {expected_id} receipt: {path}")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if receipt.get("id") != expected_id or receipt.get("status") != "complete":
        raise PlotError(f"{path} is not a completed {expected_id} receipt")
    if rule not in receipt.get("rules", []) or rule not in receipt.get("summary_points", {}):
        raise PlotError(f"{path} has no {rule} summary")
    return receipt


def point(row: dict, context: str) -> tuple[float, float]:
    try:
        mean, std = float(row["mean"]), float(row["sample_std"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PlotError(f"invalid summary row for {context}") from exc
    if not np.isfinite(mean) or not np.isfinite(std) or std < 0:
        raise PlotError(f"non-finite/negative summary row for {context}")
    return mean, std


def rank_data(receipt: dict, rule: str) -> dict:
    ranks = [int(value) for value in receipt.get("ranks", [])]
    if not ranks or ranks != sorted(set(ranks)):
        raise PlotError("T17 ranks are missing, duplicated, or unordered")
    out = {"ranks": ranks, "tasks": {}}
    for task in TASKS:
        rows = receipt["summary_points"][rule].get(task, {})
        values = [point(rows.get(str(rank), {}), f"T17/{task}/rank{rank}") for rank in ranks]
        out["tasks"][task] = {
            "mean": [value[0] for value in values],
            "sample_std": [value[1] for value in values],
        }
    return out


def depth_data(receipt: dict, rule: str) -> dict:
    task_depths = receipt.get("task_depths", {})
    calibrated = receipt.get("calibrated_depth", {})
    out = {"tasks": {}}
    for task in TASKS:
        depths = [int(value) for value in task_depths.get(task, [])]
        if not depths or depths != sorted(set(depths)):
            raise PlotError(f"T5 depths are missing, duplicated, or unordered for {task}")
        if int(calibrated.get(task, -1)) not in depths:
            raise PlotError(f"T5 calibrated depth is absent for {task}")
        rows = receipt["summary_points"][rule].get(task, {})
        values = [point(rows.get(str(depth), {}), f"T5/{task}/depth{depth}") for depth in depths]
        out["tasks"][task] = {
            "depths": depths,
            "calibrated_depth": int(calibrated[task]),
            "mean": [value[0] for value in values],
            "sample_std": [value[1] for value in values],
        }
    return out


def anchor_data(receipt: dict, rule: str) -> dict:
    ranks = [int(value) for value in receipt.get("ranks", [])]
    if not ranks or ranks != sorted(set(ranks)):
        raise PlotError("T4 ranks are missing, duplicated, or unordered")
    rows = receipt["summary_points"][rule]
    out = {"ranks": ranks, "arms": {"tacs": {}, "first_update": {}}}
    for arm in out["arms"]:
        values = [
            point(rows.get(str(rank), {}).get(arm, {}), f"T4/rank{rank}/{arm}")
            for rank in ranks
        ]
        out["arms"][arm] = {
            "mean": [value[0] for value in values],
            "sample_std": [value[1] for value in values],
        }
    return out


def style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": None,
    })


def stacked_panel(data: dict, x_key: str, xlabel: str, out: Path, *,
                  highlighted_depths: dict[str, int] | None = None,
                  max_x: int | None = None) -> None:
    fig, axes = plt.subplots(3, 1, figsize=SIZE, sharex=False)
    fig.subplots_adjust(left=.24, right=.97, bottom=.18, top=.98, hspace=.28)
    common_x = sorted(set(value for task in TASKS for value in
                          data["tasks"][task].get(x_key, data.get(x_key, []))
                          if max_x is None or value <= max_x))
    for ax, task in zip(axes, TASKS):
        row = data["tasks"][task]
        xs = row.get(x_key, data.get(x_key))
        if xs is None:
            raise PlotError(f"plot data lacks {x_key} for {task}")
        keep = [index for index, value in enumerate(xs) if max_x is None or value <= max_x]
        xs = [xs[index] for index in keep]
        positions = np.asarray([common_x.index(value) for value in xs])
        mean = np.asarray([row["mean"][index] for index in keep])
        std = np.asarray([row["sample_std"][index] for index in keep])
        ax.plot(positions, mean, color=COLORS[task], lw=1.3, marker="o", ms=3, zorder=3)
        ax.fill_between(positions, mean - std, mean + std, color=COLORS[task], alpha=.15, lw=0)
        highlighted = (highlighted_depths or {}).get(task)
        if highlighted is not None and highlighted in xs:
            index = xs.index(highlighted)
            ax.scatter([positions[index]], [mean[index]], marker="*", s=90, color=COLORS[task],
                       edgecolor="#202020", linewidth=.65, zorder=5)
        span = float((mean + std).max() - (mean - std).min())
        pad = max(.12, .18 * span)
        ax.set_ylim(float((mean - std).min() - pad), float((mean + std).max() + pad))
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
        ax.grid(axis="y", color=GRID, lw=.5, zorder=0)
        ax.tick_params(length=2.5, width=.6, pad=2)
        ax.text(.02, .96, TASK_LABEL[task], transform=ax.transAxes, ha="left", va="top",
                fontsize=8.5, color=COLORS[task])
        ax.set_xlim(-.35, len(common_x) - .65)
        ax.set_xticks(np.arange(len(common_x)))
        ax.set_xticklabels([] if ax is not axes[-1] else [str(value) for value in common_x])
    axes[-1].set_xlabel(xlabel, labelpad=3)
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=300)
    plt.close(fig)


def anchor_panel(data: dict, out: Path) -> None:
    ranks = data["ranks"]
    x = np.arange(len(ranks))
    width = .24
    fig, ax = plt.subplots(figsize=SIZE)
    fig.subplots_adjust(left=.24, right=.97, bottom=.18, top=.80)
    for offset, arm, label, color, hatch in (
        (-width / 2, "first_update", "Early checkpoint", FIRST_COLOR, "///"),
        (width / 2, "tacs", "TACS", TACS_COLOR, None),
    ):
        mean = np.asarray(data["arms"][arm]["mean"])
        std = np.asarray(data["arms"][arm]["sample_std"])
        ax.errorbar(x + offset, mean, yerr=std, color=color,
                    fmt="s" if arm == "first_update" else "o", markersize=4,
                    capsize=2.5, elinewidth=1, capthick=.8,
                    label=label, zorder=3)
    first = np.asarray(data["arms"]["first_update"]["mean"])
    tacs = np.asarray(data["arms"]["tacs"]["mean"])
    top = max(float((first + np.asarray(data["arms"]["first_update"]["sample_std"])).max()),
              float((tacs + np.asarray(data["arms"]["tacs"]["sample_std"])).max()))
    bottom = min(float((first - np.asarray(data["arms"]["first_update"]["sample_std"])).min()),
                 float((tacs - np.asarray(data["arms"]["tacs"]["sample_std"])).min()))
    pad = max(.5, .18 * (top - bottom))
    ax.set_ylim(bottom - pad, top + 1.6 * pad)
    ax.set_xlim(-.5, len(ranks) - .5)
    ax.set_xticks(x, [str(rank) for rank in ranks])
    ax.set_xlabel("Warmup LoRA rank $r$", labelpad=3)
    ax.set_ylabel("TyDiQA F1", labelpad=3)
    ax.grid(axis="y", color=GRID, lw=.5, zorder=0)
    ax.tick_params(length=2.5, width=.6, pad=2)
    ax.legend(loc="lower left", bbox_to_anchor=(-.02, 1.01), ncol=1,
              handlelength=1.2, columnspacing=1.2, handletextpad=.5, borderaxespad=0)
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=300)
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--t17", type=Path, required=True)
    parser.add_argument("--t5", type=Path, required=True)
    parser.add_argument("--t4", type=Path, required=True)
    parser.add_argument("--rule", default="validation_selected")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        t17 = load_receipt(args.t17, "T17", args.rule)
        t5 = load_receipt(args.t5, "T5", args.rule)
        t4 = load_receipt(args.t4, "T4", args.rule)
        rank = rank_data(t17, args.rule)
        depth = depth_data(t5, args.rule)
        anchor = anchor_data(t4, args.rule)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        style()
        stacked_panel(rank, "ranks", "Warmup LoRA rank $r$", args.out_dir / "figure4_rank_revised")
        stacked_panel(depth, "depths", "Warmup epochs $T$", args.out_dir / "figure4_depth_revised",
                      highlighted_depths={"tydiqa": 2, "mmlu": 64, "bbh": 64}, max_x=64)
        anchor_panel(anchor, args.out_dir / "figure4_perturb_revised")
        outputs = [
            "figure4_rank_revised.pdf", "figure4_rank_revised.png",
            "figure4_depth_revised.pdf", "figure4_depth_revised.png",
            "figure4_perturb_revised.pdf", "figure4_perturb_revised.png",
        ]
        plotting_receipt = {
            "schema_version": 1,
            "rule": args.rule,
            "inputs": {
                "T17": {"path": str(args.t17), "sha256": sha256(args.t17)},
                "T5": {"path": str(args.t5), "sha256": sha256(args.t5)},
                "T4": {"path": str(args.t4), "sha256": sha256(args.t4)},
            },
            "rank": rank,
            "depth": depth,
            "anchor": anchor,
            "outputs": outputs,
            "output_sha256": {name: sha256(args.out_dir / name) for name in outputs},
        }
        (args.out_dir / "figure4_clean_values.json").write_text(
            json.dumps(plotting_receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except (OSError, ValueError, KeyError, PlotError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "rule": args.rule, "out_dir": str(args.out_dir),
        "outputs": plotting_receipt["outputs"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

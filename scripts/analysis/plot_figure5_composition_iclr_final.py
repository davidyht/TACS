"""Rebuild Figure 5's composition panel with the three-seed full-FT TracIn result."""
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/less-figure5-composition-mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "analysis/figure3_revision_20260908"
OUT = ROOT / "iclr_final/fig/figure3_composition"
SOURCE = AUDIT / "plotted_values.csv"
TRACIN = (
    AUDIT / "tracin_fullft_noisy_seed0_local.json",
    AUDIT / "tracin_fullft_noisy_seeds1to2_local.json",
)


def main():
    with SOURCE.open(newline="") as fh:
        historical = {row["method"]: row for row in csv.DictReader(fh)}
    runs = [run for path in TRACIN for run in json.loads(path.read_text())["runs"]]
    assert sorted(run["seed"] for run in runs) == [0, 1, 2]
    assert all(run["method"] == "TracIn" and run["k"] == 500 and "error" not in run for run in runs)
    rows = []
    for label, source in [
        ("Random", "Random"), ("Embedding", "EmbedRetrieval"),
        ("EL2N", "EL2N"), ("GraNd", "GraNd"),
    ]:
        rows.append((label, float(historical[source]["clean_percent"]),
                     float(historical[source]["target_percent"])))
    rows.append(("TracIn", 100 * np.mean([r["selected_clean_fraction"] for r in runs]),
                 100 * np.mean([r["selected_target_fraction"] for r in runs])))
    for label in ("LESS", "TACS"):
        rows.append((label, float(historical[label]["clean_percent"]),
                     float(historical[label]["target_percent"])))

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif"],
        "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5,
        "ytick.labelsize": 8, "legend.fontsize": 7,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.spines.left": False, "axes.linewidth": .6,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(2.69, 2.85))
    fig.subplots_adjust(left=.28, right=.98, bottom=.18, top=.87)
    for j, (title, color) in enumerate([("Clean labels", "#999999"),
                                       ("Target classes", "#D55E00")]):
        vals = [row[j + 1] for row in rows]
        yy = y + (j - .5) * .34
        ax.barh(yy, vals, .30, label=title, color=color, edgecolor="white",
                linewidth=.4, zorder=3, hatch="///" if j == 0 else None)
        for i, (label, *_) in enumerate(rows):
            if label in {"TracIn", "LESS", "TACS"}:
                ax.text(vals[i] + 1, yy[i], f"{vals[i]:.1f}", va="center",
                        ha="left", fontsize=6.5)
    ax.set_yticks(y, [row[0] for row in rows])
    ax.set_ylim(len(rows) - .4, -.6)
    ax.set_xlim(0, 105)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.tick_params(axis="y", length=0, pad=4)
    ax.tick_params(axis="x", length=3)
    for tick in ax.get_yticklabels():
        if tick.get_text() == "TACS":
            tick.set_fontweight("bold")
    ax.set_xlabel("Selected examples (%)", labelpad=5)
    ax.grid(axis="x", color="#DDDDDD", linewidth=.5, zorder=0)
    ax.axvline(0, color="#555555", linewidth=.7, zorder=2)
    ax.legend(loc="lower left", bbox_to_anchor=(-.32, 1.01), ncol=2,
              frameon=False, handlelength=1.2, columnspacing=1.2,
              handletextpad=.5, borderaxespad=0)
    fig.savefig(OUT.with_suffix(".pdf"))
    fig.savefig(OUT.with_suffix(".png"), dpi=300)
    plt.close(fig)
    with (AUDIT / "figure5_right_iclr_final_values.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["method", "clean_percent", "target_percent"])
        writer.writerows(rows)
    print("TracIn:", rows[4])
    print("Saved:", OUT.with_suffix(".pdf"))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""T9 figures from receipts: cost crossover over (m pools, k targets), and selection GPU-hours vs test score.

crossover  cost(m, k) = per_pool_h*m + per_target_h*k + per_pair_h*m*k per method, with the coefficients of
           clean_cost_accounting.py (`cost_model`). Shades the cheapest method on an m x k grid and marks the
           measured point (m=4, k=3).
pareto     One panel per task. x = selection GPU-h attributable to the task: pool x target hours over the four
           pools, plus the task's target hours, plus pool hours / 3. y = seed mean of the source-averaged test
           score (a seed counts only when all four sources are present); error bars = SD over those seeds.
           Random selection costs nothing and sits at x = 0.
Every plotted number is written to a JSON sidecar next to the figure (receipt input).

  python plot_cost_pareto.py crossover --cost cost.json --out fig/cost_crossover.pdf
  python plot_cost_pareto.py pareto --cost cost.json --results-dir $S/results --rule validation_selected --out fig/cost_pareto.pdf
"""
import argparse
import glob
import hashlib
import json
import math
import os
import re
import statistics
import sys

METHODS = ("less", "tov", "tacs")
LABEL = {"less": "LESS", "tov": "ToV", "tacs": "TACS", "random": "Random"}
# Colors of the method schematic in the paper (plot_method_comparison_figure.py): pool trajectory brown,
# ToV perturbation green, validation-induced trajectory blue; Random is neutral.
COLOR = {"less": "#8A3A3A", "tov": "#15803D", "tacs": "#1D4ED8", "random": "#9CA3AF"}
SOURCES = ("dolly", "oasst1", "flan_v2", "cot")
TASKS = ("tydiqa", "mmlu", "bbh")
TASK_LABEL = {"tydiqa": "TyDiQA (F1)", "mmlu": "MMLU (acc, %)", "bbh": "BBH (EM, %)"}
SCALE = {"tydiqa": 1.0, "mmlu": 100.0, "bbh": 100.0}
RESULT_NAME = re.compile(r"^(?P<method>[a-z0-9_]+)__(?P<source>[a-z0-9_]+)__(?P<task>[a-z]+)__s(?P<seed>\d+)__(?P<rule>[a-z_]+)\.json$")


def sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def method_cost(model, m, k):
    return model["per_pool_h"] * m + model["per_target_h"] * k + model["per_pair_h"] * m * k


def crossover_grid(cost_model, m_max, k_max):
    grid = []
    for k in range(1, k_max + 1):
        row = []
        for m in range(1, m_max + 1):
            costs = {meth: method_cost(cost_model[meth], m, k) for meth in METHODS}
            row.append(min(METHODS, key=lambda x: (costs[x], METHODS.index(x))))
        grid.append(row)
    return grid


def task_costs(by_scope):
    out = {}
    for meth in METHODS:
        sc = by_scope.get(meth, {})
        pools, targets, pairs = sc.get("pool", {}), sc.get("target", {}), sc.get("pair", {})
        out[meth] = {task: sum(pairs.get("%s/%s" % (s, task), 0.0) for s in SOURCES) + targets.get(task, 0.0)
                     + sum(pools.values()) / len(TASKS) for task in TASKS}
    out["random"] = {task: 0.0 for task in TASKS}
    return out


def cost_receipt_errors(cost):
    """Validate the exact clean-study cost decomposition used by the Pareto plot."""
    errors = []
    complete = cost.get("complete")
    if not isinstance(complete, dict) or any(complete.get(meth) is not True for meth in METHODS):
        errors.append("incomplete LESS/ToV/TACS cost receipt")
    totals = cost.get("totals_gpu_h")
    by_scope = cost.get("by_scope_gpu_h")
    if not isinstance(totals, dict):
        errors.append("missing totals_gpu_h")
    if not isinstance(by_scope, dict):
        errors.append("missing by_scope_gpu_h")
        return errors

    pair_keys = {"%s/%s" % (source, task) for source in SOURCES for task in TASKS}
    expected = {
        "less": {"pool": set(SOURCES), "pair": pair_keys},
        "tov": {"pool": set(SOURCES), "pair": pair_keys},
        "tacs": {"target": set(TASKS), "pair": pair_keys},
    }
    for meth in METHODS:
        scopes = by_scope.get(meth)
        if not isinstance(scopes, dict):
            errors.append("missing by_scope_gpu_h.%s" % meth)
            continue
        if set(scopes) != set(expected[meth]):
            errors.append("%s cost scopes must be exactly %s" % (meth, sorted(expected[meth])))
            continue
        for scope, keys in expected[meth].items():
            values = scopes.get(scope)
            if not isinstance(values, dict) or set(values) != keys:
                errors.append("%s/%s keys do not match the registered grid" % (meth, scope))
                continue
            for key, value in values.items():
                if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                    errors.append("invalid cost at %s/%s/%s" % (meth, scope, key))
        total = totals.get(meth) if isinstance(totals, dict) else None
        if not isinstance(total, (int, float)) or not math.isfinite(total) or total < 0:
            errors.append("invalid totals_gpu_h.%s" % meth)

    if not errors:
        allocated = task_costs(by_scope)
        for meth in METHODS:
            if not math.isclose(sum(allocated[meth].values()), totals[meth], rel_tol=0.0, abs_tol=1e-9):
                errors.append("%s task allocation does not sum to totals_gpu_h" % meth)
    return errors


def score_points(results_dir, rule, aggregate=None):
    values = {}
    if aggregate:
        with open(aggregate) as fh:
            report = json.load(fh)
        if report.get("rule") != rule or report.get("missing"):
            raise ValueError("aggregate is incomplete or has a different checkpoint rule")
        for row in report["rows"]:
            if row["source"] not in SOURCES or row["method"] not in ("random",) + METHODS:
                continue
            cell = values.setdefault((row["method"], row["task"]), {}).setdefault(int(row["seed"]), {})
            if row["source"] in cell:
                raise ValueError("duplicate aggregate score row")
            cell[row["source"]] = float(row["value"])
    else:
        for path in glob.glob(os.path.join(results_dir, "*__%s.json" % rule)):
            m = RESULT_NAME.match(os.path.basename(path))
            if not m or m.group("source") not in SOURCES:
                continue
            with open(path) as fh:
                value = float(json.load(fh)["value"])
            values.setdefault((m.group("method"), m.group("task")), {}).setdefault(int(m.group("seed")), {})[m.group("source")] = value
    points = {}
    for (meth, task), seeds in sorted(values.items()):
        means = {s: SCALE[task] * sum(v.values()) / len(SOURCES) for s, v in seeds.items() if len(v) == len(SOURCES)}
        if not means:
            continue
        vals = [means[s] for s in sorted(means)]
        points.setdefault(meth, {})[task] = {"seeds": sorted(means), "mean": statistics.mean(vals),
                                              "sd": statistics.stdev(vals) if len(vals) > 1 else None}
    return points


def cmd_crossover(args):
    with open(args.cost) as fh:
        cost = json.load(fh)
    model = cost["cost_model"]
    grid = crossover_grid(model, args.m_max, args.k_max)
    at = {meth: method_cost(model[meth], args.m, args.k) for meth in METHODS}
    sidecar = {"figure": os.path.abspath(args.out), "cost_receipt": os.path.abspath(args.cost),
               "cost_receipt_sha256": sha256(args.cost), "complete": cost.get("complete"),
               "coefficients": {m: {k: model[m][k] for k in ("per_pool_h", "per_target_h", "per_pair_h")} for m in METHODS},
               "marked_point": {"m": args.m, "k": args.k, "gpu_h": at},
               "cheapest_grid_rows_k_cols_m": grid}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    idx = [[METHODS.index(c) for c in row] for row in grid]
    ax.pcolormesh(range(1, args.m_max + 2), range(1, args.k_max + 2), idx, shading="flat",
                  cmap=ListedColormap([COLOR[m] for m in METHODS]), vmin=-0.5, vmax=len(METHODS) - 0.5, alpha=0.35,
                  edgecolors="white", linewidth=0.5)
    ax.plot(args.m + 0.5, args.k + 0.5, marker="*", ms=12, color="#111827")
    ax.annotate("  ".join("%s %.1f h" % (LABEL[m], at[m]) for m in METHODS), (args.m + 0.5, args.k + 0.5),
                xytext=(6, 8), textcoords="offset points", fontsize=7, color="#111827",
                bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.9})
    ax.set_xlabel("Candidate pools $m$")
    ax.set_ylabel("Target tasks $k$")
    ax.set_xticks([x + 0.5 for x in range(1, args.m_max + 1, max(1, args.m_max // 8))])
    ax.set_xticklabels(range(1, args.m_max + 1, max(1, args.m_max // 8)))
    ax.set_yticks([y + 0.5 for y in range(1, args.k_max + 1, max(1, args.k_max // 8))])
    ax.set_yticklabels(range(1, args.k_max + 1, max(1, args.k_max // 8)))
    ax.legend(handles=[Patch(color=COLOR[m], alpha=0.35, label="%s cheapest" % LABEL[m]) for m in METHODS],
              fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(METHODS), frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.out)
    sidecar["figure_sha256"] = sha256(args.out)
    write_sidecar(args.out, sidecar)
    return 0


def cmd_pareto(args):
    with open(args.cost) as fh:
        cost = json.load(fh)
    errors = cost_receipt_errors(cost)
    if errors:
        print("ERROR: invalid cost receipt: %s" % "; ".join(errors), file=sys.stderr)
        return 2
    costs = task_costs(cost["by_scope_gpu_h"])
    points = score_points(args.results_dir, args.rule, args.aggregate)
    missing = [
        "%s/%s" % (method, task)
        for method in ("random",) + METHODS
        for task in TASKS
        if points.get(method, {}).get(task, {}).get("seeds") != [3, 7, 42]
    ]
    if missing:
        print("ERROR: incomplete Pareto score grid: %s" % ", ".join(missing), file=sys.stderr)
        return 2
    sidecar = {"figure": os.path.abspath(args.out), "cost_receipt": os.path.abspath(args.cost),
               "cost_receipt_sha256": sha256(args.cost), "complete": cost.get("complete"), "rule": args.rule,
               "task_gpu_h": costs, "scores": points,
               "aggregate_path": os.path.abspath(args.aggregate) if args.aggregate else None,
               "aggregate_sha256": sha256(args.aggregate) if args.aggregate else None}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(TASKS), figsize=(7.0, 2.4))
    for ax, task in zip(axes, TASKS):
        for meth in ("random",) + METHODS:
            p = points.get(meth, {}).get(task)
            if not p:
                continue
            ax.errorbar(costs[meth][task], p["mean"], yerr=p["sd"] or 0.0, fmt="o", ms=5, color=COLOR[meth],
                        ecolor=COLOR[meth], elinewidth=1, capsize=2, label=LABEL[meth])
        ax.set_title(TASK_LABEL[task], fontsize=9)
        ax.set_xlabel("Selection GPU-h", fontsize=8)
        ax.tick_params(labelsize=7)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(labels), fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(args.out)
    sidecar["figure_sha256"] = sha256(args.out)
    write_sidecar(args.out, sidecar)
    return 0


def write_sidecar(fig_path, data):
    path = os.path.splitext(fig_path)[0] + ".json"
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print("wrote %s and %s" % (fig_path, path))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd")
    c = sub.add_parser("crossover")
    c.add_argument("--cost", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--m-max", type=int, default=16)
    c.add_argument("--k-max", type=int, default=16)
    c.add_argument("--m", type=int, default=4)
    c.add_argument("--k", type=int, default=3)
    p = sub.add_parser("pareto")
    p.add_argument("--cost", required=True)
    p.add_argument("--results-dir", required=True)
    p.add_argument("--aggregate", help="verified aggregate to use for corrected evaluation scores")
    p.add_argument("--rule", default="validation_selected")
    p.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "crossover":
        return cmd_crossover(args)
    if args.cmd == "pareto":
        return cmd_pareto(args)
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())

"""Audit pairwise LESS subset overlap and source-composition changes.

Requires a saved reference-path sensitivity result bundle containing analyses
and selected JSONL files. Pass --bundle and --out explicitly.
"""
import argparse
import datetime
import hashlib
import itertools
import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SOURCES = ("cot", "flan_v2", "oasst1", "dolly")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class Bundle:
    def __init__(self, root):
        self.root = Path(root)
        self.inputs = {}

    def path(self, rel):
        p = self.root / rel
        self.inputs[rel] = sha256(p)
        return p

    def json(self, rel):
        return json.load(open(self.path(rel)))

    def selection(self, rel):
        # split on "\n" only: str.splitlines() also breaks on U+2028/U+2029 inside records (RESULTS.md caveat 9)
        text = open(self.path(rel), encoding="utf-8").read()
        return [json.loads(line) for line in text.split("\n") if line.strip()]


def stats(vals):
    return {"n": len(vals), "mean": statistics.fmean(vals), "min": min(vals), "max": max(vals)}


def pair_values(analysis, a_arms, b_arms, metric="overlap", task="tydiqa"):
    """Pairwise values between two arm groups; the same group gives each unordered pair once."""
    pairs = analysis["tasks"][task]["pairs"]
    combos = itertools.combinations(a_arms, 2) if a_arms == b_arms else itertools.product(a_arms, b_arms)
    return [pairs[f"{a}|{b}"][metric] if f"{a}|{b}" in pairs else pairs[f"{b}|{a}"][metric] for a, b in combos]


def pair_group(analysis, a_arms, b_arms, metric="overlap", task="tydiqa"):
    return stats(pair_values(analysis, a_arms, b_arms, metric, task))


def family_summary(analysis, typed_family, task="tydiqa"):
    """Within-family agreement for one typed family vs the random controls, recomputed from the stored pairs.

    Stored `treatment`/`sensitivity_gap`/`composition_ratio` pool *all* non-control families, so in the analyses
    that also carry a `target_aligned` arm they are not the typed-family numbers. They are kept only as a check.
    """
    t = analysis["tasks"][task]
    typed, control = analysis["families"][typed_family], analysis["control_arms"]
    out = {"pool_size": t["pool_size"], "top_k": t["top_k"], "chance_overlap": t["chance_overlap"],
           "typed_family": typed_family, "typed_arms": typed, "control_arms": control}
    for group, arms in (("typed", typed), ("random", control)):
        out[group] = {m: pair_group(analysis, arms, arms, m, task) for m in ("overlap", "spearman", "composition_tvd")}
    out["overlap_gap_random_minus_typed"] = out["random"]["overlap"]["mean"] - out["typed"]["overlap"]["mean"]
    out["tvd_ratio_typed_over_random"] = out["typed"]["composition_tvd"]["mean"] / out["random"]["composition_tvd"]["mean"]
    others = sorted(f for f, arms in analysis["families"].items() if f != typed_family and arms != control)
    if others:
        out["stored_aggregate_pools_extra_families"] = others
    else:
        out["stored_aggregate_matches"] = (
            abs(t["treatment"]["overlap"]["mean"] - out["typed"]["overlap"]["mean"]) < 1e-9
            and abs(t["sensitivity_gap"] - out["overlap_gap_random_minus_typed"]) < 1e-9
            and abs(t["composition_ratio"] - out["tvd_ratio_typed_over_random"]) < 1e-9)
    return out


def overlap(a, b, k=500):
    return len({r["id"] for r in a} & {r["id"] for r in b}) / k


def source_shares(rows):
    n = len(rows)
    return {s: sum(1 for r in rows if r.get("dataset") == s) / n for s in SOURCES}


def mean_pairwise(sel, arms):
    return stats([overlap(sel[a], sel[b]) for a, b in itertools.combinations(arms, 2)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    B = Bundle(args.bundle)
    checks, numbers = [], {}

    # --- Dolly pool (uncontaminated by the 2026-08-31 collision) ---
    dolly = B.json("dolly_study/analysis/final_all6.json")
    numbers["dolly_kmeans_r128"] = family_summary(dolly, "kmeans")
    numbers["dolly_diversity_matched_r128"] = family_summary(B.json("dolly_study/analysis/diverse_vs_random.json"), "diverse")
    diversity = B.json("dolly_study/analysis/diversity_report.json")
    numbers["dolly_warmup_internal_diversity"] = diversity
    numbers["dolly_depth_prefix_n1_per_group"] = {
        f"depth{d}": family_summary(B.json(f"dolly_study/analysis/depth{d}.json"), "kmeans") for d in (1, 2, 3, 4)}
    numbers["dolly_rank1_lr2e-5"] = family_summary(B.json("dolly_study/analysis/dolly_rank1.json"), "kmeans")
    numbers["dolly_rank1_normmatched_lr1e-4"] = family_summary(B.json("dolly_study/analysis/dolly_rank1_normmatched.json"), "kmeans")
    for tag, rel in (("r128", "divergence.json"), ("rank1", "divergence_rank1.json"),
                     ("rank1_normmatched", "divergence_rank1_normmatched.json")):
        div = B.json(f"dolly_study/analysis/{rel}")
        cos = div["cosines"]
        typed = [v for k, v in cos.items() if k.count("kmeans") == 2]
        rand = [v for k, v in cos.items() if k.count("random") == 2]
        numbers[f"dolly_update_geometry_{tag}"] = {
            "norm_range": [min(div["norms"].values()), max(div["norms"].values())],
            "cosine_typed_mean": statistics.fmean(typed) if typed else None,
            "cosine_random_mean": statistics.fmean(rand) if rand else None}
    numbers["dolly_ratio_sweep"] = {
        r: family_summary(B.json(f"dolly_study/analysis/ratio_{r}.json"), "kmeans") for r in ("0.01", "0.02", "0.05", "0.10")}

    # --- multi-source pool (controls retrained after the 2026-08-31 correction) ---
    ms = {"tydiqa": B.json("multisource_study/analysis/multisource_all.json"),
          "mmlu": B.json("multisource_study/analysis/multisource_mmlu.json"),
          "bbh": B.json("multisource_study/analysis/multisource_bbh.json")}
    numbers["multisource_provenance_r128"] = {
        task: family_summary(a, "source", task) for task, a in ms.items()}
    numbers["multisource_ratio_sweep_tydiqa"] = {
        r: family_summary(B.json(f"multisource_study/analysis/ratio_{r}.json"), "source") for r in ("0.01", "0.02", "0.05", "0.10")}
    all8 = B.json("multisource_study/analysis/multisource_all8.json")
    rand_arms = ["random_r0", "random_r1", "random_r2"]
    src_arms = [f"source_{s}" for s in SOURCES]
    numbers["multisource_target_aligned_tydiqa"] = {
        "target_aligned_vs_random": pair_group(all8, ["target_aligned"], rand_arms),
        "random_vs_random": pair_group(all8, rand_arms, rand_arms),
        "target_aligned_vs_source": pair_group(all8, ["target_aligned"], src_arms)}
    # language axis: stored pairwise TVDs only; per-arm language shares need py3langid labels not in the bundle
    lang = family_summary(B.json("multisource_study/analysis/multisource_language.json"), "source")
    numbers["multisource_language_axis_tydiqa"] = {
        "composition_tvd_typed_mean": lang["typed"]["composition_tvd"]["mean"],
        "composition_tvd_random_mean": lang["random"]["composition_tvd"]["mean"],
        "tvd_ratio_typed_over_random": lang["tvd_ratio_typed_over_random"]}

    # --- recount from the selection files: family overlaps, self-avoidance, ToV vs LESS ---
    sel_dirs = {"tydiqa": "selected_data", "mmlu": "selected_data_mmlu", "bbh": "selected_data_bbh"}
    selfavoid, tov = {}, {}
    for task, d in sel_dirs.items():
        arms = rand_arms + src_arms + ["tov_r0", "tov_r1", "tov_r2"]
        sel = {a: B.selection(f"multisource_study/{d}/{a}__selected_top500.jsonl") for a in arms}
        for a, rows in sel.items():
            assert len(rows) == 500, (task, a, len(rows))
        rec_rand = mean_pairwise(sel, rand_arms)
        rec_src = mean_pairwise(sel, src_arms)
        stored = numbers["multisource_provenance_r128"][task]
        checks.append({"what": f"multisource {task} random-vs-random overlap mean (recount vs stored)",
                       "recount": rec_rand["mean"], "stored": stored["random"]["overlap"]["mean"],
                       "ok": abs(rec_rand["mean"] - stored["random"]["overlap"]["mean"]) < 1e-9})
        checks.append({"what": f"multisource {task} source-vs-source overlap mean (recount vs stored)",
                       "recount": rec_src["mean"], "stored": stored["typed"]["overlap"]["mean"],
                       "ok": abs(rec_src["mean"] - stored["typed"]["overlap"]["mean"]) < 1e-9})
        shares = {a: source_shares(sel[a]) for a in rand_arms + src_arms + ["tov_r0", "tov_r1", "tov_r2"]}
        baseline = {s: statistics.fmean(shares[a][s] for a in rand_arms) for s in SOURCES}
        selfavoid[task] = {
            "shares": shares,
            "random_baseline": baseline,
            "own_source": {s: {"share": shares[f"source_{s}"][s], "random_baseline": baseline[s],
                               "under_selection_x": (baseline[s] / shares[f"source_{s}"][s])
                               if shares[f"source_{s}"][s] > 0 else None,
                               "is_row_minimum": shares[f"source_{s}"][s] == min(shares[f"source_{s}"].values())}
                           for s in SOURCES},
            "all_four_below_random_baseline": all(shares[f"source_{s}"][s] < baseline[s] for s in SOURCES)}
        tov[task] = {"tov_vs_tov": mean_pairwise(sel, ["tov_r0", "tov_r1", "tov_r2"]),
                     "less_vs_less": rec_rand,
                     "tov_vs_less_same_trajectory": [overlap(sel[f"tov_r{i}"], sel[f"random_r{i}"]) for i in range(3)]}
    numbers["multisource_self_avoidance"] = selfavoid
    numbers["multisource_tov_vs_less_same_trajectories"] = tov

    # --- flan_v2-only pool ---
    numbers["flan_v2_kmeans_r128"] = family_summary(B.json("flan_v2_study/analysis/flan_v2_default.json"), "kmeans")

    # --- downstream (retrain seeds 3/5/7; one warmup seed 3); recorded, not proposed for the main text ---
    ds = {}
    for rel in sorted(Path(args.bundle).glob("*_study/downstream*/summary*.json")):
        rel_s = str(rel.relative_to(args.bundle))
        data = B.json(rel_s)
        ds[rel_s] = {cfg: {k: v for k, v in vals.items() if k != "runs" and k != "values"}
                     for cfg, vals in data.items() if isinstance(vals, dict) and not cfg.startswith("_")}
    numbers["downstream_summaries"] = ds

    # pooled per selector: TyDiQA F1 (runs[*].f1), MMLU/BBH accuracy (values); one entry per retrain run
    def runs_of(entry):
        return [r["f1"] for r in entry["runs"].values()] if "runs" in entry else list(entry["values"])

    groups = {"less_random_warmup": rand_arms, "less_source_warmup": src_arms,
              "tov_random_warmup": ["tov_r0", "tov_r1", "tov_r2"], "random500": ["random500"],
              "fullpool": ["fullpool"], "base": ["base"]}
    suffix = {"tydiqa": ("", "summary_fixed.json"), "mmlu": ("_mmlu", "summary.json"), "bbh": ("_bbh", "summary.json")}
    pooled = {}
    for task, (sfx, name) in suffix.items():
        merged = {}
        for d in (f"downstream{sfx}", f"downstream_tov{sfx}"):
            merged.update(B.json(f"multisource_study/{d}/{name}"))
        pooled[task] = {}
        for g, arms in groups.items():
            arms = [a for a in arms if a in merged]
            if not arms:
                continue
            vals = [v for a in arms for v in runs_of(merged[a])]
            arm_means = [statistics.fmean(runs_of(merged[a])) for a in arms]
            pooled[task][g] = {"n_runs": len(vals), "mean": statistics.fmean(vals),
                               "arm_mean_range": [min(arm_means), max(arm_means)]}
        base = pooled[task]["base"]["mean"]
        trained = [v for g, e in pooled[task].items() if g != "base" for v in e["arm_mean_range"]]
        pooled[task]["max_abs_arm_mean_minus_base"] = max(abs(v - base) for v in trained)
    numbers["multisource_downstream_pooled"] = pooled

    def stored_checks(node, where):
        if isinstance(node, dict):
            if "stored_aggregate_matches" in node:
                checks.append({"what": f"{where}: stored treatment/gap/ratio equal the recomputed typed-family values",
                               "ok": node["stored_aggregate_matches"]})
            for k, v in node.items():
                stored_checks(v, f"{where}.{k}" if where else k)
    stored_checks(numbers, "")

    receipt = {
        "id": "T19",
        "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "agent": "claude",
        "status": "legacy_diagnostic",
        "scope": {
            "model": "Qwen2.5-1.5B (RESULTS.md 'Result 2 Configuration'; manifests do not record the model)",
            "less_config": "LoRA r=128 alpha=512 (rank-1 arms: r=1 alpha=4), lr 2e-5 (norm-matched rank 1: 1e-4), 4 epochs, "
                           "batch 1 x grad-accum 32, warmup seed 3, 4 checkpoints, uniform checkpoint weights, "
                           "TRAK dim 8192 seed 0",
            "pools": "10,000 rows: Dolly; four-source stratified (flan_v2/cot/dolly/oasst1); flan_v2 only",
            "warmup": "750 rows from a disjoint reservoir (Dolly/flan_v2 5,011; multi-source 20,000)",
            "selection": "top 500 (5%); ratio sweeps 1/2/5/10%",
            "not_clean_llama_table1": True, "not_tacs_benchmark": True,
            "compute": "collaborator's single H200 (bld), not DeltaAI"},
        "inputs": [{"path": f"result (5)/{k}", "sha256": v} for k, v in sorted(B.inputs.items())],
        "code": [{"path": "scripts/analysis/path_sensitivity_receipt.py",
                  "sha256": sha256(Path(__file__))}],
        "command": "python3 scripts/analysis/path_sensitivity_receipt.py",
        "cluster_jobs": [],
        "numbers": numbers,
        "consistency_checks": checks,
        "do_not_use": [
            {"path": "result (5)/multisource_study/analysis/divergence.json",
             "why": "multi-source update norms/cosines computed from superseded adapters (random-arm norms identical to "
                    "the Dolly study's); RESULTS.md Result 7 withdraws them"},
            {"path": "result (5)/figures/fig7_self_avoidance.png",
             "why": "random rows show pre-correction compositions (flan_v2 0.73-0.83); regenerate from this receipt"},
            {"path": "result (5)/figures/fig8_language_swing.png",
             "why": "built before the 2026-08-31 correction per RESULTS.md; regenerate before use"},
            {"path": "result (5)/superseded_precorrection/", "why": "quarantined pre-correction numbers"},
            {"what": "RESULTS.md Result 1 (LESS released seed 3/6/9 selections)",
             "why": "no artifact in the bundle; reproducible only after downloading princeton-nlp/less_data selections"}],
        "predeclared_reading": "receipts/T19_runcard.md (reading rule of T10 applied unchanged; written after the "
                               "collaborator's results existed, so it is not predeclared in the strict sense)",
        "outcome": "A: typed warmups agree less than random warmups in every setting (typed mean overlap below "
                   "random mean); full separation (typed max < random min) in all but the flan_v2 pool and the "
                   "four-source 1% budget. Downstream: no resolvable selector differences (legacy sampled eval).",
        "paper_edits": [
            "sections/appendix.tex: 'Warmup-Sensitivity Study' names Qwen2.5-1.5B (was Llama-3.2-3B) and the "
            "warmup/scoring recipe",
            "sections/appendix.tex: new subsubsection 'Robustness of Warmup Sensitivity' "
            "(label app:warmup_sensitivity_robustness, table tab:warmup_sensitivity_robustness)",
            "sections/motivation.tex: paragraph 'Reference-path construction affects selection.' names the model "
            "and points to the robustness appendix",
            "sections/motivation.tex + appendix.tex: target-aligned diagnostic described as a seventh warmup in the "
            "same study (was 'a separate legacy diagnostic')"],
    }
    out = Path(args.out)
    out.write_text(json.dumps(receipt, indent=2) + "\n")
    bad = [c for c in checks if not c["ok"]]
    print(f"wrote {out} ({len(B.inputs)} inputs hashed); consistency checks: {len(checks) - len(bad)}/{len(checks)} ok")
    for c in bad:
        print("MISMATCH:", c)


if __name__ == "__main__":
    main()

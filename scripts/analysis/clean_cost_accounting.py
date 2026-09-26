#!/usr/bin/env python3
"""Measured selection cost of the clean Llama-3.2-3B study (inputs for T9 and the T15 Table 2 update).

GPU hours = allocated GPUs x elapsed of every COMPLETED job in a method's selection pipeline at one seed,
attributed from the launch records under the study root. Retraining and evaluation are excluded (all
methods pay them). Jobs in other states are listed as attempts, never as cost.

Scopes, for the cost model over m pools and k targets:
  LESS  pool: warmup, train gradients.  pool x target: eval gradients and the match/select controller
        (each covers the pool's three tasks and is split evenly across them).
  ToV   pool: the LESS warmup it reuses.  pool x target: score, merge, select controller.
  TACS  target: calibration (seed-independent; job ids plus checkpoint-mtime warmup timing) and the target
        warmup.  pool x target: scoring of the theta1 checkpoints {1, T*}, credited to the array element that
        produced each cache (tacs/score_registry), plus merge and select.
TACS warmups also train the epochs the depth/anchor controls need; `warmup_epochs` and T* are reported so a
reader can see how far a warmup ran past T*. No pro-rating is applied to the measured hours.

  python clean_cost_accounting.py --study-root $S --sacct sacct.psv --seed 3 \
     --calibration tydiqa=3144497,3144532,3144735 --calibration mmlu=3144533 --calibration bbh=3144879,3144880,3144881 \
     --calibration-warmups mmlu=r9_timing.json --calibration-warmups bbh=r9_timing.json --out cost.json
"""
import argparse
import csv
import glob
import hashlib
import json
import os
import re
import sys

SOURCES = ("dolly", "oasst1", "flan_v2", "cot")
TASKS = ("tydiqa", "mmlu", "bbh")
POOL_ROWS = {"dolly": 15011, "oasst1": 55668, "flan_v2": 100000, "cot": 100000}
DEFAULT_TSTAR = {"tydiqa": 64, "mmlu": 2, "bbh": 2}
TACS_MAIN_TAG = re.compile(r"^clean_v1_theta1_(?P<src>dolly|oasst1|flan_v2|cot)_s(?P<seed>\d+)_(?P<group>[a-z0-9]+)_\d{8}_\d{6}$")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_sacct(path):
    jobs = {}
    with open(path) as fh:
        header = fh.readline().rstrip("\n").split("|")
        col = {h: i for i, h in enumerate(header)}
        for line in fh:
            parts = line.rstrip("\n").split("|")
            if len(parts) != len(header):
                continue
            m = re.search(r"gres/gpu=(\d+)", parts[col["AllocTRES"]])
            jobs[parts[col["JobID"]]] = {
                "job": parts[col["JobID"]], "name": parts[col["JobName"]],
                "state": (parts[col["State"]].split() or ["UNKNOWN"])[0],
                "elapsed_s": int(parts[col["ElapsedRaw"]] or 0), "gpus": int(m.group(1)) if m else 0,
            }
    return jobs


def elements(jobs, job_id):
    """The allocation rows of a job: the job itself or its array elements (pending ranges included)."""
    job_id = str(job_id)
    return [v for k, v in sorted(jobs.items()) if k == job_id or k.startswith(job_id + "_")]


class Ledger:
    def __init__(self, jobs):
        self.jobs = jobs
        self.rows = []
        self.attempts = []
        self.missing = []

    def add(self, method, source, task, stage, scope, job_rows, share=1.0, note=None):
        done = [r for r in job_rows if r["state"] == "COMPLETED"]
        for r in job_rows:
            if r["state"] != "COMPLETED":
                self.attempts.append(dict(r, method=method, source=source, task=task, stage=stage))
        if not done:
            self.missing.append({"method": method, "source": source, "task": task, "stage": stage,
                                 "jobs": [r["job"] for r in job_rows]})
            return
        for r in done:
            self.rows.append({"method": method, "source": source, "task": task, "stage": stage, "scope": scope,
                              "job": r["job"], "elapsed_s": r["elapsed_s"], "gpus": r["gpus"], "share": share,
                              "gpu_h": r["gpus"] * r["elapsed_s"] * share / 3600.0, "note": note})

    def add_job(self, method, source, task, stage, scope, job_id, share=1.0, note=None):
        rows = elements(self.jobs, job_id)
        if not rows:
            self.missing.append({"method": method, "source": source, "task": task, "stage": stage,
                                 "jobs": [str(job_id)], "reason": "not in sacct export"})
            return
        self.add(method, source, task, stage, scope, rows, share, note)


def read_lines(path):
    with open(path) as fh:
        return fh.read().splitlines()


def read_tsv(path):
    if not os.path.isfile(path):
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def replacement_ledgers(study_root):
    """Return explicit old-job -> replacement-job mappings from repair ledgers.

    These ledgers are part of the clean-run provenance.  Accounting the original
    launch records alone incorrectly marks superseded jobs as missing and omits
    the GPU time of the successful chunked/resharded replacements.
    """
    warmups = {}
    for row in read_tsv(os.path.join(study_root, "warmup_timechunk_ledger.tsv")):
        old = [x for x in row.get("old_chunks", "").split(",") if x]
        count = int(row["chunks"])
        last = int(row["last"])
        new = [str(j) for j in range(last - count + 1, last + 1)]
        # Key every original warmup/chunk id so either generation of launch
        # record resolves to the same complete execution history.
        for job in old:
            warmups[job] = old + new
    # The first repair ledger links the original monolithic warmup to the
    # epoch chunks that were themselves superseded by the time-chunk chain.
    # Resolve that second hop so the original launch record remains usable.
    for row in read_tsv(os.path.join(study_root, "less_warmup_chunk_ledger.tsv")):
        old_warmup = row.get("old_warmup")
        chunks = [x for x in row.get("chunks", "").split(",") if x]
        if old_warmup and chunks:
            warmups[old_warmup] = warmups.get(chunks[0], chunks)
    grads = {}
    for row in read_tsv(os.path.join(study_root, "less_grad_reshard_ledger.tsv")):
        jobs = [row.get("shard_array"), row.get("merge_job")]
        grads[row["old_array"]] = [x for x in jobs if x]
    controllers = {
        row["old"]: row["new"]
        for row in read_tsv(os.path.join(study_root, "resubmitted_controllers_ledger.tsv"))
        if row.get("old") and row.get("new")
    }
    return warmups, grads, controllers


def canonical_warmup_tags(study_root, seed):
    """Read the frozen shared-warmup links, even when their cache targets were reclaimed."""
    tags = {}
    for task in TASKS:
        path = os.path.join(study_root, "tacs", "warmup_roots", "s%d" % seed,
                            task, "warmup_ckpts")
        if not os.path.islink(path):
            continue
        target = os.readlink(path)
        match = re.search(r"/valwarmup_cache/([^/]+)/", target)
        if match:
            tags[task] = match.group(1)
    return tags


def canonical_selection_tags(study_root, seed):
    """Map active clean TACS cells to the selected-data tag they actually trained on."""
    registry = os.path.join(study_root, "registry.jsonl")
    out = {}
    if not os.path.isfile(registry):
        return out
    with open(registry) as fh:
        for line in fh:
            if not line.strip():
                continue
            entry = json.loads(line)
            parts = entry.get("cell", "").split("/")
            if len(parts) != 4 or parts[0] != "tacs" or int(parts[3]) != seed:
                continue
            freeze = entry.get("freeze_path")
            if not freeze or not os.path.isfile(freeze):
                continue
            run_dir = json.load(open(freeze)).get("run_dir")
            log = os.path.join(run_dir or "", "train.log")
            if not os.path.isfile(log):
                continue
            with open(log, errors="replace") as train_log:
                for row in train_log:
                    match = re.search(r"/out/selected_data/([^/]+)/", row)
                    if match:
                        out[(parts[1], parts[2])] = match.group(1)
                        break
    return out


def parse_less_wave(path):
    blocks, cur = [], None
    for line in read_lines(path):
        s = line.strip()
        m = re.match(r"^clean baseline: method=less source=(\S+) seed=(\d+)", s)
        if m:
            cur = {"source": m.group(1), "seed": int(m.group(2)), "record": path}
            blocks.append(cur)
            continue
        if cur is None:
            continue
        m = re.match(r"^(warmup|train_grads|eval_grads|match_select_retrain_eval): (\d+)$", s)
        if m:
            cur.setdefault(m.group(1), m.group(2))
            continue
        m = re.match(r"^LESS_(TRAIN_GRADS|EVAL_GRADS|MATCH_RETRAIN)_JOB_ID=(\d+)$", s)
        if m:
            key = {"TRAIN_GRADS": "train_grads", "EVAL_GRADS": "eval_grads",
                   "MATCH_RETRAIN": "match_select_retrain_eval"}[m.group(1)]
            cur.setdefault(key, m.group(2))
            continue
        m = re.match(r"^WAVE_SEED source=(\S+) seed=(\d+) .*warmup=(\d+)", s)
        if m and cur["source"] == m.group(1) and cur["seed"] == int(m.group(2)):
            cur.setdefault("warmup", m.group(3))
    return blocks


def parse_tov(path):
    out = {}
    for line in read_lines(path):
        m = re.match(r"^source=(\S+) task=(\S+) (perturb|score|merge|select\+retrain\+eval)=(\d+)$", line.strip())
        if m:
            out.setdefault(m.group(2), {})[m.group(3).split("+")[0]] = m.group(4)
    return out


def parse_tacs_record(path):
    rec = {"record": path, "tag": None, "warmup_epochs": None, "tasks": {}}
    pending, score_task = None, None
    for line in read_lines(path):
        s = line.strip()
        if s.startswith("STUDY_TAG=") and rec["tag"] is None:
            rec["tag"] = s.split("=", 1)[1]
        m = re.match(r"^WARMUP_EPOCHS=(\d+)$", s)
        if m and rec["warmup_epochs"] is None:
            rec["warmup_epochs"] = int(m.group(1))
        m = re.match(r"^\[warmup\] (\S+): submitting fresh warmup", s)
        if m:
            pending = ("warmup", m.group(1))
            continue
        m = re.match(r"^\[warmup\] (\S+): reusing from (\S+)", s)
        if m:
            rec["tasks"].setdefault(m.group(1), {})["reuse"] = m.group(2)
            continue
        m = re.match(r"^\[score\] (\S+): depth=(\d+) ckpt_ids=", s)
        if m:
            score_task = m.group(1)
            rec["tasks"].setdefault(score_task, {})["depth"] = int(m.group(2))
            continue
        m = re.match(r"^(SCORE|MERGE)_JOB_ID=(\d+)$", s)
        if m and score_task:
            rec["tasks"].setdefault(score_task, {}).setdefault(m.group(1).lower(), m.group(2))
            continue
        m = re.match(r"^\[select\] (\S+)/(\S+):(.*)$", s)
        if m:
            pending = ("select", m.group(1))
            tag = re.search(r"\bsel_tag=(\S+)", m.group(3))
            if tag:
                rec["tasks"].setdefault(m.group(1), {})["selection_tag"] = tag.group(1)
            continue
        m = re.search(r"Submitted batch job (\d+)", s)
        if m and pending:
            rec["tasks"].setdefault(pending[1], {}).setdefault(pending[0], m.group(1))
            pending = None
    return rec


def load_registry(root):
    entries = []
    for path in glob.glob(os.path.join(root, "*", "*.json")):
        if ".tmp." in path:
            continue
        with open(path) as fh:
            entries.append(json.load(fh))
    return entries


def producer_element(entries, pool, step, ckpt_marker):
    """(job, step position, steps scored by that job) for the producer of (pool, step) under ckpt_marker."""
    hits = [e for e in entries if e["key"]["pool_name"] == pool and int(e["key"]["step"]) == step
            and ckpt_marker in e["key"]["checkpoint"] and str(e.get("job", "")).isdigit()]
    if len(hits) != 1:
        return None, None, len(hits)
    e = hits[0]
    steps = sorted(int(x["key"]["step"]) for x in entries
                   if x.get("claim") == e.get("claim") and str(x.get("job")) == str(e["job"]) and x["key"]["pool_name"] == pool)
    return str(e["job"]), steps.index(step), len(steps)


def array_size(jobs, job_id):
    """Number of array elements of job_id, from element rows (J_i) and pending ranges (J_[a-b%p])."""
    size = 0
    for key in jobs:
        m = re.match(r"^%s_(\d+)$" % re.escape(job_id), key) or re.match(r"^%s_\[(?:\d+-)?(\d+)" % re.escape(job_id), key)
        if m:
            size = max(size, int(m.group(1)) + 1)
    return size


def account(args):
    jobs = load_sacct(args.sacct)
    led = Ledger(jobs)
    S = args.study_root
    tstar = dict(DEFAULT_TSTAR)
    for item in args.tstar or []:
        t, v = item.split("=")
        tstar[t] = int(v)
    records = []
    warmup_replacements, grad_replacements, controller_replacements = replacement_ledgers(S)

    # LESS and the pool warmups ToV reuses.
    less_warmup = {}
    for path in sorted(glob.glob(os.path.join(S, "wave_*_submit_stdout.txt"))):
        records.append(path)
        for b in parse_less_wave(path):
            if b["seed"] != args.seed:
                continue
            src = b["source"]
            less_warmup[src] = b.get("warmup")
            warmup = b.get("warmup")
            warmup_jobs = warmup_replacements.get(warmup)
            if warmup_jobs:
                led.add("less", src, None, "warmup", "pool",
                        [r for job in warmup_jobs for r in elements(jobs, job)],
                        note="successful chunked replacement of %s" % warmup)
            else:
                led.add_job("less", src, None, "warmup", "pool", warmup)
            train_grads = b.get("train_grads")
            grad_jobs = grad_replacements.get(train_grads)
            if grad_jobs:
                led.add("less", src, None, "train_grads", "pool",
                        [r for job in grad_jobs for r in elements(jobs, job)],
                        note="successful resharded replacement of %s" % train_grads)
            else:
                led.add_job("less", src, None, "train_grads", "pool", train_grads)
            for task in TASKS:
                led.add_job("less", src, task, "eval_grads", "pair", b.get("eval_grads"), share=1.0 / len(TASKS))
                led.add_job("less", src, task, "match_select", "pair", b.get("match_select_retrain_eval"),
                            share=1.0 / len(TASKS))

    # ToV.
    for src in SOURCES:
        path = os.path.join(S, "tov_%s_seed%d_submit_stdout.txt" % (src, args.seed))
        if not os.path.isfile(path):
            led.missing.append({"method": "tov", "source": src, "task": None, "stage": "launch", "jobs": []})
            continue
        records.append(path)
        warmup = less_warmup.get(src)
        warmup_jobs = warmup_replacements.get(warmup)
        if warmup_jobs:
            led.add("tov", src, None, "pool_warmup(less)", "pool",
                    [r for job in warmup_jobs for r in elements(jobs, job)],
                    note="the successful chunked LESS warmup of this pool and seed")
        else:
            led.add_job("tov", src, None, "pool_warmup(less)", "pool", warmup,
                        note="the LESS warmup of this pool and seed")
        for task, ids in sorted(parse_tov(path).items()):
            if ids.get("perturb"):  # sharded ToV: one saved perturbation per checkpoint, loaded by every score shard
                led.add_job("tov", src, task, "perturb", "pair", ids.get("perturb"))
            led.add_job("tov", src, task, "score", "pair", ids.get("score"))
            led.add_job("tov", src, task, "merge", "pair", ids.get("merge"))
            select = ids.get("select")
            led.add_job("tov", src, task, "select", "pair",
                        controller_replacements.get(select, select))

    # TACS main arm (theta1, no control suffix).
    tacs_records = sorted(set(glob.glob(os.path.join(S, "tacs", "launches", "tacs_*.txt"))
                              + glob.glob(os.path.join(S, "tacs_*_submit*_stdout.txt"))))
    mains = []
    for path in tacs_records:
        rec = parse_tacs_record(path)
        m = TACS_MAIN_TAG.match(rec["tag"] or "")
        if m and int(m.group("seed")) == args.seed:
            rec["source"] = m.group("src")
            mains.append(rec)
            records.append(path)
    registry = load_registry(os.path.join(S, "tacs", "score_registry"))
    preferred_tags = canonical_warmup_tags(S, args.seed)
    selected_tags = canonical_selection_tags(S, args.seed)
    warm_tags = {}
    task_records = {}
    for task in TASKS:
        candidates = [r for r in mains if task in r["tasks"]]
        preferred = preferred_tags.get(task)
        if preferred:
            chosen = [r for r in candidates if r["tag"] == preferred]
        else:
            chosen = [r for r in candidates if any(
                x["state"] == "COMPLETED"
                for x in elements(jobs, r["tasks"].get(task, {}).get("warmup", "")))]
        if len(chosen) != 1:
            led.missing.append({"method": "tacs", "source": None, "task": task,
                                "stage": "warmup", "jobs": [r["tag"] for r in chosen],
                                "reason": "expected one canonical warmup record, found %d" % len(chosen)})
            continue
        rec = chosen[0]
        ids = rec["tasks"][task]
        warmup = ids.get("warmup")
        rows = elements(jobs, warmup) if warmup else []
        if not rows:
            led.missing.append({"method": "tacs", "source": None, "task": task,
                                "stage": "warmup", "jobs": [warmup] if warmup else [],
                                "reason": "canonical warmup job absent"})
            continue
        led.add("tacs", None, task, "warmup", "target", rows,
                note="warmup_epochs=%s T*=%s" % (rec["warmup_epochs"], tstar.get(task)))
        if any(r["state"] == "COMPLETED" for r in rows):
            warm_tags[task] = rec["tag"]

    # Keep superseded failed warmups visible as attempts, but do not let them
    # make a successful canonical stage incomplete.
    canonical_jobs = {
        r["tasks"][task].get("warmup")
        for task, tag in warm_tags.items() for r in mains if r["tag"] == tag
    }
    for rec in mains:
        for task, ids in rec["tasks"].items():
            job = ids.get("warmup")
            if job and job not in canonical_jobs:
                for row in elements(jobs, job):
                    if row["state"] != "COMPLETED":
                        led.attempts.append(dict(row, method="tacs", source=None,
                                                 task=task, stage="warmup_superseded"))

    for src in SOURCES:
        for task in TASKS:
            candidates = [r for r in mains if r["source"] == src and task in r["tasks"]]
            selected = selected_tags.get((src, task))
            if selected:
                candidates = [r for r in candidates
                              if r["tasks"][task].get("selection_tag") == selected]
            elif src == "dolly" and task in warm_tags:
                candidates = [r for r in candidates if r["tag"] == warm_tags[task]]
            if len(candidates) == 1:
                task_records[(src, task)] = candidates[0]
            else:
                led.missing.append({"method": "tacs", "source": src, "task": task,
                                    "stage": "launch", "jobs": [r["tag"] for r in candidates],
                                    "reason": "expected one canonical main launch, found %d" % len(candidates)})
    for src in SOURCES:
        for task in TASKS:
            tag = warm_tags.get(task)
            for step in (1, tstar[task]):
                if not tag:
                    led.missing.append({"method": "tacs", "source": src, "task": task, "stage": "score_step%d" % step,
                                        "jobs": [], "reason": "no unique completed warmup"})
                    continue
                stage = "score_step%d" % step
                job, position, n_steps = producer_element(registry, src, step, "/%s/%s/" % (tag, task))
                if job is None:
                    led.missing.append({"method": "tacs", "source": src, "task": task, "stage": stage,
                                        "jobs": [], "reason": "registry entries found: %d" % n_steps})
                    continue
                size = array_size(jobs, job)
                if size == 0 or size % n_steps:
                    led.missing.append({"method": "tacs", "source": src, "task": task, "stage": stage, "jobs": [job],
                                        "reason": "array size %d is not a multiple of %d steps" % (size, n_steps)})
                    continue
                n_shards = size // n_steps
                ids = ["%s_%d" % (job, position * n_shards + s) for s in range(n_shards)]
                rows = [jobs.get(i, {"job": i, "state": "NOT_IN_SACCT", "elapsed_s": 0, "gpus": 0}) for i in ids]
                if sum(r["state"] == "COMPLETED" for r in rows) < n_shards:
                    led.missing.append({"method": "tacs", "source": src, "task": task, "stage": stage, "jobs": ids,
                                        "reason": "incomplete score shards"})
                led.add("tacs", src, task, stage, "pair", rows, note="%d shard(s)" % n_shards)
            for kind in ("merge", "select"):
                rec = task_records.get((src, task))
                ids = [rec["tasks"][task][kind]] if rec and kind in rec["tasks"][task] else []
                rows = [x for i in ids for x in elements(jobs, i)]
                if not rows:
                    led.missing.append({"method": "tacs", "source": src, "task": task, "stage": kind, "jobs": ids})
                else:
                    led.add("tacs", src, task, kind, "pair", rows)

    # TACS calibration.
    for item in args.calibration or []:
        task, ids = item.split("=")
        for j in ids.split(","):
            led.add_job("tacs", None, task, "calibration_job", "target", j)
    for item in args.calibration_warmups or []:
        task, path = item.split("=")
        with open(path) as fh:
            summary = json.load(fh)["summary"].get(task)
        if not summary:
            led.missing.append({"method": "tacs", "source": None, "task": task, "stage": "calibration_warmups",
                                "jobs": [], "reason": "no timing summary in %s" % path})
            continue
        led.rows.append({"method": "tacs", "source": None, "task": task, "stage": "calibration_warmups", "scope": "target",
                         "job": "mtimes:%s" % path, "elapsed_s": summary["total_train_s"], "gpus": 1, "share": 1.0,
                         "gpu_h": summary["total_train_s"] / 3600.0,
                         "note": "%d warmups, training time from checkpoint mtimes" % summary["n"]})

    return summarize(led, args, records)


def summarize(led, args, records):
    totals, scopes, cells = {}, {}, {}
    for r in led.rows:
        totals[r["method"]] = totals.get(r["method"], 0.0) + r["gpu_h"]
        sc = scopes.setdefault(r["method"], {}).setdefault(r["scope"], {})
        key = {"pool": r["source"], "target": r["task"], "pair": "%s/%s" % (r["source"], r["task"])}[r["scope"]]
        sc[key] = sc.get(key, 0.0) + r["gpu_h"]
        st = cells.setdefault(r["method"], {}).setdefault(r["stage"], 0.0)
        cells[r["method"]][r["stage"]] = st + r["gpu_h"]
    model = {}
    for method, sc in scopes.items():
        pools = sc.get("pool", {})
        targets = sc.get("target", {})
        pairs = sc.get("pair", {})
        model[method] = {
            "per_pool_h": sum(pools.values()) / len(SOURCES) if pools else 0.0,
            "per_target_h": sum(targets.values()) / len(TASKS) if targets else 0.0,
            "per_pair_h": sum(pairs.values()) / (len(SOURCES) * len(TASKS)) if pairs else 0.0,
            "note": "averages over the four measured pools and three targets (m=4, k=3); pools differ in size",
        }
    missing_by_method = {}
    for m in led.missing:
        missing_by_method.setdefault(m["method"], []).append(m)
    return {
        "seed": args.seed,
        "inputs": {"sacct": os.path.abspath(args.sacct), "sacct_sha256": sha256(args.sacct),
                   "records": sorted(set(records)), "pool_rows": POOL_ROWS},
        "totals_gpu_h": totals,
        "complete": {m: m not in missing_by_method for m in ("less", "tov", "tacs")},
        "by_stage_gpu_h": cells,
        "by_scope_gpu_h": scopes,
        "cost_model": model,
        "missing": led.missing,
        "attempts": led.attempts,
        "rows": led.rows,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--study-root", required=True)
    ap.add_argument("--sacct", required=True)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--tstar", action="append", help="task=T* (defaults: frozen D1 picks)")
    ap.add_argument("--calibration", action="append", help="task=jobid,jobid")
    ap.add_argument("--calibration-warmups", action="append", help="task=warmup_ckpt_timing.json")
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    report = account(args)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text + "\n")
    print(json.dumps({k: report[k] for k in ("totals_gpu_h", "complete", "cost_model")}, indent=2, sort_keys=True))
    print("missing stages: %d, non-completed attempts: %d" % (len(report["missing"]), len(report["attempts"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())

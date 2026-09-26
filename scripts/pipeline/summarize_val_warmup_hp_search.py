#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch


def _load_probe_scores(path: Path) -> torch.Tensor:
    obj = json.loads(path.read_text(encoding="utf-8"))
    target_valid = obj.get("target_valid")
    if not isinstance(target_valid, dict):
        raise ValueError(f"target_valid missing in {path}")
    scores = target_valid.get("scores")
    if not isinstance(scores, list) or not scores:
        raise ValueError(f"target_valid.scores missing in {path}")
    return torch.tensor([float(x) for x in scores], dtype=torch.float32)


def _load_pool_scores(path: Path) -> torch.Tensor:
    scores = torch.load(path, map_location="cpu")
    if not torch.is_tensor(scores):
        raise ValueError(f"expected tensor in {path}, got {type(scores)}")
    scores = scores.float().flatten()
    if scores.numel() == 0:
        raise ValueError(f"empty tensor in {path}")
    return scores


def _pairwise_auroc(pos: torch.Tensor, neg: torch.Tensor) -> float:
    if pos.numel() == 0 or neg.numel() == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float((((diff > 0).float()) + 0.5 * ((diff == 0).float())).mean().item())


def _stage_glob(stage: str) -> str:
    if stage == "lr":
        return "lr_search/fold_*/lr_*"
    if stage == "depth":
        return "depth_search/fold_*/depth_*"
    raise ValueError(f"unsupported stage: {stage}")


def _candidate_name(stage: str, path: Path) -> Tuple[str, float]:
    label = path.name
    if stage == "lr":
        raw = label.replace("lr_", "", 1)
        return raw, float(raw)
    raw = label.replace("depth_", "", 1)
    return raw, float(int(raw))


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize val-warmup HP-search folds with AUROC.")
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--task", required=True, choices=["tydiqa", "mmlu", "bbh"])
    ap.add_argument("--stage", required=True, choices=["lr", "depth"])
    args = ap.parse_args()

    run_root = Path(args.run_root).expanduser().resolve()
    analysis_dir = run_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    grouped: Dict[str, Dict[str, object]] = {}
    for combo_root in sorted(run_root.glob(_stage_glob(args.stage))):
        task_root = combo_root / args.task
        if not task_root.is_dir():
            continue
        fold_dir = combo_root.parent
        fold_label = fold_dir.name
        candidate, candidate_numeric = _candidate_name(args.stage, combo_root)
        probe_path = task_root / "probe_scores.json"
        pool_path = task_root / "train_probe_influence_score.pt"
        grouped.setdefault(
            candidate,
            {
                "candidate": candidate,
                "candidate_numeric": candidate_numeric,
                "folds": [],
            },
        )
        grouped[candidate]["folds"].append(
            {
                "fold": fold_label,
                "probe_path": str(probe_path),
                "pool_path": str(pool_path),
                "task_root": str(task_root),
            }
        )

    rows: List[Dict[str, object]] = []
    ranking_rows: List[Dict[str, object]] = []
    for candidate in sorted(grouped.keys(), key=lambda key: grouped[key]["candidate_numeric"]):
        fold_entries = grouped[candidate]["folds"]
        target_scores: List[torch.Tensor] = []
        pool_scores: List[torch.Tensor] = []
        fold_payload: List[Dict[str, object]] = []
        error = ""
        for fold in sorted(fold_entries, key=lambda row: row["fold"]):
            probe_path = Path(str(fold["probe_path"]))
            pool_path = Path(str(fold["pool_path"]))
            if not probe_path.exists() or not pool_path.exists():
                error = f"missing artifacts for {fold['fold']}"
                break
            try:
                pos = _load_probe_scores(probe_path)
                neg = _load_pool_scores(pool_path)
            except Exception as exc:
                error = f"{fold['fold']}: {exc}"
                break
            target_scores.append(pos)
            pool_scores.append(neg)
            fold_payload.append(
                {
                    "fold": fold["fold"],
                    "n_target": int(pos.numel()),
                    "n_pool": int(neg.numel()),
                    "target_mean": float(pos.mean().item()),
                    "pool_mean": float(neg.mean().item()),
                    "auroc": _pairwise_auroc(pos, neg),
                    "task_root": fold["task_root"],
                }
            )

        row = {
            "task": args.task,
            "stage": args.stage,
            "candidate": candidate,
            "candidate_numeric": grouped[candidate]["candidate_numeric"],
            "n_folds": len(fold_entries),
            "error": error,
        }
        if not error and target_scores and pool_scores:
            concat_target = torch.cat(target_scores, dim=0)
            mean_pool = torch.stack(pool_scores, dim=0).mean(dim=0)
            row.update(
                {
                    "n_target": int(concat_target.numel()),
                    "n_pool": int(mean_pool.numel()),
                    "target_mean": float(concat_target.mean().item()),
                    "pool_mean": float(mean_pool.mean().item()),
                    "mean_gap": float(concat_target.mean().item() - mean_pool.mean().item()),
                    "auroc": _pairwise_auroc(concat_target, mean_pool),
                    "folds": fold_payload,
                }
            )
            ranking_rows.append(row)
        else:
            row.update(
                {
                    "n_target": 0,
                    "n_pool": 0,
                    "target_mean": float("nan"),
                    "pool_mean": float("nan"),
                    "mean_gap": float("nan"),
                    "auroc": float("nan"),
                    "folds": fold_payload,
                }
            )
        rows.append(row)

    if not ranking_rows:
        raise SystemExit(f"no successful {args.stage} candidates found under {run_root}")

    ranking_rows.sort(key=lambda row: (float(row["auroc"]), float(row["mean_gap"]), -float(row["candidate_numeric"])), reverse=True)
    best = ranking_rows[0]
    summary_path = analysis_dir / f"{args.stage}_summary.json"
    best_path = analysis_dir / f"{args.stage}_best.json"
    csv_path = analysis_dir / f"{args.stage}_summary.csv"

    summary_payload = {
        "task": args.task,
        "stage": args.stage,
        "rows": rows,
        "best_candidate": best["candidate"],
        "best_candidate_numeric": best["candidate_numeric"],
        "best_value": best["auroc"],
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    best_path.write_text(json.dumps(best, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "task",
                "stage",
                "candidate",
                "candidate_numeric",
                "n_folds",
                "n_target",
                "n_pool",
                "target_mean",
                "pool_mean",
                "mean_gap",
                "auroc",
                "error",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "task": row["task"],
                    "stage": row["stage"],
                    "candidate": row["candidate"],
                    "candidate_numeric": row["candidate_numeric"],
                    "n_folds": row["n_folds"],
                    "n_target": row["n_target"],
                    "n_pool": row["n_pool"],
                    "target_mean": row["target_mean"],
                    "pool_mean": row["pool_mean"],
                    "mean_gap": row["mean_gap"],
                    "auroc": row["auroc"],
                    "error": row["error"],
                }
            )

    print(f"wrote {summary_path}")
    print(f"wrote {csv_path}")
    print(f"best_{args.stage}={best['candidate']} auroc={best['auroc']:.6f}")


if __name__ == "__main__":
    main()

# TACS: Data Selection via Target-Aligned Paths

Code for **Let the Target Select for Itself: Data Selection via Target-Aligned Paths**.

Targeted data selection depends on the model states used to judge candidate
examples. TACS studies this *reference-path dependence*: changing the warmup
data can change candidate rankings and selected subsets even when the pool,
target task, and scoring rule stay fixed.

TACS builds a short, low-capacity reference path from a compact target-validation
proxy and calibrates its depth to the target task. It ranks candidates by their
normalized loss reduction between the path's endpoints. The path can be reused
across candidate pools, and candidate scoring requires only forward passes.
The selected subset is then used for ordinary fine-tuning. The repository
includes the controlled logistic and vision studies and the instruction-tuning
workflow used in the paper.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
python -m pytest -q tests
```

The synthetic example runs on CPU and generates its own data:

```bash
python experiments/logistic_shift/run_logistic_shift_tacs.py \
  --quick --output runs/logistic_quick.json
```

## What is where?

| Path | Purpose |
| --- | --- |
| `tacs/data_selection/` | Target warmup, candidate loss scoring, LESS and ToV baselines, subset export |
| `tacs/train/` | Instruction-tuning and baseline training |
| `tacs/metrics/` | Loss-drop and ranking metrics |
| `scripts/pipeline/` | Cross-validation splits and ablation summaries |
| `scripts/rebuttal/` | Held-out specificity scoring and calibration choice |
| `experiments/logistic_shift/` | Controlled logistic experiments |
| `experiments/cv_cifar10_noisy/` | Clean and noisy CIFAR-10 experiments |
| `evaluation/eval/` | MMLU, BBH, and TyDiQA evaluation |
| `scripts/analysis/` | Paper figure and cost-analysis utilities |

## Instruction-tuning workflow

1. Put candidate JSONL files and target evaluation data in the layout shown in
   [`configs/data_layout.example.yaml`](configs/data_layout.example.yaml).
2. Create three target-proxy folds with
   `scripts/pipeline/prepare_val_warmup_cv_splits.py`.
3. Train rank-one target trajectories over the paper's learning-rate and depth
   grid. Measure held-out specificity with
   `scripts/rebuttal/alignment_calibration_scorer.py`; select the best complete
   cell with `scripts/rebuttal/summarize_specificity_calibration.py`.
4. Run `python -m tacs.data_selection.val_warmup_loss_gap` on the full target
   proxy at the selected settings, using `--score_metric final_drop` and
   `--score_normalize_by_first`. Export the top 5% with
   `python -m tacs.data_selection.select_from_scores`.
5. Fine-tune on the selected JSONL with `python -m tacs.train.train` and evaluate
   with the task runners in `evaluation/eval/`.

The final-paper Llama-3.2-3B settings, candidate sources, calibration grid,
checkpoint choices, and experiment-to-code map are in
[`docs/reproduce.md`](docs/reproduce.md) and
[`configs/paper_protocol.json`](configs/paper_protocol.json). Run each CLI with `--help` for its full
set of options. Model weights and datasets must be obtained from their original
providers; outputs are written outside this repository.

## Citation and attribution

Please cite the accompanying TACS paper. Instruction-tuning utilities adapt
code from [LESS](https://github.com/princeton-nlp/LESS); license details are in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

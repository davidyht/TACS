# Reproducing the paper

This map follows the ICLR final manuscript. It records the experimental
settings and points to the programs that implement each stage. Large model
weights, datasets, and generated scores are separate inputs.

## Paper experiments

| Paper result | Entry points |
| --- | --- |
| Controlled logistic mixtures and path comparisons | `experiments/logistic_shift/run_logistic_shift_tacs.py`, `run_logistic_tov_style_comparison.py`, plotting scripts in the same directory |
| Clean and 40% noisy CIFAR-10 | `experiments/cv_cifar10_noisy/run_cifar10_noisy.py`, `run_cifar10_noisy_fullft.py`, `make_camera_ready_figs.py` |
| Llama-3.2-3B target selection and Table 1 | `tacs/data_selection/val_warmup_loss_gap.py`, `select_from_scores.py`, `tacs/train/train.py`, `evaluation/eval/{mmlu,bbh,tydiqa}/run_eval.py` |
| LESS and ToV comparisons | `tacs/data_selection/{collect_grad_reps,matching,tov_score}.py`, `tacs/train/relaunch.py` |
| Calibration, depth and rank ablations | `scripts/pipeline/prepare_val_warmup_cv_splits.py`, `scripts/rebuttal/{alignment_calibration_scorer,summarize_specificity_calibration}.py`, `scripts/pipeline/summarize_{depth,rank}_ablation.py` |
| Cost, score and length analyses | `scripts/analysis/` and `scripts/pipeline/summarize_metric_ablation.py` |
| Reference-endpoint perturbation | `scripts/rebuttal/path_invariance_sweep.py` |
| Raw versus normalized endpoint score variants | `scripts/analysis/tacs_score_variant_ablation.py` |
| Qwen2.5-1.5B LESS reference-path diagnostic | Core LESS warmup/scoring tools plus `scripts/analysis/path_sensitivity_receipt.py` for the saved result bundle |

## Llama-3.2-3B protocol

The four candidate sources are Flan V2 (100,000 examples), CoT (100,000),
OASST1 (55,668), and Dolly (15,011). The target proxies are MMLU (285), BBH
(81), and TyDiQA (9). Use the prompt formats and source JSONL schemas expected
by `tacs.data_selection.get_training_dataset` and
`get_validation_dataset`. Set maximum sequence length to 2,048, bfloat16,
attention-projection LoRA, AdamW, and zero weight decay.

**TACS calibration.** Split each target proxy into three held-out folds.
Warm up rank-one LoRA (`r=1`, `alpha=4`, dropout `0.1`) on each fold's training
split for learning rates `{1,2,5}e-5` and depths `{2,4,8,12,16,32,64}`.
For every cell, compute

`cos(g_warmup, g_holdout) - cos(g_warmup, g_reference)`.

The generic reference is 100 fixed Latin-script Aya examples, with no string
overlap with candidate pools. Use the same reference in every cell. Average
specificity across the three folds; ties favor smaller depth, then smaller
learning rate. The final manuscript selected LR `1e-5` for all three tasks,
with depths `2` (MMLU), `2` (BBH), and `64` (TyDiQA). The selection script
rejects missing fold cells and can verify the declared full grid:

```bash
python scripts/rebuttal/summarize_specificity_calibration.py \
  --task-align mmlu=runs/mmlu_alignment.json \
  --task-align bbh=runs/bbh_alignment.json \
  --task-align tydiqa=runs/tydiqa_alignment.json \
  --expected-folds 3 \
  --expect-grid mmlu=1e-05,2e-05,5e-05:2,4,8,12,16,32,64 \
  --expect-grid bbh=1e-05,2e-05,5e-05:2,4,8,12,16,32,64 \
  --expect-grid tydiqa=1e-05,2e-05,5e-05:2,4,8,12,16,32,64 \
  --out-summary runs/calibration_summary.json \
  --out-raw-hp runs/calibration_picks.json
```

`alignment_calibration_scorer.py` defaults to at most 100 examples when
estimating each gradient; retain that setting when matching its saved runs.

**Scoring and selection.** Warm up on the full target proxy with calibrated
settings. Score the candidate pool by normalized endpoint loss drop between
checkpoint 1 and the selected depth. The scoring CLI is
`python -m tacs.data_selection.val_warmup_loss_gap`; use
`--score_metric final_drop --score_normalize_by_first` and set
`--score_ckpt_ids 1 T`. Export the highest-scoring 5% with
`python -m tacs.data_selection.select_from_scores`. Keep each source's row
order unchanged between scoring and subset export.

**Retraining.** Train the selected 5% for four epochs with LoRA `r=128`,
`alpha=512`, dropout `0.1`, effective batch size 32 and a linear warmup.
Choose source-specific learning rates from `{1,2,5}e-5` on random subsets,
then share the choice across methods. Report the validation-selected
checkpoint and the final epoch-four checkpoint on held-out test sets.

**Baselines.** LESS uses a random 5% pool warmup, four checkpoints, rank-128
LoRA and projected Adam-preconditioned gradients of dimension 8,192. ToV uses
the same pool-reference setup and a validation perturbation of `0.1` around
each checkpoint. The main paper compares methods at rank 128 for these
pool-reference warmups; TACS uses rank one for target-reference construction.

## Data and outputs

Keep model weights and datasets outside Git. `configs/data_layout.example.yaml`
shows the expected paths. `tacs/data_selection/select_from_scores.py` writes a
selected JSONL and optional metadata. The experiment runners write JSON
summaries; plotting utilities consume those summaries. Generate large scores
and checkpoints under a local `runs/` directory, which Git ignores.

"""Measure candidate-ranking changes after reference-endpoint perturbations."""

import argparse
import glob
import json
import os

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from peft import LoraConfig, TaskType, get_peft_model

from tacs.data_selection.val_warmup_loss_gap import (
    _iter_trainable_lora_params,
    _prepare_candidate_dataset,
    _load_warmup_meta,
)
from tacs.data_selection.get_info import load_model as load_checkpoint_model
from tacs.data_selection.get_validation_dataset import get_dataset
from tacs.train.model_arguments import add_padding_to_tokenizer


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--warmup-ckpt-dir", required=True,
                    help="dir holding step_*.pt and meta.json from the val warmup")
    ap.add_argument("--step-first", type=int, default=1, help="baseline checkpoint (l_1)")
    ap.add_argument("--step-last", type=int, required=True, help="endpoint checkpoint (l_T)")
    ap.add_argument("--pool-file", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--task", required=True, choices=["tydiqa", "mmlu", "bbh"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-model", default=None)
    ap.add_argument("--n-candidates", type=int, default=1500)
    ap.add_argument("--n-pool-grad", type=int, default=64,
                    help="pool examples used to estimate the pool mean gradient")
    ap.add_argument("--magnitudes", default="0.25 0.5 1.0 2.0 4.0",
                    help="multiples of ||theta_T - theta_1||")
    ap.add_argument("--top-p", type=float, default=0.05)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--dump-base-scores", action="store_true",
                    help="also save the unperturbed per-candidate score vector and the pool/val "
                         "gradient alignments for follow-up diagnostics")
    return ap.parse_args()


def build_peft(base_model, meta, trc, tokenizer=None):
    base = load_checkpoint_model(base_model, trust_remote_code=trc)
    # add_padding_to_tokenizer appends a pad token, so the tokenizer can be longer
    # than the model's embedding table; without this the pad id indexes out of range
    # and CUDA raises a device-side assert.  Mirrors val_warmup_loss_gap.py:1622.
    if tokenizer is not None:
        emb = base.get_input_embeddings().weight.shape[0]
        if len(tokenizer) > emb:
            try:
                base.resize_token_embeddings(len(tokenizer), mean_resizing=False)
            except TypeError:
                base.resize_token_embeddings(len(tokenizer))
    cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM, bias="none",
        r=int(meta["lora_r"]), lora_alpha=int(meta["lora_alpha"]),
        lora_dropout=float(meta.get("lora_dropout", 0.1)),
        target_modules=list(meta["lora_target_modules"]),
        rank_pattern=dict(meta.get("lora_rank_pattern") or {}),
        alpha_pattern=dict(meta.get("lora_alpha_pattern") or {}),
    )
    return get_peft_model(base, cfg)


def flat(model):
    return torch.cat([p.detach().float().view(-1).cpu()
                      for _, p in _iter_trainable_lora_params(model)])


def write_flat(model, vec):
    i = 0
    for _, p in _iter_trainable_lora_params(model):
        n = p.numel()
        p.data.copy_(vec[i:i + n].view_as(p).to(p.dtype).to(p.device))
        i += n
    assert i == vec.numel(), f"size mismatch {i} vs {vec.numel()}"


@torch.no_grad()
def per_example_losses(model, ds, tok, device, bs):
    """Mean NLL over response tokens, one value per example."""
    coll = DataCollatorForSeq2Seq(tokenizer=tok, model=model, padding="longest")
    dl = DataLoader(ds, batch_size=bs, shuffle=False, collate_fn=coll)
    out = []
    for batch in dl:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        logits = model(input_ids=batch["input_ids"],
                       attention_mask=batch["attention_mask"]).logits
        lab = batch["labels"]
        sl, sh = logits[:, :-1].float(), lab[:, 1:]
        tokloss = torch.nn.functional.cross_entropy(
            sl.reshape(-1, sl.size(-1)), sh.reshape(-1),
            ignore_index=-100, reduction="none").view(sh.shape)
        mask = (sh != -100).float()
        out.append(((tokloss * mask).sum(1) / mask.sum(1).clamp_min(1)).cpu())
    return torch.cat(out)


def grad_flat(model, ds, tok, device, max_ex, per_example=False):
    """Gradient of the mean loss in trainable-LoRA space (or a stack of per-example grads)."""
    if max_ex and len(ds) > max_ex:
        ds = ds.select(range(max_ex))
    coll = DataCollatorForSeq2Seq(tokenizer=tok, model=model, padding="longest")
    dl = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=coll)
    rows = []
    model.zero_grad(set_to_none=True)
    for batch in dl:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        loss = model(**batch).loss
        (loss if per_example else loss / len(dl)).backward()
        if per_example:
            rows.append(torch.cat([p.grad.detach().float().view(-1).cpu()
                                   for _, p in _iter_trainable_lora_params(model)
                                   if p.grad is not None]))
            model.zero_grad(set_to_none=True)
    if per_example:
        return torch.stack(rows)
    g = torch.cat([p.grad.detach().float().view(-1).cpu()
                   for _, p in _iter_trainable_lora_params(model) if p.grad is not None])
    model.zero_grad(set_to_none=True)
    return g


def spearman(a, b):
    ra = a.argsort().argsort().float()
    rb = b.argsort().argsort().float()
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra @ rb) / (ra.norm() * rb.norm()).clamp_min(1e-12))


def main():
    a = parse_args()
    torch.manual_seed(a.seed)
    meta = _load_warmup_meta(a.warmup_ckpt_dir)
    base_model = a.base_model or meta["model_name_or_path"]
    chat_format = meta.get("chat_format", "tokenizer")
    max_seq = int(meta.get("max_seq_length", 2048))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=a.trust_remote_code)
    add_padding_to_tokenizer(tok)
    model = build_peft(base_model, meta, a.trust_remote_code, tokenizer=tok).to(device)
    model._align_tok = tok

    ds_pool = _prepare_candidate_dataset(a.pool_file, tok, max_seq, percentage=1.0,
                                         seed=42, chat_format=chat_format)
    if len(ds_pool) > a.n_candidates:
        g = torch.Generator().manual_seed(a.seed)
        idx = torch.randperm(len(ds_pool), generator=g)[:a.n_candidates].tolist()
        ds_pool = ds_pool.select(idx)
    else:
        idx = list(range(len(ds_pool)))
    ds_val = get_dataset(a.task, data_dir=a.data_dir, tokenizer=tok, max_length=max_seq,
                         use_chat_format=True, chat_format=chat_format,
                         mmlu_n_shot=int(meta.get("mmlu_n_shot", 1) or 1))

    sd_first = torch.load(os.path.join(a.warmup_ckpt_dir, f"step_{a.step_first}.pt"),
                          map_location="cpu")
    sd_last = torch.load(os.path.join(a.warmup_ckpt_dir, f"step_{a.step_last}.pt"),
                         map_location="cpu")

    model.load_state_dict(sd_first, strict=False)
    theta_1 = flat(model)
    model.eval()
    l_first = per_example_losses(model, ds_pool, tok, device, a.batch_size)

    model.load_state_dict(sd_last, strict=False)
    theta_T = flat(model)
    scale = float((theta_T - theta_1).norm())
    model.eval()
    l_last = per_example_losses(model, ds_pool, tok, device, a.batch_size)
    base_score = (l_first - l_last) / l_first.clamp_min(1e-8)

    # ---- direction basis, all built AT theta_T ----
    model.train()
    g_val = grad_flat(model, ds_val, tok, device, max_ex=128)
    g_pool = grad_flat(model, ds_pool, tok, device, max_ex=a.n_pool_grad)
    G_val = grad_flat(model, ds_val, tok, device, max_ex=min(64, len(ds_val)),
                      per_example=True)
    _, _, Vh = torch.linalg.svd(G_val - G_val.mean(0, keepdim=True), full_matrices=False)
    val_top = Vh[0]
    model.eval()

    def u(x):
        return x / x.norm().clamp_min(1e-12)

    iso = u(torch.randn_like(theta_T))
    B = torch.stack([u(g_val), u(g_pool), u(val_top)])
    Q, _ = torch.linalg.qr(B.T)
    comp = u(iso - Q @ (Q.T @ iso))
    dirs = {"iso": iso, "val_grad": u(g_val), "val_top": u(val_top),
            "pool_grad": u(g_pool), "complement": comp}

    res = {"meta": {"base_model": base_model, "task": a.task,
                    "n_candidates": int(len(ds_pool)), "pool_file": a.pool_file,
                    "step_first": a.step_first, "step_last": a.step_last,
                    "displacement_norm_theta_T_minus_theta_1": scale,
                    "trainable_dim": int(theta_T.numel())},
           "cos_between_directions": {
               f"{i}~{j}": float(u(dirs[i]) @ u(dirs[j]))
               for n_, i in enumerate(dirs) for j in list(dirs)[n_ + 1:]},
           "sweep": []}

    k = max(1, int(round(a.top_p * len(ds_pool))))
    base_top = set(base_score.topk(k).indices.tolist())

    for name, v in dirs.items():
        for m in [float(x) for x in a.magnitudes.split()]:
            write_flat(model, theta_T + m * scale * v)
            l = per_example_losses(model, ds_pool, tok, device, a.batch_size)
            s = (l_first - l) / l_first.clamp_min(1e-8)
            top = set(s.topk(k).indices.tolist())
            row = {"direction": name, "magnitude": m,
                   "spearman_vs_base": spearman(base_score, s),
                   "topk_overlap": len(base_top & top) / k,
                   "score_sd": float(s.std()), "base_score_sd": float(base_score.std())}
            res["sweep"].append(row)
            print(json.dumps(row), flush=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)

    if a.dump_base_scores:
        # Enables the cross-model pool-contamination test: is the selected set displaced in
        # POOL-alignment beyond what its target-alignment explains?  Needs the raw per-candidate
        # score plus each candidate's cosine to the pool and target directions.
        write_flat(model, theta_T)
        model.train()
        Gc = grad_flat(model, ds_pool, tok, device, max_ex=min(256, len(ds_pool)),
                       per_example=True)
        model.eval()
        Gc = Gc / Gc.norm(dim=1, keepdim=True).clamp_min(1e-12)
        side = a.out.replace(".json", "") + "_basescores.pt"
        torch.save({"base_score": base_score,
                    "pool_align": (Gc @ u(g_pool)).cpu(),
                    "target_align": (Gc @ u(g_val)).cpu(),
                    "n_grad_examples": int(Gc.shape[0]),
                    "pool_subsample_indices": idx,
                    "meta": res["meta"]}, side)
        print(f"[done] {side}")
    print(f"[done] {a.out}")


if __name__ == "__main__":
    main()

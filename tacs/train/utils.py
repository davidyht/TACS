import json
import time
from typing import Callable, Iterable, Optional

import torch


def set_seed(seed: Optional[int]):
    if seed is None:
        return
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_loss_on_dataloader(model: torch.nn.Module, dataloader: Iterable, device: torch.device, verbose: bool = True):
    model.eval()
    total_loss = 0.0
    total_examples = 0
    skipped_batches = 0
    batch_losses = []  # for debugging
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            # prefer outputs.loss if available
            loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
            # convert to python float safely and guard against NaNs/Infs
            try:
                loss_val = float(loss.item())
            except Exception:
                # if we can't extract scalar, skip this batch silently
                skipped_batches += 1
                continue
            batch_size = batch.get("input_ids", batch.get("labels")).shape[0]
            if not (loss_val == loss_val) or loss_val == float('inf') or loss_val == float('-inf'):
                # loss is NaN or infinite; skip this batch silently
                skipped_batches += 1
                continue
            batch_losses.append(loss_val)
            total_loss += loss_val * batch_size
            total_examples += batch_size
    if total_examples == 0:
        return float('nan')
    avg_loss = total_loss / float(total_examples)
    if verbose:
        print(f"  [evaluate_loss] total_examples={total_examples}, skipped={skipped_batches}, avg_loss={avg_loss:.6f}, first_5_losses={batch_losses[:5]}", flush=True)
    return avg_loss


def estimate_full_gradient_norm(model: torch.nn.Module,
                                dataloader: Iterable,
                                device: torch.device,
                                param_filter: Optional[Callable[[str], bool]] = None,
                                transfer_interval: int = 10,
                                verbose: bool = False,
                                heartbeat_interval: int = 10,
                                precond_concat: Optional[torch.Tensor] = None):
    """
    Estimate squared L2 norm of the average gradient over the dataloader.

    By default, only parameters whose name contains 'lora' are included (to reduce memory).
    Returns: scalar (float) representing || avg_grad ||_2^2 and metadata dict.
    """
    model.train()  # need grads
    # build list of params to monitor
    params = [p for n, p in model.named_parameters() if p.requires_grad and (param_filter(n) if param_filter else ('lora' in n or 'Lora' in n))]
    if len(params) == 0:
        raise RuntimeError("No trainable parameters matched for gradient estimation. Check param_filter or model's parameter names.")

    # Strategy: accumulate gradients on GPU into accum_gpu tensors and only
    # transfer to CPU accumulators every `transfer_interval` batches to reduce
    # GPU->CPU copy frequency. CPU accumulators use double precision to improve
    # numerical stability.
    accum_gpu = [torch.zeros_like(p, device=device) for p in params]
    accum_cpu = [torch.zeros_like(p.detach().cpu(), dtype=torch.double).pin_memory() for p in params]
    total_examples = 0
    batch_count = 0
    start_time = time.time()

    # try to determine total batches for progress bar
    total_batches = None
    try:
        total_batches = len(dataloader)
    except Exception:
        total_batches = None

    # Use tqdm for clean progress display
    from tqdm.auto import tqdm
    pbar = tqdm(dataloader, total=total_batches, desc="estimate_grad", ncols=100)

    for batch in pbar:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
        batch_size = batch.get("input_ids", batch.get("labels")).shape[0]
        # compute gradients of sum(losses) over batch
        (loss * batch_size).backward()
        # accumulate grads onto GPU buffers
        idx = 0
        for n, p in model.named_parameters():
            if p.requires_grad and (param_filter(n) if param_filter else ('lora' in n or 'Lora' in n)):
                if p.grad is not None:
                    accum_gpu[idx].add_(p.grad.detach())
                idx += 1

        batch_count += 1
        total_examples += batch_size
        model.zero_grad()

        # Update progress bar with current stats
        if batch_count % max(1, heartbeat_interval) == 0:
            elapsed = time.time() - start_time
            avg_batch = elapsed / max(1, batch_count)
            pbar.set_postfix({'ex': total_examples, 'batch_sec': f'{avg_batch:.3f}'})

        # Periodically flush GPU accumulators to CPU accumulators
        if transfer_interval is not None and batch_count % max(1, transfer_interval) == 0:
            for i in range(len(params)):
                if accum_gpu[i].abs().sum().item() != 0:
                    # move chunk to CPU and add to double-precision CPU accumulator
                    tmp = accum_gpu[i].detach().cpu()
                    accum_cpu[i].add_(tmp.double())
                    accum_gpu[i].zero_()

    pbar.close()

    # flush any remaining GPU accumulators
    if batch_count % max(1, transfer_interval) != 0:
        for i in range(len(params)):
            if accum_gpu[i].abs().sum().item() != 0:
                tmp = accum_gpu[i].detach().cpu()
                accum_cpu[i].add_(tmp.double())
                accum_gpu[i].zero_()

    # average on CPU
    avg_accum = [g / float(max(1, total_examples)) for g in accum_cpu]
    # compute squared l2 norm (double precision) or quadratic form g^T P g if precond provided
    norm2_sq = 0.0
    if precond_concat is None:
        for g in avg_accum:
            norm2_sq += float(g.view(-1).double().pow(2).sum().item())
    else:
        # precond_concat expected to be a 1D tensor of length equal to total params numel
        try:
            pc = precond_concat.view(-1).double()
            pos = 0
            for g in avg_accum:
                n = g.numel()
                slice_pc = pc[pos:pos + n]
                if slice_pc.numel() != n:
                    raise RuntimeError("precond_concat length does not match parameter sizes")
                g_flat = g.view(-1).double()
                # quadratic form contribution: sum_i g_i^2 * pc_i
                norm2_sq += float((g_flat.pow(2) * slice_pc).sum().item())
                pos += n
        except Exception as e:
            print(f"estimate_full_gradient_norm: failed to apply preconditioner: {e}; falling back to standard ||g||^2", flush=True)
            for g in avg_accum:
                norm2_sq += float(g.view(-1).double().pow(2).sum().item())

    meta = {
        "num_params": len(params),
        "total_examples": int(total_examples),
        "batches": int(batch_count),
        "transfer_interval": int(transfer_interval) if transfer_interval is not None else None,
    }
    return norm2_sq, meta


def estimate_full_gradient(model: torch.nn.Module,
                           dataloader: Iterable,
                           device: torch.device,
                           param_filter: Optional[Callable[[str], bool]] = None,
                           transfer_interval: int = 10,
                           verbose: bool = False,
                           heartbeat_interval: int = 10,
                           precond_concat: Optional[torch.Tensor] = None):
    """
    Estimate the average gradient (per-parameter tensors) over the dataloader.

    Returns a tuple: (avg_grads, norm2_sq, meta)
    - avg_grads: list of CPU tensors (double) in the same order as the parameters matched
    - norm2_sq: squared L2 norm of the average gradient (float)
    - meta: metadata dict same as in estimate_full_gradient_norm
    """
    model.train()
    params = [p for n, p in model.named_parameters() if p.requires_grad and (param_filter(n) if param_filter else ('lora' in n or 'Lora' in n))]
    if len(params) == 0:
        raise RuntimeError("No trainable parameters matched for gradient estimation. Check param_filter or model's parameter names.")

    accum_gpu = [torch.zeros_like(p, device=device) for p in params]
    accum_cpu = [torch.zeros_like(p.detach().cpu(), dtype=torch.double).pin_memory() for p in params]
    total_examples = 0
    batch_count = 0
    start_time = time.time()

    total_batches = None
    try:
        total_batches = len(dataloader)
    except Exception:
        total_batches = None

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
        batch_size = batch.get("input_ids", batch.get("labels")).shape[0]
        (loss * batch_size).backward()

        idx = 0
        for n, p in model.named_parameters():
            if p.requires_grad and (param_filter(n) if param_filter else ('lora' in n or 'Lora' in n)):
                if p.grad is not None:
                    accum_gpu[idx].add_(p.grad.detach())
                idx += 1

        batch_count += 1
        total_examples += batch_size
        model.zero_grad()

        if heartbeat_interval is not None and heartbeat_interval > 0 and (batch_count % heartbeat_interval == 0):
            now = time.time()
            elapsed = now - start_time
            avg_batch = elapsed / max(1, batch_count)
            msg = f"estimate_grad: processed_batches={batch_count} processed_examples={total_examples} avg_sec_per_batch={avg_batch:.3f}"
            if total_batches is not None:
                remaining = max(0, total_batches - batch_count)
                eta = remaining * avg_batch
                msg += f" eta_s={int(eta)} total_batches={total_batches}"
            if verbose:
                print(msg, flush=True)
            else:
                print(msg, flush=True)

        if transfer_interval is not None and batch_count % max(1, transfer_interval) == 0:
            if verbose:
                pass  # Silent flush to CPU
            for i in range(len(params)):
                if accum_gpu[i].abs().sum().item() != 0:
                    tmp = accum_gpu[i].detach().cpu()
                    accum_cpu[i].add_(tmp.double())
                    accum_gpu[i].zero_()

    if batch_count % max(1, transfer_interval) != 0:
        for i in range(len(params)):
            if accum_gpu[i].abs().sum().item() != 0:
                tmp = accum_gpu[i].detach().cpu()
                accum_cpu[i].add_(tmp.double())
                accum_gpu[i].zero_()

    avg_accum = [g / float(max(1, total_examples)) for g in accum_cpu]
    # compute norm or quadratic form depending on precond
    norm2_sq = 0.0
    if precond_concat is None:
        for g in avg_accum:
            norm2_sq += float(g.view(-1).double().pow(2).sum().item())
    else:
        try:
            pc = precond_concat.view(-1).double()
            pos = 0
            for g in avg_accum:
                n = g.numel()
                slice_pc = pc[pos:pos + n]
                if slice_pc.numel() != n:
                    raise RuntimeError("precond_concat length does not match parameter sizes")
                g_flat = g.view(-1).double()
                norm2_sq += float((g_flat.pow(2) * slice_pc).sum().item())
                pos += n
        except Exception as e:
            print(f"estimate_full_gradient: failed to apply preconditioner: {e}; falling back to standard ||g||^2", flush=True)
            for g in avg_accum:
                norm2_sq += float(g.view(-1).double().pow(2).sum().item())

    meta = {
        "num_params": len(params),
        "total_examples": int(total_examples),
        "batches": int(batch_count),
        "transfer_interval": int(transfer_interval) if transfer_interval is not None else None,
    }
    return avg_accum, norm2_sq, meta

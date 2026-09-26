import json
import os
import glob
from hashlib import md5
from typing import Iterable, List, Optional

import torch
import torch.nn.functional as F
from functorch import grad, make_functional_with_buffers, vmap
from peft import PeftModel
from torch import Tensor
from torch.nn.functional import normalize
from tqdm import tqdm
from trak.projectors import BasicProjector, CudaProjector, ProjectionType
from transformers import RobertaModel


def prepare_batch(batch, device=torch.device("cuda:0")):
    """ Move the batch to the device. """
    for key in batch:
        batch[key] = batch[key].to(device)


def _clear_model_grads(model: torch.nn.Module) -> None:
    """Clear grads with a best-effort set_to_none path for lower peak memory."""
    try:
        model.zero_grad(set_to_none=True)
    except TypeError:
        model.zero_grad()


def _truncate_batch_tokens(batch: dict, target_len: int) -> int:
    """Truncate token-like tensors in batch to target_len; return previous seq len."""
    prev_len = -1
    for key in ("input_ids", "attention_mask", "labels"):
        tensor = batch.get(key)
        if torch.is_tensor(tensor) and tensor.dim() >= 2:
            if prev_len < 0:
                prev_len = int(tensor.shape[-1])
            if tensor.shape[-1] > target_len:
                batch[key] = tensor[..., :target_len].contiguous()
    return prev_len


def _get_batch_seq_len(batch: dict) -> int:
    """Best-effort current token length from common batch tensor keys."""
    for key in ("input_ids", "attention_mask", "labels"):
        tensor = batch.get(key)
        if torch.is_tensor(tensor) and tensor.dim() >= 2:
            return int(tensor.shape[-1])
    return -1


def _is_cuda_oom_or_sdpa(exc: Exception) -> bool:
    """Return True when exception looks like CUDA OOM / SDPA runtime instability."""
    msg = str(exc).lower()
    oom_type = getattr(torch, "OutOfMemoryError", None)
    return (
        (oom_type is not None and isinstance(exc, oom_type))
        or "out of memory" in msg
        or "cuda out of memory" in msg
        or "mha_graph->execute" in msg
        or "scaled_dot_product_attention" in msg
    )


def get_max_saved_index(output_dir: str, prefix="reps") -> int:
    """
    Retrieve the highest index for which the data (either representation or gradients) has been stored.

    Args:
        output_dir (str): The output directory.
        prefix (str, optional): The prefix of the files, [reps | grads]. Defaults to "reps".

    Returns:
        int: The maximum representation index, or -1 if no index is found.
    """

    files = [file for file in os.listdir(
        output_dir) if file.startswith(prefix)]
    index = [int(file.split(".")[0].split("-")[1])
             for file in files]  # e.g., output_dir/reps-100.pt
    return max(index) if len(index) > 0 else -1


def get_output(model,
               weights: Iterable[Tensor],
               buffers: Iterable[Tensor],
               input_ids=None,
               attention_mask=None,
               labels=None,
               ) -> Tensor:
    logits = model(weights, buffers, *(input_ids.unsqueeze(0),
                   attention_mask.unsqueeze(0))).logits
    labels = labels.unsqueeze(0)
    loss_fct = F.cross_entropy
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = loss_fct(
        shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1))
    return loss


def get_trak_projector(device: torch.device):
    """ Get trak projectors (see https://github.com/MadryLab/trak for details) """
    require_cuda_projector = os.environ.get("LESS_REQUIRE_CUDA_PROJECTOR", "0") in {"1", "true", "True"}
    # fast_jl may fail to import if torch/nvidia shared-library folders are not
    # in LD_LIBRARY_PATH. Best effort patch before probing the CUDA projector.
    try:
        torch_root = os.path.dirname(torch.__file__)
        site_root = os.path.dirname(torch_root)
        lib_paths = [os.path.join(torch_root, "lib")]
        lib_paths.extend(sorted(glob.glob(os.path.join(site_root, "nvidia", "*", "lib"))))
        existing = os.environ.get("LD_LIBRARY_PATH", "")
        existing_parts = [p for p in existing.split(":") if p]
        missing = [p for p in lib_paths if os.path.isdir(p) and p not in existing_parts]
        if missing:
            os.environ["LD_LIBRARY_PATH"] = ":".join(missing + existing_parts)
            print(f"Patched LD_LIBRARY_PATH for fast_jl with {len(missing)} paths")
    except Exception as e:
        print(f"Warning: failed to patch LD_LIBRARY_PATH for fast_jl: {e}")

    try:
        num_sms = torch.cuda.get_device_properties(
            device.index).multi_processor_count
        import fast_jl

        # test run to catch at init time if projection goes through
        fast_jl.project_rademacher_8(torch.zeros(
            8, 1_000, device=device), 512, 0, num_sms)
        projector = CudaProjector
        print("Using CudaProjector")
    except Exception as e:
        if require_cuda_projector:
            raise RuntimeError(
                f"LESS_REQUIRE_CUDA_PROJECTOR=1 but CudaProjector init failed: {e}"
            ) from e
        projector = BasicProjector
        print(f"Using BasicProjector (CudaProjector init failed: {e})")
    return projector


def get_number_of_params(model):
    """ Make sure that only lora parameters require gradients in peft models. """
    if isinstance(model, PeftModel):
        names = [n for n, p in model.named_parameters(
        ) if p.requires_grad and "lora" not in n]
        assert len(names) == 0
    num_params = sum([p.numel()
                     for p in model.parameters() if p.requires_grad])
    print(f"Total number of parameters that require gradients: {num_params}")
    return num_params


def obtain_gradients(model, batch):
    """ obtain gradients. """
    loss = model(**batch).loss
    loss.backward()
    vectorized_grads = torch.cat(
        [p.grad.view(-1) for p in model.parameters() if p.grad is not None])
    return vectorized_grads


def obtain_sign_gradients(model, batch):
    """ obtain gradients with sign. """
    loss = model(**batch).loss
    loss.backward()

    # Instead of concatenating the gradients, concatenate their signs
    vectorized_grad_signs = torch.cat(
        [torch.sign(p.grad).view(-1) for p in model.parameters() if p.grad is not None])

    return vectorized_grad_signs


def obtain_gradients_with_adam(model, batch, avg, avg_sq):
    """ obtain gradients with adam optimizer states. """
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-08

    loss = model(**batch).loss
    loss.backward()

    vectorized_grads = torch.cat(
        [p.grad.view(-1) for n, p in model.named_parameters() if p.grad is not None])

    updated_avg = beta1 * avg + (1 - beta1) * vectorized_grads
    updated_avg_sq = beta2 * avg_sq + (1 - beta2) * vectorized_grads ** 2
    # Guard against corrupted/invalid optimizer moments.
    updated_avg = torch.nan_to_num(updated_avg, nan=0.0, posinf=0.0, neginf=0.0)
    updated_avg_sq = torch.nan_to_num(updated_avg_sq, nan=0.0, posinf=0.0, neginf=0.0)
    updated_avg_sq = torch.clamp(updated_avg_sq, min=0.0)
    denom = torch.sqrt(updated_avg_sq + eps)
    vectorized_grads = updated_avg / denom
    vectorized_grads = torch.nan_to_num(vectorized_grads, nan=0.0, posinf=0.0, neginf=0.0)

    return vectorized_grads


def prepare_optimizer_state(model, optimizer_state, device):
    def _sanitize_moment(t: torch.Tensor, nonneg: bool = False) -> torch.Tensor:
        x = t.detach().view(-1).to(torch.float32)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if nonneg:
            x = torch.clamp(x, min=0.0)
        return x

    named_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    names = [n for n, _ in named_params]
    # If a full optimizer state dict is provided, attempt to load it into a temp optimizer
    # to map integer-keyed states onto current parameters (best-effort).
    if isinstance(optimizer_state, dict) and "state" in optimizer_state and "param_groups" in optimizer_state:
        try:
            state_keys = list(optimizer_state.get("state", {}).keys())
            # Only use this path if keys are not clearly name-keyed strings
            if len(state_keys) > 0 and not all(isinstance(k, str) for k in state_keys):
                params = [p for _, p in named_params]
                if len(params) > 0:
                    candidate_groupings = []
                    state_groups = optimizer_state.get("param_groups", []) or []

                    # Candidate 1: match saved group lengths exactly (best chance to satisfy load_state_dict).
                    try:
                        lengths = [len(g.get("params", [])) for g in state_groups]
                        if len(lengths) > 0 and sum(lengths) == len(params):
                            groups = []
                            start = 0
                            for g, ln in zip(state_groups, lengths):
                                end = start + ln
                                gg = {"params": params[start:end]}
                                if "weight_decay" in g:
                                    gg["weight_decay"] = float(g["weight_decay"])
                                groups.append(gg)
                                start = end
                            candidate_groupings.append(groups)
                    except Exception:
                        pass

                    # Candidate 2: Trainer-like decay/non-decay grouping.
                    no_decay_terms = ("bias", "layernorm.weight", "layer_norm.weight", "norm.weight")
                    decay = [p for n, p in named_params if not any(term in n.lower() for term in no_decay_terms)]
                    no_decay = [p for n, p in named_params if any(term in n.lower() for term in no_decay_terms)]
                    if len(decay) > 0 and len(no_decay) > 0:
                        candidate_groupings.append(
                            [
                                {"params": decay, "weight_decay": 0.01},
                                {"params": no_decay, "weight_decay": 0.0},
                            ]
                        )

                    # Candidate 3: single group.
                    candidate_groupings.append([{"params": params}])

                    tried = set()
                    last_err = None
                    for groups in candidate_groupings:
                        sig = tuple(len(g["params"]) for g in groups)
                        if sig in tried:
                            continue
                        tried.add(sig)
                        try:
                            tmp_optim = torch.optim.AdamW(groups, lr=1e-3)
                            tmp_optim.load_state_dict(optimizer_state)
                        except Exception as e:
                            last_err = e
                            continue

                        avg_list = []
                        avg_sq_list = []
                        missing = 0
                        for n, p in named_params:
                            st = tmp_optim.state.get(p, {})
                            if "exp_avg" in st and "exp_avg_sq" in st:
                                avg_list.append(_sanitize_moment(st["exp_avg"], nonneg=False))
                                avg_sq_list.append(_sanitize_moment(st["exp_avg_sq"], nonneg=True))
                            else:
                                avg_list.append(torch.zeros(p.numel(), dtype=torch.float32))
                                avg_sq_list.append(torch.zeros(p.numel(), dtype=torch.float32))
                                missing += 1
                        avg = torch.cat(avg_list).to(device)
                        avg_sq = torch.cat(avg_sq_list).to(device)
                        print(
                            "Prepared optimizer moments via optimizer.load_state_dict; "
                            f"group_sizes={sig}, missing {missing} params."
                        )
                        return avg, avg_sq

                    if last_err is not None:
                        print(f"Warning: failed to load optimizer state dict into temp optimizer: {last_err}")
        except Exception as e:
            print(f"Warning: failed to load optimizer state dict into temp optimizer: {e}")
    try:
        # Preferred format: optimizer_state keyed by parameter *names*
        avg = torch.cat([_sanitize_moment(optimizer_state[n]["exp_avg"], nonneg=False) for n in names])
        avg_sq = torch.cat([_sanitize_moment(optimizer_state[n]["exp_avg_sq"], nonneg=True)
                           for n in names])
        avg = avg.to(device)
        avg_sq = avg_sq.to(device)
        return avg, avg_sq
    except Exception:
        # Try more robust mapping strategies before giving up:
        # 1) Normalize common prefixes (e.g., remove 'base_model.', 'model.')
        # 2) Match by suffix (longest-suffix match) when exact name not present
        try:
            state = optimizer_state
            # state might be nested under 'state' key
            if isinstance(state, dict) and 'state' in state:
                state = state['state']

            state_keys = [str(k) for k in state.keys()]

            def normalize_key(k: str):
                # remove common wrapper prefixes that appear when saving from different wrappers
                if not isinstance(k, str):
                    k = str(k)
                # collapse repeated 'model.' occurrences
                k = k.replace('model.model.', 'model.')
                # remove common prefixes
                for pref in ['base_model.model.', 'base_model.', 'model.', 'module.']:
                    if k.startswith(pref):
                        k = k[len(pref):]
                        break
                # remove '.default' artifacts introduced by some PEFT serializers
                k = k.replace('.default', '')
                # normalize any double dots
                k = k.replace('..', '.')
                return k

            norm_state = {normalize_key(k): k for k in state_keys}

            mapped = {}
            unmapped = []
            # For each parameter name in the model (that requires grad), try to find a matching state key
            for pname, p in [(n, p) for n, p in model.named_parameters() if p.requires_grad]:
                if pname in state:
                    mapped[pname] = pname
                    continue
                npname = normalize_key(pname)
                if npname in norm_state:
                    mapped[pname] = norm_state[npname]
                    continue

                # suffix match: find state key whose normalized form endswith the normalized param name
                candidates = [k for k in norm_state.keys() if k.endswith(npname)]
                if len(candidates) == 1:
                    mapped[pname] = norm_state[candidates[0]]
                    continue
                elif len(candidates) > 1:
                    # pick the longest candidate (most specific)
                    best = max(candidates, key=len)
                    mapped[pname] = norm_state[best]
                    continue

                # no mapping found
                unmapped.append(pname)

            total_params = sum([p.numel() for p in model.parameters() if p.requires_grad])

            if len(mapped) == 0:
                print("Warning: optimizer state mapping by parameter name failed.")
                print("Falling back to zero-initialized Adam moments. This may change")
                print("the resulting projected gradients compared to using saved optimizer state.")
                print("If you expect non-zero optimizer moments, ensure your optimizer file")
                print("is saved with parameter-name keys or provide an explicit mapping.")
                avg = torch.zeros(total_params, dtype=torch.float32, device=device)
                avg_sq = torch.zeros(total_params, dtype=torch.float32, device=device)
                return avg, avg_sq

            # If some params are unmapped, provide short diagnostics to help debugging
            if len(unmapped) > 0:
                sample_unmapped = unmapped[:20]
                print(f"Optimizer mapping: {len(mapped)} matched, {len(unmapped)} unmatched parameters.")
                print("Sample unmatched parameter names:")
                for s in sample_unmapped:
                    print("  ", s)
                # show a small sample of state keys to inspect naming patterns
                sample_state_keys = list(state.keys())[:40]
                print("Sample optimizer state keys (first 40):")
                for k in sample_state_keys:
                    print("  ", k)

                # Additional diagnostics: extract layer indices present in state keys and in model param names
                import re
                def extract_layers_from_keys(keys):
                    layers = set()
                    for k in keys:
                        if not isinstance(k, str):
                            k = str(k)
                        m = re.search(r"layers\.(\d+)", k)
                        if m:
                            layers.add(int(m.group(1)))
                    return layers

                state_layers = extract_layers_from_keys(state.keys())
                model_layers = extract_layers_from_keys([n for n, _ in model.named_parameters()])
                sorted_state = sorted(state_layers)
                sorted_model = sorted(model_layers)
                print(f"Layer indices in optimizer state: {sorted_state[:20]} (count {len(sorted_state)})")
                print(f"Layer indices in model params: {sorted_model[:20]} (count {len(sorted_model)})")
                missing_layers = sorted(set(model_layers) - set(state_layers))
                extra_layers = sorted(set(state_layers) - set(model_layers))
                print(f"Layers present in model but missing in optimizer state (sample up to 20): {missing_layers[:20]}")
                print(f"Layers present in optimizer state but missing in model (sample up to 20): {extra_layers[:20]}")

            # Build concatenated avg and avg_sq according to model param order
            avg_list = []
            avg_sq_list = []
            missing_count = 0
            for pname, p in [(n, p) for n, p in model.named_parameters() if p.requires_grad]:
                numel = p.numel()
                if pname in mapped:
                    state_key = mapped[pname]
                    try:
                        e = _sanitize_moment(state[state_key]['exp_avg'], nonneg=False)
                        esq = _sanitize_moment(state[state_key]['exp_avg_sq'], nonneg=True)
                        # if shapes mismatch, fallback to zeros for this param
                        if e.numel() != numel or esq.numel() != numel:
                            print(f"Warning: shape mismatch for param {pname}: expected {numel}, got {e.numel()} / {esq.numel()}")
                            avg_list.append(torch.zeros(numel, dtype=torch.float32))
                            avg_sq_list.append(torch.zeros(numel, dtype=torch.float32))
                            missing_count += 1
                        else:
                            avg_list.append(e)
                            avg_sq_list.append(esq)
                    except Exception as ex:
                        print(f"Warning: failed to read optimizer moments for {pname}: {ex}")
                        avg_list.append(torch.zeros(numel, dtype=torch.float32))
                        avg_sq_list.append(torch.zeros(numel, dtype=torch.float32))
                        missing_count += 1
                else:
                    avg_list.append(torch.zeros(numel, dtype=torch.float32))
                    avg_sq_list.append(torch.zeros(numel, dtype=torch.float32))
                    missing_count += 1

            avg = torch.cat(avg_list).to(device)
            avg_sq = torch.cat(avg_sq_list).to(device)

            print(f"Prepared optimizer moments: mapped {len(mapped)} params, missing {missing_count} params.")
            return avg, avg_sq

        except Exception as e:
            print(f"Warning: failed robust mapping of optimizer state: {e}")
            total_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
            avg = torch.zeros(total_params, dtype=torch.float32, device=device)
            avg_sq = torch.zeros(total_params, dtype=torch.float32, device=device)
            return avg, avg_sq


def collect_grads(dataloader,
                  model,
                  output_dir,
                  proj_dim: List[int] = [8192],
                  adam_optimizer_state: Optional[dict] = None,
                  gradient_type: str = "adam",
                  max_samples: Optional[int] = None,
                  project_dtype: Optional[torch.dtype] = None):
    """
    Collects gradients from the model during evaluation and saves them to disk.

    Args:
        dataloader (torch.utils.data.DataLoader): The data loader for evaluation dataset.
        model (torch.nn.Module): The model from which gradients will be collected.
        output_dir (str): The directory where the gradients will be saved.
        proj_dim List[int]: The dimensions of the target projectors. Each dimension will be saved in a separate folder.
        gradient_type (str): The type of gradients to collect. [adam | sign | sgd]
        adam_optimizer_state (dict): The optimizer state of adam optimizers. If None, the gradients will be collected without considering Adam optimization states.
        max_samples (int, optional): The maximum number of samples to collect. Defaults to None.
        project_dtype (torch.dtype, optional): The dtype used for projection. Defaults to float16 when None.
    """

    model_id = 0  # model_id is used to draft the random seed for the projectors
    block_size = 128  # fixed block size for the projectors
    projector_batch_size = 16  # batch size for the projectors
    torch.random.manual_seed(0)  # set the random seed for torch

    # For large LoRA checkpoints (e.g., Qwen3-8B), stacking too many full
    # gradients can trigger GPU OOM. Keep these tunable via env.
    project_interval = int(os.environ.get("LESS_PROJECT_INTERVAL", "16"))
    save_interval = int(
        os.environ.get(
            "LESS_SAVE_BATCH_SIZE",
            os.environ.get("LESS_SAVE_INTERVAL", str(project_interval)),
        )
    )
    if project_interval < 1:
        project_interval = 1
    if save_interval < 1:
        save_interval = max(project_interval, 1)
    print(
        f"[grad-collect] project_interval={project_interval} save_interval={save_interval}",
        flush=True,
    )
    oom_retry_max_len = int(os.environ.get("LESS_OOM_RETRY_MAX_LENGTH", "0"))

    def _project(current_full_grads, projected_grads):
        current_full_grads = torch.stack(current_full_grads).to(project_dtype)
        for i, projector in enumerate(projectors):
            current_projected_grads = projector.project(
                current_full_grads, model_id=model_id)
            projected_grads[proj_dim[i]].append(current_projected_grads.cpu())

    def _save(projected_grads, output_dirs):
        for dim in proj_dim:
            if len(projected_grads[dim]) == 0:
                continue
            projected_grads[dim] = torch.cat(projected_grads[dim])

            output_dir = output_dirs[dim]
            outfile = os.path.join(output_dir, f"grads-{count}.pt")
            torch.save(projected_grads[dim], outfile)
            print(
                f"Saving {outfile}, {projected_grads[dim].shape}", flush=True)
            projected_grads[dim] = []

    device = next(model.parameters()).device
    if project_dtype is None:
        project_dtype = torch.float16

    # prepare optimization states
    if gradient_type == "adam":
        assert adam_optimizer_state is not None
        # first and second moment estimates
        m, v = prepare_optimizer_state(model, adam_optimizer_state, device)

    projector = get_trak_projector(device)
    number_of_params = get_number_of_params(model)

    # never made it work sadly
    # fmodel, params, buffers = make_functional_with_buffers(model)
    # grads_loss = torch.func.grad(get_output, has_aux=False, argnums=1)

    # initialize a project for each target projector dimension
    projectors = []
    for dim in proj_dim:
        # instantiate projector with a best-effort set of kwargs; some projector
        # implementations (BasicProjector) may not accept all keywords like
        # `max_batch_size` so try a fallback if needed.
        kwargs = dict(
            grad_dim=number_of_params,
            proj_dim=dim,
            seed=0,
            proj_type=ProjectionType.rademacher,
            device=device,
            dtype=project_dtype,
            block_size=block_size,
            max_batch_size=projector_batch_size,
        )
        try:
            proj = projector(**kwargs)
        except TypeError:
            # retry without max_batch_size
            kwargs.pop('max_batch_size', None)
            try:
                proj = projector(**kwargs)
            except Exception as e:
                # last resort: try minimal args
                proj = projector(number_of_params, dim)
        projectors.append(proj)

    count = 0

    # set up a output directory for each dimension
    output_dirs = {}
    for dim in proj_dim:
        output_dir_per_dim = os.path.join(output_dir, f"dim{dim}")
        output_dirs[dim] = output_dir_per_dim
        os.makedirs(output_dir_per_dim, exist_ok=True)

    # max index for each dimension
    max_index = min(get_max_saved_index(
        output_dirs[dim], "grads") for dim in proj_dim)

    # projected_gradients
    full_grads = []  # full gradients
    projected_grads = {dim: [] for dim in proj_dim}  # projected gradients

    pending_projected = 0
    for batch in tqdm(dataloader, total=len(dataloader)):
        count += 1

        if count <= max_index:
            # avoid flooding logs when resuming over many already-saved batches
            if count % 100 == 0 or count == 1:
                print("skipping count", count)
            continue

        prepare_batch(batch)

        try:
            if gradient_type == "adam":
                if count == 1:
                    print("Using Adam gradients")
                vectorized_grads = obtain_gradients_with_adam(model, batch, m, v)
            elif gradient_type == "sign":
                if count == 1:
                    print("Using Sign gradients")
                vectorized_grads = obtain_sign_gradients(model, batch)
            else:
                if count == 1:
                    print("Using SGD gradients")
                vectorized_grads = obtain_gradients(model, batch)
        except Exception as exc:
            if os.environ.get("LESS_STRICT_NO_TRUNCATION", "0") == "1":
                raise
            if not _is_cuda_oom_or_sdpa(exc):
                raise

            retried = False
            seq_len = _get_batch_seq_len(batch)
            retry_len = oom_retry_max_len if oom_retry_max_len > 0 else (seq_len // 2 if seq_len > 0 else 0)
            if retry_len > 0 and seq_len > 0:
                # Retry with progressively smaller sequence lengths (e.g., 1024->512->256->128).
                while retry_len >= 128 and retry_len < _get_batch_seq_len(batch):
                    prev_len = _truncate_batch_tokens(batch, retry_len)
                    print(
                        f"[oom-retry] count={count} truncating seq_len {prev_len}->{retry_len}",
                        flush=True,
                    )
                    _clear_model_grads(model)
                    torch.cuda.empty_cache()
                    try:
                        if gradient_type == "adam":
                            vectorized_grads = obtain_gradients_with_adam(model, batch, m, v)
                        elif gradient_type == "sign":
                            vectorized_grads = obtain_sign_gradients(model, batch)
                        else:
                            vectorized_grads = obtain_gradients(model, batch)
                        retried = True
                        break
                    except Exception as exc2:
                        if not _is_cuda_oom_or_sdpa(exc2):
                            raise
                        retry_len //= 2

            if not retried:
                print(f"[oom-skip] count={count} skipping batch due to: {exc}", flush=True)
                _clear_model_grads(model)
                torch.cuda.empty_cache()
                continue

        # add the gradients to the full_grads
        full_grads.append(vectorized_grads)
        _clear_model_grads(model)

        if len(full_grads) >= project_interval:
            chunk_size = len(full_grads)
            _project(full_grads, projected_grads)
            full_grads = []
            pending_projected += chunk_size

            # Save every small projected chunk (or every N samples) and release memory.
            if pending_projected >= save_interval:
                _save(projected_grads, output_dirs)
                pending_projected = 0
                torch.cuda.empty_cache()

        if max_samples is not None and count == max_samples:
            break

    if len(full_grads) > 0:
        chunk_size = len(full_grads)
        _project(full_grads, projected_grads)
        full_grads = []
        pending_projected += chunk_size

    for dim in proj_dim:
        _save(projected_grads, output_dirs)

    torch.cuda.empty_cache()
    for dim in proj_dim:
        output_dir = output_dirs[dim]
        merge_and_normalize_info(output_dir, prefix="grads")
        merge_info(output_dir, prefix="grads")

    print("Finished")


def merge_and_normalize_info(output_dir: str, prefix="reps"):
    """ Merge and normalize the representations and gradients into a single file. """
    info = os.listdir(output_dir)
    info = [file for file in info if file.startswith(prefix)]
    # Sort the files in ascending order
    info.sort(key=lambda x: int(x.split(".")[0].split("-")[1]))
    merged_data = []
    for file in info:
        data = torch.load(os.path.join(output_dir, file))
        normalized_data = normalize(data, dim=1)
        merged_data.append(normalized_data)
    merged_data = torch.cat(merged_data, dim=0)

    output_file = os.path.join(output_dir, f"all_orig.pt")
    torch.save(merged_data, output_file)
    print(
        f"Saving the normalized {prefix} (Shape: {merged_data.shape}) to {output_file}.")


def merge_info(output_dir: str, prefix="reps"):
    """ Merge the representations and gradients into a single file without normalization. """
    info = os.listdir(output_dir)
    info = [file for file in info if file.startswith(prefix)]
    # Sort the files in ascending order
    info.sort(key=lambda x: int(x.split(".")[0].split("-")[1]))
    merged_data = []
    for file in info:
        data = torch.load(os.path.join(output_dir, file))
        merged_data.append(data)
    merged_data = torch.cat(merged_data, dim=0)

    output_file = os.path.join(output_dir, f"all_unormalized.pt")
    torch.save(merged_data, output_file)
    print(
        f"Saving the unnormalized {prefix} (Shape: {merged_data.shape}) to {output_file}.")


def collect_reps(dataloader: torch.utils.data.DataLoader,
                 model: torch.nn.Module,
                 output_dir: str,
                 max_samples: Optional[int] = None):
    """
    Collects representations from a dataloader using a given model and saves them to the output directory.

    Args:
        dataloader (torch.utils.data.DataLoader): The dataloader containing the input data.
        model (torch.nn.Module): The model used to compute the representations.
        output_dir (str): The directory where the representations will be saved.
        max_samples (int, optional): The maximum number of samples to collect. Defaults to None.
    """

    all_reps = []
    count = 0
    save_interval = 160  # save every 160 batches

    device = next(model.parameters()).device  # only works for single gpu
    max_index = get_max_saved_index(output_dir, prefix="reps")

    for batch in tqdm(dataloader):
        count += 1
        if count <= max_index:
            # avoid flooding logs when resuming over many already-saved batches
            if count % 100 == 0 or count == 1:
                print("skipping count", count)
            continue

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.inference_mode():
            if isinstance(model, RobertaModel):
                reps = model(input_ids=input_ids,
                             attention_mask=attention_mask, output_hidden_states=True, return_dict=True).pooler_output
            else:
                hidden_states = model(input_ids,
                                      labels=input_ids,
                                      attention_mask=attention_mask,
                                      output_hidden_states=True).hidden_states
                ids = torch.arange(len(input_ids), device=input_ids.device)
                pos = attention_mask.sum(dim=1) - 1
                reps = hidden_states[-1][ids, pos]

            all_reps.append(reps.cpu())
            if count % save_interval == 0:
                all_reps = torch.cat(all_reps)
                outfile = os.path.join(output_dir, f"reps-{count}.pt")
                torch.save(all_reps, outfile)
                all_reps = []
                print(f"Saving {outfile}")

            if max_samples is not None and count >= max_samples:
                break

    if len(all_reps) > 0:
        all_reps = torch.cat(all_reps)
        outfile = os.path.join(output_dir, f"reps-{count}.pt")
        torch.save(all_reps, outfile)
        print(f"Saving {outfile}")

    torch.cuda.empty_cache()
    merge_and_normalize_info(output_dir, prefix="reps")

    print("Finished")


def get_loss(dataloader: torch.utils.data.DataLoader,
             model: torch.nn.Module,
             output_dir: str,):
    """ Get the loss of the model on the given dataset. """
    total_loss = 0
    total_tokens = 0
    for batch in tqdm(dataloader):
        prepare_batch(batch)
        num_token = (batch["labels"] != -100).sum()
        with torch.inference_mode():
            loss = model(**batch).loss * num_token
        total_loss += loss.item()
        total_tokens += num_token.item()

    print(f"Loss: {total_loss / total_tokens}")
    result = {"num_tokens": total_tokens, "loss": (
        total_loss / total_tokens)}
    with open(os.path.join(output_dir, "loss.txt"), "w") as f:
        f.write(json.dumps(result, indent=4))

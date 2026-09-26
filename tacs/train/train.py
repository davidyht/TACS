#!/usr/bin/env python
# coding=utf-8
import logging
import os
import random
import sys
import time

import datasets
import torch
import torch.distributed as dist
import transformers
import transformers.trainer
# from instruction_tuning.train.lora_trainer import LoRAFSDPTrainer, Trainer
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForSeq2Seq, HfArgumentParser, Trainer,
                          TrainerCallback, set_seed)

from tacs.data_selection.get_training_dataset import get_training_dataset
from tacs.train.data_arguments import DataArguments, get_data_statistics
from tacs.train.model_arguments import ModelArguments, add_padding_to_tokenizer
from tacs.train.training_arguments import TrainingArguments

logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
_PROCESS_START = time.time()


def _keep_epoch_when_skipping(skip_first_batches):
    """Fix the batch order of a mid-epoch resume (transformers 4.57 with accelerate's seedable sampler).

    Trainer sets the epoch on the train dataloader, then wraps it with skip_first_batches. The wrapper starts
    with iteration 0 and re-seeds the sampler with epoch 0 when iterated, so a resume inside epoch k >= 1
    replayed epoch 0's permutation (tests/test_stop_after_epoch.py). Copying the iteration keeps epoch k's
    order. Only mid-epoch resumes call this; epoch-boundary resumes are unaffected.
    """
    def wrapped(dataloader, num_batches=0):
        skipped = skip_first_batches(dataloader, num_batches)
        if hasattr(dataloader, "iteration") and hasattr(skipped, "iteration"):
            skipped.iteration = dataloader.iteration
        return skipped
    return wrapped


if hasattr(transformers.trainer, "skip_first_batches"):
    transformers.trainer.skip_first_batches = _keep_epoch_when_skipping(transformers.trainer.skip_first_batches)


class StopAfterEpochCallback(TrainerCallback):
    """Stop right after the checkpoint that ends epoch `stop_epoch`, so a later job can resume.

    Used through STOP_AFTER_EPOCH by scripts/rebuttal/less_warmup_chunk.sbatch. The learning-rate
    schedule still spans all epochs, and HF resume restores the optimizer, scheduler and RNG state
    (tests/test_stop_after_epoch.py checks bitwise equality with uninterrupted training).
    """

    def __init__(self, stop_epoch):
        if not stop_epoch > 0:
            raise ValueError(f"stop_epoch must be positive, got {stop_epoch}")
        self.stop_epoch = float(stop_epoch)

    def on_epoch_end(self, args, state, control, **kwargs):
        if state.epoch is not None and state.epoch >= self.stop_epoch - 1e-6:
            control.should_save = True
            control.should_training_stop = True
        return control


class TimeBudgetCallback(TrainerCallback):
    """Stop at any optimizer step once a wall-clock deadline (or, for tests, a global step) is reached.

    Used through STOP_AFTER_MINUTES by scripts/rebuttal/less_warmup_timechunk.sbatch for runs whose epochs are
    longer than a short job. HF performs the epoch-end save even after a mid-epoch stop; that partial state is
    kept out of the checkpoint-* set that validation reads. train.py saves it under <output_dir>/resume_state
    instead (save_resume_state). Epoch checkpoints at real epoch ends are unchanged.
    """

    def __init__(self, deadline=None, stop_step=None):
        if deadline is None and stop_step is None:
            raise ValueError("TimeBudgetCallback needs a deadline or a stop step")
        self.deadline = deadline
        self.stop_step = stop_step
        self.stopped = False

    def on_step_end(self, args, state, control, **kwargs):
        due = (self.stop_step is not None and state.global_step >= self.stop_step) or \
              (self.deadline is not None and time.time() >= self.deadline)
        if due and state.global_step < state.max_steps:
            control.should_training_stop = True
            self.stopped = True
        return control

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.stopped and state.epoch is not None and abs(state.epoch - round(state.epoch)) > 1e-6:
            control.should_save = False
        return control


def save_resume_state(trainer, root, keep=2):
    """Save a resumable checkpoint (adapter, optimizer, scheduler, RNG, trainer state) under root/checkpoint-<step>.

    Only resume states under root are rotated (the newest `keep` are kept); epoch checkpoints are never touched.
    """
    output_dir, limit = trainer.args.output_dir, trainer.args.save_total_limit
    trainer.args.output_dir, trainer.args.save_total_limit = root, keep
    try:
        trainer._save_checkpoint(trainer.model, trial=None)
    finally:
        trainer.args.output_dir, trainer.args.save_total_limit = output_dir, limit
    return os.path.join(root, f"checkpoint-{trainer.state.global_step}")


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training parameters {training_args}")
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Dataset parameters {data_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
    )
    # Load training dataset
    train_dataset = get_training_dataset(data_args.train_files,
                                         tokenizer=tokenizer,
                                         max_seq_length=data_args.max_seq_length,
                                         sample_percentage=data_args.percentage,
                                         seed=data_args.sample_data_seed,
                                         chat_format=data_args.chat_format)

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, torch_dtype=model_args.torch_dtype, trust_remote_code=model_args.trust_remote_code)
    add_padding_to_tokenizer(tokenizer)

    # resize embeddings if needed (e.g. for LlamaTokenizer)
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        try:
            model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        except TypeError:
            # Older Transformers versions do not expose mean_resizing. The
            # added pad token is loss-masked and embeddings are frozen by
            # PEFT below, so mean/covariance initialization is unnecessary.
            model.resize_token_embeddings(len(tokenizer))
        # if you load lora model and resize the token embeddings, the requires_grad flag is set to True for embeddings
        if isinstance(model, PeftModel):
            model.get_input_embeddings().weight.requires_grad = False
            model.get_output_embeddings().weight.requires_grad = False

    if not isinstance(model, PeftModel) and model_args.lora:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=model_args.lora_target_modules,
        )
        model = get_peft_model(model, lora_config)
        logger.info(
            f"Applied LoRA to model."
        )
        model.print_trainable_parameters()

        # for checkpointing
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    get_data_statistics(train_dataset)

    if "dataset" in train_dataset.features:
        train_dataset = train_dataset.remove_columns(
            ["dataset", "id", "messages"])


    for index in random.sample(range(len(train_dataset)), 1):
        logger.info(
            f"Sample {index} of the training set: {train_dataset[index]}.")

    model_params = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    logger.info(f"trainable model_params: {model_params}")

    analysis_dataset = None
    if training_args.analysis_mode:
        from tacs.data_selection.get_validation_dataset import get_dataset
        analysis_dataset = get_dataset(training_args.analysis_dataset,
                                       data_dir=data_args.data_dir,
                                       tokenizer=tokenizer,
                                       max_length=data_args.max_seq_length,
                                       chat_format=data_args.chat_format)

    # for testing if the model can go through full length
    # import torch
    # from datasets import Dataset

    # input_ids = [torch.randint(0, 32000, (2048, )) for _ in range(10000)]
    # attention_mask = [torch.ones(2048, ) for _ in range(10000)]
    # train_dataset = Dataset.from_dict({"input_ids": input_ids, "labels": input_ids, "attention_mask": attention_mask})

    if dist.is_initialized() and dist.get_rank() == 0:
        print(model)
    elif not dist.is_initialized():
        print(model)

    # STOP_AFTER_EPOCH=k trains through epoch k, saves its checkpoint and stops without the final
    # save, so the run can be resumed in a later job (one-epoch warmup chunks).
    stop_after_epoch = os.environ.get("STOP_AFTER_EPOCH", "").strip()
    callbacks = []
    if stop_after_epoch:
        strategy = getattr(training_args.save_strategy, "value", training_args.save_strategy)
        if strategy != "epoch":
            raise ValueError(f"STOP_AFTER_EPOCH needs save_strategy=epoch, got {strategy}")
        if float(stop_after_epoch) > training_args.num_train_epochs:
            raise ValueError(f"STOP_AFTER_EPOCH={stop_after_epoch} exceeds num_train_epochs={training_args.num_train_epochs}")
        callbacks.append(StopAfterEpochCallback(float(stop_after_epoch)))
    # STOP_AFTER_MINUTES=m stops at the first optimizer step after m minutes from process start and saves a
    # resumable state under <output_dir>/resume_state (time-budgeted chunks of runs with long epochs).
    stop_after_minutes = os.environ.get("STOP_AFTER_MINUTES", "").strip()
    budget = None
    if stop_after_minutes:
        if stop_after_epoch:
            raise ValueError("STOP_AFTER_EPOCH and STOP_AFTER_MINUTES are exclusive")
        budget = TimeBudgetCallback(deadline=_PROCESS_START + 60.0 * float(stop_after_minutes))
        callbacks.append(budget)

    if os.environ.get("PAD_TO_MULTIPLE_OF"):
        logger.info(f"PAD_TO_MULTIPLE_OF={os.environ['PAD_TO_MULTIPLE_OF']}: batches padded to this multiple (masked)")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=analysis_dataset,
        tokenizer=tokenizer,
        # PAD_TO_MULTIPLE_OF (opt-in, default unset = unchanged): pad each batch to a multiple of N tokens so the
        # number of distinct tensor shapes stays small (throughput benchmark, 2026-09-14). Pads are masked
        # (attention_mask 0, labels -100).
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer, model=model, padding="longest",
            pad_to_multiple_of=int(os.environ["PAD_TO_MULTIPLE_OF"]) if os.environ.get("PAD_TO_MULTIPLE_OF") else None),
        callbacks=callbacks or None,
    )

    # Training (optionally resume from a saved HF checkpoint directory).
    resume_ckpt = getattr(training_args, "resume_from_checkpoint", None)
    if resume_ckpt:
        logger.info(f"Resuming training from checkpoint: {resume_ckpt}")
    train_result = trainer.train(resume_from_checkpoint=resume_ckpt)
    if budget is not None and budget.stopped:
        saved = save_resume_state(trainer, os.path.join(training_args.output_dir, "resume_state"))
        logger.info(f"STOP_AFTER_MINUTES={stop_after_minutes}: stopped at global step {trainer.state.global_step} "
                    f"(epoch {trainer.state.epoch:.4f}); resume state {saved}")
        return
    if stop_after_epoch and trainer.state.epoch < training_args.num_train_epochs - 1e-6:
        logger.info(f"STOP_AFTER_EPOCH={stop_after_epoch}: stopped after the epoch-{trainer.state.epoch:g} "
                    "checkpoint; resume from it to continue")
        return
    trainer.save_model()  # Saves the tokenizer too for easy upload

    metrics = train_result.metrics

    metrics["train_samples"] = len(train_dataset)

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # remove the full model in the end to save space, only adapter is needed
    if isinstance(model, PeftModel):
        pytorch_model_path = os.path.join(
            training_args.output_dir, "pytorch_model_fsdp.bin")
        os.remove(pytorch_model_path) if os.path.exists(
            pytorch_model_path) else None


if __name__ == "__main__":
    main()

import json
import os
import random
from glob import glob
from typing import List, Tuple

import pandas as pd
import torch
import tqdm
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq, PreTrainedTokenizerBase

from tacs.data_selection.get_training_dataset import concat_messages
from tacs.data_selection.get_validation_dataset import (
    DEFAULT_MMLU_N_SHOT,
    resolve_mmlu_n_shot,
    tokenize,
)
from tacs.data_selection.chat_format_utils import apply_chat_template_prompt, resolve_chat_format

# llama-chat model's instruction format
B_INST, E_INST = "[INST]", "[/INST]"

def get_tydiqa_dataset(data_dir: str,
                       tokenizer: PreTrainedTokenizerBase,
                       max_length: int,
                       use_chat_format: bool = True,
                       chat_format: str = "tulu",
                       zh: bool = False,
                       **kwargs) -> Dataset:
    """
    Get the tydiqa dataset in the instruction tuning format. Each example is formatted as follows:

    Query:
    <|user|>
    <Task Prompt>
    <Passage>
    <Question>
    <|assistant|>
    Answer:

    Completion:
    <Answer>

    Args:
        data_dir (str): The main data directory.
        tokenizer (PreTrainedTokenizerBase): The tokenizer to use for tokenization.
        max_length (int): The maximum length of the input sequence.
        use_chat_format (bool, optional): Whether to use chat format. Defaults to True.
        chat_format (str, optional): The chat format to use. Defaults to "tulu".
        zh (bool, optional): Whether to use the Chinese validation examples. Defaults to False.

    Returns:
        Dataset: The tokenized TydiQA dataset.
    """

    # Same template as https://github.com/allenai/open-instruct/blob/main/eval/tydiqa/run_eval.py#L17
    encoding_templates_with_context = {
        "english": ("Answer the following question based on the information in the given passage.", "Passage:", "Question:", "Answer:"),
        "arabic": ("أجب على السؤال التالي بناءً على المعلومات في المقطع المعطى.", "المقطع:", "السؤال:", "الإجابة:"),
        "bengali": ("প্রদত্ত অধ্যায়ের তথ্যের উপর ভিত্তি করে নিম্নলিখিত প্রশ্নের উত্তর দিন।", "অধ্যায়:", "প্রশ্ন:", "উত্তর:"),
        "finnish": ("Vastaa seuraavaan kysymykseen annetun kappaleen tiedon perusteella.", "Kappale:", "Kysymys:", "Vastaus:"),
        "indonesian": ("Jawab pertanyaan berikut berdasarkan informasi di bagian yang diberikan.", "Bagian:", "Pertanyaan:", "Jawaban:"),
        "korean": ("주어진 문단의 정보에 기반하여 다음 질문에 답하십시오.", "문단:", "질문:", "답변:"),
        "russian": ("Ответьте на следующий вопрос на основе информации в данном отрывке.", "Отрывок:", "Вопрос:", "Ответ:"),
        "swahili": ("Jibu swali lifuatalo kulingana na habari kwenye kifungu kilichotolewa.", "Kifungu:", "Swali:", "Jibu:"),
        "telugu": ("ఇచ్చిన పేరాలోని సమాచారం ఆధారంగా కింది ప్రశ్నకు సమాధానం ఇవ్వండి.", "పేరా:", "ప్రశ్న:", "సమాధానం:")
    }

    # Chinese validation examples
    if zh:
        for lang in encoding_templates_with_context:
            encoding_templates_with_context[lang] = (
                "根据所给文章中的信息回答以下问题。", "文章:", "问题:", "答案:")

    file_name = "tydiqa-one-shot-zh.json" if zh else "tydiqa-one-shot.json"
    file = os.path.join(f"{data_dir}/eval/tydiqa", file_name)

    examples = json.load(open(file, "r"))
    dataset = {"input_ids": [], "attention_mask": [], "labels": []}

    for i, lang in enumerate(examples):
        example = examples[lang][0]
        prompt, p_template, q_template, a_template = encoding_templates_with_context[lang]
        prompt += p_template + " " + \
            format(example["context"]) + "\n" + q_template + \
            " " + format(example["question"]) + "\n"
        answer = " " + format(example["answers"][0]["text"])
        base_prompt = prompt + a_template
        if use_chat_format:
            fmt = resolve_chat_format(tokenizer, chat_format)
            if fmt == "tokenizer":
                prompt = apply_chat_template_prompt(tokenizer, base_prompt, add_generation_prompt=True)
            elif fmt == "tulu":
                prompt = "<|user|>\n" + prompt + "<|assistant|>\n" + a_template
            else:
                prompt = f"<s> {B_INST} {prompt} {E_INST} {a_template}"
        else:
            prompt = base_prompt
        full_input_ids, labels, attention_mask = tokenize(
            tokenizer, prompt, answer, max_length, print_ex=True)
        dataset["input_ids"].append(full_input_ids)
        dataset["labels"].append(labels)
        dataset["attention_mask"].append(attention_mask)
    dataset = Dataset.from_dict(dataset)
    return dataset


def get_tydiqa_goldp_dataset(data_dir: str,
                              tokenizer: PreTrainedTokenizerBase,
                              max_length: int,
                              use_chat_format: bool = True,
                              chat_format: str = "tulu",
                              max_examples: int = 200,
                              seed: int = 42,
                              **kwargs) -> Dataset:
    """Load a subsample of TyDiQA goldp-dev as a held-out benchmark set.

    Unlike get_tydiqa_dataset (which loads the 9 one-shot examples used for
    warmup training), this loads from the 5077-example goldp-v1.1-dev set
    and returns a stratified subsample for efficient per-step loss tracking.

    Args:
        data_dir: Root data directory (expects eval/tydiqa/tydiqa-goldp-v1.1-dev.json).
        max_examples: Max examples to return (stratified across languages).
        seed: Random seed for subsampling reproducibility.
    """
    encoding_templates_with_context = {
        "english": ("Answer the following question based on the information in the given passage.", "Passage:", "Question:", "Answer:"),
        "arabic": ("أجب على السؤال التالي بناءً على المعلومات في المقطع المعطى.", "المقطع:", "السؤال:", "الإجابة:"),
        "bengali": ("প্রদত্ত অধ্যায়ের তথ্যের উপর ভিত্তি করে নিম্নলিখিত প্রশ্নের উত্তর দিন।", "অধ্যায়:", "প্রশ্ন:", "উত্তর:"),
        "finnish": ("Vastaa seuraavaan kysymykseen annetun kappaleen tiedon perusteella.", "Kappale:", "Kysymys:", "Vastaus:"),
        "indonesian": ("Jawab pertanyaan berikut berdasarkan informasi di bagian yang diberikan.", "Bagian:", "Pertanyaan:", "Jawaban:"),
        "korean": ("주어진 문단의 정보에 기반하여 다음 질문에 답하십시오.", "문단:", "질문:", "답변:"),
        "russian": ("Ответьте на следующий вопрос на основе информации в данном отрывке.", "Отрывок:", "Вопрос:", "Ответ:"),
        "swahili": ("Jibu swali lifuatalo kulingana na habari kwenye kifungu kilichotolewa.", "Kifungu:", "Swali:", "Jibu:"),
        "telugu": ("ఇచ్చిన పేరాలోని సమాచారం ఆధారంగా కింది ప్రశ్నకు సమాధానం ఇవ్వండి.", "పేరా:", "ప్రశ్న:", "సమాధానం:")
    }

    goldp_file = os.path.join(data_dir, "eval", "tydiqa", "tydiqa-goldp-v1.1-dev.json")
    if not os.path.exists(goldp_file):
        raise FileNotFoundError(f"TyDiQA goldp-dev not found: {goldp_file}")

    goldp = json.load(open(goldp_file, "r"))

    # Parse SQuAD-format into flat list of (lang, context, question, answer)
    examples_by_lang = {}
    for entry in goldp["data"]:
        for para in entry.get("paragraphs", []):
            context = para["context"]
            for qa in para.get("qas", []):
                qid = qa.get("id", "")
                lang = qid.split("-")[0] if "-" in qid else "english"
                if lang not in encoding_templates_with_context:
                    continue
                answers = qa.get("answers", [])
                if not answers:
                    continue
                examples_by_lang.setdefault(lang, []).append({
                    "context": context,
                    "question": qa["question"],
                    "answer": answers[0]["text"],
                })

    # Stratified subsample: proportional to language count, min 5 per lang
    rng = random.Random(seed)
    sampled = []
    total_available = sum(len(v) for v in examples_by_lang.values())
    for lang, exs in examples_by_lang.items():
        n = max(5, int(round(max_examples * len(exs) / total_available)))
        n = min(n, len(exs))
        sampled.extend([(lang, e) for e in rng.sample(exs, n)])
    rng.shuffle(sampled)

    # Format into instruction-tuning tokens
    dataset = {"input_ids": [], "attention_mask": [], "labels": []}
    for lang, ex in sampled:
        prompt_tmpl, p_tmpl, q_tmpl, a_tmpl = encoding_templates_with_context[lang]
        prompt = prompt_tmpl + p_tmpl + " " + ex["context"] + "\n" + q_tmpl + " " + ex["question"] + "\n"
        answer = " " + ex["answer"]
        base_prompt = prompt + a_tmpl
        if use_chat_format:
            fmt = resolve_chat_format(tokenizer, chat_format)
            if fmt == "tokenizer":
                prompt = apply_chat_template_prompt(tokenizer, base_prompt, add_generation_prompt=True)
            elif fmt == "tulu":
                prompt = "<|user|>\n" + prompt + "<|assistant|>\n" + a_tmpl
            else:
                prompt = f"<s> {B_INST} {prompt} {E_INST} {a_tmpl}"
        else:
            prompt = base_prompt
        full_input_ids, labels, attention_mask = tokenize(
            tokenizer, prompt, answer, max_length, print_ex=False)
        # Skip examples where the answer was truncated (all labels are -100)
        if (labels == -100).all():
            continue
        dataset["input_ids"].append(full_input_ids)
        dataset["labels"].append(labels)
        dataset["attention_mask"].append(attention_mask)

    dataset = Dataset.from_dict(dataset)
    return dataset


def get_bbh_test_dataset(data_dir: str,
                         tokenizer: PreTrainedTokenizerBase,
                         max_length: int,
                         use_chat_format: bool = True,
                         chat_format: str = "tulu",
                         max_examples: int = 200,
                         seed: int = 42,
                         **kwargs) -> Dataset:
    """Load a subsample of BBH test examples as a held-out benchmark set.

    Unlike get_bbh_dataset (which uses the 81 three-shot warmup examples),
    this loads from the full BBH test set (250 examples × 27 subtasks)
    and returns a stratified subsample for efficient per-step loss tracking.

    Each test example is formatted with all 3 few-shot demonstrations as
    in-context learning context, matching the evaluation format.

    Args:
        data_dir: Root data directory (expects eval/bbh/test/*.json and
                  eval/bbh/bbh-three-shot.json).
        max_examples: Max examples to return (stratified across subtasks).
        seed: Random seed for subsampling reproducibility.
    """
    # Load few-shot examples for ICL demonstrations
    three_shot_file = os.path.join(data_dir, "eval", "bbh", "bbh-three-shot.json")
    if not os.path.exists(three_shot_file):
        raise FileNotFoundError(f"BBH three-shot file not found: {three_shot_file}")
    bbh_few_shot = json.load(open(three_shot_file, "r"))

    # Parse task prompts and ICL demos from three-shot data
    task_prompts = {}
    task_icl = {}
    for task_name, content in bbh_few_shot.items():
        parts = content.split("\n\n")
        # Last 3 parts are the Q&A examples; everything before is the task prompt
        exes = parts[-3:]
        task_prompt = "\n\n".join(parts[:-3])
        icl_string = ""
        for ex in exes:
            icl_string += ex + "\n\n"
        task_prompts[task_name] = task_prompt.strip()
        task_icl[task_name] = icl_string

    # Load test examples from each subtask
    test_dir = os.path.join(data_dir, "eval", "bbh", "test")
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"BBH test directory not found: {test_dir}")

    examples_by_task = {}
    for fname in sorted(os.listdir(test_dir)):
        if not fname.endswith(".json"):
            continue
        task_name = fname.replace(".json", "")
        if task_name not in task_prompts:
            continue
        with open(os.path.join(test_dir, fname)) as f:
            data = json.load(f)
        exs = data.get("examples", [])
        if exs:
            examples_by_task[task_name] = exs

    # Stratified subsample
    rng = random.Random(seed)
    sampled = []
    total_available = sum(len(v) for v in examples_by_task.values())
    for task_name, exs in examples_by_task.items():
        n = max(3, int(round(max_examples * len(exs) / total_available)))
        n = min(n, len(exs))
        sampled.extend([(task_name, e) for e in rng.sample(exs, n)])
    rng.shuffle(sampled)

    # Format into instruction-tuning tokens
    dataset = {"input_ids": [], "attention_mask": [], "labels": []}
    for task_name, ex in sampled:
        question = ex["input"]
        answer = " " + ex["target"]

        base_prompt = (task_prompts[task_name] + "\n\n" +
                       task_icl[task_name] +
                       f"Q: {question}\nA:")

        if use_chat_format:
            fmt = resolve_chat_format(tokenizer, chat_format)
            if fmt == "tokenizer":
                prompt = apply_chat_template_prompt(tokenizer, base_prompt, add_generation_prompt=True)
            elif fmt == "tulu":
                prompt = "<|user|>\n" + base_prompt + "\n<|assistant|>\nA:"
            else:
                prompt = f"<s> {B_INST} {base_prompt} {E_INST} A:"
        else:
            prompt = base_prompt

        full_input_ids, labels, attention_mask = tokenize(
            tokenizer, prompt, answer, max_length, print_ex=False)
        if (labels == -100).all():
            continue
        dataset["input_ids"].append(full_input_ids)
        dataset["labels"].append(labels)
        dataset["attention_mask"].append(attention_mask)

    dataset = Dataset.from_dict(dataset)
    return dataset


def get_mmlu_dataset(data_dir: str,
                     tokenizer: PreTrainedTokenizerBase,
                     max_length: int,
                     use_chat_format=True,
                     chat_format="tulu",
                     mmlu_n_shot: int = DEFAULT_MMLU_N_SHOT,
                     **kwargs):
    """
    Get the MMLU dataset in the instruction tuning format. Each example is formatted as follows:

    Query:
    <|user|>
    <Task Prompt>
    <Question>
    <|assistant|>
    The answer is:

    Completion:
    <Answer>

    Args:
        data_dir (str): The main data directory.
        tokenizer (Tokenizer): The tokenizer used to tokenize the input text.
        max_length (int): The maximum length of the input sequence.
        use_chat_format (bool, optional): Whether to use chat format for the prompts. Defaults to True.
        chat_format (str, optional): The chat format to use for the prompts. Defaults to "tulu".

    Returns:
        Dataset: The tokenized dataset containing input_ids, attention_mask, and labels.
    """

    subjects = sorted(
        [
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join(data_dir, "test"))
            if "_test.csv" in f
        ]
    )

    def format_subject(subject):
        l = subject.split("_")
        s = ""
        for entry in l:
            s += " " + entry
        return s

    def gen_prompt(train_df, subject, i=0):
        prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
            format_subject(subject)
        )
        prompt += format_example(train_df, i, include_answer=False)
        return prompt

    def format_example(df, idx, include_answer=True):
        choices = ["A", "B", "C", "D"]
        prompt = df.iloc[idx, 0]
        k = df.shape[1] - 2
        for j in range(k):
            prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
        prompt += "\nAnswer:"
        return prompt

    k = resolve_mmlu_n_shot(mmlu_n_shot)
    dataset = {"input_ids": [], "attention_mask": [], "labels": []}
    for subject in subjects:
        dev_df = pd.read_csv(
            os.path.join(data_dir, "dev", subject + "_dev.csv"), header=None
        )
        if len(dev_df) < k:
            raise ValueError(
                f"MMLU subject {subject} only has {len(dev_df)} dev examples, "
                f"cannot build {k}-shot prompts."
            )
        dev_df = dev_df[:k]
        for i in range(k):
            prompt = gen_prompt(dev_df, subject, i)
            answer = " " + dev_df.iloc[i, dev_df.shape[1] - 2 + 1]

            if use_chat_format:
                fmt = resolve_chat_format(tokenizer, chat_format)
                if fmt == "tokenizer":
                    prompt = apply_chat_template_prompt(tokenizer, prompt + "\nThe answer is:", add_generation_prompt=True)
                elif fmt == "tulu":
                    prompt = "<|user|>\n" + prompt + "\n<|assistant|>\nThe answer is:"
                else:
                    # f"<s> {B_INST} {task_prompt.strip()} {question} {E_INST} A:"
                    prompt = f"<s> {B_INST} {prompt} {E_INST} The answer is:"

            full_input_ids, labels, attention_mask = tokenize(
                tokenizer, prompt, answer, max_length, print_ex=True if i == 0 else False)
            dataset["input_ids"].append(full_input_ids)
            dataset["labels"].append(labels)
            dataset["attention_mask"].append(attention_mask)
    dataset = Dataset.from_dict(dataset)
    return dataset

import json
import os
from typing import List, Tuple

import pandas as pd
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq, PreTrainedTokenizerBase

from tacs.data_selection.chat_format_utils import apply_chat_template_prompt, resolve_chat_format

# llama-chat model's instruction format
B_INST, E_INST = "[INST]", "[/INST]"
DEFAULT_MMLU_N_SHOT = 5


def resolve_mmlu_n_shot(mmlu_n_shot: int = DEFAULT_MMLU_N_SHOT) -> int:
    n_shot = int(mmlu_n_shot)
    if n_shot <= 0:
        raise ValueError(f"mmlu_n_shot must be positive, got {mmlu_n_shot}")
    return n_shot


def tokenize(tokenizer: PreTrainedTokenizerBase,
             query: str,
             completion: str,
             max_length: int,
             print_ex: bool = False) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """
    Formats a chat conversation into input tensors for a transformer model.

    Args:
        tokenizer (PreTrainedTokenizerBase): The tokenizer used to encode the input.
        query (str): The question part of the chat conversation.
        completion (str): The answer part of the chat conversation.
        max_length (int): The maximum length of the input tensors.
        print_ex (bool, optional): Whether to print the example. Defaults to False.

    Returns:
        tuple: A tuple containing the full input IDs, labels, and attention mask tensors.
    """
    full_prompt = query + completion

    # Disabled verbose example printing
    # if print_ex:
    #     print("******** Example starts ********")
    #     print(full_prompt)
    #     print("******** Example ends ********")

    prompt_input_ids = torch.tensor(
        tokenizer.encode(query, max_length=max_length))
    full_input_ids = torch.tensor(
        tokenizer.encode(full_prompt, max_length=max_length))
    labels = torch.tensor(tokenizer.encode(full_prompt, max_length=max_length))
    labels[:len(prompt_input_ids)] = -100
    attention_mask = [1] * len(full_input_ids)

    return full_input_ids, labels, attention_mask


def get_bbh_dataset(data_dir: str,
                    tokenizer: PreTrainedTokenizerBase,
                    max_length: int,
                    use_chat_format: bool = True,
                    chat_format: str = "tulu",
                    **kwargs):
    """
    Get the bbh dataset in the instruction tuning format. Each example is formatted as follows:

    Query:
    <|user|>
    <Task Prompt>
    <Ex1>
    <Ex2>
    <Question of Ex3>
    <|assistant|>
    A:

    Completion:
    <Answer of Ex3>

    Args:
        data_dir (str): The main data directory.
        tokenizer (Tokenizer): The tokenizer used to tokenize the input text.
        max_length (int): The maximum length of the input sequence.
        use_chat_format (bool, optional): Whether to use chat format for the input. Defaults to True.
        chat_format (str, optional): The chat format to use. Defaults to "tulu".
        n_shot (int, optional): The number of shots for few-shot learning. Defaults to 3 for bbh.

    Returns:
        Dataset: The BBH dataset containing input_ids, attention_mask, and labels.
    """
    file = f"{data_dir}/eval/bbh/bbh-three-shot.json"

    bbh_few_shot_examples = json.load(open(file, "r"))
    dataset = {"input_ids": [], "attention_mask": [], "labels": []}

    # there are multiple tasks in the bbh dataset
    # each task has 3 examples
    for task in bbh_few_shot_examples:
        few_shot_exs = bbh_few_shot_examples[task]

        stuff = few_shot_exs.split("\n\n")
        exes = stuff[-3:]
        task_prompt = "\n\n".join(stuff[:-3])

        def form_icl(exs):
            string = ""
            for ex in exs:
                question, answer = ex.split("\nA:")
                string += question + "\nA:" + answer
                string += "\n\n"
            return string

        for i in range(len(exes)):
            target_ex = exes[i]
            other_exes = exes[:i] + exes[i+1:]
            icl = form_icl(other_exes)
            question, answer = target_ex.split("\nA:")

            base_prompt = task_prompt.strip() + "\n\n" + icl + f"{question}" + "\nA:"
            if use_chat_format:
                fmt = resolve_chat_format(tokenizer, chat_format)
                if fmt == "tokenizer":
                    question = apply_chat_template_prompt(tokenizer, base_prompt, add_generation_prompt=True)
                elif fmt == "tulu":  # we follow the tulu instruction tuning format
                    question = "<|user|>\n" + task_prompt.strip() + "\n\n" + icl + \
                        f"{question}" + "\n<|assistant|>\nA:"
                else:
                    question = f"<s> {B_INST} {task_prompt.strip()} {question} {E_INST} A:"
            else:
                question = base_prompt
            full_input_ids, labels, attention_mask = tokenize(
                tokenizer, question, answer, max_length, print_ex=True if i == 0 else False)
            dataset["input_ids"].append(full_input_ids)
            dataset["labels"].append(labels)
            dataset["attention_mask"].append(attention_mask)

    dataset = Dataset.from_dict(dataset)
    return dataset


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
    # prefer common locations: eval/tydiqa/, eval/tydiqa/dev/, eval/tydiqa/test/
    candidate_dirs = [os.path.join(data_dir, "eval", "tydiqa"),
                      os.path.join(data_dir, "eval", "tydiqa", "dev"),
                      os.path.join(data_dir, "eval", "tydiqa", "test")]
    candidate_files = [os.path.join(d, file_name) for d in candidate_dirs]

    # also allow .jsonl variant
    candidate_files += [p + '.l' if p.endswith('.json') else p for p in []]  # placeholder to keep structure

    found = None
    for p in candidate_files:
        try:
            if os.path.exists(p):
                found = p
                break
        except Exception:
            continue
    if found is None:
        # try a looser glob search under data_dir/eval/tydiqa
        try:
            import glob
            pattern = os.path.join(data_dir, "eval", "tydiqa", "**", "*one-shot*.json*")
            matches = glob.glob(pattern, recursive=True)
            matches = [m for m in matches if 'one-shot' in os.path.basename(m)]
            if len(matches) > 0:
                found = matches[0]
        except Exception:
            found = None

    if found is None:
        msg = f"tydiqa one-shot file not found. Tried: {candidate_files} and recursive search under {os.path.join(data_dir, 'eval', 'tydiqa')}"
        raise FileNotFoundError(msg)

    examples = json.load(open(found, "r"))
    dataset = {"input_ids": [], "attention_mask": [], "labels": []}

    for i, lang in enumerate(examples):
        lang_examples = examples[lang]
        if not isinstance(lang_examples, list):
            lang_examples = [lang_examples]
        for example in lang_examples:
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


def get_mmlu_dataset(data_dir: str,
                     tokenizer: PreTrainedTokenizerBase,
                     max_length: int,
                     use_chat_format=True,
                     chat_format="tulu",
                     mmlu_n_shot: int = DEFAULT_MMLU_N_SHOT,
                     mmlu_subjects=None,
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
    mmlu_data_dir = os.path.join(data_dir, "eval", "mmlu")
    available_subjects = sorted(
        [
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join(mmlu_data_dir, "test"))
            if "_test.csv" in f
        ]
    )
    if mmlu_subjects is None:
        env_subjects = os.environ.get("TACS_MMLU_SUBJECTS", "").strip()
        mmlu_subjects = env_subjects.split() if env_subjects else None
    if mmlu_subjects is None:
        subjects = available_subjects
    else:
        subjects = [str(subject).strip() for subject in mmlu_subjects if str(subject).strip()]
        if not subjects:
            raise ValueError("mmlu_subjects was provided but is empty")
        if len(subjects) != len(set(subjects)):
            raise ValueError(f"mmlu_subjects contains duplicates: {subjects}")
        unknown = sorted(set(subjects) - set(available_subjects))
        if unknown:
            raise ValueError(
                f"Unknown MMLU subjects: {unknown}. "
                f"Available subjects: {available_subjects}"
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
            os.path.join(mmlu_data_dir, "dev", subject + "_dev.csv"), header=None
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


def get_generic_dataset(data_dir: str,
                        tokenizer: PreTrainedTokenizerBase,
                        max_length: int,
                        use_chat_format: bool = True,
                        chat_format: str = "tulu",
                        generic_file: str = None,
                        n: int = 9,
                        seed: int = 0,
                        **kwargs) -> Dataset:
    """Build a target-agnostic warmup set: a random sample of generic instruction data.

    Used as the matched negative model for contrastive scoring: a warmup on this set
    has the same rank, learning rate and step budget as the target warmup but carries
    no target signal, so subtracting its endpoint loss cancels the task-agnostic part
    of a candidate's loss movement (formatting drift, per-source loss offsets).

    ``generic_file`` is a JSONL file in the standard training format
    (``{"messages": [...]}``); it defaults to ``$LESS_GENERIC_WARMUP_FILE``.
    """
    path = generic_file or os.environ.get("LESS_GENERIC_WARMUP_FILE")
    if not path:
        raise ValueError(
            "generic warmup needs a source file: pass generic_file=... or set "
            "LESS_GENERIC_WARMUP_FILE")
    if not os.path.isabs(path) and data_dir:
        cand = os.path.join(data_dir, path)
        if os.path.exists(cand):
            path = cand
    if not os.path.exists(path):
        raise FileNotFoundError(f"generic warmup file not found: {path}")

    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    rng = torch.Generator().manual_seed(int(seed))
    order = torch.randperm(len(rows), generator=rng)[:int(n)].tolist()

    dataset = {"input_ids": [], "attention_mask": [], "labels": []}
    for i in order:
        msgs = rows[i].get("messages", [])
        query = "".join(m["content"] for m in msgs if m.get("role") != "assistant")
        answer = " " + "".join(m["content"] for m in msgs if m.get("role") == "assistant")
        if use_chat_format:
            fmt = resolve_chat_format(tokenizer, chat_format)
            if fmt == "tokenizer":
                query = apply_chat_template_prompt(tokenizer, query, add_generation_prompt=True)
            elif fmt == "tulu":
                query = "<|user|>\n" + query + "\n<|assistant|>\n"
            else:
                query = f"<s> {B_INST} {query} {E_INST}"
        full_input_ids, labels, attention_mask = tokenize(
            tokenizer, query, answer, max_length)
        dataset["input_ids"].append(full_input_ids)
        dataset["labels"].append(labels)
        dataset["attention_mask"].append(attention_mask)
    return Dataset.from_dict(dataset)


def _get_jsonl_target_dataset(data_dir: str,
                              task: str,
                              tokenizer: PreTrainedTokenizerBase,
                              max_length: int,
                              use_chat_format: bool = True,
                              chat_format: str = "tulu",
                              **kwargs) -> Dataset:
    """Load a frozen, locally materialized target proxy.

    Extended-target studies deliberately consume a checked-in-style JSONL proxy
    rather than downloading data inside a GPU job.  Each row must contain
    ``question`` and ``answer`` and is formatted consistently across methods.
    """
    path = os.path.join(data_dir, "eval", task, "target_proxy.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"frozen {task} target proxy not found: {path}")
    dataset = {"input_ids": [], "attention_mask": [], "labels": []}
    with open(path, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if len(rows) < 3:
        raise ValueError(f"{task} target proxy needs at least 3 examples, got {len(rows)}")
    for row in rows:
        question = str(row["question"]).strip()
        answer = str(row["answer"]).strip()
        base_prompt = f"Question: {question}\nAnswer:"
        if use_chat_format:
            fmt = resolve_chat_format(tokenizer, chat_format)
            if fmt == "tokenizer":
                prompt = apply_chat_template_prompt(
                    tokenizer, base_prompt, add_generation_prompt=True
                )
            elif fmt == "tulu":
                prompt = "<|user|>\n" + base_prompt + "\n<|assistant|>\n"
            else:
                prompt = f"<s> {B_INST} {base_prompt} {E_INST}"
        else:
            prompt = base_prompt
        full_input_ids, labels, attention_mask = tokenize(
            tokenizer, prompt, " " + answer, max_length
        )
        dataset["input_ids"].append(full_input_ids)
        dataset["labels"].append(labels)
        dataset["attention_mask"].append(attention_mask)
    return Dataset.from_dict(dataset)


def get_dataset(task, **kwargs):
    """
    Get the dataset for the given task.

    Args:
        task_name (str): The name of the task.

    Raises:
        ValueError: If the task name is not valid.

    Returns:
        Dataset: The dataset.
    """
    if task == "bbh":
        return get_bbh_dataset(**kwargs)
    elif task == "tydiqa":
        return get_tydiqa_dataset(**kwargs)
    elif task == "mmlu":
        return get_mmlu_dataset(**kwargs)
    elif task == "generic":
        return get_generic_dataset(**kwargs)
    elif task in {"gsm8k", "truthfulqa"}:
        return _get_jsonl_target_dataset(task=task, **kwargs)
    else:
        raise ValueError("Invalid task name")


def get_dataloader(dataset, tokenizer, batch_size=1):
    data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer, padding="longest")
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,  # When getting gradients, we only do this single batch process
                            collate_fn=data_collator)
    print("There are {} examples in the dataset".format(len(dataset)))
    return dataloader

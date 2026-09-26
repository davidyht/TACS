from typing import List, Dict


def supports_chat_template(tokenizer) -> bool:
    if not hasattr(tokenizer, "apply_chat_template"):
        return False
    template = getattr(tokenizer, "chat_template", None)
    return bool(template)


def resolve_chat_format(tokenizer, chat_format: str, fallback: str = "tulu") -> str:
    if chat_format == "tokenizer" and not supports_chat_template(tokenizer):
        return fallback
    return chat_format


def apply_chat_template_prompt(tokenizer, user_content: str, add_generation_prompt: bool = True) -> str:
    messages = [{"role": "user", "content": user_content}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False)


def chat_template_ids(tokenizer, messages: List[Dict], max_seq_length: int):
    try:
        ids = tokenizer.apply_chat_template(
            messages, tokenize=True, max_length=max_seq_length, truncation=True
        )
    except TypeError:
        ids = tokenizer.apply_chat_template(messages, tokenize=True)

    if isinstance(ids, dict):
        ids = ids.get("input_ids", ids)
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if isinstance(ids, list) and ids and isinstance(ids[0], list):
        ids = ids[0]
    if max_seq_length is not None and len(ids) > max_seq_length:
        ids = ids[:max_seq_length]
    return ids

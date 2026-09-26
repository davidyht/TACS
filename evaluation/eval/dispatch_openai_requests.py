"""Optional OpenAI request helpers for legacy evaluation scripts.

The paper experiments use local Hugging Face models. These helpers are kept so
the evaluation utilities remain importable when the optional OpenAI path is used.
No API keys, endpoints, or deployment names are stored in this package.
"""

from __future__ import annotations

import asyncio
from typing import Any


async def dispatch_openai_chat_requesets(
    messages_list: list[list[dict[str, Any]]],
    model: str,
    **completion_kwargs: Any,
) -> list[Any]:
    """Dispatch chat-completion requests with the legacy async OpenAI API."""
    openai = _import_openai()
    async_responses = [
        openai.ChatCompletion.acreate(
            engine=model,
            messages=messages,
            **completion_kwargs,
        )
        for messages in messages_list
    ]
    return await asyncio.gather(*async_responses)


async def dispatch_openai_prompt_requesets(
    prompt_list: list[str],
    model: str,
    **completion_kwargs: Any,
) -> list[Any]:
    """Dispatch prompt-completion requests with the legacy async OpenAI API."""
    openai = _import_openai()
    async_responses = [
        openai.Completion.acreate(
            engine=model,
            prompt=prompt,
            **completion_kwargs,
        )
        for prompt in prompt_list
    ]
    return await asyncio.gather(*async_responses)


def _import_openai() -> Any:
    try:
        import openai
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError(
            "OpenAI evaluation requires installing the optional `openai` package "
            "and configuring credentials through the environment."
        ) from exc
    return openai

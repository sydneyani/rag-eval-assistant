"""Turn token counts into a dollar estimate.

Prices live on settings so a model change does not require a code change.
The figure is a list-price estimate. Offline providers cost nothing to run,
and the basis string says so instead of pretending the estimate came from a
billing API.
"""

from __future__ import annotations

from citeground.config import Settings


def estimate_cost(
    settings: Settings,
    *,
    prompt_tokens: int,
    completion_tokens: int,
    embedding_tokens: int,
    chat_provider: str,
    embedding_provider: str,
) -> tuple[float, str]:
    chat_cost = 0.0
    embedding_cost = 0.0
    if chat_provider == "openai":
        chat_cost = (
            prompt_tokens * settings.price_chat_input_per_mtok
            + completion_tokens * settings.price_chat_output_per_mtok
        ) / 1_000_000
    if embedding_provider == "openai":
        embedding_cost = embedding_tokens * settings.price_embedding_per_mtok / 1_000_000
    if chat_provider != "openai" and embedding_provider != "openai":
        return 0.0, "offline-zero"
    return chat_cost + embedding_cost, "list-price-estimate"

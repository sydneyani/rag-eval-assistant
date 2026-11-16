"""Select providers from settings.

`auto` uses OpenAI when an API key is present and the offline provider
otherwise. An explicit choice never silently falls back: asking for OpenAI
without a key is a configuration error.
"""

from __future__ import annotations

from citeground.config import Settings
from citeground.errors import ProviderError
from citeground.llm.base import ChatModel, Embedder
from citeground.llm.hashing import HashingEmbedder
from citeground.llm.mock import MockChatModel
from citeground.llm.openai import OpenAIChatModel, OpenAIEmbedder


def resolve_provider_name(choice: str, *, api_key: str, offline_name: str) -> str:
    if choice == "auto":
        return "openai" if api_key else offline_name
    if choice in {offline_name, "openai"}:
        return choice
    raise ProviderError(f"Unknown provider '{choice}'. Expected auto, openai, or {offline_name}.")


def build_embedder(settings: Settings) -> Embedder:
    name = resolve_provider_name(
        settings.embedding_provider,
        api_key=settings.openai_api_key,
        offline_name="hashing",
    )
    if name == "hashing":
        return HashingEmbedder(dim=settings.embedding_dim)
    return OpenAIEmbedder(
        api_key=settings.openai_api_key,
        model=settings.embedding_model,
        base_url=settings.openai_base_url,
    )


def build_chat_model(settings: Settings) -> ChatModel:
    name = resolve_provider_name(
        settings.llm_provider,
        api_key=settings.openai_api_key,
        offline_name="mock",
    )
    if name == "mock":
        return MockChatModel()
    return OpenAIChatModel(
        api_key=settings.openai_api_key,
        model=settings.chat_model,
        base_url=settings.openai_base_url,
    )

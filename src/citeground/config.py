"""Runtime settings.

Environment variables use the `CITEGROUND_` prefix. `OPENAI_API_KEY` is also
read unprefixed because that is the name operators already have exported.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def project_root() -> Path:
    """Find the checkout root so relative data paths work from any cwd.

    `CITEGROUND_ROOT` wins. Otherwise walk upward from the working directory
    looking for this repo's pyproject and corpus. Fall back to the cwd when
    the package is installed without the corpus checkout.
    """

    override = os.environ.get("CITEGROUND_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    cwd = Path.cwd().resolve()
    for candidate in [cwd, *cwd.parents]:
        if (candidate / "pyproject.toml").exists() and (candidate / "data" / "corpus").exists():
            return candidate
    return cwd


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CITEGROUND_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    llm_provider: str = "auto"
    embedding_provider: str = "auto"
    openai_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("OPENAI_API_KEY", "CITEGROUND_OPENAI_API_KEY"),
    )
    openai_base_url: str = "https://api.openai.com/v1"
    chat_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 256

    corpus_dir: Path = Path("data/corpus")
    index_dir: Path = Path("data/index")
    vector_store: str = "numpy"

    chunker: str = "recursive"
    chunk_size: int = 900
    chunk_overlap: int = 150
    sentence_window: int = 3
    sentence_stride: int = 2

    retrieval_mode: str = "hybrid"
    top_k: int = 5
    candidate_k: int = 20
    rrf_k: int = 60
    rerank: bool = True

    traces_path: Path = Path("data/traces/traces.jsonl")
    eval_dataset: Path = Path("data/eval/qa.jsonl")
    eval_output_dir: Path = Path("artifacts/eval")

    price_chat_input_per_mtok: float = 0.15
    price_chat_output_per_mtok: float = 0.60
    price_embedding_per_mtok: float = 0.02

    cors_origins: str = "http://127.0.0.1:8501,http://localhost:8501"

    def resolve(self, path: Path) -> Path:
        if path.is_absolute():
            return path
        return project_root() / path

    @property
    def cors_origin_list(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

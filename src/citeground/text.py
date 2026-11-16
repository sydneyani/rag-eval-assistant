"""Tokenization and light text helpers shared by retrieval and evaluation.

The tokenizer is deliberately small. Financial amounts, percents, and
hyphenated product names have to survive as single tokens or BM25 and the
hashing embedder will miss the facts this corpus is built around.
"""

from __future__ import annotations

import re

_TOKEN = re.compile(
    r"\$?\d+(?:\.\d+)?(?:-[a-z0-9]+)+%?"
    r"|\$?\d+(?:\.\d+)?%?"
    r"|[a-z][a-z0-9]*(?:-[a-z0-9]+)*"
)

# Closed-class words only. Domain words such as "revenue", "percent", and
# "million" are content and must not be dropped.
_STOP = frozenset(
    {
        "a",
        "an",
        "the",
        "of",
        "and",
        "or",
        "to",
        "for",
        "in",
        "on",
        "at",
        "by",
        "from",
        "with",
        "as",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "that",
        "this",
        "these",
        "those",
        "it",
        "its",
        "their",
        "his",
        "her",
        "our",
        "your",
        "they",
        "them",
        "we",
        "you",
        "not",
        "no",
        "into",
        "over",
        "under",
        "than",
        "then",
        "also",
        "about",
        "across",
        "after",
        "before",
    }
)

_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"])")


def normalize_token(token: str) -> str:
    return token.lstrip("$").rstrip("%")


def tokenize(text: str, *, drop_stop: bool = True) -> list[str]:
    lowered = text.lower().replace("\u2019", "'").replace("\u2014", " ")
    tokens = [normalize_token(match) for match in _TOKEN.findall(lowered)]
    tokens = [token for token in tokens if token]
    if drop_stop:
        tokens = [token for token in tokens if token not in _STOP]
    return tokens


def split_sentences(text: str) -> list[str]:
    collapsed = " ".join(text.split())
    if not collapsed:
        return []
    parts = _SENTENCE.split(collapsed)
    return [part.strip() for part in parts if part.strip()]


def estimate_tokens(text: str) -> int:
    """Rough token estimate used when a provider does not return usage.

    Four characters per token is the usual English rule of thumb. It is an
    estimate, and cost tracking labels it as such for the offline provider.
    """

    stripped = text.strip()
    if not stripped:
        return 0
    return max(1, (len(stripped) + 3) // 4)


def windows(text: str, size: int, overlap: int) -> list[tuple[int, int]]:
    """Character windows as half-open spans. Overlap must be smaller than size."""

    if size <= 0:
        raise ValueError("chunk size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be >= 0 and < chunk size")
    if not text:
        return []
    spans: list[tuple[int, int]] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + size, length)
        spans.append((start, end))
        if end == length:
            break
        start = end - overlap
    return spans

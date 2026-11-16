"""Retrieval and answer metrics.

Retrieval is section-level. Recall@k is the fraction of gold sections that
appear anywhere in the top k chunks. MRR is the reciprocal rank of the first
chunk whose section is gold. Both ignore duplicate chunks from the same
section for the set test, and MRR still uses chunk rank, so a wrong chunk
from a neighboring section costs rank.

Answer metrics are lexical proxies for the RAGAS-style judgments, computed
with the same tokenizer as retrieval. They are not an LLM's opinion. The
optional LLM judge lives next to them and is not used for the committed
offline numbers.

* Faithfulness: mean, over answer sentences, of the best token-F1 against any
  context sentence. An abstention scores 1. It made no unsupported claim.
* Answer relevance: token-F1 between the answer and the gold answer. A
  correct abstention on a question that should abstain scores 1. A wrong
  abstention, or an answer to an unanswerable question, scores 0.
* Citation support: mean token-recall of each citing sentence against the
  passage it cites. An abstention with no citation scores 1.
"""

from __future__ import annotations

import re
from collections import Counter

from citeground.generation.answer import is_abstention
from citeground.text import split_sentences, tokenize

_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def recall_at_k(gold: set[str], retrieved_sections: list[str], k: int) -> float:
    if not gold:
        return 0.0
    found = set(retrieved_sections[:k])
    return len(gold & found) / len(gold)


def mean_reciprocal_rank(gold: set[str], retrieved_sections: list[str]) -> float:
    if not gold:
        return 0.0
    for rank, section_id in enumerate(retrieved_sections, start=1):
        if section_id in gold:
            return 1.0 / rank
    return 0.0


def strip_citations(text: str) -> str:
    return _CITATION.sub("", text).strip()


def token_f1(left: str, right: str) -> float:
    left_counts = Counter(tokenize(left))
    right_counts = Counter(tokenize(right))
    if not left_counts or not right_counts:
        return 0.0
    overlap = sum((left_counts & right_counts).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(left_counts.values())
    recall = overlap / sum(right_counts.values())
    return 2 * precision * recall / (precision + recall)


def token_recall(claim: str, source: str) -> float:
    claim_counts = Counter(tokenize(claim))
    if not claim_counts:
        return 0.0
    source_counts = Counter(tokenize(source))
    overlap = sum((claim_counts & source_counts).values())
    return overlap / sum(claim_counts.values())


def _content_sentences(text: str) -> list[str]:
    return [
        sentence
        for sentence in split_sentences(strip_citations(text))
        if len(tokenize(sentence)) >= 4
    ]


def faithfulness(answer: str, contexts: list[str]) -> float:
    if is_abstention(answer):
        return 1.0
    sentences = _content_sentences(answer)
    if not sentences:
        return 0.0
    context_sentences = [
        sentence
        for context in contexts
        for sentence in split_sentences(context)
        if tokenize(sentence)
    ]
    if not context_sentences:
        return 0.0
    scores = [
        max(token_f1(sentence, context) for context in context_sentences) for sentence in sentences
    ]
    return sum(scores) / len(scores)


def answer_relevance(answer: str, gold_answer: str, *, expect_abstention: bool) -> float:
    abstained = is_abstention(answer)
    if expect_abstention:
        return 1.0 if abstained else 0.0
    if abstained or not gold_answer.strip():
        return 0.0
    return token_f1(strip_citations(answer), gold_answer)


def citation_support(answer: str, cited_texts: list[tuple[str, str]]) -> float:
    """`cited_texts` is (sentence that carries the marker, text of the cited chunk)."""

    if is_abstention(answer) and not _CITATION.search(answer):
        return 1.0
    if not cited_texts:
        return 0.0
    scores = [token_recall(sentence, source) for sentence, source in cited_texts]
    return sum(scores) / len(scores)


def answer_units(answer: str) -> list[str]:
    """Split an answer on citation markers so the next sentence is not glued on.

    The offline reader writes `sentence. [1] Next sentence. [2]`. A splitter
    that only looks for a capital letter after a period never cuts at `[1]`,
    and the two claims get scored as one.
    """

    units: list[str] = []
    for part in re.split(r"(?<=\])\s+", answer.strip()):
        units.extend(sentence for sentence in split_sentences(part) if sentence.strip())
    return units


def citing_pairs(answer: str, chunk_texts: list[str]) -> list[tuple[str, str]]:
    """Pair each citing sentence with the first passage it marks.

    Markers are 1-based indexes into `chunk_texts`, matching the prompt.
    """

    pairs: list[tuple[str, str]] = []
    for sentence in answer_units(answer):
        markers = [
            int(raw.strip()) for match in _CITATION.findall(sentence) for raw in match.split(",")
        ]
        markers = [marker for marker in markers if 1 <= marker <= len(chunk_texts)]
        if not markers:
            continue
        pairs.append((strip_citations(sentence), chunk_texts[markers[0] - 1]))
    return pairs

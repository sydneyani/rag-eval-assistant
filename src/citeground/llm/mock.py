"""Extractive offline chat model.

Given the prompt format produced by the generator, this model scores context
sentences against the question and returns the best-supported sentences with
the citation markers the prompt assigned. It does not invent numbers.

If no sentence overlaps the question, it abstains. It also abstains when two
or more content words from the question never appear in the chosen passage,
so a sentence about the chief executive officer is not offered as the answer
to a question about that officer's favorite color. Abstention is a product
behavior: a filing analyst that guesses is worse than one that says the
passages do not state the answer.

The implementation is deterministic, which is what makes the evaluation
harness reproducible without an API key.
"""

from __future__ import annotations

import re

from citeground.llm.base import Completion
from citeground.text import estimate_tokens, split_sentences, tokenize

_QUESTION = re.compile(r"QUESTION:\n(.*?)(?:\n\nCONTEXT:|\Z)", re.DOTALL)
_BLOCK = re.compile(r"\[(\d+)\][^\n]*\n(.*?)(?=\n\[(\d+)\]|\Z)", re.DOTALL)
_ABSTAIN = "The retrieved passages do not state this directly."


def _covers(needle: str, haystack: set[str]) -> bool:
    if needle in haystack:
        return True
    if len(needle) < 5:
        return False
    return any(
        token.startswith(needle) or needle.startswith(token)
        for token in haystack
        if len(token) >= 5
    )


def _unsupported(query_terms: set[str], passage: str) -> list[str]:
    """Content words from the question that the passage never uses."""

    passage_terms = set(tokenize(passage))
    missing = [
        term
        for term in query_terms
        if len(term) >= 5 and not term[:1].isdigit() and not _covers(term, passage_terms)
    ]
    return sorted(set(missing))


def _overlap(query_terms: set[str], sentence: str) -> float:
    if not query_terms:
        return 0.0
    terms = set(tokenize(sentence))
    if not terms:
        return 0.0
    return len(query_terms & terms) / len(query_terms)


class MockChatModel:
    """Sentence-extractive reader. Provider name `mock`."""

    provider_name = "mock"
    model_name = "mock-extractive"

    def __init__(self, overlap_threshold: float = 0.2) -> None:
        self.overlap_threshold = overlap_threshold

    def complete(self, messages: list[dict[str, str]], *, temperature: float = 0.0) -> Completion:
        del temperature
        user = ""
        for message in reversed(messages):
            if message.get("role") == "user":
                user = message.get("content", "")
                break
        answer = self._answer(user)
        prompt = "\n".join(message.get("content", "") for message in messages)
        return Completion(
            text=answer,
            prompt_tokens=estimate_tokens(prompt),
            completion_tokens=estimate_tokens(answer),
            model=self.model_name,
        )

    def _answer(self, prompt: str) -> str:
        question_match = _QUESTION.search(prompt)
        question = question_match.group(1).strip() if question_match else ""
        blocks = [(int(marker), body.strip()) for marker, body, _rest in _BLOCK.findall(prompt)]
        if not blocks or not question:
            return _ABSTAIN
        query_terms = set(tokenize(question))
        ranked: list[tuple[float, int, str]] = []
        for marker, body in blocks:
            for sentence in _sentences(body):
                score = _overlap(query_terms, sentence)
                if score <= 0:
                    continue
                ranked.append((score, marker, sentence))
        ranked.sort(key=lambda item: (-item[0], item[1], item[2]))
        if not ranked or ranked[0][0] < self.overlap_threshold:
            return _ABSTAIN
        passages = {marker: body for marker, body in blocks}
        if len(_unsupported(query_terms, passages[ranked[0][1]])) >= 2:
            return _ABSTAIN
        chosen: list[tuple[int, str]] = []
        seen: set[str] = set()
        for score, marker, sentence in ranked:
            if score < self.overlap_threshold:
                break
            key = sentence.lower()
            if key in seen:
                continue
            seen.add(key)
            chosen.append((marker, sentence))
            if len(chosen) == 2:
                break
        lines = [f"{sentence} [{marker}]" for marker, sentence in chosen]
        return " ".join(lines)


def _sentences(text: str) -> list[str]:
    return [sentence for sentence in split_sentences(text) if len(tokenize(sentence)) >= 4]

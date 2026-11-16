"""Two judges, one result shape.

The deterministic judge is the offline mode. It is what CI runs and what the
committed table reports. The LLM judge asks a chat model for the same two
scores as JSON. It is available when the chat provider is OpenAI. It is not
mixed into the offline numbers.
"""

from __future__ import annotations

import json
import re
from typing import Protocol

from citeground.errors import ProviderError
from citeground.eval.metrics import answer_relevance, citation_support, citing_pairs, faithfulness
from citeground.llm.base import ChatModel

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


class Judge(Protocol):
    name: str

    def score(
        self,
        *,
        question: str,
        answer: str,
        contexts: list[str],
        gold_answer: str,
        expect_abstention: bool,
    ) -> dict[str, float]: ...


class DeterministicJudge:
    name = "offline-deterministic"

    def score(
        self,
        *,
        question: str,
        answer: str,
        contexts: list[str],
        gold_answer: str,
        expect_abstention: bool,
    ) -> dict[str, float]:
        del question
        return {
            "faithfulness": faithfulness(answer, contexts),
            "answer_relevance": answer_relevance(
                answer, gold_answer, expect_abstention=expect_abstention
            ),
            "citation_support": citation_support(answer, citing_pairs(answer, contexts)),
        }


_JUDGE_SYSTEM = (
    "You score one grounded answer. Faithfulness is how well the answer is supported "
    "by the context, from 0 to 1. Answer relevance is how well it addresses the "
    "question relative to the gold answer, from 0 to 1. Reply with a JSON object and "
    'nothing else, shaped {"faithfulness": <float>, "answer_relevance": <float>}.'
)


class LLMJudge:
    name = "llm"

    def __init__(self, chat: ChatModel) -> None:
        if chat.provider_name == "mock":
            raise ProviderError(
                "The LLM judge needs an OpenAI chat model. "
                "The offline table uses the deterministic judge."
            )
        self.chat = chat

    def score(
        self,
        *,
        question: str,
        answer: str,
        contexts: list[str],
        gold_answer: str,
        expect_abstention: bool,
    ) -> dict[str, float]:
        context = "\n\n".join(contexts) if contexts else "(none)"
        user = (
            f"QUESTION:\n{question}\n\nGOLD:\n{gold_answer or '(abstain)'}\n\n"
            f"EXPECT_ABSTENTION:\n{expect_abstention}\n\nANSWER:\n{answer}\n\nCONTEXT:\n{context}\n"
        )
        completion = self.chat.complete(
            [
                {"role": "system", "content": _JUDGE_SYSTEM},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
        )
        return _parse_judge_json(completion.text)


def _parse_judge_json(text: str) -> dict[str, float]:
    match = _JSON_OBJECT.search(text)
    if not match:
        raise ProviderError("LLM judge did not return a JSON object")
    try:
        payload = json.loads(match.group(0))
        faithfulness_score = float(payload["faithfulness"])
        relevance_score = float(payload["answer_relevance"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ProviderError("LLM judge JSON was missing faithfulness or answer_relevance") from exc
    for score in (faithfulness_score, relevance_score):
        if not 0.0 <= score <= 1.0:
            raise ProviderError("LLM judge scores must be between 0 and 1")
    return {"faithfulness": faithfulness_score, "answer_relevance": relevance_score}

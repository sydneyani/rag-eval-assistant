"""Retrieve, then ask a chat model to answer only from those passages."""

from __future__ import annotations

import logging
import re
import time
import uuid
from collections.abc import Callable

from citeground.config import Settings
from citeground.generation.prompts import SYSTEM_PROMPT, build_user_prompt
from citeground.llm.base import ChatModel
from citeground.observability.cost import estimate_cost
from citeground.observability.tracing import TraceSink
from citeground.retrieval.service import HybridRetriever
from citeground.schemas import AnswerResult, Citation, ScoredChunk, StageTiming, Usage
from citeground.text import estimate_tokens

logger = logging.getLogger("citeground.answer")

_ABSTAIN_MARKERS = (
    "do not state",
    "does not state",
    "cannot answer",
    "can't answer",
)


def is_abstention(answer: str) -> bool:
    lowered = answer.lower()
    return any(marker in lowered for marker in _ABSTAIN_MARKERS)


def parse_citations(answer: str, hits: list[ScoredChunk]) -> list[Citation]:
    citations: list[Citation] = []
    seen: set[int] = set()
    for match in re.finditer(r"\[(\d+(?:\s*,\s*\d+)*)\]", answer):
        for raw in match.group(1).split(","):
            marker = int(raw.strip())
            if marker in seen or marker < 1 or marker > len(hits):
                continue
            seen.add(marker)
            hit = hits[marker - 1]
            quote = hit.chunk.text.strip()
            if len(quote) > 360:
                quote = quote[:357].rstrip() + "..."
            citations.append(
                Citation(
                    marker=marker,
                    chunk_id=hit.chunk.chunk_id,
                    doc_id=hit.chunk.doc_id,
                    section_id=hit.chunk.section_id,
                    doc_title=hit.chunk.doc_title,
                    section_title=hit.chunk.section_title,
                    score=hit.score,
                    quote=quote,
                )
            )
    citations.sort(key=lambda item: item.marker)
    return citations


class AnswerService:
    def __init__(
        self,
        retriever: HybridRetriever,
        chat: ChatModel,
        settings: Settings,
        traces: TraceSink,
        *,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.retriever = retriever
        self.chat = chat
        self.settings = settings
        self.traces = traces
        self._clock = clock

    def answer(
        self,
        question: str,
        *,
        top_k: int | None = None,
        mode: str | None = None,
        rerank: bool | None = None,
        trace_id: str | None = None,
    ) -> AnswerResult:
        started = self._clock()
        trace = trace_id or uuid.uuid4().hex
        selected_mode = mode or self.retriever.mode
        limit = top_k if top_k is not None else self.settings.top_k

        retrieve_started = self._clock()
        hits = self.retriever.retrieve(
            question,
            limit,
            mode=selected_mode,
            rerank_results=rerank,
        )
        retrieve_ms = _elapsed_ms(self._clock, retrieve_started)

        generate_started = self._clock()
        completion = self.chat.complete(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(question, hits)},
            ],
            temperature=0.0,
        )
        generate_ms = _elapsed_ms(self._clock, generate_started)

        embedding_tokens = 0
        if selected_mode in {"dense", "hybrid"}:
            embedding_tokens = estimate_tokens(question)
        embedding_provider = self.retriever.embedder.provider_name
        cost, basis = estimate_cost(
            self.settings,
            prompt_tokens=completion.prompt_tokens,
            completion_tokens=completion.completion_tokens,
            embedding_tokens=embedding_tokens,
            chat_provider=self.chat.provider_name,
            embedding_provider=embedding_provider,
        )
        usage = Usage(
            prompt_tokens=completion.prompt_tokens,
            completion_tokens=completion.completion_tokens,
            embedding_tokens=embedding_tokens,
            cost_usd=cost,
            cost_basis=basis,
        )
        stages = [
            StageTiming(name="retrieve", duration_ms=retrieve_ms),
            StageTiming(name="generate", duration_ms=generate_ms),
        ]
        result = AnswerResult(
            question=question,
            answer=completion.text,
            citations=parse_citations(completion.text, hits),
            retrieved=hits,
            provider=self.chat.provider_name,
            model=completion.model,
            usage=usage,
            trace_id=trace,
            latency_ms=_elapsed_ms(self._clock, started),
            abstained=is_abstention(completion.text),
            stages=stages,
        )
        self.traces.write(
            {
                "trace_id": result.trace_id,
                "question": question,
                "provider": result.provider,
                "model": result.model,
                "retrieval_mode": selected_mode,
                "abstained": result.abstained,
                "latency_ms": result.latency_ms,
                "stages": [stage.model_dump() for stage in stages],
                "usage": usage.model_dump(),
                "citation_section_ids": [citation.section_id for citation in result.citations],
                "retrieved_section_ids": [hit.chunk.section_id for hit in hits],
            }
        )
        logger.info(
            "answer",
            extra={
                "trace_id": result.trace_id,
                "provider": result.provider,
                "latency_ms": round(result.latency_ms, 2),
            },
        )
        return result


def _elapsed_ms(clock: Callable[[], float], started: float) -> float:
    return round((clock() - started) * 1000, 3)

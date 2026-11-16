"""In-process metrics and a JSONL trace sink.

The sink is append-only. Each answer is one line: the trace id, stage
latencies, retrieval mode, provider, and token usage. The metrics registry
is the aggregate a `/v1/metrics` caller reads. Neither store is a tracing
vendor. The trace id is the join key if you later export these lines.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from citeground.schemas import Usage


class MetricsRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.requests = 0
        self.http_errors = 0
        self.answers = 0
        self.abstentions = 0
        self.latency_ms_sum = 0.0
        self.answer_latency_ms_sum = 0.0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.embedding_tokens = 0
        self.cost_usd = 0.0

    def record_http(self, status: int, latency_ms: float) -> None:
        with self._lock:
            self.requests += 1
            self.latency_ms_sum += latency_ms
            if status >= 400:
                self.http_errors += 1

    def record_answer(self, *, latency_ms: float, usage: Usage, abstained: bool) -> None:
        with self._lock:
            self.answers += 1
            self.answer_latency_ms_sum += latency_ms
            self.prompt_tokens += usage.prompt_tokens
            self.completion_tokens += usage.completion_tokens
            self.embedding_tokens += usage.embedding_tokens
            self.cost_usd += usage.cost_usd
            if abstained:
                self.abstentions += 1

    def snapshot(self) -> dict[str, float | int]:
        with self._lock:
            requests = self.requests
            answers = self.answers
            return {
                "requests": requests,
                "http_errors": self.http_errors,
                "answers": answers,
                "abstentions": self.abstentions,
                "mean_http_latency_ms": (self.latency_ms_sum / requests) if requests else 0.0,
                "mean_answer_latency_ms": (self.answer_latency_ms_sum / answers)
                if answers
                else 0.0,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "embedding_tokens": self.embedding_tokens,
                "cost_usd": round(self.cost_usd, 8),
            }


class TraceSink:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def write(self, record: dict[str, object]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, default=str)
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

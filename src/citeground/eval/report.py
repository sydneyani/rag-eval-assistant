"""Render the experiment table.

The evaluate command and `scripts/compare_experiments.py` both call this, so
the markdown in the repo and the markdown in the terminal cannot drift apart
from the JSON.
"""

from __future__ import annotations


def format_metric(value: float) -> str:
    return f"{value:.3f}"


def render_comparison(payload: dict) -> str:
    lines = [
        "# Evaluation comparison",
        "",
        f"Judge: `{payload['judge']}`.",
        f"Embeddings: `{payload['embedding_provider']}` / `{payload['embedding_model']}`.",
        f"Reader: `{payload['llm_provider']}` / `{payload['llm_model']}`.",
        f"Questions: {payload['n_questions']} "
        f"({payload['n_answerable']} answerable, {payload['n_unanswerable']} meant to abstain).",
        "",
        payload["label"],
        "",
        "| Experiment | Recall@5 | Recall@10 | MRR | Faithfulness | Answer relevance | Citation support | Abstention accuracy |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for experiment in payload["experiments"]:
        metrics = experiment["metrics"]
        lines.append(
            "| {name} | {recall_at_5} | {recall_at_10} | {mrr} | {faithfulness} | "
            "{answer_relevance} | {citation_support} | {abstention_accuracy} |".format(
                name=experiment["name"],
                recall_at_5=format_metric(metrics["recall_at_5"]),
                recall_at_10=format_metric(metrics["recall_at_10"]),
                mrr=format_metric(metrics["mrr"]),
                faithfulness=format_metric(metrics["faithfulness"]),
                answer_relevance=format_metric(metrics["answer_relevance"]),
                citation_support=format_metric(metrics["citation_support"]),
                abstention_accuracy=format_metric(metrics["abstention_accuracy"]),
            )
        )
    lines.extend(
        [
            "",
            "Recall and MRR are averaged over answerable questions only. "
            "An empty gold set would make those metrics zero for a correct abstention.",
            "Faithfulness, answer relevance, and citation support are averaged over every question.",
            "Abstention accuracy is the fraction of questions marked `expect_abstention` "
            "where the reader abstained. It is not a retrieval metric.",
            "",
        ]
    )
    return "\n".join(lines)

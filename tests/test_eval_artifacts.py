import json
from pathlib import Path

from citeground.eval.report import format_metric

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "artifacts" / "eval" / "results.json"
README = ROOT / "README.md"


def test_committed_table_matches_the_offline_run() -> None:
    payload = json.loads(RESULTS.read_text(encoding="utf-8"))
    readme = README.read_text(encoding="utf-8")
    assert payload["judge"] == "offline-deterministic"
    assert payload["embedding_provider"] == "hashing"
    assert payload["llm_provider"] == "mock"
    assert payload["n_questions"] == 28
    assert payload["n_answerable"] == 27
    assert "offline-deterministic" in readme
    assert "not RAGAS" in readme or "not an LLM" in readme or "not RAGAS scores" in readme
    names = [experiment["name"] for experiment in payload["experiments"]]
    assert names == [
        "bm25-recursive-900",
        "dense-hash-recursive-900",
        "hybrid-rerank-recursive-900",
        "hybrid-rerank-fixed-500",
    ]
    for experiment in payload["experiments"]:
        assert experiment["name"] in readme
        metrics = experiment["metrics"]
        for key in (
            "recall_at_5",
            "recall_at_10",
            "mrr",
            "faithfulness",
            "answer_relevance",
            "citation_support",
            "abstention_accuracy",
        ):
            assert format_metric(metrics[key]) in readme

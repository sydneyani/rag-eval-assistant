from citeground.eval.metrics import (
    answer_relevance,
    citation_support,
    citing_pairs,
    faithfulness,
    mean_reciprocal_rank,
    recall_at_k,
    token_f1,
)
from citeground.llm.base import Completion


def test_recall_and_mrr_are_section_level() -> None:
    gold = {"sec-a", "sec-b"}
    retrieved = ["sec-c", "sec-a", "sec-a", "sec-b"]
    assert recall_at_k(gold, retrieved, 1) == 0.0
    assert recall_at_k(gold, retrieved, 2) == 0.5
    assert recall_at_k(gold, retrieved, 4) == 1.0
    assert mean_reciprocal_rank(gold, retrieved) == 0.5
    assert mean_reciprocal_rank(gold, ["other"]) == 0.0
    assert recall_at_k(set(), retrieved, 5) == 0.0


def test_exact_extractive_sentence_is_faithful_and_relevant() -> None:
    sentence = (
        "Fiscal 2024 revenue was $4.82 billion, compared with fiscal 2023 revenue of $4.08 billion."
    )
    assert token_f1(sentence, sentence) == 1.0
    assert faithfulness(f"{sentence} [1]", [sentence]) == 1.0
    assert answer_relevance(f"{sentence} [1]", sentence, expect_abstention=False) == 1.0


def test_abstention_is_faithful_and_relevant_only_when_expected() -> None:
    answer = "The retrieved passages do not state this directly."
    assert faithfulness(answer, ["Unrelated filing text about revenue."]) == 1.0
    assert answer_relevance(answer, "Revenue was $4.82 billion.", expect_abstention=False) == 0.0
    assert answer_relevance(answer, "", expect_abstention=True) == 1.0
    assert citation_support(answer, []) == 1.0


def test_citation_markers_do_not_glue_two_claims_together() -> None:
    answer = (
        "Energy cost at Quarry Ridge was $146 million in fiscal 2024. [1] "
        "The power purchase agreement expires on June 30, 2026. [1]"
    )
    contexts = [
        "Energy cost at Quarry Ridge was $146 million in fiscal 2024. "
        "The power purchase agreement expires on June 30, 2026."
    ]
    pairs = citing_pairs(answer, contexts)
    assert len(pairs) == 2
    assert citation_support(answer, pairs) == 1.0


def test_citation_support_requires_the_claim_to_be_in_the_cited_passage() -> None:
    answer = "Backlog was $2.05 billion as of February 2, 2025. [1]"
    contexts = ["Backlog was $2.05 billion as of February 2, 2025, compared with the prior year."]
    pairs = citing_pairs(answer, contexts)
    assert len(pairs) == 1
    assert citation_support(answer, pairs) > 0.8


def test_llm_judge_parses_a_json_object() -> None:
    from citeground.eval.judge import LLMJudge

    class StubChat:
        provider_name = "openai"
        model_name = "stub"

        def complete(self, messages, *, temperature: float = 0.0) -> Completion:
            del messages, temperature
            return Completion(
                text='Here you go {"faithfulness": 0.4, "answer_relevance": 0.7}',
                prompt_tokens=1,
                completion_tokens=1,
                model="stub",
            )

    scores = LLMJudge(StubChat()).score(
        question="q",
        answer="a",
        contexts=["c"],
        gold_answer="g",
        expect_abstention=False,
    )
    assert scores == {"faithfulness": 0.4, "answer_relevance": 0.7}

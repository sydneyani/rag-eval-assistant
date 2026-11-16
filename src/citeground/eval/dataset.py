"""Load the labeled question file.

Each line is one question. `gold_section_ids` are the sections that state the
answer. Retrieval is scored against those ids, not against chunk offsets, so
a chunking experiment does not force a relabel. `expect_abstention` marks a
question the corpus does not answer.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field


class LabeledQuestion(BaseModel):
    id: str
    question: str
    gold_answer: str = ""
    gold_section_ids: list[str] = Field(default_factory=list)
    expect_abstention: bool = False


def load_dataset(path: Path) -> list[LabeledQuestion]:
    questions: list[LabeledQuestion] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            questions.append(LabeledQuestion.model_validate_json(line))
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number} is not a labeled question") from exc
    if not questions:
        raise ValueError(f"No questions in {path}")
    seen: set[str] = set()
    for question in questions:
        if question.id in seen:
            raise ValueError(f"Duplicate question id {question.id}")
        seen.add(question.id)
        if not question.expect_abstention and not question.gold_section_ids:
            raise ValueError(f"{question.id} needs gold sections or expect_abstention")
    return questions


def dump_dataset(path: Path, questions: list[LabeledQuestion]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(
        json.dumps(question.model_dump(), ensure_ascii=True) for question in questions
    )
    path.write_text(payload + "\n", encoding="utf-8")

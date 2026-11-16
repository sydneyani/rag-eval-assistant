"""Prompt format shared by the OpenAI model and the offline reader.

The offline reader parses `QUESTION:` and numbered `CONTEXT:` blocks. Keep
that shape stable. A change here is a change to the mock model's contract
and to the citation markers the UI expects.
"""

from __future__ import annotations

from citeground.schemas import ScoredChunk

SYSTEM_PROMPT = (
    "You are CiteGround, a filing analyst. Answer using only the numbered context "
    "passages. After each claim, cite the supporting passage with its marker, such "
    "as [1]. If the passages do not state the answer, say that you cannot answer "
    "from the retrieved passages and do not guess. Do not use outside knowledge. "
    "Write a few sentences at most."
)


def build_user_prompt(question: str, hits: list[ScoredChunk]) -> str:
    blocks: list[str] = []
    for marker, hit in enumerate(hits, start=1):
        header = (
            f"[{marker}] title={hit.chunk.doc_title} | "
            f"section={hit.chunk.section_title} | id={hit.chunk.section_id}"
        )
        blocks.append(f"{header}\n{hit.chunk.text}")
    context = "\n\n".join(blocks) if blocks else "(no passages retrieved)"
    return f"QUESTION:\n{question.strip()}\n\nCONTEXT:\n{context}\n"

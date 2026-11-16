"""Run the labeled evaluation and write the comparison artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from citeground.config import Settings
from citeground.errors import ProviderError
from citeground.eval.experiments import specs_by_name
from citeground.eval.judge import DeterministicJudge, LLMJudge
from citeground.eval.report import render_comparison
from citeground.eval.runner import run_evaluation
from citeground.llm.factory import build_chat_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="citeground-eval",
        description="Compare retrieval stacks on the labeled Helios questions.",
    )
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--experiments", nargs="*", default=None)
    parser.add_argument("--judge", choices=["deterministic", "llm"], default="deterministic")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    settings = Settings()
    if args.judge == "deterministic":
        settings = settings.model_copy(
            update={"llm_provider": "mock", "embedding_provider": "hashing"}
        )
        judge = DeterministicJudge()
    else:
        chat = build_chat_model(settings)
        try:
            judge = LLMJudge(chat)
        except ProviderError as exc:
            raise SystemExit(str(exc)) from exc
    dataset = args.dataset or settings.resolve(settings.eval_dataset)
    out = args.out or settings.resolve(settings.eval_output_dir)
    payload = run_evaluation(
        settings=settings,
        dataset_path=dataset,
        specs=specs_by_name(args.experiments),
        work_dir=out / "indexes",
        judge=judge,
    )
    payload["dataset"] = _portable(dataset)
    _write(out, payload)
    print(render_comparison(payload))
    print(f"Wrote {out / 'results.json'} and {out / 'comparison.md'}")


def _portable(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(resolved)


def _write(out: Path, payload: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    public = {key: value for key, value in payload.items() if key != "experiments"}
    public["experiments"] = []
    per_question: list[dict] = []
    for experiment in payload["experiments"]:
        summary = {key: value for key, value in experiment.items() if key != "questions"}
        public["experiments"].append(summary)
        for row in experiment["questions"]:
            per_question.append({"experiment": experiment["name"], **row})
    (out / "results.json").write_text(json.dumps(public, indent=2), encoding="utf-8")
    (out / "per_question.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in per_question),
        encoding="utf-8",
    )
    (out / "comparison.md").write_text(render_comparison(public), encoding="utf-8")


if __name__ == "__main__":
    main()

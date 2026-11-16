"""Re-render the experiment table from a results file.

The evaluate command writes artifacts/eval/results.json. This script prints
the same markdown the command writes, without re-running retrieval.

    python scripts/compare_experiments.py
    python scripts/compare_experiments.py artifacts/eval/results.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from citeground.eval.report import render_comparison


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    path = Path(args[0] if args else "artifacts/eval/results.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    print(render_comparison(payload), end="")


if __name__ == "__main__":
    main()

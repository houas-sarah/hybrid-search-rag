from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

EVAL_SET_PATH = Path(__file__).parent / "eval_set.json"


@dataclass(slots=True)
class EvalCase:
    question: str
    relevant_sources: list[str]  # filenames that should answer the question
    expected_keywords: list[str] = field(default_factory=list)


def load_cases(path: Path = EVAL_SET_PATH) -> list[EvalCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [EvalCase(**case) for case in raw]

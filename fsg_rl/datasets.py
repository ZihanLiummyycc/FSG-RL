"""Dataset loading for algorithmic mathematics problems."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .schemas import JsonDict, Problem


class DatasetError(ValueError):
    """Raised when a configured dataset cannot be parsed or validated."""


def load_dataset(config: Dict[str, Any]) -> List[Problem]:
    section = config.get("dataset", {})
    raw_path = section.get("algorithm_problem_dataset_path")
    if not raw_path:
        raise DatasetError("dataset.algorithm_problem_dataset_path is required")
    path = Path(str(raw_path)).expanduser()
    if not path.is_file():
        raise DatasetError(f"Dataset file does not exist: {path}")

    rows = list(_read_records(path))
    problems = [Problem.from_dict(_normalize_problem(row, index)) for index, row in enumerate(rows)]

    split = section.get("split")
    if split:
        problems = [problem for problem in problems if problem.split == split]
    if section.get("shuffle", False):
        random.Random(int(section.get("seed", 42))).shuffle(problems)
    offset = int(section.get("offset", 0))
    if offset < 0:
        raise DatasetError("dataset.offset must be non-negative")
    if offset:
        problems = problems[offset:]
    limit = section.get("limit")
    if limit is not None:
        problems = problems[: int(limit)]
    if not problems:
        raise DatasetError(f"No problems remained after loading/filtering {path}")
    return problems


def load_json_records(path: Path) -> List[JsonDict]:
    return list(_read_records(path))


def _read_records(path: Path) -> Iterable[JsonDict]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise DatasetError(f"Invalid JSONL at {path}:{line_number}") from exc
                if not isinstance(value, dict):
                    raise DatasetError(f"Expected JSON object at {path}:{line_number}")
                yield value
        return

    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if isinstance(value, dict):
        value = value.get("data", value.get("problems"))
    if not isinstance(value, list):
        raise DatasetError(f"JSON dataset must be a list or contain data/problems: {path}")
    for index, row in enumerate(value):
        if not isinstance(row, dict):
            raise DatasetError(f"Expected object at index {index} in {path}")
        yield row


def _normalize_problem(row: JsonDict, index: int) -> JsonDict:
    problem_id = row.get("id", row.get("problem_id", row.get("uid", f"problem-{index}")))
    text = row.get("text", row.get("problem", row.get("question")))
    answer = row.get("gold_answer", row.get("answer", row.get("final_answer")))
    if not isinstance(text, str) or not text.strip():
        raise DatasetError(f"Problem {problem_id!r} has no non-empty text/question field")
    if answer is None:
        raise DatasetError(f"Problem {problem_id!r} has no gold answer")
    metadata = dict(row.get("metadata", {}))
    for key in ("subject", "difficulty", "source", "solution"):
        if key in row and key not in metadata:
            metadata[key] = row[key]
    return {
        "id": str(problem_id),
        "text": text.strip(),
        "gold_answer": str(answer),
        "split": str(row.get("split", "train")),
        "metadata": metadata,
    }

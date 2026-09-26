#!/usr/bin/env python3
"""Export only evaluation prompt fields from a locked FSG-RL JSONL file."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


EXPECTED_ROWS = 400
FORBIDDEN_GRAPH_KEYS = {
    "gold_answer", "answer", "target_call", "tests", "expected",
    "reference_implementation", "mutant_implementation", "hidden_tests",
    "verification_graph", "verifier_annotation", "verifier_execution_audit",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_public_graph(value: object) -> None:
    if isinstance(value, dict):
        forbidden = FORBIDDEN_GRAPH_KEYS.intersection(value)
        if forbidden:
            raise ValueError(f"Private graph keys detected: {sorted(forbidden)}")
        for child in value.values():
            check_public_graph(child)
    elif isinstance(value, list):
        for child in value:
            check_public_graph(child)


def public_row(row: dict) -> dict:
    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Missing metadata")
    graph = metadata.get("function_graph")
    if not isinstance(graph, dict):
        raise ValueError("Missing public function_graph")
    check_public_graph(graph)
    problem_id = row.get("id")
    text = row.get("text")
    if not isinstance(problem_id, str) or not isinstance(text, str):
        raise ValueError("Missing problem ID or text")
    if row.get("split") != "locked_test" or metadata.get("do_not_train") is not True:
        raise ValueError("Expected a locked evaluation-only row")
    family = "omni" if problem_id.startswith("omni_math_") else "medium"
    return {
        "id": problem_id,
        "text": text,
        "source": row.get("source"),
        "difficulty": row.get("difficulty"),
        "family": family,
        "split": "test",
        "do_not_train": True,
        "function_graph": graph,
    }


def export(source: Path, output: Path, expected_hash: str) -> dict:
    actual_hash = sha256(source)
    if actual_hash != expected_hash:
        raise ValueError("Locked source SHA256 differs from the expected original")
    rows = []
    with source.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(public_row(json.loads(line)))
    ids = [row["id"] for row in rows]
    if len(rows) != EXPECTED_ROWS or len(set(ids)) != EXPECTED_ROWS:
        raise ValueError("Expected 400 unique evaluation problems")
    if Counter(row["family"] for row in rows) != {"medium": 320, "omni": 80}:
        raise ValueError("Medium/Omni family counts differ from the locked set")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    with output.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return {
        "rows": len(rows),
        "family_counts": dict(Counter(row["family"] for row in rows)),
        "source_counts": dict(sorted(Counter(row["source"] for row in rows).items())),
        "locked_source_sha256": actual_hash,
        "public_file_sha256": sha256(output),
        "contains_gold_answers_or_private_tests": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-sha256", required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.source, args.output, args.expected_sha256), indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Strip annotation audits while retaining GRPO training verifier records."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


DROP_METADATA = {"verifier_annotation", "verifier_execution_audit"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def training_row(row: dict) -> dict:
    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Missing training metadata")
    if not isinstance(metadata.get("function_graph"), dict):
        raise ValueError("Missing public function graph")
    if not isinstance(metadata.get("verification_graph"), dict):
        raise ValueError("Missing training verification graph")
    if not row.get("id") or not row.get("text") or not row.get("gold_answer"):
        raise ValueError("Missing training problem, question, or answer")
    cleaned = dict(row)
    cleaned["metadata"] = {
        key: value for key, value in metadata.items() if key not in DROP_METADATA
    }
    return cleaned


def export(source: Path, output: Path, expected_rows: int) -> dict:
    if output.exists():
        raise FileExistsError(output)
    rows = []
    with source.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(training_row(json.loads(line)))
    ids = [row["id"] for row in rows]
    if len(rows) != expected_rows or len(set(ids)) != expected_rows:
        raise ValueError(f"Expected {expected_rows} unique training records")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return {
        "records": len(rows),
        "source_sha256": sha256(source),
        "export_sha256": sha256(output),
        "annotation_audits_removed": True,
        "training_verification_graph_retained": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-rows", required=True, type=int)
    args = parser.parse_args()
    print(json.dumps(export(args.source, args.output, args.expected_rows), indent=2))


if __name__ == "__main__":
    main()

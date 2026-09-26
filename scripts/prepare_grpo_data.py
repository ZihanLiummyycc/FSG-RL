#!/usr/bin/env python3
"""Build a leakage-audited GRPO split from successful Omni-MATH SFT labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set

from fsg_rl.decomposition import validate_function_graph
from fsg_rl.schemas import FunctionGraph


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--master", required=True)
    parser.add_argument("--graph-train", required=True)
    parser.add_argument("--graph-validation", required=True)
    parser.add_argument("--holdout", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--audit-output", required=True)
    parser.add_argument("--expected-train-rows", type=int, default=798)
    args = parser.parse_args()

    master_rows = _read_jsonl(Path(args.master))
    graph_train_rows = _read_jsonl(Path(args.graph_train))
    graph_validation_rows = _read_jsonl(Path(args.graph_validation))
    holdout_rows = _read_jsonl(Path(args.holdout))

    masters = _unique_by_id(master_rows, "master")
    train_ids = _source_ids(graph_train_rows, "graph train")
    validation_ids = _source_ids(graph_validation_rows, "graph validation")
    holdout_ids = {str(row["id"]) for row in holdout_rows}

    _require_disjoint("train", train_ids, "validation", validation_ids)
    _require_disjoint("train", train_ids, "holdout", holdout_ids)
    _require_disjoint("validation", validation_ids, "holdout", holdout_ids)

    missing_master = train_ids - set(masters)
    if missing_master:
        raise ValueError(
            "Graph-train IDs missing from teacher master: "
            f"{sorted(missing_master)[:10]}"
        )
    if len(train_ids) != args.expected_train_rows:
        raise ValueError(
            f"Expected {args.expected_train_rows} unique train IDs, got {len(train_ids)}"
        )

    records = [_to_grpo_problem(masters[problem_id]) for problem_id in sorted(train_ids)]
    output_path = Path(args.output).expanduser().resolve()
    audit_path = Path(args.audit_output).expanduser().resolve()
    _write_jsonl(output_path, records)

    output_ids = {record["id"] for record in records}
    audit = {
        "master_rows": len(master_rows),
        "master_unique_ids": len(masters),
        "graph_train_rows": len(graph_train_rows),
        "graph_train_unique_ids": len(train_ids),
        "graph_validation_rows": len(graph_validation_rows),
        "graph_validation_unique_ids": len(validation_ids),
        "holdout_rows": len(holdout_rows),
        "holdout_unique_ids": len(holdout_ids),
        "grpo_output_rows": len(records),
        "train_validation_overlap": len(train_ids & validation_ids),
        "train_holdout_overlap": len(train_ids & holdout_ids),
        "validation_holdout_overlap": len(validation_ids & holdout_ids),
        "output_validation_overlap": len(output_ids & validation_ids),
        "output_holdout_overlap": len(output_ids & holdout_ids),
        "contains_reference_solution": any(
            _contains_forbidden_key(record, {"reference_solution", "tagged_solution"})
            for record in records
        ),
        "decomposition_backend": "dataset",
        "note": (
            "Each record contains the independent teacher function graph and gold final "
            "answer, but never the reference solution or tagged teacher solution. The "
            "current GRPO implementation updates graph-conditioned solve rollouts only."
        ),
        "output": str(output_path),
    }
    if audit["contains_reference_solution"]:
        raise ValueError("Reference-solution leakage detected in GRPO output")
    if any(
        audit[key]
        for key in (
            "train_validation_overlap",
            "train_holdout_overlap",
            "validation_holdout_overlap",
            "output_validation_overlap",
            "output_holdout_overlap",
        )
    ):
        raise ValueError(f"Split leakage detected: {audit}")
    _write_json(audit_path, audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


def _to_grpo_problem(master: Dict[str, Any]) -> Dict[str, Any]:
    graph = master.get("function_graph")
    if not isinstance(graph, dict):
        raise ValueError(f"Master {master.get('id')!r} has no function_graph object")
    graph_object = FunctionGraph.from_dict(graph)
    if graph_object.problem_id != str(master["id"]):
        raise ValueError(
            f"Master {master['id']!r} graph problem_id is {graph_object.problem_id!r}"
        )
    validate_function_graph(graph_object)
    problem = str(master.get("problem", "")).strip()
    answer = str(master.get("gold_answer", "")).strip()
    if not problem or not answer:
        raise ValueError(f"Master {master.get('id')!r} has empty problem/gold_answer")
    return {
        "id": str(master["id"]),
        "text": problem,
        "gold_answer": answer,
        "split": "train",
        "metadata": {
            "source": "KbsdJames/Omni-MATH",
            "domain": list(master.get("domain", [])),
            "difficulty": master.get("difficulty"),
            "function_graph": graph_object.to_dict(),
            "verification_graph_source": "gpt-5.6-terra-sft-label",
        },
    }


def _source_ids(rows: List[Dict[str, Any]], label: str) -> Set[str]:
    result = set()
    for row in rows:
        metadata = row.get("metadata", {})
        source_id = metadata.get("source_problem_id") if isinstance(metadata, dict) else None
        if not source_id:
            record_id = str(row.get("id", ""))
            source_id = record_id.removesuffix(":graph")
        source_id = str(source_id).strip()
        if not source_id:
            raise ValueError(f"{label} row has no source problem ID")
        if source_id in result:
            raise ValueError(f"Duplicate source ID in {label}: {source_id}")
        result.add(source_id)
    return result


def _unique_by_id(rows: List[Dict[str, Any]], label: str) -> Dict[str, Dict[str, Any]]:
    result = {}
    for row in rows:
        row_id = str(row.get("id", "")).strip()
        if not row_id:
            raise ValueError(f"{label} row has no ID")
        if row_id in result:
            raise ValueError(f"Duplicate ID in {label}: {row_id}")
        result[row_id] = row
    return result


def _require_disjoint(
    left_name: str,
    left: Set[str],
    right_name: str,
    right: Set[str],
) -> None:
    overlap = left & right
    if overlap:
        raise ValueError(
            f"{left_name}/{right_name} leakage: count={len(overlap)} "
            f"examples={sorted(overlap)[:10]}"
        )


def _contains_forbidden_key(value: Any, forbidden: Set[str]) -> bool:
    if isinstance(value, dict):
        return bool(set(value) & forbidden) or any(
            _contains_forbidden_key(child, forbidden) for child in value.values()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_key(child, forbidden) for child in value)
    return False


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def _write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _write_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Summarize retrieval and teacher-repair effects from inference trajectories."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _verification_pass(verification: Dict[str, Any], key: str) -> bool:
    return float(verification.get(key, 0.0)) >= 1.0


def _failure_free(verification: Dict[str, Any]) -> bool:
    return not list(verification.get("failures", []))


def _mean(values: Iterable[bool]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _memory_preview(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    memory = dict(row.get("memory_context", {}))
    result = []
    for rank, item in enumerate(memory.get("theorem_items", [])[:3], 1):
        metadata = dict(item.get("metadata", {}))
        text = " ".join(str(item.get("text", "")).split())
        result.append(
            {
                "rank": rank,
                "id": item.get("id"),
                "source": item.get("source"),
                "lean_name": metadata.get("lean_name"),
                "source_file": metadata.get("source_file"),
                "keywords": list(item.get("keywords", [])),
                "text_preview": text[:500],
            }
        )
    return result


def build_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    per_problem = []
    repair_statuses: Counter[str] = Counter()
    before_verifications = []
    after_verifications = []
    teacher_api_calls = 0
    repaired_problems = 0
    improved_problems = 0
    regressed_problems = 0

    for row in rows:
        records = list(row.get("rollout_records", []))
        if not records:
            continue
        record = records[0]
        repair = dict(record.get("repair", {}))
        status = str(repair.get("status", "not_needed"))
        repair_statuses[status] += 1
        attempts = list(repair.get("attempts", []))
        teacher_api_calls += len(attempts)
        if status in {"succeeded", "failed"}:
            repaired_problems += 1

        after = dict(record.get("verification", {}))
        before = dict(repair.get("verification_before_repair", after))
        before_verifications.append(before)
        after_verifications.append(after)

        before_full = _failure_free(before)
        after_full = _failure_free(after)
        if after_full and not before_full:
            improved_problems += 1
        elif before_full and not after_full:
            regressed_problems += 1

        problem = dict(row.get("problem", {}))
        per_problem.append(
            {
                "id": problem.get("id", row.get("problem_id")),
                "question": problem.get("text", ""),
                "gold_answer": problem.get("gold_answer"),
                "top3_theory": _memory_preview(row),
                "teacher_called": bool(attempts),
                "repair_status": status,
                "teacher_diagnosis": repair.get("teacher_diagnosis"),
                "teacher_repair_instructions": repair.get(
                    "teacher_repair_instructions"
                ),
                "before": {
                    "final_answer_score": before.get("final_answer_score", 0.0),
                    "backward_score": before.get("backward_score", 0.0),
                    "failure_count": len(before.get("failures", [])),
                },
                "after": {
                    "final_answer_score": after.get("final_answer_score", 0.0),
                    "backward_score": after.get("backward_score", 0.0),
                    "failure_count": len(after.get("failures", [])),
                },
                "generated_text": dict(record.get("rollout", {})).get(
                    "raw_text", ""
                ),
            }
        )

    memory_counts = [len(item["top3_theory"]) for item in per_problem]
    return {
        "problems": len(per_problem),
        "memory": {
            "top_k": 3,
            "problems_with_three_theories": sum(count == 3 for count in memory_counts),
            "coverage_rate": _mean(count == 3 for count in memory_counts),
        },
        "teacher": {
            "api_calls": teacher_api_calls,
            "repair_statuses": dict(sorted(repair_statuses.items())),
            "repaired_problems": repaired_problems,
        },
        "before_teacher": {
            "final_answer_accuracy": _mean(
                _verification_pass(item, "final_answer_score")
                for item in before_verifications
            ),
            "backward_accuracy": _mean(
                _verification_pass(item, "backward_score")
                for item in before_verifications
            ),
            "failure_free_rate": _mean(
                _failure_free(item) for item in before_verifications
            ),
        },
        "after_teacher": {
            "final_answer_accuracy": _mean(
                _verification_pass(item, "final_answer_score")
                for item in after_verifications
            ),
            "backward_accuracy": _mean(
                _verification_pass(item, "backward_score")
                for item in after_verifications
            ),
            "failure_free_rate": _mean(
                _failure_free(item) for item in after_verifications
            ),
        },
        "full_success_transitions": {
            "improved": improved_problems,
            "regressed": regressed_problems,
            "unchanged": len(per_problem) - improved_problems - regressed_problems,
        },
        "per_problem": per_problem,
        "interpretation": (
            "The teacher is called only after verifier-detected failure. Rows marked "
            "not_needed use memory-augmented Qwen inference without a teacher API call."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()

    summary = build_summary(load_jsonl(Path(args.trajectory).expanduser()))
    rendered = json.dumps(summary, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        output = Path(args.output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

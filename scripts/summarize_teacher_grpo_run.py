#!/usr/bin/env python3
"""Summarize teacher usage and before/after signals in a guided GRPO run."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()

    rows = list(read_jsonl(Path(args.trajectories)))
    summary = summarize(rows)
    rendered = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


def summarize(rows: list[Dict[str, Any]]) -> Dict[str, Any]:
    teacher_calls = sum(int(row.get("teacher_calls_used", 0)) for row in rows)
    statuses = Counter()
    probe_final = []
    trained_final = []
    trained_backward = []
    teacher_scores = []
    kls = []
    for row in rows:
        probe = row.get("teacher_probe")
        if isinstance(probe, dict):
            probe_final.append(
                float(probe.get("verification", {}).get("final_answer_score", 0.0))
            )
        rollout_records = list(row.get("rollout_records", []))
        for record in rollout_records:
            statuses[str(record.get("repair", {}).get("status", "unknown"))] += 1
            reward = record.get("reward", {})
            if reward.get("teacher_reward") is not None:
                teacher_scores.append(float(reward.get("teacher_reward", 0.0)))
            verification = record.get("verification", {})
            trained_final.append(float(verification.get("final_answer_score", 0.0)))
            trained_backward.append(float(verification.get("backward_score", 0.0)))
        update = row.get("grpo_update")
        if isinstance(update, dict) and update.get("approx_kl") is not None:
            kls.append(float(update["approx_kl"]))

    problems_with_feedback = sum(
        any(
            record.get("repair", {}).get("status") == "shared_feedback"
            for record in row.get("rollout_records", [])
        )
        for row in rows
    )
    problems_with_group_judgment = sum(
        isinstance(row.get("teacher_group_judgment"), dict)
        and isinstance(
            row.get("teacher_group_judgment", {}).get("rollout_judgments"),
            list,
        )
        for row in rows
    )
    return {
        "problems": len(rows),
        "teacher_api_calls": teacher_calls,
        "teacher_calls_per_problem": round(teacher_calls / len(rows), 6) if rows else 0.0,
        "problems_with_shared_teacher_feedback": problems_with_feedback,
        "problems_with_group_teacher_judgment": problems_with_group_judgment,
        "problems_without_teacher_feedback": len(rows) - problems_with_feedback,
        "rollout_repair_statuses": dict(statuses),
        "probe_final_answer_accuracy": round(_mean(probe_final), 6),
        "teacher_conditioned_rollout_accuracy": round(_mean(trained_final), 6),
        "teacher_conditioned_backward_accuracy": round(_mean(trained_backward), 6),
        "teacher_free_training_rollout_accuracy": round(_mean(trained_final), 6),
        "mean_teacher_judge_score": round(_mean(teacher_scores), 6),
        "mean_update_kl": round(_mean(kls), 8),
        "max_update_kl": round(max(kls, default=0.0), 8),
        "interpretation": (
            "These are training-time signals, not the final result. Group-judge runs keep "
            "student rollout prompts teacher-free and use the teacher only as a bounded reward "
            "judge. Effectiveness must be decided by teacher-free held-out evaluation."
        ),
    }


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("Trajectory JSONL rows must be objects")
                yield value


if __name__ == "__main__":
    main()

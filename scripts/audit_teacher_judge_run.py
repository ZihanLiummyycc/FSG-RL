#!/usr/bin/env python3
"""Audit that group-judge GRPO stayed teacher-free and produced usable signals."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--expected-problems", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--require-teacher-success", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()

    rows = list(read_jsonl(Path(args.trajectories)))
    result = audit(rows, max_new_tokens=args.max_new_tokens)
    if args.expected_problems is not None and len(rows) != args.expected_problems:
        raise AssertionError(
            f"Expected {args.expected_problems} problems, found {len(rows)}"
        )
    if result["student_prompts_containing_teacher_feedback"]:
        raise AssertionError("Teacher feedback leaked into a student rollout prompt")
    if result["teacher_calls_over_budget"]:
        raise AssertionError("A problem used more than one teacher API call")
    if args.require_teacher_success and result["teacher_error_problems"]:
        raise AssertionError("At least one teacher group judgment failed")

    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


def audit(rows: list[Dict[str, Any]], *, max_new_tokens: int) -> Dict[str, Any]:
    teacher_scores = []
    final_scores = []
    backward_scores = []
    kls = []
    prompts_with_feedback = 0
    teacher_error_problems = 0
    judged_problems = 0
    calls_over_budget = 0
    groups_with_score_variance = 0
    capped_generations = 0

    for row in rows:
        calls = int(row.get("teacher_calls_used", 0))
        calls_over_budget += int(calls > 1)
        judgment = row.get("teacher_group_judgment")
        if isinstance(judgment, dict) and isinstance(
            judgment.get("rollout_judgments"), list
        ):
            judged_problems += 1
        else:
            teacher_error_problems += 1

        row_scores = []
        for record in row.get("rollout_records", []):
            rollout = record.get("rollout", {})
            prompts_with_feedback += int(
                "repair_feedback" in str(rollout.get("prompt_text", ""))
            )
            capped_generations += int(
                int(rollout.get("completion_token_count", 0)) >= max_new_tokens
            )
            reward = record.get("reward", {})
            score = float(reward.get("teacher_reward", 0.0))
            teacher_scores.append(score)
            row_scores.append(score)
            verification = record.get("verification", {})
            final_scores.append(float(verification.get("final_answer_score", 0.0)))
            backward_scores.append(float(verification.get("backward_score", 0.0)))
        if row_scores and max(row_scores) - min(row_scores) > 1e-9:
            groups_with_score_variance += 1

        update = row.get("grpo_update")
        if isinstance(update, dict) and update.get("approx_kl") is not None:
            kls.append(float(update["approx_kl"]))

    return {
        "problems": len(rows),
        "rollouts": len(teacher_scores),
        "teacher_group_judged_problems": judged_problems,
        "teacher_error_problems": teacher_error_problems,
        "teacher_calls_over_budget": calls_over_budget,
        "student_prompts_containing_teacher_feedback": prompts_with_feedback,
        "groups_with_teacher_score_variance": groups_with_score_variance,
        "mean_teacher_score": round(_mean(teacher_scores), 6),
        "teacher_score_min": min(teacher_scores, default=0.0),
        "teacher_score_max": max(teacher_scores, default=0.0),
        "training_rollout_final_accuracy": round(_mean(final_scores), 6),
        "training_rollout_backward_accuracy": round(_mean(backward_scores), 6),
        "generations_at_token_cap": capped_generations,
        "mean_update_kl": round(_mean(kls), 8),
        "max_update_kl": round(max(kls, default=0.0), 8),
    }


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("Trajectory rows must be JSON objects")
                yield row


if __name__ == "__main__":
    main()

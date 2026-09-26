#!/usr/bin/env python3
"""Compare two sampled GRPO trajectory files with paired pass@k metrics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, help="Baseline trajectories JSONL")
    parser.add_argument("--candidate", required=True, help="Candidate trajectories JSONL")
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    baseline = load_problem_views(Path(args.baseline))
    candidate = load_problem_views(Path(args.candidate))
    if set(baseline) != set(candidate):
        raise ValueError("Baseline and candidate trajectory problem IDs differ")

    report = compare_problem_views(
        baseline,
        candidate,
        baseline_label=args.baseline_label,
        candidate_label=args.candidate_label,
    )
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def load_problem_views(path: Path) -> Dict[str, Dict[str, Any]]:
    rows = _read_jsonl(path.expanduser().resolve())
    views: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        problem = row.get("problem", {})
        problem_id = str(problem.get("id", row.get("problem_id", ""))).strip()
        if not problem_id:
            raise ValueError(f"Trajectory row in {path} has no problem ID")
        if problem_id in views:
            raise ValueError(f"Duplicate problem ID in {path}: {problem_id}")
        rollout_views = [_rollout_view(item) for item in row.get("rollout_records", [])]
        if not rollout_views:
            raise ValueError(f"Trajectory {problem_id} has no rollout records")
        views[problem_id] = {
            "id": problem_id,
            "rollouts": rollout_views,
            "final_pass_at_k": any(item["final"] for item in rollout_views),
            "strict_pass_at_k": any(item["strict"] for item in rollout_views),
            "executable_pass_at_k": any(item["execution"] for item in rollout_views),
            "backward_pass_at_k": any(item["backward"] for item in rollout_views),
        }
    return views


def compare_problem_views(
    baseline: Dict[str, Dict[str, Any]],
    candidate: Dict[str, Dict[str, Any]],
    *,
    baseline_label: str,
    candidate_label: str,
) -> Dict[str, Any]:
    ids = sorted(baseline)
    metrics = (
        "final_pass_at_k",
        "strict_pass_at_k",
        "executable_pass_at_k",
        "backward_pass_at_k",
    )
    transitions: Dict[str, Dict[str, Any]] = {}
    for metric in metrics:
        improved = [
            problem_id
            for problem_id in ids
            if not baseline[problem_id][metric] and candidate[problem_id][metric]
        ]
        regressed = [
            problem_id
            for problem_id in ids
            if baseline[problem_id][metric] and not candidate[problem_id][metric]
        ]
        transitions[metric] = {
            "improved": len(improved),
            "regressed": len(regressed),
            "unchanged": len(ids) - len(improved) - len(regressed),
            "improved_ids": improved,
            "regressed_ids": regressed,
            "mcnemar_exact_p": _mcnemar_exact_p(len(improved), len(regressed)),
        }

    return {
        "problems": len(ids),
        "baseline": {
            "label": baseline_label,
            **_summarize(baseline.values()),
        },
        "candidate": {
            "label": candidate_label,
            **_summarize(candidate.values()),
        },
        "paired_transitions": transitions,
        "selection_warning": (
            "This is a 17-problem development split. Confirm any selected checkpoint "
            "on an untouched evaluation set and additional sampling seeds."
        ),
    }


def _rollout_view(record: Dict[str, Any]) -> Dict[str, Any]:
    reward = record.get("reward", {})
    rollout = record.get("rollout", {})
    final = float(reward.get("final_reward", 0.0)) == 1.0
    backward = float(reward.get("backward_reward", 0.0)) == 1.0
    format_ok = float(reward.get("format_gate", 0.0)) == 1.0
    signature = float(reward.get("signature_reward", 0.0)) == 1.0
    execution = float(reward.get("execution_reward", 0.0)) == 1.0
    unit = float(reward.get("unit_test_reward", 0.0)) == 1.0
    property_ok = float(reward.get("property_test_reward", 0.0)) == 1.0
    return {
        "final": final,
        "backward": backward,
        "execution": execution,
        "strict": all(
            (format_ok, signature, execution, unit, property_ok, backward, final)
        ),
        "unit_score": float(reward.get("unit_test_reward", 0.0)),
        "property_score": float(reward.get("property_test_reward", 0.0)),
        "completion_tokens": len(rollout.get("completion_token_ids", [])),
    }


def _summarize(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    problems = list(rows)
    rollouts = [item for problem in problems for item in problem["rollouts"]]
    count = len(problems)
    return {
        "rollouts": len(rollouts),
        "sampled_rollout_accuracy": mean(float(item["final"]) for item in rollouts),
        "final_pass_at_k": sum(item["final_pass_at_k"] for item in problems) / count,
        "strict_pass_at_k": sum(item["strict_pass_at_k"] for item in problems) / count,
        "executable_pass_at_k": sum(
            item["executable_pass_at_k"] for item in problems
        )
        / count,
        "backward_pass_at_k": sum(item["backward_pass_at_k"] for item in problems)
        / count,
        "mean_unit_score": mean(item["unit_score"] for item in rollouts),
        "mean_property_score": mean(item["property_score"] for item in rollouts),
        "mean_completion_tokens": mean(item["completion_tokens"] for item in rollouts),
    }


def _mcnemar_exact_p(improved: int, regressed: int) -> float:
    discordant = improved + regressed
    if discordant == 0:
        return 1.0
    tail = min(improved, regressed)
    probability = sum(math.comb(discordant, value) for value in range(tail + 1))
    return min(1.0, 2.0 * probability / (2**discordant))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


if __name__ == "__main__":
    main()

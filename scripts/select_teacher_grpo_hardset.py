#!/usr/bin/env python3
"""Select hard-but-learnable training rows from an earlier GRPO trajectory."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--eval-data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--audit-output", required=True)
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--omni-rows", type=int, default=15)
    parser.add_argument("--max-final-rate", type=float, default=0.5)
    parser.add_argument("--min-learnability", type=float, default=0.2)
    args = parser.parse_args()

    train_rows = list(read_jsonl(Path(args.train_data)))
    trajectories = list(read_jsonl(Path(args.trajectories)))
    eval_rows = list(read_jsonl(Path(args.eval_data)))
    selected, audit = select_hard_rows(
        train_rows,
        trajectories,
        eval_rows,
        limit=args.limit,
        omni_rows=args.omni_rows,
        max_final_rate=args.max_final_rate,
        min_learnability=args.min_learnability,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output, selected)
    audit_output = Path(args.audit_output)
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    audit.update(
        {
            "train_data": str(Path(args.train_data).resolve()),
            "trajectory_data": str(Path(args.trajectories).resolve()),
            "eval_data": str(Path(args.eval_data).resolve()),
            "output": str(output.resolve()),
        }
    )
    audit_output.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


def select_hard_rows(
    train_rows: List[Dict[str, Any]],
    trajectories: List[Dict[str, Any]],
    eval_rows: List[Dict[str, Any]],
    *,
    limit: int,
    omni_rows: int,
    max_final_rate: float,
    min_learnability: float,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if limit <= 0:
        raise ValueError("limit must be positive")
    if not 0.0 <= max_final_rate <= 1.0:
        raise ValueError("max_final_rate must be between 0 and 1")
    if not 0.0 <= min_learnability <= 1.0:
        raise ValueError("min_learnability must be between 0 and 1")

    train_by_id = _unique_by_id(train_rows, "train data")
    eval_ids = {_row_id(row) for row in eval_rows}
    trajectory_by_id = _unique_trajectories(trajectories)
    missing_trajectories = sorted(set(train_by_id) - set(trajectory_by_id))

    candidates = []
    rejected = Counter()
    for problem_id, train_row in train_by_id.items():
        trajectory = trajectory_by_id.get(problem_id)
        if trajectory is None:
            rejected["missing_trajectory"] += 1
            continue
        metrics = trajectory_metrics(trajectory)
        if metrics["rollouts"] < 2:
            rejected["fewer_than_two_rollouts"] += 1
            continue
        if metrics["final_rate"] > max_final_rate:
            rejected["not_hard_enough"] += 1
            continue
        if metrics["learnability"] < min_learnability:
            rejected["no_process_signal"] += 1
            continue
        source_group = (
            "omni" if problem_id.startswith("omni_math_") else "medium"
        )
        mixed_outcome_bonus = float(0.0 < metrics["final_rate"] < 1.0)
        score = (
            3.0 * mixed_outcome_bonus
            + 2.0 * (1.0 - metrics["final_rate"])
            + 2.0 * metrics["learnability"]
            + min(1.0, metrics["reward_std"])
        )
        candidates.append(
            {
                "id": problem_id,
                "row": train_row,
                "source_group": source_group,
                "selection_score": round(score, 6),
                **metrics,
            }
        )

    candidates.sort(
        key=lambda item: (
            -item["selection_score"],
            item["final_rate"],
            item["id"],
        )
    )
    requested_omni = max(0, min(limit, omni_rows))
    selected_candidates = _take_with_source_quota(
        candidates,
        limit=limit,
        omni_rows=requested_omni,
    )
    if len(selected_candidates) < limit:
        raise ValueError(
            f"Only {len(selected_candidates)} eligible hard rows were found; "
            f"requested {limit}. Lower --min-learnability or raise --max-final-rate."
        )

    selected_ids = {item["id"] for item in selected_candidates}
    leakage = sorted(selected_ids & eval_ids)
    if leakage:
        raise ValueError(
            "Selected training rows overlap evaluation data: " + ", ".join(leakage[:10])
        )

    selected_rows = [dict(item["row"]) for item in selected_candidates]
    for row in selected_rows:
        row["split"] = "train"

    selection_records = [
        {
            key: value
            for key, value in item.items()
            if key != "row"
        }
        for item in selected_candidates
    ]
    audit = {
        "train_rows": len(train_rows),
        "trajectory_rows": len(trajectories),
        "unique_trajectory_problems": len(trajectory_by_id),
        "eval_rows": len(eval_rows),
        "eligible_rows": len(candidates),
        "selected_rows": len(selected_rows),
        "selected_source_counts": dict(
            Counter(item["source_group"] for item in selected_candidates)
        ),
        "selected_final_rate_buckets": dict(
            Counter(_rate_bucket(item["final_rate"]) for item in selected_candidates)
        ),
        "selected_difficulty_counts": dict(
            Counter(
                str(item["row"].get("metadata", {}).get("difficulty", "unknown"))
                for item in selected_candidates
            )
        ),
        "selected_mean_final_rate": round(
            statistics.fmean(item["final_rate"] for item in selected_candidates), 6
        ),
        "selected_mean_learnability": round(
            statistics.fmean(item["learnability"] for item in selected_candidates), 6
        ),
        "eval_overlap": len(leakage),
        "missing_trajectory_rows": len(missing_trajectories),
        "rejected_reasons": dict(rejected),
        "selection_records": selection_records,
        "note": (
            "Rows come only from the prior GRPO training set. Difficulty is based on "
            "teacher-free rollout outcomes; Eval100 overlap is forbidden."
        ),
    }
    return selected_rows, audit


def trajectory_metrics(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    records = list(trajectory.get("rollout_records", []))
    final_scores = [_nested_score(row, "verification", "final_answer_score") for row in records]
    execution_scores = [_nested_score(row, "reward", "execution_reward") for row in records]
    unit_scores = [_nested_score(row, "reward", "unit_test_reward") for row in records]
    property_scores = [_nested_score(row, "reward", "property_test_reward") for row in records]
    backward_scores = [_nested_score(row, "verification", "backward_score") for row in records]
    rewards = [_nested_score(row, "reward", "total_reward") for row in records]
    process_per_rollout = [
        statistics.fmean(values)
        for values in zip(execution_scores, unit_scores, property_scores, backward_scores)
    ] if records else []
    return {
        "rollouts": len(records),
        "final_rate": round(_mean(final_scores), 6),
        "execution_rate": round(_mean(execution_scores), 6),
        "unit_rate": round(_mean(unit_scores), 6),
        "property_rate": round(_mean(property_scores), 6),
        "backward_rate": round(_mean(backward_scores), 6),
        "learnability": round(max(process_per_rollout, default=0.0), 6),
        "reward_std": round(statistics.pstdev(rewards) if len(rewards) > 1 else 0.0, 6),
    }


def _take_with_source_quota(
    candidates: List[Dict[str, Any]],
    *,
    limit: int,
    omni_rows: int,
) -> List[Dict[str, Any]]:
    omni = [row for row in candidates if row["source_group"] == "omni"]
    medium = [row for row in candidates if row["source_group"] == "medium"]
    selected = omni[:omni_rows]
    selected.extend(medium[: max(0, limit - len(selected))])
    selected_ids = {row["id"] for row in selected}
    if len(selected) < limit:
        selected.extend(
            row
            for row in candidates
            if row["id"] not in selected_ids
        )
    selected = selected[:limit]
    selected.sort(key=lambda item: (-item["selection_score"], item["id"]))
    return selected


def _nested_score(row: Dict[str, Any], section: str, key: str) -> float:
    value = row.get(section, {}).get(key, 0.0)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def _row_id(row: Dict[str, Any]) -> str:
    if isinstance(row.get("problem"), dict):
        value = row["problem"].get("id", row["problem"].get("problem_id"))
    else:
        value = row.get("id", row.get("problem_id"))
    if value is None:
        raise ValueError("Record has no id/problem_id")
    return str(value)


def _unique_by_id(
    rows: Iterable[Dict[str, Any]],
    label: str,
) -> Dict[str, Dict[str, Any]]:
    result = {}
    for row in rows:
        problem_id = _row_id(row)
        if problem_id in result:
            raise ValueError(f"Duplicate {label} id: {problem_id}")
        result[problem_id] = row
    return result


def _unique_trajectories(
    rows: Iterable[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    result = {}
    for row in rows:
        result[_row_id(row)] = row
    return result


def _rate_bucket(value: float) -> str:
    if value == 0.0:
        return "0"
    if value <= 0.25:
        return "(0,0.25]"
    if value <= 0.5:
        return "(0.25,0.5]"
    return ">0.5"


def _mean(values: List[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            yield value


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

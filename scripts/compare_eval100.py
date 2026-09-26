#!/usr/bin/env python3
"""Create a statistically explicit paired report for two eval100 runs."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence


METRICS = (
    "tags_exact",
    "signature_exact",
    "python_executable",
    "all_hidden_tests_pass",
    "backward_target_correct",
    "final_answer_correct",
    "full_success",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--baseline-results", required=True)
    parser.add_argument("--candidate-results", required=True)
    parser.add_argument("--baseline-label", default="stage2")
    parser.add_argument("--candidate-label", default="grpo")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown", required=True)
    args = parser.parse_args()

    data = _unique(_read_jsonl(Path(args.data)), "evaluation data")
    baseline = _unique(_read_jsonl(Path(args.baseline_results)), "baseline")
    candidate = _unique(_read_jsonl(Path(args.candidate_results)), "candidate")
    report = build_report(
        data,
        baseline,
        candidate,
        baseline_label=args.baseline_label,
        candidate_label=args.candidate_label,
    )
    _write_json(Path(args.output_json), report)
    Path(args.output_markdown).expanduser().resolve().write_text(
        render_markdown(report), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def build_report(
    data: Mapping[str, Dict[str, Any]],
    baseline: Mapping[str, Dict[str, Any]],
    candidate: Mapping[str, Dict[str, Any]],
    *,
    baseline_label: str,
    candidate_label: str,
) -> Dict[str, Any]:
    if set(data) != set(baseline) or set(data) != set(candidate):
        raise ValueError(
            "Evaluation data and result IDs differ: "
            f"data={len(data)} baseline={len(baseline)} candidate={len(candidate)}"
        )
    ids = sorted(data)
    metrics = {}
    for metric in METRICS:
        left = [bool(baseline[row_id].get(metric)) for row_id in ids]
        right = [bool(candidate[row_id].get(metric)) for row_id in ids]
        improved = sum(not old and new for old, new in zip(left, right))
        regressed = sum(old and not new for old, new in zip(left, right))
        metrics[metric] = {
            "baseline": _rate_summary(left),
            "candidate": _rate_summary(right),
            "difference": sum(right) / len(right) - sum(left) / len(left),
            "improved": improved,
            "regressed": regressed,
            "unchanged": len(ids) - improved - regressed,
            "mcnemar_exact_p": _mcnemar_exact(improved, regressed),
        }

    strata: Dict[str, list[str]] = defaultdict(list)
    for row_id, row in data.items():
        strata[_family(row)].append(row_id)
        strata[f"source:{_source(row)}"].append(row_id)
    breakdown = {}
    for name, row_ids in sorted(strata.items()):
        breakdown[name] = {
            metric: {
                "n": len(row_ids),
                "baseline_rate": sum(
                    bool(baseline[row_id].get(metric)) for row_id in row_ids
                )
                / len(row_ids),
                "candidate_rate": sum(
                    bool(candidate[row_id].get(metric)) for row_id in row_ids
                )
                / len(row_ids),
            }
            for metric in ("final_answer_correct", "all_hidden_tests_pass", "full_success")
        }
    return {
        "records": len(ids),
        "baseline": baseline_label,
        "candidate": candidate_label,
        "family_counts": dict(Counter(_family(row) for row in data.values())),
        "source_counts": dict(Counter(_source(row) for row in data.values())),
        "metrics": metrics,
        "breakdown": breakdown,
        "interpretation": (
            "Wilson intervals describe uncertainty of each rate. McNemar exact p-values use "
            "paired disagreements; p<0.05 is conventional evidence of a difference."
        ),
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Eval100 paired evaluation",
        "",
        f"Records: {report['records']}",
        "",
        "| Metric | Baseline | Candidate | Difference | Improved | Regressed | McNemar p |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for metric, value in report["metrics"].items():
        left = value["baseline"]
        right = value["candidate"]
        lines.append(
            f"| {metric} | {left['rate']:.1%} [{left['wilson95_low']:.1%}, "
            f"{left['wilson95_high']:.1%}] | {right['rate']:.1%} "
            f"[{right['wilson95_low']:.1%}, {right['wilson95_high']:.1%}] | "
            f"{value['difference']:+.1%} | {value['improved']} | "
            f"{value['regressed']} | {value['mcnemar_exact_p']:.4f} |"
        )
    lines.extend(["", "## Stratified results", ""])
    for name, values in report["breakdown"].items():
        n = next(iter(values.values()))["n"]
        lines.append(f"### {name} (n={n})")
        lines.append("")
        for metric, value in values.items():
            lines.append(
                f"- {metric}: {value['baseline_rate']:.1%} → "
                f"{value['candidate_rate']:.1%}"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _rate_summary(values: Sequence[bool]) -> Dict[str, Any]:
    successes = sum(values)
    low, high = _wilson(successes, len(values))
    return {
        "successes": successes,
        "total": len(values),
        "rate": successes / len(values),
        "wilson95_low": low,
        "wilson95_high": high,
    }


def _wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        raise ValueError("Wilson interval needs a positive sample size")
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def _mcnemar_exact(improved: int, regressed: int) -> float:
    disagreements = improved + regressed
    if disagreements == 0:
        return 1.0
    tail = sum(
        math.comb(disagreements, value)
        for value in range(min(improved, regressed) + 1)
    ) / (2**disagreements)
    return min(1.0, 2 * tail)


def _family(row: Mapping[str, Any]) -> str:
    return "medium" if str(row.get("id", "")).startswith("medium_") else "omni"


def _source(row: Mapping[str, Any]) -> str:
    metadata = row.get("metadata", {})
    return str(
        row.get("source")
        or (metadata.get("source") if isinstance(metadata, dict) else None)
        or "unknown"
    )


def _read_jsonl(path: Path) -> list[Dict[str, Any]]:
    with path.expanduser().open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _unique(
    rows: Iterable[Dict[str, Any]], label: str
) -> Dict[str, Dict[str, Any]]:
    result = {}
    for row in rows:
        row_id = str(row.get("id", "")).strip()
        if not row_id or row_id in result:
            raise ValueError(f"Missing or duplicate ID in {label}: {row_id!r}")
        result[row_id] = row
    return result


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

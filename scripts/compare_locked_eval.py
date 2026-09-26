#!/usr/bin/env python3
"""Build three-way paired statistics for a locked executable evaluation set."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

from fsg_rl.executable_eval import summarize_evaluations


METRICS = (
    "tags_exact",
    "code_blocks_complete",
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
    parser.add_argument(
        "--result",
        action="append",
        required=True,
        metavar="LABEL=PATH",
    )
    parser.add_argument("--baseline-label", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown", required=True)
    args = parser.parse_args()

    data = _unique(_read_jsonl(Path(args.data)), "locked evaluation")
    result_specs = [_parse_result(value) for value in args.result]
    if len(result_specs) < 2:
        parser.error("At least two --result LABEL=PATH arguments are required")
    if len({label for label, _ in result_specs}) != len(result_specs):
        parser.error("Result labels must be unique")
    results = {
        label: _unique(_read_jsonl(Path(path)), label)
        for label, path in result_specs
    }
    report = build_report(
        data,
        results,
        baseline_label=args.baseline_label,
    )
    _write_json(Path(args.output_json), report)
    markdown_path = Path(args.output_markdown).expanduser().resolve()
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def build_report(
    data: Mapping[str, Dict[str, Any]],
    results: Mapping[str, Mapping[str, Dict[str, Any]]],
    *,
    baseline_label: str,
) -> Dict[str, Any]:
    if baseline_label not in results:
        raise ValueError(f"Unknown baseline label {baseline_label!r}")
    expected_ids = set(data)
    for label, values in results.items():
        if set(values) != expected_ids:
            raise ValueError(
                f"Locked data/{label} IDs differ: data={len(data)} result={len(values)}"
            )
    ids = sorted(expected_ids)
    labels = list(results)
    summaries = {
        label: summarize_evaluations([results[label][row_id] for row_id in ids])
        for label in labels
    }

    pairwise = {}
    primary_p_values = []
    for candidate_label in labels:
        if candidate_label == baseline_label:
            continue
        name = f"{baseline_label}_vs_{candidate_label}"
        comparison = _paired_comparison(
            ids,
            results[baseline_label],
            results[candidate_label],
        )
        pairwise[name] = {
            "baseline": baseline_label,
            "candidate": candidate_label,
            "metrics": comparison,
        }
        primary_p_values.append(
            (name, comparison["final_answer_correct"]["mcnemar_exact_p"])
        )

    non_baseline = [label for label in labels if label != baseline_label]
    if len(non_baseline) >= 2:
        for left_index, left in enumerate(non_baseline):
            for right in non_baseline[left_index + 1 :]:
                name = f"{left}_vs_{right}"
                pairwise[name] = {
                    "baseline": left,
                    "candidate": right,
                    "metrics": _paired_comparison(
                        ids, results[left], results[right]
                    ),
                }

    holm = _holm_adjust(primary_p_values)
    for name, adjusted in holm.items():
        pairwise[name]["metrics"]["final_answer_correct"][
            "holm_adjusted_p"
        ] = adjusted

    strata: Dict[str, list[str]] = defaultdict(list)
    for row_id, row in data.items():
        strata[f"family:{_family(row)}"].append(row_id)
        strata[f"source:{_source(row)}"].append(row_id)
        strata[f"difficulty:{_difficulty(row)}"].append(row_id)
    breakdown = {}
    for name, row_ids in sorted(strata.items()):
        breakdown[name] = {
            "n": len(row_ids),
            "models": {
                label: {
                    metric: sum(
                        bool(results[label][row_id].get(metric))
                        for row_id in row_ids
                    )
                    / len(row_ids)
                    for metric in (
                        "final_answer_correct",
                        "all_hidden_tests_pass",
                        "full_success",
                    )
                }
                for label in labels
            },
        }

    return {
        "records": len(ids),
        "baseline": baseline_label,
        "labels": labels,
        "family_counts": dict(Counter(_family(row) for row in data.values())),
        "source_counts": dict(Counter(_source(row) for row in data.values())),
        "difficulty_counts": dict(
            Counter(str(_difficulty(row)) for row in data.values())
        ),
        "summaries": summaries,
        "pairwise": pairwise,
        "breakdown": breakdown,
        "decision_rule": (
            "Treat teacher guidance as confirmed only if the locked-set final-answer "
            "gain is at least 0.02, paired improvements exceed regressions, full_success "
            "does not decrease, Python executability decreases by at most 0.01, and the "
            "direction is not confined to one source or difficulty stratum. McNemar and "
            "Holm p-values are reported as uncertainty evidence, not as the sole criterion."
        ),
    }


def _paired_comparison(
    ids: Sequence[str],
    baseline: Mapping[str, Dict[str, Any]],
    candidate: Mapping[str, Dict[str, Any]],
) -> Dict[str, Any]:
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
    return metrics


def render_markdown(report: Mapping[str, Any]) -> str:
    labels = list(report["labels"])
    lines = [
        "# Locked Eval400 three-way evaluation",
        "",
        f"Records: {report['records']}",
        "",
        "## Aggregate results",
        "",
        "| Metric | " + " | ".join(labels) + " |",
        "|---|" + "---:|" * len(labels),
    ]
    for metric in METRICS:
        values = [
            f"{report['summaries'][label][_summary_key(metric)]:.1%}"
            for label in labels
        ]
        lines.append(f"| {metric} | " + " | ".join(values) + " |")
    lines.extend(["", "## Paired comparisons", ""])
    for name, comparison in report["pairwise"].items():
        lines.extend(
            [
                f"### {name}",
                "",
                "| Metric | Difference | Improved | Regressed | McNemar p |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for metric, value in comparison["metrics"].items():
            p_value = value.get("holm_adjusted_p")
            p_text = f"{value['mcnemar_exact_p']:.4f}"
            if p_value is not None:
                p_text += f" (Holm {p_value:.4f})"
            lines.append(
                f"| {metric} | {value['difference']:+.1%} | "
                f"{value['improved']} | {value['regressed']} | {p_text} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Decision rule",
            "",
            str(report["decision_rule"]),
            "",
            "This is a custom executable FSG evaluation, not an official source-benchmark score.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _summary_key(metric: str) -> str:
    return {
        "tags_exact": "tags_exact_rate",
        "code_blocks_complete": "code_blocks_complete_rate",
        "signature_exact": "signature_exact_rate",
        "python_executable": "python_executable_rate",
        "all_hidden_tests_pass": "all_hidden_tests_pass_rate",
        "backward_target_correct": "backward_target_accuracy",
        "final_answer_correct": "final_answer_accuracy",
        "full_success": "full_success_rate",
    }[metric]


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


def _wilson(
    successes: int, total: int, z: float = 1.959963984540054
) -> tuple[float, float]:
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


def _holm_adjust(values: Sequence[tuple[str, float]]) -> Dict[str, float]:
    ordered = sorted(values, key=lambda item: item[1])
    adjusted = {}
    previous = 0.0
    total = len(ordered)
    for index, (name, value) in enumerate(ordered):
        current = min(1.0, (total - index) * value)
        previous = max(previous, current)
        adjusted[name] = previous
    return adjusted


def _parse_result(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--result must use LABEL=PATH")
    label, path = value.split("=", 1)
    if not label.strip() or not path.strip():
        raise argparse.ArgumentTypeError("--result requires non-empty label/path")
    return label.strip(), path.strip()


def _family(row: Mapping[str, Any]) -> str:
    return "medium" if str(row.get("id", "")).startswith("medium_") else "omni"


def _source(row: Mapping[str, Any]) -> str:
    metadata = row.get("metadata", {})
    return str(
        row.get("source")
        or (metadata.get("source") if isinstance(metadata, dict) else None)
        or "unknown"
    )


def _difficulty(row: Mapping[str, Any]) -> Any:
    metadata = row.get("metadata", {})
    return row.get(
        "difficulty",
        metadata.get("difficulty") if isinstance(metadata, dict) else None,
    )


def _read_jsonl(path: Path) -> list[Dict[str, Any]]:
    with path.expanduser().resolve().open(encoding="utf-8") as stream:
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

"""Metrics for executable function-graph rollout evaluation."""

from __future__ import annotations

import ast
import re
from typing import Any, Dict, Iterable, Mapping, Sequence

from .schemas import ExecutionResult, FunctionGraph, ParsedFunctionSpan


TAG_RE = re.compile(r"<(/?)a_([A-Za-z0-9_]+)>")
PAIR_METRICS = (
    "tags_exact",
    "code_blocks_complete",
    "signature_exact",
    "python_executable",
    "all_hidden_tests_pass",
    "backward_target_correct",
    "final_answer_correct",
    "full_success",
)


def check_exact_tags(raw_text: str, graph: FunctionGraph) -> Dict[str, Any]:
    """Require exactly the graph spans, with no text outside those spans."""

    expected = [node.id for node in graph.nodes]
    opening = []
    closing = []
    events = []
    for match in TAG_RE.finditer(raw_text):
        is_closing = bool(match.group(1))
        node_id = match.group(2)
        (closing if is_closing else opening).append(node_id)
        events.append(("close" if is_closing else "open", node_id))

    expected_events = []
    for node_id in expected:
        expected_events.extend([("open", node_id), ("close", node_id)])
    structure_exact = (
        opening == expected and closing == expected and events == expected_events
    )
    outside_segments = []
    if structure_exact:
        matches = list(TAG_RE.finditer(raw_text))
        cursor = 0
        for index in range(0, len(matches), 2):
            opening_match = matches[index]
            closing_match = matches[index + 1]
            outside_segments.append(raw_text[cursor : opening_match.start()])
            cursor = closing_match.end()
        outside_segments.append(raw_text[cursor:])
    outside_text = "".join(outside_segments)
    has_outside_text = bool(outside_text.strip())
    return {
        "passed": structure_exact and not has_outside_text,
        "expected": expected,
        "opening": opening,
        "closing": closing,
        "events": [list(event) for event in events],
        "outside_text_present": has_outside_text,
    }


def check_python_signatures(
    spans: Sequence[ParsedFunctionSpan],
    graph: FunctionGraph,
) -> Dict[str, Any]:
    """Check executable nodes for one block and an exact callable signature.

    Type annotations and defaults are intentionally ignored. The public graph's function
    name, parameter names, ordering, and parameter kinds are treated as the executable API.
    """

    spans_by_id = {span.node_id: span for span in spans}
    details: Dict[str, Any] = {}
    for node in graph.nodes:
        if not _expects_execution(node.expected_output_type, node.verification_spec):
            continue
        span = spans_by_id.get(node.id)
        code_blocks = list(span.code_blocks) if span else []
        detail: Dict[str, Any] = {
            "expected_signature": node.signature,
            "code_block_count": len(code_blocks),
            "code_block_complete": len(code_blocks) == 1,
            "passed": False,
        }
        try:
            expected = _parse_expected_signature(node.signature)
            detail["expected_callable"] = expected
            if len(code_blocks) != 1:
                detail["error"] = "expected exactly one Python code block"
            else:
                actual = _find_candidate_signature(code_blocks[0], expected["name"])
                detail["actual_callable"] = actual
                detail["passed"] = actual == expected
                if actual != expected:
                    detail["error"] = "function name or parameter structure differs"
        except (SyntaxError, ValueError) as exc:
            detail["error"] = str(exc)
        details[node.id] = detail

    return {
        "passed": bool(details) and all(item["passed"] for item in details.values()),
        "code_blocks_complete": bool(details)
        and all(item["code_block_complete"] for item in details.values()),
        "executable_node_count": len(details),
        "nodes": details,
    }


def execution_test_counts(
    execution: ExecutionResult,
    graph: FunctionGraph | None = None,
) -> Dict[str, int]:
    counts = {
        "unit_passed": 0,
        "unit_total": 0,
        "property_passed": 0,
        "property_total": 0,
    }
    if graph is None:
        for node_result in execution.node_results.values():
            for test in node_result.test_results:
                kind = "property" if test.get("kind") == "property" else "unit"
                counts[f"{kind}_total"] += 1
                counts[f"{kind}_passed"] += int(bool(test.get("passed")))
        return counts

    for node in graph.nodes:
        expected_tests = list(node.verification_spec.get("tests", []))
        actual_tests = (
            execution.node_results[node.id].test_results
            if node.id in execution.node_results
            else []
        )
        for index, expected in enumerate(expected_tests):
            kind = "property" if expected.get("kind") == "property" else "unit"
            counts[f"{kind}_total"] += 1
            actual = actual_tests[index] if index < len(actual_tests) else {}
            counts[f"{kind}_passed"] += int(bool(actual.get("passed")))
    return counts


def summarize_evaluations(rows: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    records = list(rows)
    count = len(records)
    totals = {
        "unit_passed": 0,
        "unit_total": 0,
        "property_passed": 0,
        "property_total": 0,
    }
    for row in records:
        tests = row.get("test_counts", {})
        for key in totals:
            totals[key] += int(tests.get(key, 0))

    summary: Dict[str, Any] = {
        "records": count,
        "tags_exact_rate": _record_rate(records, "tags_exact"),
        "code_blocks_complete_rate": _record_rate(records, "code_blocks_complete"),
        "signature_exact_rate": _record_rate(records, "signature_exact"),
        "python_executable_rate": _record_rate(records, "python_executable"),
        "all_hidden_tests_pass_rate": _record_rate(records, "all_hidden_tests_pass"),
        "backward_target_accuracy": _record_rate(records, "backward_target_correct"),
        "final_answer_accuracy": _record_rate(records, "final_answer_correct"),
        "full_success_rate": _record_rate(records, "full_success"),
        "unit_tests": {
            "passed": totals["unit_passed"],
            "total": totals["unit_total"],
            "pass_rate": _ratio(totals["unit_passed"], totals["unit_total"]),
        },
        "property_tests": {
            "passed": totals["property_passed"],
            "total": totals["property_total"],
            "pass_rate": _ratio(
                totals["property_passed"], totals["property_total"]
            ),
        },
        "mean_generation_seconds": _mean(
            float(row.get("generation_seconds", 0.0)) for row in records
        ),
        "mean_prompt_tokens": _mean(
            float(row.get("prompt_token_count", 0)) for row in records
        ),
        "mean_completion_tokens": _mean(
            float(row.get("completion_token_count", 0)) for row in records
        ),
        "error_rows": sum(bool(row.get("error")) for row in records),
    }
    return summary


def metric_deltas(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> Dict[str, float]:
    keys = [
        "tags_exact_rate",
        "code_blocks_complete_rate",
        "signature_exact_rate",
        "python_executable_rate",
        "all_hidden_tests_pass_rate",
        "backward_target_accuracy",
        "final_answer_accuracy",
        "full_success_rate",
    ]
    result = {
        key: round(float(candidate[key]) - float(baseline[key]), 6) for key in keys
    }
    result["unit_test_pass_rate"] = round(
        float(candidate["unit_tests"]["pass_rate"])
        - float(baseline["unit_tests"]["pass_rate"]),
        6,
    )
    result["property_test_pass_rate"] = round(
        float(candidate["property_tests"]["pass_rate"])
        - float(baseline["property_tests"]["pass_rate"]),
        6,
    )
    return result


def pair_evaluations(
    baseline: Sequence[Mapping[str, Any]],
    candidate: Sequence[Mapping[str, Any]],
) -> tuple[list[Dict[str, Any]], Dict[str, Dict[str, int]]]:
    """Pair identical problem IDs and report improvements and regressions."""

    baseline_by_id = _unique_by_id(baseline, "baseline")
    candidate_by_id = _unique_by_id(candidate, "candidate")
    if set(baseline_by_id) != set(candidate_by_id):
        raise ValueError("Baseline and candidate result IDs differ")
    counts = {
        metric: {"improved": 0, "regressed": 0, "unchanged": 0}
        for metric in PAIR_METRICS
    }
    pairs = []
    for problem_id in baseline_by_id:
        before = baseline_by_id[problem_id]
        after = candidate_by_id[problem_id]
        transitions = {}
        for metric in PAIR_METRICS:
            old = bool(before.get(metric))
            new = bool(after.get(metric))
            transition = "unchanged"
            if not old and new:
                transition = "improved"
            elif old and not new:
                transition = "regressed"
            counts[metric][transition] += 1
            transitions[metric] = {
                "baseline": old,
                "candidate": new,
                "transition": transition,
            }
        pairs.append(
            {
                "id": problem_id,
                "metrics": transitions,
                "baseline_generated_answer": before.get("generated_answer"),
                "candidate_generated_answer": after.get("generated_answer"),
                "gold_answer": after.get("gold_answer", before.get("gold_answer")),
                "baseline_error": before.get("error"),
                "candidate_error": after.get("error"),
            }
        )
    return pairs, counts


def _expects_execution(output_type: str, spec: Mapping[str, Any]) -> bool:
    return output_type == "python_function" or bool(spec.get("tests"))


def _parse_expected_signature(signature: str) -> Dict[str, Any]:
    signature = signature.strip()
    if not signature:
        raise ValueError("graph signature is empty")
    tree = ast.parse(f"def {signature}:\n    pass\n")
    function = tree.body[0]
    if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
        raise ValueError("graph signature is not a Python function signature")
    return _callable_shape(function)


def _find_candidate_signature(code: str, expected_name: str) -> Dict[str, Any]:
    tree = ast.parse(code)
    matches = [
        item
        for item in tree.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        and item.name == expected_name
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one top-level function named {expected_name!r}, "
            f"found {len(matches)}"
        )
    return _callable_shape(matches[0])


def _callable_shape(function: ast.FunctionDef | ast.AsyncFunctionDef) -> Dict[str, Any]:
    arguments = function.args
    positional_only = [argument.arg for argument in arguments.posonlyargs]
    positional_or_keyword = [argument.arg for argument in arguments.args]
    keyword_only = [argument.arg for argument in arguments.kwonlyargs]
    return {
        "name": function.name,
        "async": isinstance(function, ast.AsyncFunctionDef),
        "positional_only": positional_only,
        "positional_or_keyword": positional_or_keyword,
        "vararg": arguments.vararg.arg if arguments.vararg else None,
        "keyword_only": keyword_only,
        "kwarg": arguments.kwarg.arg if arguments.kwarg else None,
    }


def _record_rate(records: Sequence[Mapping[str, Any]], key: str) -> float:
    return _ratio(sum(bool(record.get(key)) for record in records), len(records))


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return round(sum(items) / len(items), 6) if items else 0.0


def _unique_by_id(
    records: Sequence[Mapping[str, Any]], label: str
) -> Dict[str, Mapping[str, Any]]:
    result: Dict[str, Mapping[str, Any]] = {}
    for record in records:
        problem_id = str(record.get("id", ""))
        if not problem_id:
            raise ValueError(f"{label} result is missing an ID")
        if problem_id in result:
            raise ValueError(f"{label} contains duplicate ID {problem_id!r}")
        result[problem_id] = record
    return result

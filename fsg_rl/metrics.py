"""Aggregate the accuracy, efficiency, verification, and tool metrics in the proposal."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List


def summarize_metrics(problem_summaries: List[Dict[str, Any]]) -> Dict[str, float]:
    records = [
        record
        for problem in problem_summaries
        for record in problem.get("rollout_records", [])
    ]
    if not records:
        return {}

    correct = [float(record["reward"].get("final_reward", 0.0)) for record in records]
    pass_at_k = [
        float(
            any(
                float(record["reward"].get("final_reward", 0.0)) >= 1.0
                for record in problem.get("rollout_records", [])
            )
        )
        for problem in problem_summaries
    ]
    solving_times = [float(problem.get("wall_time_seconds", 0.0)) for problem in problem_summaries]
    sandbox_times = [float(record["execution"].get("runtime_seconds", 0.0)) for record in records]
    token_counts = [
        float(record["rollout"].get("completion_token_count", 0.0)) for record in records
    ]
    tool_calls = [
        int(
            record["rollout"].get(
                "tool_call_count",
                sum(
                    len(span.get("code_blocks", []))
                    for span in record["rollout"].get("parsed_spans", [])
                ),
            )
        )
        for record in records
    ]
    node_scores = [
        float(score)
        for record in records
        for score in record["verification"].get("node_scores", {}).values()
    ]
    edge_scores = [
        float(score)
        for record in records
        for score in record["verification"].get("edge_scores", {}).values()
    ]
    repairs_attempted = [
        record for record in records if record.get("repair", {}).get("status") != "not_needed"
    ]
    repairs_succeeded = [
        record for record in repairs_attempted if record.get("repair", {}).get("status") == "succeeded"
    ]
    executable_wrong = [
        record
        for record in records
        if record["execution"].get("executable", False)
        and float(record["reward"].get("final_reward", 0.0)) < 1.0
    ]
    verifier_disagreements = [
        record
        for record in records
        if _structured_pass(record) != (
            float(record["reward"].get("final_reward", 0.0)) >= 1.0
        )
    ]
    test_results = [
        test
        for record in records
        for node in record["execution"].get("node_results", {}).values()
        for test in node.get("test_results", [])
    ]
    invalid_tools = [
        node
        for record in records
        for node in record["execution"].get("node_results", {}).values()
        if not node.get("executable", False)
    ]
    timeouts = [record for record in records if record["execution"].get("timeout", False)]

    accuracy = _mean(correct)
    average_solving_time = _mean(solving_times)
    total_tool_calls = sum(tool_calls)
    return {
        "rollout_accuracy": accuracy,
        "pass_at_k": _mean(pass_at_k),
        "average_solving_time_seconds": average_solving_time,
        "average_sandbox_time_seconds": _mean(sandbox_times),
        "average_tool_calls": _mean([float(value) for value in tool_calls]),
        "average_generated_tokens": _mean(token_counts),
        "cost_normalized_accuracy": accuracy / average_solving_time if average_solving_time else 0.0,
        "node_verification_pass_rate": _mean(node_scores),
        "edge_consistency_rate": _mean(edge_scores),
        "repair_success_rate": (
            len(repairs_succeeded) / len(repairs_attempted) if repairs_attempted else 0.0
        ),
        "verifier_disagreement_rate": len(verifier_disagreements) / len(records),
        "executable_but_wrong_rate": len(executable_wrong) / len(records),
        "tool_productivity": sum(correct) / (1 + total_tool_calls),
        "useful_tool_ratio": (
            sum(bool(test.get("passed")) for test in test_results) / len(test_results)
            if test_results
            else 0.0
        ),
        "invalid_tool_rate": len(invalid_tools) / total_tool_calls if total_tool_calls else 0.0,
        "timeout_rate": len(timeouts) / len(records),
    }


def _structured_pass(record: Dict[str, Any]) -> bool:
    verification = record["verification"]
    node_scores = list(verification.get("node_scores", {}).values())
    edge_scores = list(verification.get("edge_scores", {}).values())
    return all(float(value) >= 1.0 for value in [*node_scores, *edge_scores])


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0

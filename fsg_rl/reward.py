"""Proposal-aligned rollout and function-span reward assignment."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .executable_eval import check_exact_tags, check_python_signatures
from .schemas import (
    ExecutionResult,
    FunctionGraph,
    ParsedFunctionSpan,
    PolicyRollout,
    Problem,
    RewardRecord,
    VerificationResult,
)


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


class RewardAssigner:
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("reward", {})

    def assign(
        self,
        problem: Problem,
        graph: FunctionGraph,
        verification: VerificationResult,
        execution: ExecutionResult,
        parsed_spans: List[ParsedFunctionSpan],
        repaired: bool = False,
        rollout: Optional[PolicyRollout] = None,
        teacher_score: float = 0.0,
        teacher_node_scores: Optional[Dict[str, float]] = None,
    ) -> RewardRecord:
        del problem
        teacher_score = min(1.0, max(0.0, float(teacher_score)))
        teacher_node_scores = {
            str(node_id): min(1.0, max(0.0, float(score)))
            for node_id, score in dict(teacher_node_scores or {}).items()
        }
        parsed_format = bool(
            len(parsed_spans) == len(graph.nodes)
            and all(span.raw_text and span.start_char >= 0 for span in parsed_spans)
        )
        exact_format = (
            bool(check_exact_tags(rollout.raw_text, graph)["passed"])
            if rollout is not None
            else parsed_format
        )
        format_gate = float(parsed_format and exact_format)
        signature_check = check_python_signatures(parsed_spans, graph)
        signature_values = [
            float(bool(detail.get("passed")))
            for detail in signature_check["nodes"].values()
        ]
        signature_score = _mean(signature_values)
        node_mean = _mean(list(verification.node_scores.values()))
        edge_mean = _mean(list(verification.edge_scores.values()))
        execution_mean = _applicable_score_mean(
            graph,
            verification.node_execution_scores,
            kind="execution",
        )
        unit_test_mean = _applicable_score_mean(
            graph,
            verification.node_test_scores,
            kind="unit",
        )
        property_test_mean = _applicable_score_mean(
            graph,
            verification.node_property_scores,
            kind="property",
        )
        efficiency_penalty = self._efficiency_penalty(execution, parsed_spans, rollout)
        repair_penalty = float(repaired)
        answer_gate_floor = float(self.config.get("answer_gate_floor", 1.0))
        answer_gate = answer_gate_floor + (
            1.0 - answer_gate_floor
        ) * verification.final_answer_score

        structural_reward = (
            self._weight("lambda_format") * format_gate
            + self._weight("lambda_signature") * signature_score
        )
        process_reward = (
            self._weight("lambda_execution") * execution_mean
            + self._weight("lambda_unit_test") * unit_test_mean
            + self._weight("lambda_property_test") * property_test_mean
            + self._weight("lambda_node") * node_mean
            + self._weight("lambda_edge") * edge_mean
        )
        outcome_reward = (
            self._weight("lambda_final") * verification.final_answer_score
            + self._weight("lambda_backward") * verification.backward_score
            + self._weight("lambda_consensus") * verification.consensus_score
        )
        teacher_reward = self._weight("lambda_teacher") * teacher_score
        verified_reward = (
            structural_reward
            + answer_gate * process_reward
            + outcome_reward
            + teacher_reward
        )
        total = format_gate * verified_reward
        total -= self._weight("lambda_efficiency") * efficiency_penalty
        total -= self._weight("lambda_repair") * repair_penalty

        span_answer_credit = float(self.config.get("span_answer_credit", 0.0))
        shared_outcome_credit = span_answer_credit * (
            self._weight("lambda_final") * verification.final_answer_score
            + self._weight("lambda_backward") * verification.backward_score
        )
        span_rewards = {}
        for node in graph.nodes:
            node_id = node.id
            signature_detail = signature_check["nodes"].get(node_id, {})
            span_rewards[node_id] = (
                self._weight("lambda_signature")
                * float(bool(signature_detail.get("passed")))
                + answer_gate
                * (
                    self._weight("lambda_node")
                    * verification.node_scores.get(node_id, 0.0)
                    + self._weight("lambda_execution")
                    * verification.node_execution_scores.get(node_id, 0.0)
                    + self._weight("lambda_unit_test")
                    * verification.node_test_scores.get(node_id, 0.0)
                    + self._weight("lambda_property_test")
                    * verification.node_property_scores.get(node_id, 0.0)
                )
            )
            if node_id != "main":
                span_rewards[node_id] += shared_outcome_credit
            span_rewards[node_id] += self._weight("lambda_teacher") * (
                teacher_node_scores.get(node_id, teacher_score)
            )
        for edge in graph.edges:
            edge_score = verification.edge_scores.get(f"{edge.source}->{edge.target}", 0.0)
            credit = 0.5 * answer_gate * self._weight("lambda_edge") * edge_score
            span_rewards[edge.source] = span_rewards.get(edge.source, 0.0) + credit
            span_rewards[edge.target] = span_rewards.get(edge.target, 0.0) + credit
        span_rewards["main"] = span_rewards.get("main", 0.0) + (
            self._weight("lambda_final") * verification.final_answer_score
            + self._weight("lambda_backward") * verification.backward_score
            + self._weight("lambda_consensus") * verification.consensus_score
        )
        if format_gate == 0.0:
            span_rewards = {node_id: 0.0 for node_id in span_rewards}

        return RewardRecord(
            total_reward=round(total, 6),
            node_rewards=dict(verification.node_scores),
            edge_rewards=dict(verification.edge_scores),
            final_reward=verification.final_answer_score,
            backward_reward=verification.backward_score,
            consensus_reward=verification.consensus_score,
            format_gate=format_gate,
            efficiency_penalty=round(efficiency_penalty, 6),
            repair_penalty=repair_penalty,
            signature_reward=round(signature_score, 6),
            execution_reward=round(execution_mean, 6),
            unit_test_reward=round(unit_test_mean, 6),
            property_test_reward=round(property_test_mean, 6),
            teacher_reward=round(teacher_score, 6),
            answer_gate=round(answer_gate, 6),
            span_rewards={key: round(value, 6) for key, value in span_rewards.items()},
        )

    def _weight(self, key: str) -> float:
        return float(self.config.get(key, 0.0))

    def _efficiency_penalty(
        self,
        execution: ExecutionResult,
        parsed_spans: List[ParsedFunctionSpan],
        rollout: Optional[PolicyRollout],
    ) -> float:
        runtime_budget = float(self.config.get("runtime_budget_seconds", 5.0))
        tool_budget = float(self.config.get("tool_call_budget", 8))
        token_budget = float(self.config.get("token_budget", 2048))
        tool_calls = sum(len(span.code_blocks) for span in parsed_spans)
        tokens = (
            len(rollout.completion_token_ids)
            if rollout and rollout.completion_token_ids
            else sum(len(span.raw_text.split()) for span in parsed_spans)
        )
        runtime_component = min(1.0, execution.runtime_seconds / runtime_budget) if runtime_budget else 0.0
        tool_component = min(1.0, tool_calls / tool_budget) if tool_budget else 0.0
        token_component = min(1.0, tokens / token_budget) if token_budget else 0.0
        return (runtime_component + tool_component + token_component) / 3.0


def _applicable_score_mean(
    graph: FunctionGraph,
    scores: Dict[str, float],
    *,
    kind: str,
) -> float:
    values = []
    for node in graph.nodes:
        tests = list(node.verification_spec.get("tests", []))
        if kind == "execution":
            applicable = node.expected_output_type == "python_function" or bool(tests)
        elif kind == "unit":
            applicable = any(test.get("kind", "unit") == "unit" for test in tests)
        elif kind == "property":
            applicable = any(test.get("kind") == "property" for test in tests)
        else:
            raise ValueError(f"Unknown reward score kind: {kind!r}")
        if applicable:
            values.append(float(scores.get(node.id, 0.0)))
    return _mean(values)

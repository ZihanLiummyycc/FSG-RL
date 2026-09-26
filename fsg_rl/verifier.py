"""Layered node, edge, answer, backward, and consensus verification."""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .answer_equivalence import answers_equivalent, check_answer_equivalence
from .parsing import span_by_node
from .schemas import (
    ExecutionResult,
    FunctionGraph,
    ParsedFunctionSpan,
    Problem,
    VerificationResult,
)


class Verifier:
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("verifier", {})
        self.execution_weight = float(self.config.get("node_execution_weight", 0.25))
        self.test_weight = float(self.config.get("node_test_weight", 0.5))
        self.property_weight = float(self.config.get("node_property_weight", 0.25))

    def verify(
        self,
        problem: Problem,
        graph: FunctionGraph,
        parsed_spans: List[ParsedFunctionSpan],
        execution_result: ExecutionResult,
    ) -> VerificationResult:
        failures: List[Dict[str, Any]] = []
        execution_scores, test_scores, property_scores, node_scores = self._node_scores(
            graph,
            parsed_spans,
            execution_result,
            failures,
        )
        edge_scores = self._edge_scores(graph, parsed_spans, execution_result, failures)
        final_score = self._final_answer_score(problem, parsed_spans, failures)
        backward_score, backward_applicable = self._backward_score(
            problem,
            graph,
            parsed_spans,
            execution_result,
        )
        if backward_applicable and backward_score < 1.0:
            failures.append({"type": "backward_constraint_failure"})

        return VerificationResult(
            node_scores=node_scores,
            node_execution_scores=execution_scores,
            node_test_scores=test_scores,
            node_property_scores=property_scores,
            edge_scores=edge_scores,
            final_answer_score=final_score,
            backward_score=backward_score,
            consensus_score=0.0,
            failures=failures,
        )

    def apply_group_consensus(
        self,
        span_groups: Sequence[List[ParsedFunctionSpan]],
        verifications: Sequence[VerificationResult],
        graph: FunctionGraph,
    ) -> None:
        """Assign weak per-rollout agreement after the whole rollout group exists."""

        node_values: Dict[str, List[Optional[str]]] = {}
        for node in graph.nodes:
            node_values[node.id] = [
                _span_value(span_by_node(spans, node.id)) for spans in span_groups
            ]

        for rollout_index, verification in enumerate(verifications):
            agreement_scores = []
            for node in graph.nodes:
                values = [value for value in node_values[node.id] if value is not None]
                own = node_values[node.id][rollout_index]
                if own is None or not values:
                    continue
                counts = Counter(values)
                agreement_scores.append(counts[own] / len(values))
            verification.consensus_score = (
                sum(agreement_scores) / len(agreement_scores) if agreement_scores else 0.0
            )

    def _node_scores(
        self,
        graph: FunctionGraph,
        parsed_spans: List[ParsedFunctionSpan],
        execution_result: ExecutionResult,
        failures: List[Dict[str, Any]],
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:
        execution_scores: Dict[str, float] = {}
        test_scores: Dict[str, float] = {}
        property_scores: Dict[str, float] = {}
        combined_scores: Dict[str, float] = {}

        for node in graph.nodes:
            span = span_by_node(parsed_spans, node.id)
            if not span or not span.raw_text:
                execution_scores[node.id] = 0.0
                test_scores[node.id] = 0.0
                property_scores[node.id] = 0.0
                combined_scores[node.id] = 0.0
                failures.append({"type": "parse_failure", "node": node.id})
                continue

            execution = execution_result.node_results.get(node.id)
            tests = execution.test_results if execution else []
            unit_tests = [test for test in tests if test.get("kind", "unit") == "unit"]
            property_tests = [test for test in tests if test.get("kind") == "property"]
            expects_execution = bool(node.verification_spec.get("tests")) or (
                node.expected_output_type == "python_function"
            )

            execution_score = (
                float(bool(execution and execution.executable)) if expects_execution else 0.0
            )
            test_score = _pass_rate(unit_tests, default=0.0)
            has_static_property_spec = node.expected_output_type in {"formula", "values"} and bool(
                node.verification_spec.get(node.expected_output_type)
            )
            property_score = _pass_rate(
                property_tests,
                default=(
                    self._non_executable_property_score(node, span)
                    if has_static_property_spec
                    else 0.0
                ),
            )
            execution_scores[node.id] = execution_score
            test_scores[node.id] = test_score
            property_scores[node.id] = property_score
            weighted_components = []
            if expects_execution:
                weighted_components.append((self.execution_weight, execution_score))
            if unit_tests:
                weighted_components.append((self.test_weight, test_score))
            if property_tests or has_static_property_spec:
                weighted_components.append((self.property_weight, property_score))
            applicable_weight = sum(weight for weight, _ in weighted_components)
            if applicable_weight:
                combined_scores[node.id] = sum(
                    weight * score for weight, score in weighted_components
                ) / applicable_weight

            if expects_execution and execution_score < 1.0:
                failures.append(
                    {
                        "type": "execution_failure",
                        "node": node.id,
                        "stderr": execution.stderr if execution else "missing execution result",
                    }
                )
            if unit_tests and test_score < 1.0:
                failures.append({"type": "test_failure", "node": node.id, "tests": unit_tests})
            if property_tests and property_score < 1.0:
                failures.append(
                    {"type": "property_failure", "node": node.id, "tests": property_tests}
                )
            if (
                node.id in combined_scores
                and combined_scores[node.id] < 1.0
                and not expects_execution
            ):
                failures.append({"type": "node_verification_failure", "node": node.id})

        return execution_scores, test_scores, property_scores, combined_scores

    def _non_executable_property_score(
        self,
        node: Any,
        span: ParsedFunctionSpan,
    ) -> Tuple[float, bool]:
        if node.expected_output_type == "formula":
            expected = node.verification_spec.get("formula")
            if expected:
                return float(_normalize_math_text(str(expected)) in _normalize_math_text(span.raw_text))
            return 0.0
        if node.expected_output_type == "values":
            expected_values = dict(node.verification_spec.get("values", {}))
            if not expected_values:
                return 0.0
            matches = [
                answers_equivalent(span.extracted_values.get(key), value)
                for key, value in expected_values.items()
            ]
            return sum(matches) / len(matches) if matches else 0.0
        return 0.0

    def _edge_scores(
        self,
        graph: FunctionGraph,
        parsed_spans: List[ParsedFunctionSpan],
        execution_result: ExecutionResult,
        failures: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        for edge in graph.edges:
            key = f"{edge.source}->{edge.target}"
            if edge.relation_type == "uses_value":
                score = self._check_uses_value(
                    edge.source, edge.target, parsed_spans, execution_result
                )
            elif edge.relation_type == "implements_formula":
                score = self._check_implements_formula(
                    edge.source, edge.target, parsed_spans, execution_result
                )
            elif edge.relation_type == "aggregates_result":
                score = self._check_aggregates_result(
                    edge.source, edge.target, parsed_spans, execution_result
                )
            elif edge.relation_type == "equivalent_output":
                score = self._check_equivalent_output(
                    edge.source, edge.target, execution_result
                )
            else:
                score = 0.0
            scores[key] = score
            if score < 1.0 and edge.severity == "critical":
                failures.append(
                    {
                        "type": "dependency_failure",
                        "edge": key,
                        "relation_type": edge.relation_type,
                    }
                )
        return scores

    def _check_uses_value(
        self,
        source: str,
        target: str,
        parsed_spans: List[ParsedFunctionSpan],
        execution_result: ExecutionResult,
    ) -> float:
        parent = span_by_node(parsed_spans, source)
        child = span_by_node(parsed_spans, target)
        child_result = execution_result.node_results.get(target)
        if not parent or not child:
            return 0.0
        values = [str(value) for value in parent.extracted_values.values()]
        if not values:
            return 0.0
        target_evidence = child.raw_text
        if child_result:
            target_evidence += " " + json.dumps(child_result.outputs, ensure_ascii=False)
        return sum(value in target_evidence for value in values) / len(values)

    def _check_implements_formula(
        self,
        source: str,
        target: str,
        parsed_spans: List[ParsedFunctionSpan],
        execution_result: ExecutionResult,
    ) -> float:
        parent = span_by_node(parsed_spans, source)
        child_result = execution_result.node_results.get(target)
        if not parent or not parent.raw_text or not child_result or not child_result.executable:
            return 0.0
        properties = [
            test for test in child_result.test_results if test.get("kind") == "property"
        ]
        tests = properties or child_result.test_results
        return _pass_rate(tests, default=0.0)

    def _check_aggregates_result(
        self,
        source: str,
        target: str,
        parsed_spans: List[ParsedFunctionSpan],
        execution_result: ExecutionResult,
    ) -> float:
        parent_result = execution_result.node_results.get(source)
        target_span = span_by_node(parsed_spans, target)
        if not parent_result or not target_span or target_span.extracted_answer is None:
            return 0.0
        actual_values = [
            str(test.get("actual", ""))
            for test in parent_result.test_results
            if test.get("actual") is not None
        ]
        return float(any(answers_equivalent(value, target_span.extracted_answer) for value in actual_values))

    @staticmethod
    def _check_equivalent_output(
        source: str,
        target: str,
        execution_result: ExecutionResult,
    ) -> float:
        left = execution_result.node_results.get(source)
        right = execution_result.node_results.get(target)
        if not left or not right:
            return 0.0
        return float(left.outputs == right.outputs)

    def _final_answer_score(
        self,
        problem: Problem,
        parsed_spans: List[ParsedFunctionSpan],
        failures: List[Dict[str, Any]],
    ) -> float:
        predicted = self._main_answer(parsed_spans)
        if predicted is None or problem.gold_answer is None:
            failures.append({"type": "final_answer_failure", "predicted": predicted})
            return 0.0
        equivalence = check_answer_equivalence(predicted, problem.gold_answer)
        score = float(equivalence.equivalent)
        if score < 1.0:
            failures.append(
                {
                    "type": "final_answer_failure",
                    "predicted": predicted,
                    "gold": problem.gold_answer,
                    "equivalence_method": equivalence.method,
                    "normalized_predicted": equivalence.normalized_left,
                    "normalized_gold": equivalence.normalized_right,
                    "equivalence_error": equivalence.error,
                }
            )
        return score

    def _backward_score(
        self,
        problem: Problem,
        graph: FunctionGraph,
        parsed_spans: List[ParsedFunctionSpan],
        execution_result: ExecutionResult,
    ) -> float:
        main = graph.node_by_id("main")
        target_call = str(
            problem.metadata.get(
                "target_call",
                main.verification_spec.get("target_call", "") if main else "",
            )
        )
        main_answer = self._main_answer(parsed_spans)
        if not target_call:
            return 0.0, False
        for result in execution_result.node_results.values():
            if target_call in result.outputs and answers_equivalent(
                result.outputs[target_call], main_answer
            ):
                return 1.0, True
        return 0.0, True

    @staticmethod
    def _main_answer(parsed_spans: List[ParsedFunctionSpan]) -> Optional[str]:
        main = span_by_node(parsed_spans, "main")
        return main.extracted_answer if main else None


def _pass_rate(tests: Sequence[Dict[str, Any]], default: float) -> float:
    if not tests:
        return default
    return sum(bool(test.get("passed")) for test in tests) / len(tests)


def _span_value(span: Optional[ParsedFunctionSpan]) -> Optional[str]:
    if span is None or not span.raw_text:
        return None
    if span.extracted_answer is not None:
        return "answer:" + _normalize_math_text(span.extracted_answer)
    if span.extracted_formula is not None:
        return "formula:" + _normalize_math_text(span.extracted_formula)
    if span.extracted_values:
        return "values:" + json.dumps(span.extracted_values, sort_keys=True)
    return "text:" + _normalize_math_text(span.raw_text)


def _normalize_math_text(value: str) -> str:
    value = value.strip().replace("\\,", "").replace(" ", "")
    value = re.sub(r"^\$|\$$", "", value)
    return value.lower()

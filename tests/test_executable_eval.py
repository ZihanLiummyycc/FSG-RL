from __future__ import annotations

import unittest

from fsg_rl.executable_eval import (
    check_exact_tags,
    check_python_signatures,
    execution_test_counts,
    metric_deltas,
    pair_evaluations,
    summarize_evaluations,
)
from fsg_rl.parsing import parse_tagged_function_spans
from fsg_rl.rollout import build_policy_messages
from fsg_rl.schemas import (
    ExecutionResult,
    FunctionGraph,
    FunctionNode,
    MemoryContext,
    NodeExecutionResult,
    Problem,
)


def _graph() -> FunctionGraph:
    return FunctionGraph(
        problem_id="p1",
        nodes=[
            FunctionNode(
                id="f1",
                name="double",
                question="Double n.",
                signature="double(n: int, *, offset: int = 0) -> int",
                expected_output_type="python_function",
                verification_spec={},
            ),
            FunctionNode(
                id="main",
                name="answer",
                question="Answer.",
                signature="main() -> int",
                expected_output_type="final_answer",
                verification_spec={},
            ),
        ],
        edges=[],
    )


class ExecutableEvalTests(unittest.TestCase):
    def test_rollout_prompt_requires_self_contained_stdlib_only_code(self):
        messages = build_policy_messages(
            Problem(id="p1", text="Double five.", gold_answer="10"),
            _graph(),
            MemoryContext(),
        )
        system = messages[0]["content"][0]["text"]
        normalized = " ".join(system.split())
        self.assertIn("executed alone in a fresh Python 3 interpreter", normalized)
        self.assertIn("do not share runtime state", normalized)
        self.assertIn("include every required standard-library import", normalized)
        self.assertIn("including sympy, are forbidden", normalized)
        self.assertIn("every referenced name is", normalized)
        self.assertIn("Begin immediately with the first required <a_X> tag", normalized)
        self.assertIn("Do not emit a plan, scratchpad, meta-analysis", normalized)

    def test_exact_tags_require_order_and_no_extras(self):
        graph = _graph()
        valid = (
            "<a_f1>```python\ndef double(n, *, offset=0):\n "
            "   return 2*n + offset\n```</a_f1>"
            "<a_main>\\boxed{10}</a_main>"
        )
        self.assertTrue(check_exact_tags(valid, graph)["passed"])
        invalid = valid + "<a_extra>x</a_extra>"
        self.assertFalse(check_exact_tags(invalid, graph)["passed"])

    def test_exact_tags_reject_text_outside_spans(self):
        graph = _graph()
        valid = (
            "<a_f1>```python\ndef double(n, *, offset=0):\n"
            "    return 2*n + offset\n```</a_f1>"
            "<a_main>\\boxed{10}</a_main>"
        )
        for invalid in (
            "I should plan first. " + valid,
            valid.replace("</a_f1><a_main>", "</a_f1>more planning<a_main>"),
            valid + " done",
        ):
            result = check_exact_tags(invalid, graph)
            self.assertFalse(result["passed"])
            self.assertTrue(result["outside_text_present"])

    def test_signature_uses_name_and_parameter_structure(self):
        graph = _graph()
        valid = (
            "<a_f1>```python\ndef double(n, *, offset=0):\n"
            "    return 2*n + offset\n```</a_f1>"
            "<a_main>\\boxed{10}</a_main>"
        )
        spans = parse_tagged_function_spans(valid, graph)
        result = check_python_signatures(spans, graph)
        self.assertTrue(result["passed"])
        wrong = valid.replace("double(n, *, offset=0)", "double(value, offset=0)")
        spans = parse_tagged_function_spans(wrong, graph)
        self.assertFalse(check_python_signatures(spans, graph)["passed"])

    def test_test_counts_and_summary(self):
        graph = _graph()
        graph.nodes[0].verification_spec = {
            "tests": [
                {"kind": "unit", "call": "double(1)", "expected": 2},
                {"kind": "unit", "call": "double(2)", "expected": 4},
                {"kind": "property", "call": "double(0) == 0", "expected": True},
            ]
        }
        execution = ExecutionResult(
            node_results={
                "f1": NodeExecutionResult(
                    node_id="f1",
                    executable=True,
                    test_results=[
                        {"kind": "unit", "passed": True},
                        {"kind": "unit", "passed": False},
                        {"kind": "property", "passed": True},
                    ],
                )
            }
        )
        counts = execution_test_counts(execution, graph)
        self.assertEqual(counts["unit_passed"], 1)
        self.assertEqual(counts["unit_total"], 2)
        summary = summarize_evaluations(
            [
                {
                    "tags_exact": True,
                    "code_blocks_complete": True,
                    "signature_exact": True,
                    "python_executable": True,
                    "all_hidden_tests_pass": False,
                    "backward_target_correct": True,
                    "final_answer_correct": True,
                    "full_success": False,
                    "test_counts": counts,
                }
            ]
        )
        self.assertEqual(summary["unit_tests"]["pass_rate"], 0.5)
        self.assertEqual(summary["property_tests"]["pass_rate"], 1.0)
        delta = metric_deltas(summary, summary)
        self.assertTrue(all(value == 0.0 for value in delta.values()))

    def test_missing_execution_counts_hidden_tests_as_failed(self):
        graph = _graph()
        graph.nodes[0].verification_spec = {
            "tests": [
                {"kind": "unit", "call": "double(1)", "expected": 2},
                {"kind": "property", "call": "double(0) == 0", "expected": True},
            ]
        }
        counts = execution_test_counts(ExecutionResult(), graph)
        self.assertEqual(counts["unit_total"], 1)
        self.assertEqual(counts["unit_passed"], 0)
        self.assertEqual(counts["property_total"], 1)
        self.assertEqual(counts["property_passed"], 0)

    def test_pair_evaluations_tracks_improvement_and_regression(self):
        before = [{"id": "p1", "tags_exact": False, "full_success": True}]
        after = [{"id": "p1", "tags_exact": True, "full_success": False}]
        pairs, counts = pair_evaluations(before, after)
        self.assertEqual(
            pairs[0]["metrics"]["tags_exact"]["transition"], "improved"
        )
        self.assertEqual(counts["tags_exact"]["improved"], 1)
        self.assertEqual(counts["full_success"]["regressed"], 1)


if __name__ == "__main__":
    unittest.main()

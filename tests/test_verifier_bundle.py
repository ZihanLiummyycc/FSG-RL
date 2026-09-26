from __future__ import annotations

import unittest

from fsg_rl.verifier_bundle import VerifierBundleError, compile_verifier_bundle
from fsg_rl.verifier_execution import validate_compiled_verifier_record
from fsg_rl.tool_execution import ToolExecutor


def _source_record():
    return {
        "id": "p1",
        "text": "Compute f(5).",
        "gold_answer": "10",
        "split": "train",
        "metadata": {
            "function_graph": {
                "problem_id": "p1",
                "nodes": [
                    {
                        "id": "f1",
                        "name": "double",
                        "question": "Double n.",
                        "signature": "double(n: int) -> int",
                        "expected_output_type": "int",
                        "verification_spec": {"method": "arithmetic"},
                    },
                    {
                        "id": "main",
                        "name": "answer",
                        "question": "Compute f(5).",
                        "signature": "main() -> int",
                        "expected_output_type": "int",
                        "verification_spec": {},
                    },
                ],
                "edges": [
                    {
                        "source": "f1",
                        "target": "main",
                        "relation_type": "aggregates_result",
                        "check_method": "same result",
                        "severity": "critical",
                    }
                ],
            }
        },
    }


def _payload():
    return {
        "eligible": True,
        "confidence": 0.9,
        "reason": "finite arithmetic",
        "node_verifiers": [
            {
                "node_id": "f1",
                "canonical_output_type": "python_function",
                "check_types": ["unit", "property", "small_bruteforce"],
                "tests": [
                    {"kind": "unit", "call": "double(0)", "expected": 0},
                    {"kind": "unit", "call": "double(1)", "expected": 2},
                    {"kind": "unit", "call": "double(5)", "expected": 10},
                    {
                        "kind": "property",
                        "expression": "all(double(n) == 2*n for n in range(10))",
                        "expected": True,
                        "source": "small_bruteforce",
                    },
                ],
                "reference_code": "def double(n):\n    return 2*n",
                "mutant_code": "def double(n):\n    return 2*n + 1",
                "oracle_strategy": "direct small integers",
                "estimated_runtime_seconds": 0.1,
            },
            {
                "node_id": "main",
                "canonical_output_type": "final_answer",
                "check_types": ["backward"],
                "target_call": "double(5)",
            },
        ],
    }


class VerifierBundleTests(unittest.TestCase):
    def test_compiles_separate_public_and_hidden_graphs(self):
        compiled = compile_verifier_bundle(
            _source_record(), _payload(), annotation_model="teacher"
        )
        metadata = compiled["metadata"]
        public_f1 = metadata["function_graph"]["nodes"][0]
        hidden_f1 = metadata["verification_graph"]["nodes"][0]
        self.assertEqual(public_f1["expected_output_type"], "python_function")
        self.assertNotIn("tests", public_f1["verification_spec"])
        self.assertEqual(len(hidden_f1["verification_spec"]["tests"]), 4)
        self.assertNotIn("reference_solution", compiled)

    def test_rejects_too_few_tests(self):
        payload = _payload()
        payload["node_verifiers"][0]["tests"] = payload["node_verifiers"][0]["tests"][:3]
        with self.assertRaisesRegex(VerifierBundleError, "at least four"):
            compile_verifier_bundle(_source_record(), payload, annotation_model="teacher")

    def test_rejects_forbidden_code(self):
        payload = _payload()
        payload["node_verifiers"][0]["reference_code"] = "import os\ndef double(n): return n"
        with self.assertRaisesRegex(VerifierBundleError, "forbidden module"):
            compile_verifier_bundle(_source_record(), payload, annotation_model="teacher")

    def test_rejects_undefined_placeholder_in_tests(self):
        payload = _payload()
        payload["node_verifiers"][0]["tests"][3]["expression"] = (
            "all(f(n) == 2*n for n in range(10))"
        )
        with self.assertRaisesRegex(VerifierBundleError, "undefined functions.*f"):
            compile_verifier_bundle(_source_record(), payload, annotation_model="teacher")

    def test_generation_time_execution_audit(self):
        compiled = compile_verifier_bundle(
            _source_record(), _payload(), annotation_model="teacher"
        )
        executor = ToolExecutor(
            {
                "sandbox": {
                    "backend": "subprocess",
                    "allow_unsafe_subprocess": True,
                    "timeout_seconds": 2,
                }
            }
        )
        audit = validate_compiled_verifier_record(compiled, executor)
        self.assertTrue(audit["reference_all_tests_passed"])
        self.assertTrue(audit["backward_target_passed"])

    def test_backward_audit_accepts_numeric_answer_with_unit(self):
        source = _source_record()
        source["gold_answer"] = "10 hours"
        compiled = compile_verifier_bundle(
            source, _payload(), annotation_model="teacher"
        )
        executor = ToolExecutor(
            {
                "sandbox": {
                    "backend": "subprocess",
                    "allow_unsafe_subprocess": True,
                    "timeout_seconds": 2,
                }
            }
        )
        audit = validate_compiled_verifier_record(compiled, executor)
        self.assertTrue(audit["backward_target_passed"])

    def test_execution_audit_rejects_wrong_expected_value(self):
        payload = _payload()
        payload["node_verifiers"][0]["tests"][1]["expected"] = 999
        compiled = compile_verifier_bundle(
            _source_record(), payload, annotation_model="teacher"
        )
        executor = ToolExecutor(
            {
                "sandbox": {
                    "backend": "subprocess",
                    "allow_unsafe_subprocess": True,
                    "timeout_seconds": 2,
                }
            }
        )
        with self.assertRaisesRegex(VerifierBundleError, "failed tests"):
            validate_compiled_verifier_record(compiled, executor)


if __name__ == "__main__":
    unittest.main()

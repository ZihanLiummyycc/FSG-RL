from __future__ import annotations

import json
import unittest

from scripts.prepare_medium_math_sft import (
    build_medium_teacher_messages,
    normalize_teacher_signatures,
    validate_medium_annotation,
)


class MediumMathSftTests(unittest.TestCase):
    def setUp(self):
        self.record = {
            "id": "medium-test",
            "problem": "Given x=1, compute x+1.",
            "reference_solution": "Add one to obtain 2.",
            "gold_answer": "2",
            "domain": ["Arithmetic"],
            "difficulty": 1,
            "source": "openai/gsm8k",
            "source_metadata": {},
            "problem_sha256": "abc",
            "split": "train",
        }

    def test_prompt_requires_self_contained_code_and_gold_constraint(self):
        messages = build_medium_teacher_messages(
            self.record,
            [],
            max_nodes=4,
        )
        system = messages[0]["content"]
        self.assertIn("fresh Python 3 interpreter", system)
        self.assertIn("gold_answer is a hard constraint", system)
        self.assertIn("sympy", system)
        payload = json.loads(messages[1]["content"])
        self.assertEqual(payload["source"], "openai/gsm8k")

    def test_valid_code_aware_annotation_passes(self):
        master = validate_medium_annotation(
            self.record,
            self._payload(),
            max_nodes=4,
        )
        self.assertEqual(master["code_validation"]["python_node_count"], 1)
        self.assertTrue(master["code_validation"]["signature_exact"])

    def test_missing_python_node_is_rejected(self):
        payload = self._payload()
        payload["function_graph"]["nodes"][0]["expected_output_type"] = "formula"
        payload["tagged_solution"] = (
            "<a_calc>Use x+1.</a_calc>"
            "<a_main>The answer is \\boxed{2}</a_main>"
        )
        with self.assertRaisesRegex(ValueError, "python_function"):
            validate_medium_annotation(self.record, payload, max_nodes=4)

    def test_wrong_signature_is_rejected(self):
        payload = self._payload()
        payload["tagged_solution"] = payload["tagged_solution"].replace(
            "def increment(x: int)",
            "def increment(y: int)",
        )
        with self.assertRaisesRegex(ValueError, "signatures"):
            validate_medium_annotation(self.record, payload, max_nodes=4)

    def test_def_style_graph_signature_is_normalized(self):
        payload = self._payload()
        payload["function_graph"]["nodes"][0]["signature"] = (
            "def increment(x: int) -> int:"
        )
        payload["function_graph"]["nodes"][1]["signature"] = "def main() -> int:"
        normalized = normalize_teacher_signatures(payload)
        self.assertEqual(
            normalized["function_graph"]["nodes"][0]["signature"],
            "increment(x: int) -> int",
        )
        master = validate_medium_annotation(self.record, payload, max_nodes=4)
        self.assertEqual(
            master["function_graph"]["nodes"][0]["signature"],
            "increment(x: int) -> int",
        )

    def test_python_node_must_directly_feed_main(self):
        payload = self._payload()
        payload["function_graph"]["nodes"].insert(
            1,
            {
                "id": "format",
                "name": "format",
                "question": "Format the intermediate result.",
                "signature": "format(value: int) -> int",
                "expected_output_type": "formula",
                "verification_spec": {},
            },
        )
        payload["function_graph"]["edges"] = [
            {
                "source": "calc",
                "target": "format",
                "relation_type": "uses_value",
                "check_method": "Pass the intermediate value.",
                "severity": "normal",
            },
            {
                "source": "format",
                "target": "main",
                "relation_type": "uses_value",
                "check_method": "Use the formatted value.",
                "severity": "normal",
            },
        ]
        payload["tagged_solution"] = payload["tagged_solution"].replace(
            "</a_calc>",
            "</a_calc><a_format>Format the result.</a_format>",
        )
        with self.assertRaisesRegex(ValueError, "directly feed main"):
            validate_medium_annotation(self.record, payload, max_nodes=4)

    def _payload(self):
        return {
            "function_graph": {
                "nodes": [
                    {
                        "id": "calc",
                        "name": "increment",
                        "question": "Return x plus one.",
                        "signature": "increment(x: int) -> int",
                        "expected_output_type": "python_function",
                        "verification_spec": {},
                    },
                    {
                        "id": "main",
                        "name": "main",
                        "question": "Give the final answer.",
                        "signature": "main() -> int",
                        "expected_output_type": "final_answer",
                        "verification_spec": {},
                    },
                ],
                "edges": [
                    {
                        "source": "calc",
                        "target": "main",
                        "relation_type": "uses_value",
                        "check_method": "Use the computed increment.",
                        "severity": "normal",
                    }
                ],
            },
            "tagged_solution": (
                "<a_calc>```python\n"
                "def increment(x: int) -> int:\n"
                "    return x + 1\n"
                "```</a_calc>"
                "<a_main>Substitute x=1. The answer is \\boxed{2}</a_main>"
            ),
            "annotation_model": "test",
            "retrieved_memory": [],
        }


if __name__ == "__main__":
    unittest.main()

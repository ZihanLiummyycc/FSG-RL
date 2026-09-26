from __future__ import annotations

import json
import unittest

from fsg_rl.code_sft import build_code_sft_record, build_replay_mix


def _private_record():
    public_graph = {
        "problem_id": "p1",
        "nodes": [
            {
                "id": "f1",
                "name": "double",
                "question": "Double n.",
                "signature": "double(n: int) -> int",
                "expected_output_type": "python_function",
                "verification_spec": {
                    "check_types": ["unit", "property"],
                    "semantic_requirements": ["Return twice n."],
                },
            },
            {
                "id": "main",
                "name": "answer",
                "question": "Compute the answer.",
                "signature": "main() -> int",
                "expected_output_type": "final_answer",
                "verification_spec": {
                    "check_types": ["backward"],
                    "semantic_requirements": [],
                },
            },
        ],
        "edges": [
            {
                "source": "f1",
                "target": "main",
                "relation_type": "uses_value",
                "check_method": "same result",
                "severity": "critical",
            }
        ],
    }
    hidden_graph = json.loads(json.dumps(public_graph))
    hidden_graph["nodes"][0]["verification_spec"] = {
        "tests": [
            {"kind": "unit", "call": "double(5)", "expected": 10},
        ]
    }
    hidden_graph["nodes"][1]["verification_spec"] = {
        "target_call": "double(5)"
    }
    return {
        "id": "p1",
        "text": "What is twice five?",
        "gold_answer": "10",
        "metadata": {
            "function_graph": public_graph,
            "verification_graph": hidden_graph,
            "verifier_implementations": {
                "f1": {
                    "reference_code": "def double(n):\n    return 2*n",
                    "mutant_code": "def double(n):\n    return 2*n + 1",
                }
            },
        },
    }


def _master():
    return {
        "id": "p1",
        "problem": "What is twice five?",
        "tagged_solution": (
            "<a_f1>Twice five is ten.</a_f1>"
            "<a_main>Therefore \\boxed{10}.</a_main>"
        ),
    }


class CodeSftTests(unittest.TestCase):
    def test_builds_code_target_without_prompt_leakage(self):
        record = build_code_sft_record(_private_record(), _master())
        target = record["messages"][-1]["content"]
        self.assertIn("```python\ndef double(n):", target)
        self.assertIn(r"\boxed{10}", target)
        prompt = json.dumps(record["messages"][:-1], ensure_ascii=False)
        self.assertNotIn("mutant_code", prompt)
        self.assertNotIn("target_call", prompt)
        self.assertNotIn('"tests"', prompt)
        self.assertNotIn("return 2*n + 1", target)
        system_content = record["messages"][0]["content"]
        normalized_system = " ".join(
            " ".join(
                str(part.get("text", ""))
                for part in system_content
                if isinstance(part, dict)
            ).split()
        )
        self.assertIn(
            "executed alone in a fresh Python 3 interpreter",
            normalized_system,
        )

    def test_replay_mix_is_50_25_25(self):
        code = [build_code_sft_record(_private_record(), _master())]
        graph = [{"id": "g1", "task": "construct_function_graph"}]
        solve = [{"id": "s1", "task": "solve_with_function_graph"}]
        mixed = build_replay_mix(code, graph, solve, seed=42, code_repeats=2)
        counts = {}
        for row in mixed:
            counts[row["task"]] = counts.get(row["task"], 0) + 1
        self.assertEqual(counts["solve_with_executable_function_graph"], 2)
        self.assertEqual(counts["construct_function_graph"], 1)
        self.assertEqual(counts["solve_with_function_graph"], 1)


if __name__ == "__main__":
    unittest.main()

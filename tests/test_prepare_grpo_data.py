from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "prepare_grpo_data.py"
SPEC = importlib.util.spec_from_file_location("prepare_grpo_data", SCRIPT_PATH)
assert SPEC and SPEC.loader
prepare_grpo_data = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare_grpo_data)


def _graph(problem_id: str):
    return {
        "problem_id": problem_id,
        "nodes": [
            {
                "id": "main",
                "name": "answer",
                "question": "Answer.",
                "signature": "main() -> int",
                "expected_output_type": "final_answer",
                "verification_spec": {},
            }
        ],
        "edges": [],
    }


class PrepareGrpoDataTests(unittest.TestCase):
    def test_output_keeps_graph_but_drops_teacher_solution(self):
        record = prepare_grpo_data._to_grpo_problem(
            {
                "id": "p1",
                "problem": "What is 1+1?",
                "gold_answer": "2",
                "reference_solution": "This must never be copied.",
                "tagged_solution": "This must never be copied either.",
                "domain": ["algebra"],
                "difficulty": 1,
                "function_graph": _graph("p1"),
            }
        )
        self.assertEqual(record["gold_answer"], "2")
        self.assertIn("function_graph", record["metadata"])
        self.assertFalse(
            prepare_grpo_data._contains_forbidden_key(
                record, {"reference_solution", "tagged_solution"}
            )
        )

    def test_overlap_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "train/holdout leakage"):
            prepare_grpo_data._require_disjoint(
                "train", {"p1", "p2"}, "holdout", {"p2", "p3"}
            )

    def test_invalid_graph_is_rejected(self):
        invalid = _graph("p1")
        invalid["nodes"][0]["id"] = "not_main"
        with self.assertRaises(ValueError):
            prepare_grpo_data._to_grpo_problem(
                {
                    "id": "p1",
                    "problem": "What is 1+1?",
                    "gold_answer": "2",
                    "function_graph": invalid,
                }
            )


if __name__ == "__main__":
    unittest.main()

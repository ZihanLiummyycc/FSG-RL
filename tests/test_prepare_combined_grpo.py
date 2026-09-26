from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).parents[1] / "scripts" / "prepare_combined_grpo.py"
SPEC = importlib.util.spec_from_file_location("prepare_combined_grpo", SCRIPT)
assert SPEC and SPEC.loader
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)


def _graph(problem_id: str, *, hidden: bool):
    spec = {"target_call": "helper(2)"} if hidden else {"check_types": []}
    helper_spec = (
        {
            "tests": [
                {"kind": "unit", "call": "helper(2)", "expected": 2},
                {"kind": "unit", "call": "helper(3)", "expected": 3},
                {"kind": "unit", "call": "helper(4)", "expected": 4},
                {"kind": "property", "expression": "helper(5)==5", "expected": True},
            ]
        }
        if hidden
        else {"check_types": ["unit", "property"]}
    )
    return {
        "problem_id": problem_id,
        "nodes": [
            {
                "id": "helper",
                "name": "helper",
                "question": "Return n.",
                "signature": "helper(n: int) -> int",
                "expected_output_type": "python_function",
                "verification_spec": helper_spec,
            },
            {
                "id": "main",
                "name": "answer",
                "question": "Answer.",
                "signature": "main() -> int",
                "expected_output_type": "final_answer",
                "verification_spec": spec,
            },
        ],
        "edges": [
            {
                "source": "helper",
                "target": "main",
                "relation_type": "uses_value",
                "check_method": "hidden",
                "severity": "hard",
            }
        ],
    }


def _record(problem_id: str, *, private: bool, source: str = "openai/gsm8k"):
    metadata = {
        "source": source,
        "difficulty": 2,
        "function_graph": _graph(problem_id, hidden=False),
        "verification_graph": _graph(problem_id, hidden=True),
    }
    if private:
        metadata["verifier_implementations"] = {
            "helper": {
                "reference_code": "def helper(n): return n",
                "mutant_code": "def helper(n): return n+1",
            }
        }
    return {
        "id": problem_id,
        "text": "Return two.",
        "gold_answer": "2",
        "split": "train",
        "source": source,
        "difficulty": 2,
        "metadata": metadata,
    }


class PrepareCombinedGrpoTests(unittest.TestCase):
    def test_deterministic_split_and_private_code_removal(self):
        medium = [_record(f"medium_{index}", private=True) for index in range(20)]
        stage = [
            {"id": row["id"], "split": "train"}
            for row in medium
        ]
        omni_train = [_record(f"omni_train_{index}", private=False) for index in range(4)]
        omni_validation = [_record("omni_validation_1", private=False)]

        first = prepare.build_combined_splits(
            medium,
            stage,
            [],
            [],
            omni_train,
            omni_validation,
            validation_fraction=0.2,
            seed=42,
        )
        second = prepare.build_combined_splits(
            medium,
            stage,
            [],
            [],
            omni_train,
            omni_validation,
            validation_fraction=0.2,
            seed=42,
        )

        self.assertEqual(first["medium_grpo_validation"], second["medium_grpo_validation"])
        self.assertEqual(len(first["medium_grpo_train"]), 16)
        self.assertEqual(len(first["medium_grpo_validation"]), 4)
        self.assertEqual(len(first["combined_grpo_train"]), 20)
        self.assertEqual(len(first["combined_grpo_validation"]), 5)
        self.assertFalse(
            prepare._contains_forbidden_key(
                first["combined_grpo_train"],
                prepare.PRIVATE_KEYS,
            )
        )
        self.assertIn(
            "verifier_implementations",
            first["medium_private_train"][0]["metadata"],
        )
        self.assertEqual(first["audit"]["train_validation_overlap"], 0)

    def test_pool_leakage_is_rejected(self):
        medium = [_record("medium_1", private=True)]
        with self.assertRaisesRegex(ValueError, "pool validation leakage"):
            prepare.build_combined_splits(
                medium,
                [{"id": "medium_1", "split": "train"}],
                [{"id": "medium_1"}],
                [],
                [],
                [],
                validation_fraction=0.2,
                seed=42,
            )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = _load("prepare_eval100", ROOT / "scripts" / "prepare_eval100.py")
compare = _load("compare_eval100", ROOT / "scripts" / "compare_eval100.py")


def _graph(problem_id: str, hidden: bool):
    helper_spec = (
        {"tests": [{"kind": "unit", "call": "helper(2)", "expected": 2}]}
        if hidden
        else {"check_types": ["unit"]}
    )
    main_spec = {"target_call": "helper(2)"} if hidden else {"check_types": []}
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
                "name": "main",
                "question": "Give the answer.",
                "signature": "main() -> int",
                "expected_output_type": "final_answer",
                "verification_spec": main_spec,
            },
        ],
        "edges": [
            {
                "source": "helper",
                "target": "main",
                "relation_type": "uses_value",
                "check_method": "Use helper.",
                "severity": "hard",
            }
        ],
    }


def _record(problem_id: str, source: str = "openai/gsm8k", private: bool = False):
    metadata = {
        "source": source,
        "difficulty": 2,
        "function_graph": _graph(problem_id, False),
        "verification_graph": _graph(problem_id, True),
    }
    if private:
        metadata["verifier_implementations"] = {
            "helper": {
                "reference_code": "def helper(n): return n",
                "mutant_code": "def helper(n): return n + 1",
            }
        }
    return {
        "id": problem_id,
        "text": "Return two.",
        "gold_answer": "2",
        "split": "validation",
        "source": source,
        "difficulty": 2,
        "metadata": metadata,
    }


class Eval100Tests(unittest.TestCase):
    def test_final_set_is_exactly_17_plus_79_plus_4_without_leakage(self):
        base = [
            _record(f"omni_{index}", "KbsdJames/Omni-MATH")
            for index in range(17)
        ] + [_record(f"medium_base_{index}") for index in range(79)]
        extras = [_record(f"medium_extra_{index}", private=True) for index in range(8)]
        train = [_record(f"medium_train_{index}") for index in range(20)]

        rows, audit = prepare.finalize_eval_set(
            base,
            extras,
            train,
            target_size=100,
            expected_omni=17,
            expected_existing_medium=79,
            seed=3030,
        )

        self.assertEqual(len(rows), 100)
        self.assertEqual(audit["family_counts"], {"omni": 17, "medium": 83})
        self.assertEqual(audit["new_internal_test_rows"], 4)
        self.assertEqual(audit["train_overlap"], 0)
        self.assertFalse(prepare._contains_forbidden_key(rows, prepare.PRIVATE_KEYS))

    def test_pool_selection_rejects_train_and_validation_ids(self):
        pool = [
            {
                "id": f"medium_{index}",
                "problem": "p",
                "reference_solution": "s",
                "gold_answer": "1",
                "source": "openai/gsm8k",
                "difficulty": 2,
                "split": "test",
            }
            for index in range(10)
        ]
        rows = prepare.select_extra_candidates(
            pool,
            [{"id": "medium_0"}],
            [{"id": "medium_1"}],
            count=4,
            seed=3030,
        )
        self.assertEqual(len(rows), 4)
        self.assertFalse({"medium_0", "medium_1"} & {row["id"] for row in rows})

    def test_paired_report_counts_improvements_and_regressions(self):
        data = {"a": _record("a"), "b": _record("b")}
        baseline = {
            "a": {"id": "a", "final_answer_correct": False},
            "b": {"id": "b", "final_answer_correct": True},
        }
        candidate = {
            "a": {"id": "a", "final_answer_correct": True},
            "b": {"id": "b", "final_answer_correct": False},
        }
        report = compare.build_report(
            data,
            baseline,
            candidate,
            baseline_label="before",
            candidate_label="after",
        )
        metric = report["metrics"]["final_answer_correct"]
        self.assertEqual(metric["improved"], 1)
        self.assertEqual(metric["regressed"], 1)
        self.assertEqual(metric["mcnemar_exact_p"], 1.0)


if __name__ == "__main__":
    unittest.main()

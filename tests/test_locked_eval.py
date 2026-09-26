from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = _load(
    "prepare_locked_eval", ROOT / "scripts" / "prepare_locked_eval.py"
)
compare = _load(
    "compare_locked_eval", ROOT / "scripts" / "compare_locked_eval.py"
)
evaluate = _load(
    "evaluate_executable_sft_script",
    ROOT / "scripts" / "evaluate_executable_sft.py",
)


def _graph(problem_id: str, *, hidden: bool) -> dict:
    helper_spec = (
        {
            "tests": [
                {"kind": "unit", "call": "helper(1)", "expected": 1},
                {"kind": "unit", "call": "helper(2)", "expected": 2},
                {"kind": "unit", "call": "helper(3)", "expected": 3},
                {
                    "kind": "property",
                    "expression": "all(helper(n) == n for n in range(4))",
                    "expected": True,
                },
            ]
        }
        if hidden
        else {"check_types": ["unit", "property"]}
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
                "question": "Return two.",
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


def _raw(problem_id: str, *, split: str, source: str) -> dict:
    return {
        "id": problem_id,
        "problem": f"Compute the value for {problem_id}.",
        "reference_solution": "The value is two.",
        "gold_answer": "2",
        "source": source,
        "difficulty": 2,
        "domain": ["Algebra"],
        "split": split,
    }


def _accepted(problem_id: str, *, source: str) -> dict:
    return {
        "id": problem_id,
        "text": f"Compute the value for {problem_id}.",
        "gold_answer": "2",
        "source": source,
        "difficulty": 2,
        "metadata": {
            "source": source,
            "difficulty": 2,
            "domain": ["Algebra"],
            "function_graph": _graph(problem_id, hidden=False),
            "verification_graph": _graph(problem_id, hidden=True),
            "verifier_implementations": {
                "helper": {
                    "reference_code": "def helper(n): return n",
                    "mutant_code": "def helper(n): return n + 1",
                }
            },
        },
    }


class LockedEvalTests(unittest.TestCase):
    def test_select_excludes_training_and_prior_eval_by_id_and_text(self):
        medium = [
            _raw(f"medium_{index}", split="test", source="openai/gsm8k")
            for index in range(6)
        ]
        omni = [
            _raw(f"omni_math_{index}", split="holdout", source="Omni-MATH")
            for index in range(6)
        ]
        training = [{"id": "medium_0"}]
        prior_eval = [
            {
                "id": "old_eval",
                "text": medium[1]["problem"],
            }
        ]
        rows, audit = prepare.select_candidates(
            medium,
            omni,
            training,
            prior_eval,
            medium_count=3,
            omni_count=3,
            seed=7,
        )
        selected_ids = {row["id"] for row in rows}
        self.assertEqual(len(rows), 6)
        self.assertNotIn("medium_0", selected_ids)
        self.assertNotIn("medium_1", selected_ids)
        self.assertEqual(audit["id_overlap_with_training_or_prior_eval"], 0)

    def test_finalize_enforces_family_quota_and_removes_private_code(self):
        raw = [
            _raw(f"medium_{index}", split="test", source="openai/gsm8k")
            for index in range(3)
        ] + [
            _raw(f"omni_math_{index}", split="holdout", source="Omni-MATH")
            for index in range(2)
        ]
        accepted = [
            _accepted(row["id"], source=row["source"])
            for row in raw
        ]
        rows, audit = prepare.finalize_locked_eval(
            raw,
            accepted,
            [],
            [],
            medium_target=2,
            omni_target=1,
            seed=11,
        )
        self.assertEqual(len(rows), 3)
        self.assertEqual(audit["family_counts"], {"medium": 2, "omni": 1})
        self.assertTrue(all(row["split"] == "locked_test" for row in rows))
        self.assertFalse(
            prepare._contains_forbidden_key(rows, prepare.PRIVATE_KEYS)
        )

    def test_three_way_report_counts_paired_changes(self):
        data = {
            "a": {"id": "a", "source": "s"},
            "b": {"id": "b", "source": "s"},
        }
        template = {
            "test_counts": {
                "unit_passed": 0,
                "unit_total": 0,
                "property_passed": 0,
                "property_total": 0,
            }
        }
        results = {
            "start": {
                "a": {"id": "a", **template, "final_answer_correct": False},
                "b": {"id": "b", **template, "final_answer_correct": True},
            },
            "teacher15": {
                "a": {"id": "a", **template, "final_answer_correct": True},
                "b": {"id": "b", **template, "final_answer_correct": True},
            },
            "teacher_final": {
                "a": {"id": "a", **template, "final_answer_correct": False},
                "b": {"id": "b", **template, "final_answer_correct": False},
            },
        }
        report = compare.build_report(data, results, baseline_label="start")
        first = report["pairwise"]["start_vs_teacher15"]["metrics"]
        self.assertEqual(first["final_answer_correct"]["improved"], 1)
        self.assertEqual(first["final_answer_correct"]["regressed"], 0)
        self.assertIn("holm_adjusted_p", first["final_answer_correct"])

    def test_resume_loader_rejects_adapter_mismatch_and_retries_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.jsonl"
            path.write_text(
                json.dumps(
                    {"id": "a", "adapter": "model", "status": "error"}
                )
                + "\n",
                encoding="utf-8",
            )
            loaded = evaluate._load_resumable_results(
                path,
                label="model",
                expected_ids={"a"},
                retry_errors=True,
            )
            self.assertEqual(loaded, {})
            with self.assertRaisesRegex(ValueError, "adapter mismatch"):
                evaluate._load_resumable_results(
                    path,
                    label="other",
                    expected_ids={"a"},
                    retry_errors=False,
                )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import unittest

from fsg_rl.configuration import ConfigurationError, validate_config
from fsg_rl.schemas import (
    ExecutionResult,
    FunctionGraph,
    FunctionNode,
    KnowledgeItem,
    MemoryContext,
    PolicyRollout,
    Problem,
    VerificationResult,
)
from fsg_rl.teacher import TeacherClient
from scripts.summarize_memory_teacher_inference import build_summary


def _config() -> dict:
    return {
        "dataset": {"algorithm_problem_dataset_path": "data.jsonl"},
        "policy": {
            "backend": "transformers",
            "model_name_or_path": "model",
        },
        "rollout": {"num_rollouts": 1},
        "decomposition": {"backend": "dataset"},
        "teacher": {
            "enabled": True,
            "api": {
                "model": "teacher",
                "base_url": "https://teacher.invalid/v1",
                "api_key_env": "TEACHER_API_KEY",
            },
        },
        "repair": {"enabled": True},
        "sandbox": {"backend": "docker"},
    }


def _graph() -> FunctionGraph:
    return FunctionGraph(
        problem_id="p1",
        nodes=[
            FunctionNode(
                id="main",
                name="answer",
                question="Answer.",
                signature="main() -> int",
                expected_output_type="final_answer",
                verification_spec={},
            )
        ],
        edges=[],
    )


class _FakeChatClient:
    def __init__(self):
        self.messages = None

    def complete_json(self, messages, **kwargs):
        del kwargs
        self.messages = messages
        return {
            "diagnosis": "Use mathlib4:t0.",
            "repair_instructions": "Correct the arithmetic and regenerate all spans.",
            "failed_nodes": ["main"],
            "failed_edges": [],
            "memory_item": "Check the final arithmetic.",
            "repaired_graph": None,
        }


class InferenceMemoryTeacherTests(unittest.TestCase):
    def test_inference_repair_requires_teacher_api(self):
        config = _config()
        validate_config(config, "inference")
        config["teacher"]["enabled"] = False
        with self.assertRaises(ConfigurationError):
            validate_config(config, "inference")

    def test_teacher_receives_only_top_three_theories(self):
        teacher = TeacherClient(_config())
        fake = _FakeChatClient()
        teacher.client = fake
        memory = MemoryContext(
            theorem_items=[
                KnowledgeItem(
                    id=f"mathlib4:t{index}",
                    source="mathlib4",
                    item_type="theorem",
                    text=f"Theorem {index}",
                    keywords=["number_theory"],
                    metadata={"lean_name": f"T{index}"},
                )
                for index in range(4)
            ]
        )
        teacher.diagnose_and_plan(
            Problem(id="p1", text="Compute 2+2.", gold_answer="4"),
            _graph(),
            PolicyRollout(problem_id="p1", raw_text="<a_main>\\boxed{5}</a_main>"),
            ExecutionResult(),
            VerificationResult(
                final_answer_score=0.0,
                failures=[{"type": "final_answer_failure", "node": "main"}],
            ),
            "verification_failure",
            memory_context=memory,
        )
        payload = json.loads(fake.messages[1]["content"])
        ids = [item["id"] for item in payload["retrieved_memory"]["theorem_items"]]
        self.assertEqual(ids, ["mathlib4:t0", "mathlib4:t1", "mathlib4:t2"])

    def test_summary_compares_before_and_after_repair(self):
        memory = {
            "theorem_items": [
                {
                    "id": f"mathlib4:t{index}",
                    "source": "mathlib4",
                    "item_type": "theorem",
                    "text": f"Theorem {index}",
                    "keywords": [],
                    "metadata": {},
                }
                for index in range(3)
            ]
        }
        rows = [
            {
                "problem": {"id": "p1", "text": "q1", "gold_answer": "4"},
                "memory_context": memory,
                "rollout_records": [
                    {
                        "rollout": {"raw_text": "corrected"},
                        "verification": {
                            "final_answer_score": 1.0,
                            "backward_score": 1.0,
                            "failures": [],
                        },
                        "repair": {
                            "status": "succeeded",
                            "teacher_diagnosis": "arithmetic",
                            "attempts": [{"status": "repaired_by_policy"}],
                            "verification_before_repair": {
                                "final_answer_score": 0.0,
                                "backward_score": 0.0,
                                "failures": [{"type": "final_answer_failure"}],
                            },
                        },
                    }
                ],
            },
            {
                "problem": {"id": "p2", "text": "q2", "gold_answer": "5"},
                "memory_context": memory,
                "rollout_records": [
                    {
                        "rollout": {"raw_text": "already correct"},
                        "verification": {
                            "final_answer_score": 1.0,
                            "backward_score": 1.0,
                            "failures": [],
                        },
                        "repair": {"status": "not_needed", "attempts": []},
                    }
                ],
            },
        ]
        summary = build_summary(rows)
        self.assertEqual(summary["problems"], 2)
        self.assertEqual(summary["memory"]["problems_with_three_theories"], 2)
        self.assertEqual(summary["teacher"]["api_calls"], 1)
        self.assertEqual(summary["before_teacher"]["final_answer_accuracy"], 0.5)
        self.assertEqual(summary["after_teacher"]["final_answer_accuracy"], 1.0)
        self.assertEqual(summary["full_success_transitions"]["improved"], 1)


if __name__ == "__main__":
    unittest.main()

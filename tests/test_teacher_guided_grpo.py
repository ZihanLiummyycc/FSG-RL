from __future__ import annotations

import unittest

from fsg_rl.configuration import ConfigurationError, validate_config
from fsg_rl.repair import RepairModule, _public_failure_summaries
from fsg_rl.schemas import (
    ExecutionResult,
    FunctionGraph,
    FunctionNode,
    NodeExecutionResult,
    PolicyRollout,
    Problem,
    VerificationResult,
)
from fsg_rl.teacher import TeacherClient, _normalize_group_judgment
from scripts.select_teacher_grpo_hardset import select_hard_rows
from scripts.summarize_teacher_grpo_run import summarize


def _config() -> dict:
    return {
        "dataset": {"algorithm_problem_dataset_path": "data.jsonl"},
        "policy": {"backend": "transformers", "model_name_or_path": "model"},
        "rollout": {"num_rollouts": 4},
        "decomposition": {"backend": "dataset"},
        "teacher": {
            "enabled": True,
            "feedback_visibility": "aggregate",
            "api": {
                "model": "teacher",
                "base_url": "https://teacher.invalid/v1",
                "api_key_env": "TEACHER_API_KEY",
            },
        },
        "repair": {
            "enabled": True,
            "repair_budget": 1,
            "max_teacher_calls_per_problem": 1,
        },
        "sandbox": {"backend": "docker"},
    }


def _trajectory(problem_id: str, final_scores: list[float], process: float) -> dict:
    records = []
    for index, final_score in enumerate(final_scores):
        records.append(
            {
                "verification": {
                    "final_answer_score": final_score,
                    "backward_score": final_score,
                },
                "reward": {
                    "total_reward": process + final_score + index * 0.01,
                    "execution_reward": process,
                    "unit_test_reward": process,
                    "property_test_reward": process,
                },
            }
        )
    return {"problem": {"id": problem_id}, "rollout_records": records}


class TeacherGuidedGrpoTests(unittest.TestCase):
    def test_teacher_call_budget_is_configured_and_validated(self):
        config = _config()
        config["repair"]["group_mode"] = "shared_feedback"
        validate_config(config, "train")
        self.assertEqual(RepairModule(config).max_teacher_calls_per_problem, 1)
        self.assertEqual(RepairModule(config).group_mode, "shared_feedback")

        config["repair"]["max_teacher_calls_per_problem"] = -1
        with self.assertRaises(ConfigurationError):
            validate_config(config, "train")

        config["repair"]["max_teacher_calls_per_problem"] = 1
        with self.assertRaises(ConfigurationError):
            validate_config(config, "inference")

    def test_group_judge_requires_one_call_and_positive_reward(self):
        config = _config()
        config["repair"]["group_mode"] = "group_judge"
        config["reward"] = {"lambda_teacher": 0.25}
        validate_config(config, "train")
        self.assertEqual(RepairModule(config).group_mode, "group_judge")

        config["repair"]["max_teacher_calls_per_problem"] = 2
        with self.assertRaisesRegex(ConfigurationError, "exactly one"):
            validate_config(config, "train")

        config["repair"]["max_teacher_calls_per_problem"] = 1
        config["reward"]["lambda_teacher"] = 0.0
        with self.assertRaisesRegex(ConfigurationError, "lambda_teacher"):
            validate_config(config, "train")

    def test_group_judgment_is_complete_bounded_and_api_matched(self):
        result = _normalize_group_judgment(
            {
                "rollout_judgments": [
                    {
                        "index": 1,
                        "overall_score": 0.2,
                        "node_scores": {"main": 0.0},
                        "diagnosis": "wrong answer",
                    },
                    {
                        "index": 0,
                        "overall_score": 0.9,
                        "node_scores": {"f1": 1.0, "main": 1.0},
                        "diagnosis": "correct",
                    },
                ],
                "group_lesson": "Check the invariant.",
            },
            rollout_count=2,
            node_ids={"f1", "main"},
        )
        self.assertEqual(
            [item["index"] for item in result["rollout_judgments"]],
            [0, 1],
        )
        with self.assertRaisesRegex(ValueError, "unknown node"):
            _normalize_group_judgment(
                {
                    "rollout_judgments": [
                        {
                            "index": 0,
                            "overall_score": 0.5,
                            "node_scores": {"private": 1.0},
                        }
                    ]
                },
                rollout_count=1,
                node_ids={"main"},
            )

    def test_group_judge_request_contains_no_hidden_expected_values(self):
        class Client:
            def __init__(self):
                self.messages = None

            def complete_json(self, messages, **kwargs):
                del kwargs
                self.messages = messages
                return {
                    "rollout_judgments": [
                        {
                            "index": 0,
                            "overall_score": 0.5,
                            "node_scores": {"main": 0.5},
                            "diagnosis": "uncertain",
                        }
                    ],
                    "group_lesson": "Check the arithmetic.",
                }

        client = Client()
        teacher = object.__new__(TeacherClient)
        teacher.temperature = 0.0
        teacher.max_tokens = 256
        teacher.feedback_visibility = "aggregate"
        teacher.client = client
        graph = FunctionGraph(
            problem_id="p1",
            nodes=[
                FunctionNode(
                    id="main",
                    name="main",
                    question="Answer.",
                    signature="main() -> int",
                    expected_output_type="final_answer",
                    verification_spec={},
                )
            ],
            edges=[],
        )
        teacher.judge_rollout_group(
            Problem(id="p1", text="Compute 1+1."),
            graph,
            [PolicyRollout(problem_id="p1", raw_text="<a_main>2</a_main>")],
            [
                ExecutionResult(
                    node_results={
                        "main": NodeExecutionResult(
                            node_id="main",
                            executable=True,
                            test_results=[
                                {
                                    "kind": "unit",
                                    "passed": False,
                                    "expected": "SECRET_EXPECTED",
                                }
                            ],
                        )
                    }
                )
            ],
        )
        request_text = str(client.messages)
        self.assertNotIn("SECRET_EXPECTED", request_text)
        self.assertNotIn("gold_answer", request_text)

    def test_policy_feedback_hides_private_expected_values(self):
        summaries = _public_failure_summaries(
            VerificationResult(
                failures=[
                    {
                        "type": "final_answer_failure",
                        "node": "main",
                        "gold": "SECRET_GOLD",
                        "expected": "SECRET_EXPECTED",
                    }
                ]
            )
        )
        self.assertEqual(
            summaries,
            [{"type": "final_answer_failure", "node": "main"}],
        )

    def test_hardset_uses_only_train_rows_and_respects_omni_quota(self):
        train = [
            {
                "id": problem_id,
                "text": "question",
                "gold_answer": "1",
                "split": "train",
                "metadata": {"difficulty": 3},
            }
            for problem_id in (
                "omni_math_a",
                "omni_math_b",
                "medium_a",
                "medium_b",
                "medium_easy",
            )
        ]
        trajectories = [
            _trajectory("omni_math_a", [0, 1, 0, 0], 0.8),
            _trajectory("omni_math_b", [0, 0, 0, 0], 0.7),
            _trajectory("medium_a", [0, 1, 0, 0], 0.9),
            _trajectory("medium_b", [0, 0, 0, 0], 0.6),
            _trajectory("medium_easy", [1, 1, 1, 1], 1.0),
        ]
        selected, audit = select_hard_rows(
            train,
            trajectories,
            [{"id": "held_out"}],
            limit=3,
            omni_rows=1,
            max_final_rate=0.5,
            min_learnability=0.2,
        )
        selected_ids = {row["id"] for row in selected}
        self.assertEqual(len(selected_ids), 3)
        self.assertEqual(sum(value.startswith("omni_math_") for value in selected_ids), 1)
        self.assertNotIn("medium_easy", selected_ids)
        self.assertEqual(audit["eval_overlap"], 0)

    def test_hardset_rejects_eval_overlap(self):
        train = [
            {
                "id": "medium_a",
                "text": "question",
                "gold_answer": "1",
                "split": "train",
                "metadata": {},
            }
        ]
        with self.assertRaisesRegex(ValueError, "overlap evaluation"):
            select_hard_rows(
                train,
                [_trajectory("medium_a", [0, 1], 0.8)],
                [{"id": "medium_a"}],
                limit=1,
                omni_rows=0,
                max_final_rate=0.5,
                min_learnability=0.2,
            )

    def test_run_summary_counts_teacher_calls_and_conditioned_accuracy(self):
        rows = [
            {
                "teacher_calls_used": 1,
                "teacher_probe": {
                    "verification": {"final_answer_score": 0.0}
                },
                "rollout_records": [
                    {
                        "repair": {"status": "shared_feedback"},
                        "verification": {
                            "final_answer_score": 1.0,
                            "backward_score": 1.0,
                        },
                    },
                    {
                        "repair": {"status": "shared_feedback"},
                        "verification": {
                            "final_answer_score": 0.0,
                            "backward_score": 0.0,
                        },
                    },
                ],
                "grpo_update": {"approx_kl": 0.01},
            }
        ]
        summary = summarize(rows)
        self.assertEqual(summary["teacher_api_calls"], 1)
        self.assertEqual(summary["problems_with_shared_teacher_feedback"], 1)
        self.assertEqual(summary["probe_final_answer_accuracy"], 0.0)
        self.assertEqual(summary["teacher_conditioned_rollout_accuracy"], 0.5)

    def test_run_summary_counts_teacher_free_group_judgments(self):
        rows = [
            {
                "teacher_calls_used": 1,
                "teacher_group_judgment": {
                    "rollout_judgments": [
                        {"index": 0, "overall_score": 0.8},
                        {"index": 1, "overall_score": 0.2},
                    ]
                },
                "rollout_records": [
                    {
                        "repair": {"status": "group_judged"},
                        "reward": {"teacher_reward": 0.8},
                        "verification": {
                            "final_answer_score": 1.0,
                            "backward_score": 1.0,
                        },
                    },
                    {
                        "repair": {"status": "group_judged"},
                        "reward": {"teacher_reward": 0.2},
                        "verification": {
                            "final_answer_score": 0.0,
                            "backward_score": 0.0,
                        },
                    },
                ],
                "grpo_update": {"approx_kl": 0.001},
            }
        ]
        summary = summarize(rows)
        self.assertEqual(summary["problems_with_group_teacher_judgment"], 1)
        self.assertEqual(summary["teacher_free_training_rollout_accuracy"], 0.5)
        self.assertEqual(summary["mean_teacher_judge_score"], 0.5)


if __name__ == "__main__":
    unittest.main()

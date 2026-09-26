from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import main as pipeline
from fsg_rl.rollout import TransformersPolicy
from fsg_rl.schemas import (
    ExecutionResult,
    FunctionGraph,
    FunctionNode,
    MemoryContext,
    PolicyRollout,
    Problem,
    VerificationResult,
)


class _Policy(TransformersPolicy):
    trainable = True
    reference_policy_description = "fake_reference"

    def __init__(self):
        self.calls = []

    def rollout(self, problem, graph, memory_context, rollout_index, repair_feedback=None):
        del memory_context
        self.calls.append(
            {
                "problem": problem.id,
                "graph": graph.problem_id,
                "rollout_index": rollout_index,
                "repair_feedback": repair_feedback,
            }
        )
        return PolicyRollout(
            problem_id=problem.id,
            raw_text="<a_main>\\boxed{2}</a_main>",
            prompt_token_ids=[1],
            completion_token_ids=[2],
        )


class _Memory:
    def retrieve(self, problem):
        del problem
        return MemoryContext()

    def update(self, *args):
        del args

    def size_summary(self):
        return {}


class _Decomposer:
    def __init__(self, graph):
        self.graph = graph

    def construct(self, problem, memory_context):
        del problem, memory_context
        return self.graph


class _Executor:
    def execute(self, spans, graph):
        del spans, graph
        return ExecutionResult(executable=True)


class _Verifier:
    def verify(self, problem, graph, spans, execution):
        del problem, graph, spans, execution
        return VerificationResult(final_answer_score=1.0, backward_score=1.0)

    def apply_group_consensus(self, *args):
        del args


class _Repair:
    enabled = True
    repair_budget = 1
    max_teacher_calls_per_problem = 1
    group_mode = "group_judge"
    fail_open = False

    def __init__(self):
        self.calls = 0

    def judge_group(self, problem, graph, memory_context, rollouts, executions):
        del problem, graph, memory_context, executions
        self.calls += 1
        return {
            "rollout_judgments": [
                {
                    "index": index,
                    "overall_score": index / max(1, len(rollouts) - 1),
                    "node_scores": {"main": 1.0},
                    "diagnosis": "checked",
                }
                for index in range(len(rollouts))
            ],
            "group_lesson": "Use the direct calculation.",
        }


class _Trainer:
    def __init__(self, config, policy):
        del config, policy

    def update(self, rollouts, rewards):
        return {
            "update_count": 1,
            "num_rollouts": len(rollouts),
            "group_advantages": [record.total_reward for record in rewards],
            "loss": 0.0,
            "policy_objective": 0.0,
            "approx_kl": 0.0,
            "epsilon": 0.2,
            "beta_kl": 0.0,
            "grad_norm": 0.0,
            "checkpoint": None,
        }

    def save_final(self):
        return Path("final")


class TeacherFreeGroupJudgeIntegrationTest(unittest.TestCase):
    def test_group_judge_never_enters_student_rollout_prompt(self):
        graph = FunctionGraph(
            problem_id="p1",
            nodes=[
                FunctionNode(
                    id="main",
                    name="main",
                    question="Compute 1+1.",
                    signature="main() -> int",
                    expected_output_type="final_answer",
                    verification_spec={},
                )
            ],
            edges=[],
        )
        problem = Problem(id="p1", text="Compute 1+1.", gold_answer="2")
        policy = _Policy()
        repair = _Repair()

        with tempfile.TemporaryDirectory() as directory:
            trajectory = Path(directory) / "trajectories.jsonl"
            config = {
                "dataset": {"algorithm_problem_dataset_path": "unused.jsonl"},
                "policy": {
                    "backend": "transformers",
                    "model_name_or_path": "model",
                },
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
                    "group_mode": "group_judge",
                },
                "reward": {"lambda_teacher": 0.25},
                "sandbox": {"backend": "docker"},
                "output": {"trajectory_path": str(trajectory)},
            }
            with (
                patch.object(pipeline, "load_dataset", return_value=[problem]),
                patch.object(pipeline, "MemoryBank", return_value=_Memory()),
                patch.object(pipeline, "ToolExecutor", return_value=_Executor()),
                patch.object(pipeline, "Verifier", return_value=_Verifier()),
                patch.object(pipeline, "build_policy", return_value=policy),
                patch.object(
                    pipeline,
                    "build_decomposer",
                    return_value=_Decomposer(graph),
                ),
                patch.object(pipeline, "RepairModule", return_value=repair),
                patch.object(pipeline, "GRPOTrainer", _Trainer),
            ):
                summary = pipeline.run(config, "train")

        self.assertEqual(len(policy.calls), 4)
        self.assertTrue(
            all(call["repair_feedback"] is None for call in policy.calls)
        )
        self.assertEqual(repair.calls, 1)
        self.assertEqual(summary["num_problems"], 1)


if __name__ == "__main__":
    unittest.main()

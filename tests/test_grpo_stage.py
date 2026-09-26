from __future__ import annotations

import json
import unittest

from fsg_rl.configuration import ConfigurationError, validate_config
from fsg_rl.parsing import parse_tagged_function_spans
from fsg_rl.reward import RewardAssigner
from fsg_rl.rollout import TransformersPolicy
from fsg_rl.schemas import (
    ExecutionResult,
    FunctionGraph,
    NodeExecutionResult,
    PolicyRollout,
    Problem,
    RewardRecord,
    VerificationResult,
)
from fsg_rl.training_graphs import resolve_verification_graph


def _graphs() -> tuple[dict, dict]:
    public = {
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
                    "semantic_requirements": [],
                },
            },
            {
                "id": "main",
                "name": "answer",
                "question": "Return the answer.",
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
                "relation_type": "aggregates_result",
                "check_method": "same output",
                "severity": "critical",
            }
        ],
    }
    hidden = json.loads(json.dumps(public))
    hidden["nodes"][0]["verification_spec"] = {
        "tests": [
            {"kind": "unit", "call": "double(5)", "expected": 10},
            {
                "kind": "property",
                "expression": "double(0) == 0",
                "expected": True,
            },
        ]
    }
    hidden["nodes"][1]["verification_spec"] = {
        "target_call": "double(5)"
    }
    return public, hidden


class GrpoStageTests(unittest.TestCase):
    def test_invalid_answer_gate_configuration_is_rejected(self):
        config = {
            "dataset": {"algorithm_problem_dataset_path": "data.jsonl"},
            "policy": {
                "backend": "transformers",
                "model_name_or_path": "model",
            },
            "rollout": {"num_rollouts": 2},
            "decomposition": {"backend": "dataset"},
            "repair": {"enabled": False},
            "sandbox": {"backend": "docker"},
            "reward": {"answer_gate_floor": 1.1},
        }
        with self.assertRaises(ConfigurationError):
            validate_config(config, "train")

    def test_reference_adapter_context_restores_trainable_policy(self):
        class Parameter:
            def __init__(self):
                self.requires_grad = False

            def requires_grad_(self, enabled):
                self.requires_grad = enabled

        class Model:
            def __init__(self):
                self.active_adapter = "default"
                self.training = True
                self.reference_parameter = Parameter()

            def set_adapter(self, name):
                self.active_adapter = name

            def named_parameters(self):
                return [("layer.lora_A.fsg_reference.weight", self.reference_parameter)]

            def eval(self):
                self.training = False

            def train(self):
                self.training = True

        policy = object.__new__(TransformersPolicy)
        policy.model = Model()
        policy.policy_adapter_name = "default"
        policy.reference_adapter_name = "fsg_reference"

        with policy.reference_adapter_context():
            self.assertEqual(policy.model.active_adapter, "fsg_reference")
            self.assertFalse(policy.model.training)
            self.assertFalse(policy.model.reference_parameter.requires_grad)

        self.assertEqual(policy.model.active_adapter, "default")
        self.assertTrue(policy.model.training)
        self.assertEqual(policy.reference_policy_description, "frozen_initial_adapter")

    def test_hidden_graph_is_separate_and_api_matched(self):
        public_data, hidden_data = _graphs()
        problem = Problem(
            id="p1",
            text="What is twice five?",
            gold_answer="10",
            metadata={
                "function_graph": public_data,
                "verification_graph": hidden_data,
            },
        )
        public = FunctionGraph.from_dict(public_data)
        hidden = resolve_verification_graph(problem, public)
        self.assertNotIn("tests", public.nodes[0].verification_spec)
        self.assertEqual(len(hidden.nodes[0].verification_spec["tests"]), 2)

    def test_hidden_graph_api_mismatch_is_rejected(self):
        public_data, hidden_data = _graphs()
        hidden_data["nodes"][0]["signature"] = "double(value: int) -> int"
        problem = Problem(
            id="p1",
            text="What is twice five?",
            gold_answer="10",
            metadata={"verification_graph": hidden_data},
        )
        with self.assertRaisesRegex(ValueError, "public/hidden node APIs differ"):
            resolve_verification_graph(problem, FunctionGraph.from_dict(public_data))

    def test_reward_exposes_dense_execution_and_test_components(self):
        _, hidden_data = _graphs()
        graph = FunctionGraph.from_dict(hidden_data)
        raw = (
            "<a_f1>```python\ndef double(n):\n    return 2 * n\n```</a_f1>"
            "<a_main>\\boxed{10}</a_main>"
        )
        spans = parse_tagged_function_spans(raw, graph)
        execution = ExecutionResult(
            executable=True,
            node_results={
                "f1": NodeExecutionResult(
                    node_id="f1",
                    executable=True,
                    test_results=[
                        {"kind": "unit", "passed": True},
                        {"kind": "property", "passed": True},
                    ],
                )
            },
        )
        verification = VerificationResult(
            node_scores={"f1": 1.0},
            node_execution_scores={"f1": 1.0},
            node_test_scores={"f1": 1.0},
            node_property_scores={"f1": 1.0},
            final_answer_score=1.0,
            backward_score=1.0,
        )
        rollout = PolicyRollout(
            problem_id="p1",
            raw_text=raw,
            completion_token_ids=[1] * 10,
        )
        reward = RewardAssigner(
            {
                "reward": {
                    "lambda_format": 0.1,
                    "lambda_signature": 0.15,
                    "lambda_execution": 0.2,
                    "lambda_unit_test": 0.35,
                    "lambda_property_test": 0.25,
                    "lambda_final": 1.0,
                    "lambda_backward": 0.25,
                }
            }
        ).assign(
            Problem(id="p1", text="What is twice five?", gold_answer="10"),
            graph,
            verification,
            execution,
            spans,
            rollout=rollout,
        )
        self.assertEqual(reward.format_gate, 1.0)
        self.assertEqual(reward.signature_reward, 1.0)
        self.assertEqual(reward.execution_reward, 1.0)
        self.assertEqual(reward.unit_test_reward, 1.0)
        self.assertEqual(reward.property_test_reward, 1.0)
        self.assertEqual(reward.answer_gate, 1.0)
        self.assertAlmostEqual(reward.total_reward, 2.3)
        self.assertGreater(reward.span_rewards["f1"], 0.0)
        restored = RewardRecord.from_dict(reward.to_dict())
        self.assertEqual(restored.unit_test_reward, 1.0)

    def test_answer_gate_makes_correct_outcome_outrank_perfect_wrong_process(self):
        _, hidden_data = _graphs()
        graph = FunctionGraph.from_dict(hidden_data)
        raw = (
            "<a_f1>```python\ndef double(n):\n    return 2 * n\n```</a_f1>"
            "<a_main>\\boxed{10}</a_main>"
        )
        spans = parse_tagged_function_spans(raw, graph)
        rollout = PolicyRollout(
            problem_id="p1",
            raw_text=raw,
            completion_token_ids=[1] * 10,
        )
        config = {
            "reward": {
                "lambda_format": 0.1,
                "lambda_signature": 0.15,
                "lambda_execution": 0.5,
                "lambda_unit_test": 0.75,
                "lambda_property_test": 0.5,
                "lambda_final": 1.5,
                "lambda_backward": 0.75,
                "answer_gate_floor": 0.25,
                "span_answer_credit": 0.5,
            }
        }
        assigner = RewardAssigner(config)
        execution = ExecutionResult(executable=True)
        wrong = assigner.assign(
            Problem(id="p1", text="What is twice five?", gold_answer="10"),
            graph,
            VerificationResult(
                node_execution_scores={"f1": 1.0},
                node_test_scores={"f1": 1.0},
                node_property_scores={"f1": 1.0},
                final_answer_score=0.0,
                backward_score=0.0,
            ),
            execution,
            spans,
            rollout=rollout,
        )
        correct = assigner.assign(
            Problem(id="p1", text="What is twice five?", gold_answer="10"),
            graph,
            VerificationResult(
                node_execution_scores={"f1": 0.0},
                node_test_scores={"f1": 0.0},
                node_property_scores={"f1": 0.0},
                final_answer_score=1.0,
                backward_score=0.0,
            ),
            execution,
            spans,
            rollout=rollout,
        )

        self.assertEqual(wrong.answer_gate, 0.25)
        self.assertEqual(correct.answer_gate, 1.0)
        self.assertGreater(correct.total_reward, wrong.total_reward)
        self.assertGreater(correct.span_rewards["f1"], wrong.span_rewards["f1"])

    def test_group_teacher_score_adds_bounded_total_and_node_credit(self):
        _, hidden_data = _graphs()
        graph = FunctionGraph.from_dict(hidden_data)
        raw = (
            "<a_f1>```python\ndef double(n):\n    return 2 * n\n```</a_f1>"
            "<a_main>\\boxed{10}</a_main>"
        )
        spans = parse_tagged_function_spans(raw, graph)
        rollout = PolicyRollout(problem_id="p1", raw_text=raw)
        assigner = RewardAssigner(
            {
                "reward": {
                    "lambda_format": 0.1,
                    "lambda_teacher": 0.25,
                }
            }
        )
        baseline = assigner.assign(
            Problem(id="p1", text="What is twice five?", gold_answer="10"),
            graph,
            VerificationResult(),
            ExecutionResult(executable=False),
            spans,
            rollout=rollout,
        )
        judged = assigner.assign(
            Problem(id="p1", text="What is twice five?", gold_answer="10"),
            graph,
            VerificationResult(),
            ExecutionResult(executable=False),
            spans,
            rollout=rollout,
            teacher_score=0.8,
            teacher_node_scores={"f1": 1.0, "main": 0.5},
        )
        self.assertEqual(judged.teacher_reward, 0.8)
        self.assertAlmostEqual(judged.total_reward - baseline.total_reward, 0.2)
        self.assertAlmostEqual(
            judged.span_rewards["f1"] - baseline.span_rewards["f1"],
            0.25,
        )
        restored = RewardRecord.from_dict(judged.to_dict())
        self.assertEqual(restored.teacher_reward, 0.8)


if __name__ == "__main__":
    unittest.main()

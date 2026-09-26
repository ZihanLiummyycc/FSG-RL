from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from fsg_rl.api_client import build_chat_completion_request, extract_json_object
from fsg_rl.configuration import ConfigurationError, validate_config
from fsg_rl.datasets import DatasetError, load_dataset
from fsg_rl.decomposition import (
    DecompositionError,
    build_decomposer,
    validate_function_graph,
)
from fsg_rl.grpo import GRPOTrainer
from fsg_rl.metrics import summarize_metrics
from fsg_rl.mathlib_memory import (
    formalize_traced_theorem,
    merge_enrichment,
    normalize_traced_theorem,
)
from fsg_rl.parsing import parse_tagged_function_spans
from fsg_rl.reward import RewardAssigner
from fsg_rl.rollout import (
    PolicyGeneration,
    _render_chat_messages,
    _sampling_generation_kwargs,
    find_tagged_token_ranges,
)
from fsg_rl.schemas import (
    DependencyEdge,
    FunctionGraph,
    FunctionNode,
    MemoryContext,
    PolicyRollout,
    Problem,
)
from fsg_rl.sft_data import (
    build_sft_records,
    canonicalize_omni_record,
    deterministic_stratified_sample,
    validate_teacher_annotation,
)
from fsg_rl.sft_training import encode_sft_messages
from fsg_rl.tool_execution import ToolExecutor
from fsg_rl.verifier import Verifier


class CharacterTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False):
        del add_special_tokens
        return [ord(character) for character in text]


class ProcessorWithoutChatTemplate:
    def apply_chat_template(self, *args, **kwargs):
        del args, kwargs
        raise ValueError(
            "Cannot use apply_chat_template because this processor does not have a chat template."
        )


class TokenizerWithChatTemplate:
    def __init__(self):
        self.template_kwargs = []

    def apply_chat_template(self, messages, *, tokenize, **kwargs):
        del messages
        self.template_kwargs.append(kwargs)
        return {"input_ids": [[1, 2, 3]]} if tokenize else "tokenizer-rendered-prompt"


class CharacterChatTokenizer(CharacterTokenizer):
    pad_token_id = 0
    eos_token_id = 3

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt=False, **kwargs):
        del kwargs
        text = "".join(
            f"<{message['role']}>{message['content']}</{message['role']}>"
            for message in messages
        )
        if add_generation_prompt:
            text += "<assistant>"
        if tokenize:
            return self.encode(text)
        return text


class FakePolicy:
    trainable = False

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate_messages(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        return PolicyGeneration(
            text=self.responses.pop(0),
            prompt_text="prompt",
            backend="fake",
            generation_seconds=0.0,
            prompt_token_ids=[],
            completion_token_ids=[],
        )


def build_graph() -> FunctionGraph:
    return FunctionGraph(
        problem_id="p1",
        nodes=[
            FunctionNode(
                id="f1",
                name="solver",
                question="Implement addition.",
                signature="solve(a: int, b: int) -> int",
                expected_output_type="python_function",
                verification_spec={
                    "tests": [
                        {"kind": "unit", "call": "solve(2, 3)", "expected": 5},
                        {"kind": "property", "expression": "solve(0, 7) == 7", "expected": True},
                    ]
                },
            ),
            FunctionNode(
                id="main",
                name="answer",
                question="Answer the problem.",
                signature="main() -> int",
                expected_output_type="final_answer",
                verification_spec={"target_call": "solve(2, 3)"},
            ),
        ],
        edges=[
            DependencyEdge(
                source="f1",
                target="main",
                relation_type="aggregates_result",
                check_method="solver_output_matches_final_answer",
                severity="critical",
            )
        ],
    )


class DatasetTests(unittest.TestCase):
    def test_load_jsonl_and_filter_split(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "problems.jsonl"
            path.write_text(
                json.dumps({"id": "a", "question": "1+1?", "answer": 2, "split": "train"})
                + "\n"
                + json.dumps({"id": "b", "question": "2+2?", "answer": 4, "split": "test"})
                + "\n",
                encoding="utf-8",
            )
            problems = load_dataset(
                {
                    "dataset": {
                        "algorithm_problem_dataset_path": str(path),
                        "split": "train",
                    }
                }
            )
        self.assertEqual([problem.id for problem in problems], ["a"])
        self.assertEqual(problems[0].gold_answer, "2")

    def test_missing_gold_answer_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "problems.json"
            path.write_text(json.dumps([{"id": "a", "text": "question"}]), encoding="utf-8")
            with self.assertRaises(DatasetError):
                load_dataset({"dataset": {"algorithm_problem_dataset_path": str(path)}})

    def test_dataset_offset_is_applied_after_deterministic_shuffle(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "problems.jsonl"
            rows = [
                {"id": str(index), "text": f"q{index}", "gold_answer": str(index)}
                for index in range(8)
            ]
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            full = load_dataset(
                {
                    "dataset": {
                        "algorithm_problem_dataset_path": str(path),
                        "shuffle": True,
                        "seed": 42,
                    }
                }
            )
            resumed = load_dataset(
                {
                    "dataset": {
                        "algorithm_problem_dataset_path": str(path),
                        "shuffle": True,
                        "seed": 42,
                        "offset": 3,
                        "limit": 2,
                    }
                }
            )
        self.assertEqual(
            [problem.id for problem in resumed],
            [problem.id for problem in full[3:5]],
        )


class GraphAndParsingTests(unittest.TestCase):
    def test_policy_decomposer_reuses_policy_and_retries_invalid_json(self):
        policy = FakePolicy(["not json", json.dumps(build_graph().to_dict())])
        decomposer = build_decomposer(
            {
                "decomposition": {
                    "backend": "policy",
                    "max_nodes": 4,
                    "json_retries": 1,
                }
            },
            policy,
        )
        graph = decomposer.construct(
            Problem(id="p1", text="What is 2+3?"),
            MemoryContext(),
        )
        self.assertEqual([node.id for node in graph.nodes], ["f1", "main"])
        self.assertEqual(len(policy.calls), 2)
        self.assertEqual(policy.calls[0]["seed_offset"], 100_000)
        self.assertEqual(policy.calls[1]["seed_offset"], 100_001)

    def test_graph_cycle_is_rejected(self):
        graph = build_graph()
        graph.edges.append(
            DependencyEdge(
                source="main",
                target="f1",
                relation_type="uses_value",
                check_method="test",
            )
        )
        with self.assertRaises(DecompositionError):
            validate_function_graph(graph)

    def test_span_offsets_and_token_ranges(self):
        graph = build_graph()
        raw = "<a_f1>work</a_f1><a_main>answer \\boxed{5}</a_main>"
        spans = parse_tagged_function_spans(raw, graph)
        self.assertEqual(spans[0].raw_text, "work")
        self.assertGreaterEqual(spans[0].start_char, 0)
        token_ids = [ord(character) for character in raw]
        ranges = find_tagged_token_ranges(token_ids, ["f1", "main"], CharacterTokenizer())
        self.assertEqual("".join(chr(value) for value in token_ids[slice(*ranges["f1"])]), "work")

    def test_nested_boxed_answer(self):
        graph = build_graph()
        spans = parse_tagged_function_spans(
            "<a_f1>work</a_f1><a_main>\\boxed{\\frac{1}{2}}</a_main>",
            graph,
        )
        self.assertEqual(spans[1].extracted_answer, "\\frac{1}{2}")


class VerificationAndRewardTests(unittest.TestCase):
    def test_layered_verification_and_reward(self):
        graph = build_graph()
        problem = Problem(id="p1", text="What is 2+3?", gold_answer="5")
        raw = """<a_f1>
```python
def solve(a, b):
    return a + b
```
</a_f1>
<a_main>The final answer is \\boxed{5}.</a_main>"""
        spans = parse_tagged_function_spans(raw, graph)
        executor = ToolExecutor(
            {
                "sandbox": {
                    "backend": "subprocess",
                    "allow_unsafe_subprocess": True,
                    "timeout_seconds": 2,
                }
            }
        )
        execution = executor.execute(spans, graph)
        verification = Verifier({}).verify(problem, graph, spans, execution)
        self.assertEqual(verification.final_answer_score, 1.0)
        self.assertEqual(verification.backward_score, 1.0)
        self.assertEqual(verification.node_property_scores["f1"], 1.0)
        self.assertEqual(verification.edge_scores["f1->main"], 1.0)

        rollout = PolicyRollout(problem_id="p1", raw_text=raw, completion_token_ids=[1] * 20)
        reward = RewardAssigner(
            {
                "reward": {
                    "lambda_final": 1.0,
                    "lambda_node": 0.4,
                    "lambda_edge": 0.3,
                    "lambda_backward": 0.2,
                    "lambda_consensus": 0.05,
                    "lambda_efficiency": 0.0,
                    "lambda_repair": 0.0,
                }
            }
        ).assign(problem, graph, verification, execution, spans, rollout=rollout)
        self.assertEqual(reward.format_gate, 1.0)
        self.assertGreater(reward.total_reward, 1.0)
        self.assertGreater(reward.span_rewards["main"], reward.span_rewards["f1"])


class UtilityTests(unittest.TestCase):
    def test_gpt5_api_uses_gateway_compatible_parameters(self):
        request = build_chat_completion_request(
            model="gpt-5.6-sol",
            messages=[{"role": "user", "content": "hello"}],
            temperature=0.0,
            max_tokens=512,
        )
        self.assertEqual(request["max_completion_tokens"], 512)
        self.assertNotIn("max_tokens", request)
        self.assertNotIn("temperature", request)

    def test_greedy_generation_omits_sampling_only_kwargs(self):
        self.assertEqual(
            _sampling_generation_kwargs(
                do_sample=False,
                temperature=0.0,
                top_p=0.95,
            ),
            {},
        )

    def test_chat_template_falls_back_from_processor_to_tokenizer(self):
        tokenizer = TokenizerWithChatTemplate()
        prompt, inputs = _render_chat_messages(
            ProcessorWithoutChatTemplate(),
            tokenizer,
            [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
            enable_thinking=False,
        )
        self.assertEqual(prompt, "tokenizer-rendered-prompt")
        self.assertEqual(inputs["input_ids"], [[1, 2, 3]])
        self.assertEqual(
            [kwargs["enable_thinking"] for kwargs in tokenizer.template_kwargs],
            [False, False],
        )

    def test_json_fence_parsing(self):
        self.assertEqual(extract_json_object('```json\n{"ok": true}\n```'), {"ok": True})

    def test_group_normalize(self):
        values = GRPOTrainer.group_normalize([1.0, 2.0, 3.0])
        self.assertAlmostEqual(sum(values), 0.0)
        self.assertEqual(GRPOTrainer.group_normalize([2.0, 2.0]), [0.0, 0.0])

    def test_training_rejects_vllm_and_single_rollout(self):
        config = {
            "dataset": {"algorithm_problem_dataset_path": "data.jsonl"},
            "policy": {"backend": "vllm", "model_name_or_path": "model"},
            "rollout": {"num_rollouts": 1},
            "decomposition": {"backend": "dataset"},
            "repair": {"enabled": False},
            "sandbox": {"backend": "docker"},
        }
        with self.assertRaises(ConfigurationError):
            validate_config(config, "train")

    def test_rollout_enable_thinking_requires_boolean(self):
        config = {
            "dataset": {"algorithm_problem_dataset_path": "data.jsonl"},
            "policy": {
                "backend": "transformers",
                "model_name_or_path": "model",
            },
            "rollout": {"num_rollouts": 2, "enable_thinking": "false"},
            "decomposition": {"backend": "dataset"},
            "repair": {"enabled": False},
            "sandbox": {"backend": "local_limited"},
        }
        with self.assertRaisesRegex(
            ConfigurationError, "rollout.enable_thinking must be a boolean"
        ):
            validate_config(config, "train")

    def test_policy_decomposition_requires_no_api_config(self):
        config = {
            "dataset": {"algorithm_problem_dataset_path": "data.jsonl"},
            "policy": {"backend": "transformers", "model_name_or_path": "model"},
            "rollout": {"num_rollouts": 1},
            "decomposition": {"backend": "policy"},
            "repair": {"enabled": False},
            "sandbox": {"backend": "docker"},
        }
        validate_config(config, "inference")

    def test_metric_summary(self):
        summary = summarize_metrics(
            [
                {
                    "wall_time_seconds": 2.0,
                    "rollout_records": [
                        {
                            "rollout": {"completion_token_count": 10, "parsed_spans": []},
                            "execution": {
                                "runtime_seconds": 0.1,
                                "executable": True,
                                "timeout": False,
                                "node_results": {},
                            },
                            "verification": {"node_scores": {"main": 1.0}, "edge_scores": {}},
                            "reward": {"final_reward": 1.0},
                            "repair": {"status": "not_needed"},
                        }
                    ],
                }
            ]
        )
        self.assertEqual(summary["rollout_accuracy"], 1.0)
        self.assertEqual(summary["pass_at_k"], 1.0)
        self.assertEqual(summary["cost_normalized_accuracy"], 0.5)


class DataPreparationTests(unittest.TestCase):
    def test_mathlib_trace_normalization_and_enrichment(self):
        trace = {
            "full_name": "Nat.gcd_comm",
            "file_path": "Mathlib/Data/Nat/GCD/Basic.lean",
            "theorem_statement": "∀ (m n : Nat), Nat.gcd m n = Nat.gcd n m",
            "commit": "abc123",
            "traced_tactics": [
                {
                    "annotated_tactic": [
                        "exact Nat.gcd_comm n m",
                        [
                            {"full_name": "Nat.gcd_comm"},
                            {"full_name": "Nat.gcd_eq_right_iff_dvd"},
                        ],
                    ]
                }
            ],
        }
        formal = formalize_traced_theorem(trace)
        self.assertEqual(
            list(formal),
            ["lean_name", "namespace", "source_file", "formal_statement", "docstring"],
        )
        normalized = normalize_traced_theorem(trace)
        self.assertEqual(normalized["memory_id"], "mathlib4:Nat.gcd_comm")
        self.assertEqual(normalized["namespace"], "Nat")
        self.assertEqual(normalized["source_revision"], "abc123")
        self.assertEqual(normalized["premise_ids"], ["Nat.gcd_eq_right_iff_dvd"])

        enriched = merge_enrichment(
            normalized,
            {
                "title": "Commutativity of gcd",
                "informal_statement": "The gcd of two natural numbers is symmetric.",
                "domain_path": ["number_theory", "gcd"],
                "keywords": ["gcd", "commutative"],
                "preconditions": [],
            },
        )
        self.assertEqual(enriched["domain_path"], ["number_theory", "gcd"])
        self.assertIn("Commutativity of gcd", enriched["text"])

    def test_function_graph_schema_errors_are_actionable(self):
        graph = build_graph().to_dict()
        graph["nodes"][0]["verification_spec"] = "check algebra"
        with self.assertRaisesRegex(
            ValueError, "verification_spec must be a JSON object, got str"
        ):
            FunctionGraph.from_dict(graph)

        graph = build_graph().to_dict()
        graph["nodes"] = ["main"]
        with self.assertRaisesRegex(
            ValueError, "FunctionNode item at index 0 must be a JSON object"
        ):
            FunctionGraph.from_dict(graph)

        graph = build_graph().to_dict()
        graph["edges"][0]["check_method"] = {
            "method": "result_flow",
            "requirement": "Use the intermediate result",
        }
        parsed = FunctionGraph.from_dict(graph)
        self.assertEqual(
            parsed.edges[0].check_method,
            '{"method":"result_flow","requirement":"Use the intermediate result"}',
        )

    def test_omni_sampling_and_two_task_records(self):
        records = [
            canonicalize_omni_record(
                {
                    "problem": f"Compute {index}+1.",
                    "solution": f"It is {index + 1}.",
                    "answer": str(index + 1),
                    "domain": ["Algebra -> Arithmetic"],
                    "difficulty": index % 10 + 1,
                    "source": "test",
                }
            )
            for index in range(10)
        ]
        selected, holdout = deterministic_stratified_sample(records, 0.2, 42)
        selected_again, _ = deterministic_stratified_sample(records, 0.2, 42)
        self.assertEqual(len(selected), 2)
        self.assertEqual(len(holdout), 8)
        self.assertEqual(
            [record["id"] for record in selected],
            [record["id"] for record in selected_again],
        )

        source = dict(records[0])
        graph = build_graph().to_dict()
        graph["problem_id"] = source["id"]
        master = validate_teacher_annotation(
            source,
            {
                "function_graph": graph,
                "tagged_solution": (
                    "<a_f1>Compute the needed value.</a_f1>"
                    f"<a_main>The answer is \\boxed{{{source['gold_answer']}}}</a_main>"
                ),
                "retrieved_memory": [],
            },
            max_nodes=8,
        )
        graph_record, solve_record = build_sft_records(master)
        self.assertEqual(graph_record["task"], "construct_function_graph")
        self.assertEqual(solve_record["task"], "solve_with_function_graph")
        self.assertNotIn("gold_answer", graph_record["messages"][1]["content"])
        self.assertIn("<a_main>", solve_record["messages"][-1]["content"])

        aliased = validate_teacher_annotation(
            source,
            {
                "graph": graph,
                "solution": (
                    "<a_f1>Compute the needed value.</a_f1>"
                    f"<a_main>The answer is \\boxed{{{source['gold_answer']}}}</a_main>"
                ),
                "retrieved_memory": [],
            },
            max_nodes=8,
        )
        self.assertEqual(aliased["function_graph"]["problem_id"], source["id"])
        self.assertIn("<a_main>", aliased["tagged_solution"])

    def test_sft_encoding_masks_prompt(self):
        tokenizer = CharacterChatTokenizer()
        encoded = encode_sft_messages(
            [
                {"role": "system", "content": "rules"},
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer"},
            ],
            tokenizer,
            max_seq_length=256,
        )
        first_label = next(index for index, value in enumerate(encoded["labels"]) if value != -100)
        self.assertGreater(first_label, 0)
        self.assertTrue(all(value == -100 for value in encoded["labels"][:first_label]))
        learned_text = "".join(
            chr(value) for value in encoded["labels"] if value != -100
        )
        self.assertEqual(learned_text, "answer</assistant>")


if __name__ == "__main__":
    unittest.main()

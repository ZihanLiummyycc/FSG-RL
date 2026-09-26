"""Function-graph construction with the shared policy model or dataset graphs."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional, Protocol, Set

from .api_client import extract_json_object
from .rollout import PolicyBackend
from .schemas import FunctionGraph, MemoryContext, Problem


class DecompositionError(ValueError):
    """Raised when a decomposer response is not a valid function DAG."""


class Decomposer(Protocol):
    def construct(self, problem: Problem, memory_context: MemoryContext) -> FunctionGraph: ...


class PolicyDecomposer:
    """Uses the already-loaded Qwen3.5 policy to construct verifiable DAGs."""

    def __init__(self, config: Dict[str, Any], policy: PolicyBackend):
        section = config.get("decomposition", {})
        self.policy = policy
        self.max_nodes = int(section.get("max_nodes", 8))
        self.temperature = float(section.get("temperature", 0.1))
        self.top_p = float(section.get("top_p", 0.9))
        self.max_new_tokens = int(section.get("max_new_tokens", 4096))
        self.json_retries = int(section.get("json_retries", 2))
        self.seed_offset = int(section.get("seed_offset", 100_000))
        if self.max_nodes < 1:
            raise ValueError("decomposition.max_nodes must be positive")
        if self.max_new_tokens < 1:
            raise ValueError("decomposition.max_new_tokens must be positive")
        if self.json_retries < 0:
            raise ValueError("decomposition.json_retries cannot be negative")

    def construct(self, problem: Problem, memory_context: MemoryContext) -> FunctionGraph:
        base_messages: list[Dict[str, Any]] = [
            {
                "role": "system",
                "content": [{"type": "text", "text": _DECOMPOSER_SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "problem_id": problem.id,
                                "problem": problem.text,
                                "memory": memory_context.to_dict(),
                                "max_nodes": self.max_nodes,
                            },
                            ensure_ascii=False,
                        ),
                    }
                ],
            },
        ]
        previous_text = ""
        last_error: Optional[Exception] = None
        for attempt in range(self.json_retries + 1):
            messages = list(base_messages)
            if previous_text and last_error is not None:
                messages.extend(
                    [
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": previous_text}],
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        "The previous graph was invalid: "
                                        f"{last_error}. Return a corrected JSON object only."
                                    ),
                                }
                            ],
                        },
                    ]
                )
            try:
                generation = self.policy.generate_messages(
                    messages,
                    temperature=self.temperature,
                    max_new_tokens=self.max_new_tokens,
                    seed_offset=self.seed_offset + attempt,
                    top_p=self.top_p,
                )
                previous_text = generation.text
                payload = extract_json_object(previous_text)
                graph_data = payload.get("function_graph", payload)
                if not isinstance(graph_data, dict):
                    raise DecompositionError(
                        "Decomposer response must contain a function_graph object"
                    )
                graph_data = dict(graph_data)
                graph_data["problem_id"] = problem.id
                graph = FunctionGraph.from_dict(graph_data)
                validate_function_graph(graph, max_nodes=self.max_nodes)
                return graph
            except (KeyError, TypeError, ValueError) as exc:
                last_error = exc

        assert last_error is not None
        raise DecompositionError(
            f"Qwen3.5 policy failed to produce a valid function graph after "
            f"{self.json_retries + 1} attempts: {last_error}"
        ) from last_error


class DatasetDecomposer:
    """Loads a precomputed graph from problem metadata for controlled ablations."""

    def construct(self, problem: Problem, memory_context: MemoryContext) -> FunctionGraph:
        del memory_context
        graph_data = problem.metadata.get("function_graph")
        if not isinstance(graph_data, dict):
            raise DecompositionError(
                f"Problem {problem.id!r} has no metadata.function_graph for dataset backend"
            )
        graph_data = dict(graph_data)
        graph_data["problem_id"] = problem.id
        graph = FunctionGraph.from_dict(graph_data)
        validate_function_graph(graph)
        return graph


def build_decomposer(
    config: Dict[str, Any], policy: Optional[PolicyBackend] = None
) -> Decomposer:
    backend = config.get("decomposition", {}).get("backend", "policy")
    if backend == "policy":
        if policy is None:
            raise ValueError("decomposition.backend='policy' requires a loaded policy")
        return PolicyDecomposer(config, policy)
    if backend == "dataset":
        return DatasetDecomposer()
    raise ValueError(f"Unsupported decomposition backend: {backend!r}")


def validate_function_graph(graph: FunctionGraph, max_nodes: int = 64) -> None:
    if not graph.nodes:
        raise DecompositionError("Function graph must contain at least one node")
    if len(graph.nodes) > max_nodes:
        raise DecompositionError(f"Function graph exceeds max_nodes={max_nodes}")

    node_ids = [node.id for node in graph.nodes]
    if len(node_ids) != len(set(node_ids)):
        raise DecompositionError("Function graph contains duplicate node IDs")
    if "main" not in node_ids:
        raise DecompositionError("Function graph must contain a final node with id='main'")
    for node in graph.nodes:
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", node.id):
            raise DecompositionError(f"Invalid node id: {node.id!r}")
        if not node.question or not node.signature or not node.expected_output_type:
            raise DecompositionError(f"Node {node.id!r} is missing required schema fields")

    adjacency: Dict[str, Set[str]] = {node_id: set() for node_id in node_ids}
    reverse_adjacency: Dict[str, Set[str]] = {node_id: set() for node_id in node_ids}
    for edge in graph.edges:
        if edge.source not in adjacency or edge.target not in adjacency:
            raise DecompositionError(
                f"Edge {edge.source!r}->{edge.target!r} references an unknown node"
            )
        if edge.source == edge.target:
            raise DecompositionError(f"Self-edge is not allowed for node {edge.source!r}")
        if edge.source == "main":
            raise DecompositionError("The final main node must be a sink")
        adjacency[edge.source].add(edge.target)
        reverse_adjacency[edge.target].add(edge.source)

    visiting: Set[str] = set()
    visited: Set[str] = set()

    def visit(node_id: str) -> None:
        if node_id in visiting:
            raise DecompositionError("Function graph must be acyclic")
        if node_id in visited:
            return
        visiting.add(node_id)
        for child in adjacency[node_id]:
            visit(child)
        visiting.remove(node_id)
        visited.add(node_id)

    for node_id in node_ids:
        visit(node_id)

    ancestors: Set[str] = set()

    def collect_ancestors(node_id: str) -> None:
        if node_id in ancestors:
            return
        ancestors.add(node_id)
        for parent in reverse_adjacency[node_id]:
            collect_ancestors(parent)

    collect_ancestors("main")
    disconnected = set(node_ids) - ancestors
    if disconnected:
        raise DecompositionError(
            f"Every node must contribute to main; disconnected nodes: {sorted(disconnected)}"
        )


_DECOMPOSER_SYSTEM_PROMPT = """You construct a verifiable function graph for an algorithmic
mathematics problem. Return one JSON object only, with keys problem_id, nodes, and edges.

Each node must contain:
- id: short identifier such as f1, f2, or main
- name
- question: a self-contained mathematical subquestion
- signature: a Python-like function signature
- expected_output_type: one of formula, values, python_function, proof_claim, final_answer
- verification_spec: machine-checkable data. For executable nodes include tests, each with
  kind (unit or property), call, and expected. Property tests should compare small instances,
  boundary cases, invariants, or brute-force equivalents.

Each edge must contain source, target, relation_type, check_method, and severity. Supported
relation types are uses_value, implements_formula, aggregates_result, and equivalent_output.
Use severity=critical when failure invalidates the solution.

Requirements:
1. The graph is a DAG and has at most max_nodes nodes.
2. The final node id is exactly main and corresponds to the original question.
3. Dependencies point from prerequisite to consumer.
4. Prefer executable, unit-testable nodes; never include API calls or filesystem access in tests.
5. Do not use or guess a hidden gold answer. Verification specs must follow from the problem,
   small-case enumeration, or explicit constraints.
6. Do not include Markdown fences or prose outside the JSON object.
"""

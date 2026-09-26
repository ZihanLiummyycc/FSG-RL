"""Resolve public rollout graphs and private verifier graphs without leakage."""

from __future__ import annotations

from .decomposition import validate_function_graph
from .schemas import FunctionGraph, Problem


def resolve_verification_graph(
    problem: Problem,
    public_graph: FunctionGraph,
) -> FunctionGraph:
    """Return the hidden graph when present and enforce an identical public API."""

    raw = problem.metadata.get("verification_graph")
    if raw is None:
        return public_graph
    if not isinstance(raw, dict):
        raise ValueError(
            f"Problem {problem.id!r} metadata.verification_graph must be an object"
        )

    data = dict(raw)
    data["problem_id"] = problem.id
    hidden_graph = FunctionGraph.from_dict(data)
    validate_function_graph(hidden_graph)

    public_nodes = [
        (node.id, node.signature, node.expected_output_type)
        for node in public_graph.nodes
    ]
    hidden_nodes = [
        (node.id, node.signature, node.expected_output_type)
        for node in hidden_graph.nodes
    ]
    if public_nodes != hidden_nodes:
        raise ValueError(
            f"Problem {problem.id!r} public/hidden node APIs differ"
        )

    public_edges = [
        (edge.source, edge.target, edge.relation_type)
        for edge in public_graph.edges
    ]
    hidden_edges = [
        (edge.source, edge.target, edge.relation_type)
        for edge in hidden_graph.edges
    ]
    if public_edges != hidden_edges:
        raise ValueError(
            f"Problem {problem.id!r} public/hidden graph edges differ"
        )
    return hidden_graph

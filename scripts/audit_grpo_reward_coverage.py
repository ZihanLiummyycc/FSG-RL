#!/usr/bin/env python3
"""Report which proposal reward signals are actually available in GRPO data."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Dict, List


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()

    rows = _read_jsonl(Path(args.input))
    node_types: Counter[str] = Counter()
    edge_types: Counter[str] = Counter()
    test_kinds: Counter[str] = Counter()
    graphs_with_executable_nodes = 0
    graphs_with_unit_tests = 0
    graphs_with_property_tests = 0
    graphs_with_backward_call = 0
    machine_verifiable_nodes = 0
    unverified_nodes = 0
    total_nodes = 0
    total_edges = 0

    for row in rows:
        metadata = row.get("metadata", {})
        graph = (
            metadata.get("verification_graph", metadata.get("function_graph", {}))
            if isinstance(metadata, dict)
            else {}
        )
        nodes = graph.get("nodes", []) if isinstance(graph, dict) else []
        edges = graph.get("edges", []) if isinstance(graph, dict) else []
        total_nodes += len(nodes)
        total_edges += len(edges)
        graph_has_exec = False
        graph_has_unit = False
        graph_has_property = False
        graph_has_backward = False

        for node in nodes:
            output_type = str(node.get("expected_output_type", "unknown"))
            node_types[output_type] += 1
            spec = node.get("verification_spec", {})
            spec = spec if isinstance(spec, dict) else {}
            tests = spec.get("tests", [])
            tests = tests if isinstance(tests, list) else []
            kinds = [str(test.get("kind", "unit")) for test in tests if isinstance(test, dict)]
            test_kinds.update(kinds)
            has_exec = bool(tests) or output_type == "python_function"
            has_unit = "unit" in kinds
            has_property = "property" in kinds
            has_static = output_type in {"formula", "values"} and bool(
                spec.get(output_type)
            )
            has_backward = bool(spec.get("target_call"))
            graph_has_exec = graph_has_exec or has_exec
            graph_has_unit = graph_has_unit or has_unit
            graph_has_property = graph_has_property or has_property
            graph_has_backward = graph_has_backward or has_backward
            if has_exec or has_unit or has_property or has_static:
                machine_verifiable_nodes += 1
            else:
                unverified_nodes += 1

        edge_types.update(str(edge.get("relation_type", "unknown")) for edge in edges)
        graphs_with_executable_nodes += int(graph_has_exec)
        graphs_with_unit_tests += int(graph_has_unit)
        graphs_with_property_tests += int(graph_has_property)
        graphs_with_backward_call += int(graph_has_backward)

    problem_count = len(rows)
    report: Dict[str, Any] = {
        "problems": problem_count,
        "total_nodes": total_nodes,
        "total_edges": total_edges,
        "node_types": dict(sorted(node_types.items())),
        "edge_types": dict(sorted(edge_types.items())),
        "test_kinds": dict(sorted(test_kinds.items())),
        "machine_verifiable_nodes": machine_verifiable_nodes,
        "unverified_nodes": unverified_nodes,
        "machine_verifiable_node_rate": _ratio(machine_verifiable_nodes, total_nodes),
        "graphs_with_executable_nodes": graphs_with_executable_nodes,
        "graphs_with_executable_node_rate": _ratio(graphs_with_executable_nodes, problem_count),
        "graphs_with_unit_tests": graphs_with_unit_tests,
        "graphs_with_unit_test_rate": _ratio(graphs_with_unit_tests, problem_count),
        "graphs_with_property_tests": graphs_with_property_tests,
        "graphs_with_property_test_rate": _ratio(graphs_with_property_tests, problem_count),
        "graphs_with_backward_call": graphs_with_backward_call,
        "graphs_with_backward_call_rate": _ratio(graphs_with_backward_call, problem_count),
        "interpretation": {
            "final_answer_reward": "available for every row",
            "format_reward": "available for every generated rollout",
            "node_execution_test_property_reward": "available only on machine-verifiable nodes",
            "edge_reward": "available when the relation-specific checker has evidence",
            "backward_reward": "available only when target_call is present",
            "consensus_reward": "available only after at least two rollouts",
        },
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(value)
    return rows


if __name__ == "__main__":
    main()

"""Omni-MATH sampling, teacher-label validation, and two-task SFT records."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from .decomposition import validate_function_graph
from .parsing import parse_tagged_function_spans
from .schemas import FunctionGraph


GRAPH_TASK = "construct_function_graph"
SOLVE_TASK = "solve_with_function_graph"


def canonicalize_omni_record(row: Dict[str, Any]) -> Dict[str, Any]:
    problem = str(row.get("problem", "")).strip()
    solution = str(row.get("solution", "")).strip()
    answer = str(row.get("answer", "")).strip()
    if not problem or not solution or not answer:
        raise ValueError("Omni-MATH row needs non-empty problem, solution, and answer")
    problem_hash = hashlib.sha256(problem.encode("utf-8")).hexdigest()
    domains = row.get("domain", [])
    if isinstance(domains, str):
        domains = [domains]
    if not isinstance(domains, list):
        domains = []
    return {
        "id": f"omni_math_{problem_hash[:16]}",
        "problem": problem,
        "reference_solution": solution,
        "gold_answer": answer,
        "domain": [str(value) for value in domains],
        "difficulty": row.get("difficulty"),
        "source": str(row.get("source", "Omni-MATH")),
        "problem_sha256": problem_hash,
    }


def deterministic_stratified_sample(
    records: Sequence[Dict[str, Any]], fraction: float, seed: int
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Select exactly round(N*fraction), preserving broad domain/difficulty mix."""

    if not 0.0 < fraction < 1.0:
        raise ValueError("sample fraction must be between 0 and 1")
    target = int(math.floor(len(records) * fraction + 0.5))
    strata: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        strata[_stratum(record)].append(record)

    allocation: Dict[Tuple[str, str], int] = {}
    remainders = []
    allocated = 0
    for key, values in strata.items():
        exact = len(values) * fraction
        count = int(math.floor(exact))
        allocation[key] = count
        allocated += count
        remainders.append((exact - count, _stable_hash(str(key), seed), key))
    for _, _, key in sorted(remainders, reverse=True)[: target - allocated]:
        allocation[key] += 1

    selected_ids = set()
    for key, values in strata.items():
        ranked = sorted(values, key=lambda row: _stable_hash(row["id"], seed))
        selected_ids.update(row["id"] for row in ranked[: allocation[key]])

    selected = sorted(
        (row for row in records if row["id"] in selected_ids),
        key=lambda row: row["id"],
    )
    holdout = sorted(
        (row for row in records if row["id"] not in selected_ids),
        key=lambda row: row["id"],
    )
    if len(selected) != target:
        raise AssertionError(f"Expected {target} selected rows, got {len(selected)}")
    return selected, holdout


def build_teacher_messages(
    record: Dict[str, Any], retrieved_memory: Sequence[Dict[str, Any]], max_nodes: int
) -> List[Dict[str, str]]:
    payload = {
        "problem_id": record["id"],
        "problem": record["problem"],
        "reference_solution": record["reference_solution"],
        "gold_answer": record["gold_answer"],
        "domain": record.get("domain", []),
        "difficulty": record.get("difficulty"),
        "retrieved_memory": list(retrieved_memory),
        "max_nodes": max_nodes,
    }
    return [
        {"role": "system", "content": _OMNI_TEACHER_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def validate_teacher_annotation(
    record: Dict[str, Any], payload: Dict[str, Any], max_nodes: int
) -> Dict[str, Any]:
    graph_data = payload.get("function_graph")
    if not isinstance(graph_data, dict):
        graph_data = payload.get("graph")
    if not isinstance(graph_data, dict) and isinstance(payload.get("nodes"), list):
        graph_data = payload
    if not isinstance(graph_data, dict):
        field_types = {
            str(key): type(value).__name__
            for key, value in payload.items()
        }
        raise ValueError(
            "Teacher output needs a function_graph object; received "
            f"top-level fields={field_types}"
        )
    graph_data = dict(graph_data)
    graph_data["problem_id"] = record["id"]
    graph = FunctionGraph.from_dict(graph_data)
    validate_function_graph(graph, max_nodes=max_nodes)

    tagged_solution = str(
        payload.get("tagged_solution", payload.get("solution", ""))
    ).strip()
    if not tagged_solution:
        raise ValueError("Teacher output needs tagged_solution")
    _validate_exact_tags(tagged_solution, [node.id for node in graph.nodes])
    spans = parse_tagged_function_spans(tagged_solution, graph)
    main = next((span for span in spans if span.node_id == "main"), None)
    if main is None or main.extracted_answer is None:
        raise ValueError("tagged_solution main span needs a boxed final answer")
    if _normalize_answer(main.extracted_answer) != _normalize_answer(record["gold_answer"]):
        raise ValueError(
            "tagged_solution boxed answer must copy the supplied gold_answer exactly"
        )
    if any(not span.raw_text for span in spans):
        raise ValueError("Every function node needs a non-empty tagged solution span")

    return {
        **record,
        "function_graph": graph.to_dict(),
        "tagged_solution": tagged_solution,
        "retrieved_memory": list(payload.get("retrieved_memory", [])),
        "annotation_model": payload.get("annotation_model"),
    }


def build_sft_records(master: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    memory = master.get("retrieved_memory", [])
    graph_user = {
        "problem_id": master["id"],
        "problem": master["problem"],
        "memory": memory,
        "max_nodes": len(master["function_graph"]["nodes"]),
    }
    graph_record = {
        "id": master["id"] + ":graph",
        "task": GRAPH_TASK,
        "messages": [
            {"role": "system", "content": _GRAPH_SFT_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(graph_user, ensure_ascii=False)},
            {
                "role": "assistant",
                "content": json.dumps(master["function_graph"], ensure_ascii=False),
            },
        ],
        "metadata": _sft_metadata(master),
    }

    solve_user = {
        "problem": master["problem"],
        "function_graph": master["function_graph"],
        "retrieved_memory": memory,
    }
    solve_record = {
        "id": master["id"] + ":solve",
        "task": SOLVE_TASK,
        "messages": [
            {"role": "system", "content": _SOLVE_SFT_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(solve_user, ensure_ascii=False)},
            {"role": "assistant", "content": master["tagged_solution"]},
        ],
        "metadata": _sft_metadata(master),
    }
    return graph_record, solve_record


class LexicalMemoryIndex:
    """Small dependency-free inverted index for offline theorem retrieval."""

    def __init__(self, records: Iterable[Dict[str, Any]]):
        self.records = []
        self.document_tokens = []
        self.postings: Dict[str, set[int]] = defaultdict(set)
        for record in records:
            text = " ".join(
                [
                    str(record.get("title", "")),
                    str(record.get("informal_statement", "")),
                    str(record.get("formal_statement", "")),
                    " ".join(str(value) for value in record.get("keywords", [])),
                ]
            )
            tokens = _tokens(text)
            index = len(self.records)
            self.records.append(record)
            self.document_tokens.append(tokens)
            for token in tokens:
                self.postings[token].add(index)

    def retrieve(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        query_tokens = _tokens(query)
        candidates: Dict[int, float] = defaultdict(float)
        total = max(1, len(self.records))
        for token in query_tokens:
            posting = self.postings.get(token, set())
            weight = math.log((total + 1) / (len(posting) + 1)) + 1.0
            for index in posting:
                candidates[index] += weight
        ranked = sorted(
            candidates,
            key=lambda index: (
                -candidates[index] / max(1.0, math.sqrt(len(self.document_tokens[index]))),
                str(self.records[index].get("id", "")),
            ),
        )
        return [self._compact(self.records[index]) for index in ranked[:top_k]]

    @staticmethod
    def _compact(record: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "memory_id": record.get("memory_id", record.get("id")),
            "title": record.get("title"),
            "informal_statement": record.get("informal_statement"),
            "formal_statement": record.get("formal_statement"),
            "preconditions": record.get("preconditions", []),
        }


def _validate_exact_tags(text: str, node_ids: Sequence[str]) -> None:
    expected = set(node_ids)
    open_tags = re.findall(r"<a_([A-Za-z][A-Za-z0-9_]*)>", text)
    close_tags = re.findall(r"</a_([A-Za-z][A-Za-z0-9_]*)>", text)
    if set(open_tags) != expected or set(close_tags) != expected:
        raise ValueError(
            "tagged_solution tags do not match function graph node IDs: "
            f"expected={sorted(expected)}, open={open_tags}, close={close_tags}"
        )
    for node_id in expected:
        if open_tags.count(node_id) != 1 or close_tags.count(node_id) != 1:
            raise ValueError(f"tagged_solution must contain node {node_id!r} exactly once")


def _normalize_answer(value: Any) -> str:
    text = str(value).strip()
    text = text.replace("\\left", "").replace("\\right", "")
    text = re.sub(r"\s+", "", text)
    return text.strip("$.").lower()


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[A-Za-z][A-Za-z0-9_]*|\d+", text.lower()))


def _stable_hash(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def _stratum(record: Dict[str, Any]) -> Tuple[str, str]:
    domain = "other"
    values = record.get("domain", [])
    if values:
        parts = [part.strip() for part in str(values[0]).split("->")]
        domain = parts[1] if len(parts) > 1 else parts[0]
    try:
        difficulty = float(record.get("difficulty"))
    except (TypeError, ValueError):
        bucket = "unknown"
    else:
        if difficulty <= 3:
            bucket = "1-3"
        elif difficulty <= 6:
            bucket = "3.5-6"
        elif difficulty <= 8:
            bucket = "6.5-8"
        else:
            bucket = "8.5-10"
    return domain, bucket


def _sft_metadata(master: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "source": "KbsdJames/Omni-MATH",
        "source_problem_id": master["id"],
        "domain": master.get("domain", []),
        "difficulty": master.get("difficulty"),
        "original_source": master.get("source"),
        "problem_sha256": master.get("problem_sha256"),
    }


_GRAPH_SFT_SYSTEM_PROMPT = """You construct a verifiable function graph for an algorithmic
mathematics problem. Return one JSON object only with problem_id, nodes, and edges. Every node
must have id, name, question, signature, expected_output_type, and verification_spec. The graph
must be a connected DAG whose final sink is exactly main. Dependencies point from prerequisite to
consumer. Use only relation types uses_value, implements_formula, aggregates_result, or
equivalent_output. Do not include Markdown or prose outside JSON."""

_SOLVE_SFT_SYSTEM_PROMPT = """Solve every function-graph node and respect all dependency edges.
For each node with id X, output exactly one <a_X>...</a_X> span and no extra node tags. The main
span must end with the final answer in \\boxed{...}. Use retrieved memory only when applicable.
Do not call external APIs, access files, or use the network."""

_OMNI_TEACHER_SYSTEM_PROMPT = """You create supervised training labels for function-structured
mathematical reasoning. Return one JSON object only with function_graph and tagged_solution.

The exact top-level shape is:
{"function_graph": {"nodes": [...], "edges": [...]},
 "tagged_solution": "<a_node_id>...</a_node_id><a_main>...</a_main>"}
Do not put nodes or edges at the top level and do not rename either top-level field.

The function_graph must be a connected DAG with at most max_nodes nodes and a final sink named
main. Each node needs id, name, a self-contained question, a Python-like signature,
expected_output_type, and verification_spec. Each edge needs source, target, relation_type,
check_method, and severity. Supported relation types are uses_value, implements_formula,
aggregates_result, and equivalent_output. Every node must contribute to main.

The nodes and edges fields must be JSON arrays whose elements are JSON objects. Every
verification_spec must be a JSON object such as {}, never a string or list. Prefer check_method
as a concise JSON string; a structured JSON object is also accepted.

Use the supplied reference solution to preserve mathematical correctness. The tagged_solution must
contain exactly one <a_X>...</a_X> span for every graph node X, no extra tags, and the main span
must contain \\boxed{gold_answer}, copying the supplied gold_answer verbatim. Do not put the hidden
gold answer in graph verification specs. Do not invent theorem references. Return no Markdown
fence or prose outside the JSON object."""

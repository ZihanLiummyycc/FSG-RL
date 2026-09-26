#!/usr/bin/env python3
"""Generate code-aware Function Graph labels for a prepared medium-math pool."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import json
import os
from pathlib import Path
import threading
from typing import Any, Dict, Iterable, List, Sequence

from fsg_rl.api_client import ChatAPIConfig, OpenAICompatibleChatClient
from fsg_rl.executable_eval import check_python_signatures
from fsg_rl.parsing import parse_tagged_function_spans
from fsg_rl.sandbox_safety import validate_python_code
from fsg_rl.schemas import FunctionGraph
from fsg_rl.sft_data import (
    LexicalMemoryIndex,
    build_sft_records,
    build_teacher_messages as build_base_teacher_messages,
    validate_teacher_annotation,
)


CODE_AWARE_TEACHER_RULES = """

Additional executable-label requirements for this medium-math curriculum:
- Create at least one non-main node with expected_output_type exactly python_function.
- Keep main as expected_output_type final_answer. Do not make main a python_function.
- At least one python_function node must directly feed main and expose a natural representative
  call whose return value is mathematically equivalent to gold_answer. Make the final difference,
  square root, aggregation, discount, unit conversion, or formatting step executable when it is
  needed to reach the final answer. Do not use contrived unrelated arguments that merely make a
  partial helper return the gold value.
- Write graph signature fields as name(arg: type) -> return_type only. Never include a def
  prefix, trailing colon, function body, or Markdown in a graph signature.
- For every python_function node, its tagged-solution span must contain exactly one fenced
  ```python block defining the exact function name and parameter structure in that node's
  signature. Non-python nodes must not contain Python blocks.
- Every Python block is executed alone in a fresh Python 3 interpreter. It must include every
  required import, constant, and helper and must not rely on another answer span.
- Allowed imports are collections, fractions, functools, heapq, itertools, math, operator, and
  statistics. Third-party modules including sympy, file access, subprocesses, APIs, and network
  access are forbidden.
- Prefer parameterized functions that implement the mathematical method instead of returning a
  memorized final constant. Check referenced names, argument signatures, dictionary keys, return
  type, and boundary cases before returning the label.
- At this stage every verification_spec must omit tests. Hidden unit/property tests are generated
  separately and must never be exposed in this public graph.
- The supplied gold_answer is a hard constraint. The main span must copy it verbatim inside
  \\boxed{...}. Independently verify the reference solution before writing code.
- For MathQA, the operation program and rationale are noisy optional hints. Never copy them
  blindly; derive the method from the problem and verify it against gold_answer.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--memory-jsonl")
    parser.add_argument("--memory-top-k", type=int, default=0)
    parser.add_argument("--max-nodes", type=int, default=6)
    parser.add_argument(
        "--api-model",
        default=os.environ.get("TEACHER_MODEL", "gpt-5.6-terra"),
    )
    parser.add_argument(
        "--api-base-url",
        default=os.environ.get("TEACHER_BASE_URL"),
    )
    parser.add_argument("--api-key-env", default="TEACHER_API_KEY")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-tokens", type=int, default=8192)
    args = parser.parse_args()
    if not args.api_base_url:
        parser.error("--api-base-url or TEACHER_BASE_URL is required")
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.attempts < 1:
        parser.error("--attempts must be positive")

    records = _read_jsonl(Path(args.input))
    _validate_input_records(records)
    record_ids = {record["id"] for record in records}
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    master_path = output_dir / "medium_teacher_master.jsonl"
    failed_path = output_dir / "medium_teacher_failed.jsonl"
    completed = {
        row["id"]: row
        for row in _read_jsonl(master_path, missing_ok=True)
        if row.get("id") in record_ids
    }
    previous_failures = _read_jsonl(failed_path, missing_ok=True)
    recovered_payloads = 0
    failure_by_id: Dict[str, Dict[str, Any]] = {}
    records_by_id = {record["id"]: record for record in records}
    for failure in previous_failures:
        record_id = str(failure.get("id", ""))
        if not record_id or record_id in completed or record_id not in records_by_id:
            continue
        previous_payload = failure.get("teacher_payload")
        if not isinstance(previous_payload, dict):
            failure_by_id[record_id] = failure
            continue
        try:
            previous_payload = dict(previous_payload)
            previous_payload.setdefault("retrieved_memory", [])
            previous_payload["annotation_model"] = args.api_model
            completed[record_id] = validate_medium_annotation(
                records_by_id[record_id],
                previous_payload,
                max_nodes=args.max_nodes,
            )
            recovered_payloads += 1
        except Exception:
            failure_by_id[record_id] = failure
    pending = [record for record in records if record["id"] not in completed]
    if args.limit is not None:
        pending = pending[: args.limit]

    memory_index = None
    if args.memory_jsonl and args.memory_top_k > 0:
        memory_index = LexicalMemoryIndex(_read_jsonl(Path(args.memory_jsonl)))

    api_config = ChatAPIConfig(
        model=str(args.api_model),
        base_url=str(args.api_base_url).rstrip("/"),
        api_key_env=args.api_key_env,
        timeout_seconds=300.0,
        max_retries=2,
    )
    local = threading.local()

    def client() -> OpenAICompatibleChatClient:
        if not hasattr(local, "client"):
            local.client = OpenAICompatibleChatClient(api_config)
        return local.client

    failed_payloads: Dict[str, Dict[str, Any]] = {}

    def annotate(record: Dict[str, Any]) -> Dict[str, Any]:
        memory = (
            memory_index.retrieve(record["problem"], args.memory_top_k)
            if memory_index is not None
            else []
        )
        messages = build_medium_teacher_messages(
            record,
            memory,
            max_nodes=args.max_nodes,
        )
        previous_payload: Dict[str, Any] | None = None
        last_error: Exception | None = None
        for _ in range(args.attempts):
            retry_messages = list(messages)
            if last_error is not None:
                if previous_payload is not None:
                    retry_messages.append(
                        {
                            "role": "assistant",
                            "content": json.dumps(
                                previous_payload,
                                ensure_ascii=False,
                            ),
                        }
                    )
                retry_messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"The previous label failed validation: {last_error}. "
                            "Return one corrected complete JSON object only."
                        ),
                    }
                )
            try:
                previous_payload = client().complete_json(
                    retry_messages,
                    temperature=0.0,
                    max_tokens=args.max_tokens,
                )
                previous_payload["retrieved_memory"] = memory
                previous_payload["annotation_model"] = args.api_model
                return validate_medium_annotation(
                    record,
                    previous_payload,
                    max_nodes=args.max_nodes,
                )
            except Exception as exc:
                last_error = exc
        assert last_error is not None
        if previous_payload is not None:
            failed_payloads[record["id"]] = previous_payload
        raise last_error

    processed = 0
    for offset in range(0, len(pending), args.workers):
        batch = pending[offset : offset + args.workers]
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(annotate, row): row for row in batch}
            for future in as_completed(futures):
                source = futures[future]
                try:
                    completed[source["id"]] = future.result()
                    failure_by_id.pop(source["id"], None)
                except Exception as exc:
                    failure: Dict[str, Any] = {
                        "id": source["id"],
                        "source": source.get("source"),
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:4000],
                    }
                    if source["id"] in failed_payloads:
                        failure["teacher_payload"] = failed_payloads[source["id"]]
                    failure_by_id[source["id"]] = failure
                processed += 1
        _write_jsonl(master_path, sorted(completed.values(), key=lambda row: row["id"]))
        _write_jsonl(
            failed_path,
            sorted(failure_by_id.values(), key=lambda row: row["id"]),
        )
        print(
            f"annotated={processed}/{len(pending)} success={len(completed)} "
            f"failed={len(failure_by_id)} recovered={recovered_payloads}",
            flush=True,
        )

    masters = sorted(completed.values(), key=lambda row: row["id"])
    _write_jsonl(master_path, masters)
    failures = sorted(failure_by_id.values(), key=lambda row: row["id"])
    _write_jsonl(failed_path, failures)
    graph_records = []
    solve_records = []
    for master in masters:
        graph, solve = build_sft_records(master)
        graph_records.append(graph)
        solve_records.append(solve)
    graph_path = output_dir / "medium_sft_graph_provisional.jsonl"
    solve_path = output_dir / "medium_sft_solve_code_provisional.jsonl"
    _write_jsonl(graph_path, graph_records)
    _write_jsonl(solve_path, solve_records)

    manifest = _build_manifest(
        records,
        masters,
        failures,
        model=str(args.api_model),
        graph_path=graph_path,
        solve_path=solve_path,
        recovered_payloads=recovered_payloads,
    )
    _write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


def build_medium_teacher_messages(
    record: Dict[str, Any],
    retrieved_memory: Sequence[Dict[str, Any]],
    *,
    max_nodes: int,
) -> List[Dict[str, str]]:
    messages = build_base_teacher_messages(record, retrieved_memory, max_nodes)
    messages[0] = {
        "role": "system",
        "content": messages[0]["content"] + CODE_AWARE_TEACHER_RULES,
    }
    payload = json.loads(messages[1]["content"])
    payload["source"] = record.get("source")
    payload["source_metadata"] = record.get("source_metadata", {})
    messages[1] = {
        "role": "user",
        "content": json.dumps(payload, ensure_ascii=False),
    }
    return messages


def validate_medium_annotation(
    record: Dict[str, Any],
    payload: Dict[str, Any],
    *,
    max_nodes: int,
) -> Dict[str, Any]:
    payload = normalize_teacher_signatures(payload)
    master = validate_teacher_annotation(record, payload, max_nodes=max_nodes)
    graph = FunctionGraph.from_dict(master["function_graph"])
    if graph.node_by_id("main").expected_output_type != "final_answer":
        raise ValueError("main must have expected_output_type final_answer")
    python_nodes = [
        node
        for node in graph.nodes
        if node.id != "main" and node.expected_output_type == "python_function"
    ]
    if not python_nodes:
        raise ValueError("At least one non-main python_function node is required")
    python_ids = {node.id for node in python_nodes}
    if not any(
        edge.source in python_ids and edge.target == "main"
        for edge in graph.edges
    ):
        raise ValueError(
            "At least one python_function node must directly feed main so a natural "
            "backward target_call can evaluate to the final answer"
        )
    for node in graph.nodes:
        if node.verification_spec.get("tests"):
            raise ValueError("Public function graph must not contain hidden tests")

    spans = parse_tagged_function_spans(master["tagged_solution"], graph)
    spans_by_id = {span.node_id: span for span in spans}
    signature_result = check_python_signatures(spans, graph)
    if not signature_result["passed"]:
        raise ValueError(
            "Python code blocks do not match graph signatures: "
            + json.dumps(signature_result["nodes"], ensure_ascii=False)
        )
    for node in graph.nodes:
        span = spans_by_id[node.id]
        if node.id in python_ids:
            if span.raw_text.lower().count("```python") != 1:
                raise ValueError(
                    f"Node {node.id!r} needs exactly one fenced python block"
                )
            validate_python_code(
                span.code_blocks[0],
                label=f"node {node.id!r} teacher code",
            )
        elif span.code_blocks:
            raise ValueError(f"Non-python node {node.id!r} contains a code block")
    master["code_validation"] = {
        "python_node_count": len(python_nodes),
        "signature_exact": True,
        "static_safety_passed": True,
        "hidden_tests_passed": None,
    }
    return master


def normalize_teacher_signatures(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Accept harmless teacher variants such as ``def f(x):`` in graph JSON."""
    normalized = copy.deepcopy(payload)
    graph_data = normalized.get("function_graph")
    if not isinstance(graph_data, dict):
        graph_data = normalized.get("graph")
    if not isinstance(graph_data, dict) and isinstance(normalized.get("nodes"), list):
        graph_data = normalized
    if not isinstance(graph_data, dict):
        return normalized
    nodes = graph_data.get("nodes", [])
    if not isinstance(nodes, list):
        return normalized
    for node in nodes:
        if not isinstance(node, dict):
            continue
        signature = str(node.get("signature", "")).strip().strip("`").strip()
        if signature.startswith("def "):
            signature = signature[4:].strip()
        if signature.endswith(":"):
            signature = signature[:-1].rstrip()
        node["signature"] = signature
    return normalized


def _validate_input_records(records: Sequence[Dict[str, Any]]) -> None:
    if not records:
        raise ValueError("Input JSONL is empty")
    required = {"id", "problem", "reference_solution", "gold_answer", "source"}
    seen = set()
    for row in records:
        missing = required - set(row)
        if missing:
            raise ValueError(f"Input row is missing fields: {sorted(missing)}")
        if row["id"] in seen:
            raise ValueError(f"Duplicate input ID: {row['id']}")
        seen.add(row["id"])


def _build_manifest(
    inputs: Sequence[Dict[str, Any]],
    masters: Sequence[Dict[str, Any]],
    failures: Sequence[Dict[str, Any]],
    *,
    model: str,
    graph_path: Path,
    solve_path: Path,
    recovered_payloads: int,
) -> Dict[str, Any]:
    python_nodes = sum(
        int(row.get("code_validation", {}).get("python_node_count", 0))
        for row in masters
    )
    return {
        "input_rows": len(inputs),
        "successfully_annotated": len(masters),
        "failed_this_run": len(failures),
        "recovered_saved_payloads": recovered_payloads,
        "success_rate": round(len(masters) / len(inputs), 6),
        "teacher_model": model,
        "source_counts": dict(Counter(row.get("source") for row in inputs)),
        "successful_source_counts": dict(
            Counter(row.get("source") for row in masters)
        ),
        "function_graph_nodes": sum(
            len(row["function_graph"]["nodes"]) for row in masters
        ),
        "python_function_nodes": python_nodes,
        "graph_sft_provisional_rows": len(masters),
        "solve_code_sft_provisional_rows": len(masters),
        "outputs": {
            "graph_provisional": str(graph_path),
            "solve_code_provisional": str(solve_path),
        },
        "warning": (
            "Provisional Python passed schema, tag, signature, and static-safety checks only. "
            "Do not train on these files until private hidden verifier execution passes."
        ),
    }


def _read_jsonl(
    path: Path,
    *,
    missing_ok: bool = False,
) -> List[Dict[str, Any]]:
    if not path.is_file():
        if missing_ok:
            return []
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _write_json(path: Path, value: Dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


if __name__ == "__main__":
    main()

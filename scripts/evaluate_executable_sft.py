#!/usr/bin/env python3
"""Compare adapters with real rollouts and private executable verification."""

from __future__ import annotations

import argparse
import gc
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from fsg_rl.executable_eval import (
    check_exact_tags,
    check_python_signatures,
    execution_test_counts,
    metric_deltas,
    pair_evaluations,
    summarize_evaluations,
)
from fsg_rl.parsing import parse_tagged_function_spans, span_by_node
from fsg_rl.rollout import TransformersPolicy
from fsg_rl.schemas import FunctionGraph, MemoryContext, Problem
from fsg_rl.tool_execution import ToolExecutor
from fsg_rl.verifier import Verifier


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-9B-Base")
    parser.add_argument(
        "--adapter",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Repeat for each adapter; evaluation is sequential.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--timeout-seconds", type=float, default=5.0)
    parser.add_argument("--memory-fraction", type=float, default=0.40)
    parser.add_argument("--local-memory-max", default="8g")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed per-adapter rows and flush every new row atomically.",
    )
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="With --resume, rerun saved rows whose status is error.",
    )
    args = parser.parse_args()

    data_path = Path(args.data).expanduser().resolve()
    records = _read_jsonl(data_path)
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        raise SystemExit("Evaluation data is empty")
    adapters = [_parse_adapter(value) for value in args.adapter]
    if len({label for label, _ in adapters}) != len(adapters):
        parser.error("Adapter labels must be unique")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: Dict[str, Any] = {}
    all_results: Dict[str, List[Dict[str, Any]]] = {}
    for label, adapter_path in adapters:
        safe_label = _safe_label(label)
        result_path = output_dir / f"{safe_label}_results.jsonl"
        print("=" * 72, flush=True)
        print(f"Evaluating {label}: {adapter_path}", flush=True)
        results, summary = _evaluate_adapter(
            label=label,
            adapter_path=adapter_path,
            base_model=args.base_model,
            records=records,
            max_new_tokens=args.max_new_tokens,
            timeout_seconds=args.timeout_seconds,
            memory_fraction=args.memory_fraction,
            local_memory_max=args.local_memory_max,
            seed=args.seed,
            result_path=result_path,
            resume=args.resume,
            retry_errors=args.retry_errors,
        )
        _write_jsonl(result_path, results)
        _write_json(output_dir / f"{safe_label}_summary.json", summary)
        all_results[label] = results
        summaries[label] = summary
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

    comparison: Dict[str, Any] = {
        "data": str(data_path),
        "records": len(records),
        "base_model": args.base_model,
        "adapters": {
            label: {"path": path, "summary": summaries[label]}
            for label, path in adapters
        },
        "privacy": (
            "Rollouts receive only metadata.function_graph. Hidden tests and target_call "
            "from metadata.verification_graph are used only after generation."
        ),
    }
    if len(adapters) == 2:
        baseline_label = adapters[0][0]
        candidate_label = adapters[1][0]
        pairs, transition_counts = pair_evaluations(
            all_results[baseline_label], all_results[candidate_label]
        )
        paired_path = output_dir / "paired_results.jsonl"
        _write_jsonl(paired_path, pairs)
        comparison["comparison"] = {
            "baseline": baseline_label,
            "candidate": candidate_label,
            "candidate_minus_baseline": metric_deltas(
                summaries[baseline_label], summaries[candidate_label]
            ),
            "paired_transition_counts": transition_counts,
            "paired_results": str(paired_path),
        }
    _write_json(output_dir / "comparison.json", comparison)
    print("=" * 72, flush=True)
    print(json.dumps(comparison, ensure_ascii=False, indent=2), flush=True)


def _evaluate_adapter(
    *,
    label: str,
    adapter_path: str,
    base_model: str,
    records: List[Dict[str, Any]],
    max_new_tokens: int,
    timeout_seconds: float,
    memory_fraction: float,
    local_memory_max: str,
    seed: int,
    result_path: Path | None = None,
    resume: bool = False,
    retry_errors: bool = False,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    config = {
        "policy": {
            "backend": "transformers",
            "model_name_or_path": base_model,
            "adapter_name_or_path": adapter_path,
            "dtype": "bfloat16",
            "device_map": "auto",
            "quantization": "4bit",
            "trust_remote_code": False,
        },
        "rollout": {
            "max_new_tokens": max_new_tokens,
            "temperature": 0.0,
            "top_p": 0.95,
            "do_sample": False,
            "enable_thinking": False,
            "repetition_penalty": 1.0,
            "seed": seed,
        },
        "sandbox": {
            "backend": "local_limited",
            "allow_unsafe_subprocess": True,
            "timeout_seconds": timeout_seconds,
            "memory_fraction": memory_fraction,
            "local_memory_max": local_memory_max,
            "pids_limit": 32,
        },
        "verifier": {
            "node_execution_weight": 0.25,
            "node_test_weight": 0.50,
            "node_property_weight": 0.25,
        },
    }
    executor = ToolExecutor(config)
    verifier = Verifier(config)
    record_ids = [str(row["id"]) for row in records]
    if len(set(record_ids)) != len(record_ids):
        raise ValueError("Evaluation data contains duplicate IDs")
    saved_by_id: Dict[str, Dict[str, Any]] = {}
    if resume and result_path is not None and result_path.is_file():
        saved_by_id = _load_resumable_results(
            result_path,
            label=label,
            expected_ids=set(record_ids),
            retry_errors=retry_errors,
        )
        print(
            f"Resuming {label}: completed={len(saved_by_id)}/{len(records)}",
            flush=True,
        )
    if len(saved_by_id) == len(records):
        results = [saved_by_id[row_id] for row_id in record_ids]
        print(f"{label} already complete; model loading skipped", flush=True)
        return results, summarize_evaluations(results)

    print("Loading policy and adapter...", flush=True)
    policy = TransformersPolicy(config, trainable=False)
    try:
        for index, source in enumerate(records, start=1):
            row_id = str(source["id"])
            if row_id in saved_by_id:
                continue
            result = _evaluate_one(
                source,
                label=label,
                policy=policy,
                executor=executor,
                verifier=verifier,
                rollout_index=index - 1,
            )
            saved_by_id[row_id] = result
            ordered_partial = [
                saved_by_id[value]
                for value in record_ids
                if value in saved_by_id
            ]
            if result_path is not None:
                _write_jsonl(result_path, ordered_partial)
            succeeded = sum(
                bool(item.get("full_success")) for item in ordered_partial
            )
            print(
                f"{label} evaluated={len(ordered_partial)}/{len(records)} "
                f"full_success={succeeded}",
                flush=True,
            )
    finally:
        del policy
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    results = [saved_by_id[row_id] for row_id in record_ids]
    return results, summarize_evaluations(results)


def _evaluate_one(
    source: Dict[str, Any],
    *,
    label: str,
    policy: TransformersPolicy,
    executor: ToolExecutor,
    verifier: Verifier,
    rollout_index: int,
) -> Dict[str, Any]:
    problem_id = str(source["id"])
    base_result: Dict[str, Any] = {
        "id": problem_id,
        "adapter": label,
        "status": "error",
        "tags_exact": False,
        "code_blocks_complete": False,
        "signature_exact": False,
        "python_executable": False,
        "all_hidden_tests_pass": False,
        "backward_target_correct": False,
        "final_answer_correct": False,
        "full_success": False,
        "test_counts": {
            "unit_passed": 0,
            "unit_total": 0,
            "property_passed": 0,
            "property_total": 0,
        },
    }
    try:
        problem, public_graph, hidden_graph = _load_problem(source)
        rollout = policy.rollout(
            problem,
            public_graph,
            MemoryContext(),
            rollout_index=rollout_index,
        )
        spans = parse_tagged_function_spans(rollout.raw_text, public_graph)
        tags = check_exact_tags(rollout.raw_text, public_graph)
        signatures = check_python_signatures(spans, public_graph)
        execution = executor.execute(spans, hidden_graph)
        verification = verifier.verify(problem, hidden_graph, spans, execution)
        counts = execution_test_counts(execution, hidden_graph)
        expected_executable_ids = [
            node.id
            for node in hidden_graph.nodes
            if node.expected_output_type == "python_function"
            or bool(node.verification_spec.get("tests"))
        ]
        python_executable = bool(expected_executable_ids) and all(
            bool(execution.node_results.get(node_id))
            and execution.node_results[node_id].executable
            for node_id in expected_executable_ids
        )
        all_tests_pass = (
            counts["unit_total"] + counts["property_total"] > 0
            and counts["unit_passed"] == counts["unit_total"]
            and counts["property_passed"] == counts["property_total"]
        )
        main_span = span_by_node(spans, "main")
        base_result.update(
            {
                "status": "completed",
                "generated_text": rollout.raw_text,
                "generated_answer": main_span.extracted_answer if main_span else None,
                "gold_answer": problem.gold_answer,
                "generation_seconds": rollout.generation_seconds,
                "prompt_token_count": len(rollout.prompt_token_ids),
                "completion_token_count": len(rollout.completion_token_ids),
                "tags_exact": bool(tags["passed"]),
                "tag_check": tags,
                "code_blocks_complete": bool(signatures["code_blocks_complete"]),
                "signature_exact": bool(signatures["passed"]),
                "signature_check": signatures,
                "python_executable": python_executable,
                "all_hidden_tests_pass": all_tests_pass,
                "test_counts": counts,
                "backward_target_correct": verification.backward_score == 1.0,
                "final_answer_correct": verification.final_answer_score == 1.0,
                "execution": execution.to_dict(),
                "verification": verification.to_dict(),
                "error": None,
            }
        )
        base_result["full_success"] = all(
            bool(base_result[key])
            for key in (
                "tags_exact",
                "code_blocks_complete",
                "signature_exact",
                "python_executable",
                "all_hidden_tests_pass",
                "backward_target_correct",
                "final_answer_correct",
            )
        )
    except Exception as exc:  # Keep all 17 rows auditable even if one fails.
        base_result["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
    return base_result


def _load_problem(
    source: Dict[str, Any],
) -> tuple[Problem, FunctionGraph, FunctionGraph]:
    metadata = source.get("metadata", {})
    public_data = metadata.get("function_graph")
    hidden_data = metadata.get("verification_graph")
    if not isinstance(public_data, dict) or not isinstance(hidden_data, dict):
        raise ValueError("Evaluation row needs public function_graph and verification_graph")
    public_graph = FunctionGraph.from_dict(public_data)
    hidden_graph = FunctionGraph.from_dict(hidden_data)
    public_ids = [node.id for node in public_graph.nodes]
    hidden_ids = [node.id for node in hidden_graph.nodes]
    if public_ids != hidden_ids:
        raise ValueError("Public and hidden graph node order differs")
    problem = Problem(
        id=str(source["id"]),
        text=str(source.get("text", source.get("problem", ""))).strip(),
        gold_answer=str(source.get("gold_answer", "")).strip(),
        split="validation",
    )
    if not problem.text or not problem.gold_answer:
        raise ValueError("Evaluation row needs problem text and gold_answer")
    return problem, public_graph, hidden_graph


def _parse_adapter(value: str) -> Tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--adapter must have LABEL=PATH format")
    label, path = value.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label or not path:
        raise argparse.ArgumentTypeError("--adapter needs non-empty LABEL and PATH")
    return label, path


def _safe_label(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_") or "adapter"


def _load_resumable_results(
    path: Path,
    *,
    label: str,
    expected_ids: set[str],
    retry_errors: bool,
) -> Dict[str, Dict[str, Any]]:
    rows = _read_jsonl(path)
    result: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        row_id = str(row.get("id", "")).strip()
        if not row_id or row_id in result:
            raise ValueError(f"Invalid or duplicate resumable result ID: {row_id!r}")
        if row_id not in expected_ids:
            raise ValueError(f"Saved result {row_id!r} is absent from evaluation data")
        if str(row.get("adapter", "")) != label:
            raise ValueError(
                f"Saved result adapter mismatch for {row_id!r}: "
                f"expected={label!r} actual={row.get('adapter')!r}"
            )
        if retry_errors and row.get("status") == "error":
            continue
        result[row_id] = row
    return result


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _write_json(path: Path, value: Dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

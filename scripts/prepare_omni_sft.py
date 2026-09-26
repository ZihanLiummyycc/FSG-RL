#!/usr/bin/env python3
"""Sample 20% of Omni-MATH and create graph/solve SFT labels with a teacher."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List

from fsg_rl.api_client import ChatAPIConfig, OpenAICompatibleChatClient
from fsg_rl.sft_data import (
    LexicalMemoryIndex,
    build_sft_records,
    build_teacher_messages,
    canonicalize_omni_record,
    deterministic_stratified_sample,
    validate_teacher_annotation,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repo-id", default="KbsdJames/Omni-MATH")
    parser.add_argument("--split", default="test")
    parser.add_argument("--fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--memory-jsonl")
    parser.add_argument("--memory-top-k", type=int, default=5)
    parser.add_argument("--max-nodes", type=int, default=8)
    parser.add_argument("--sample-only", action="store_true")
    parser.add_argument(
        "--api-model",
        default=os.environ.get("TEACHER_MODEL", "gpt-5.6-sol"),
    )
    parser.add_argument("--api-base-url", default=os.environ.get("TEACHER_BASE_URL"))
    parser.add_argument("--api-key-env", default="TEACHER_API_KEY")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-tokens", type=int, default=12288)
    args = parser.parse_args()
    if not args.sample_only and (not args.api_model or not args.api_base_url):
        parser.error("Set TEACHER_BASE_URL, pass --api-base-url, or use --sample-only")

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install the datasets package before preparing Omni-MATH") from exc

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(args.repo_id, split=args.split)
    canonical = []
    rejected = 0
    for row in dataset:
        try:
            canonical.append(canonicalize_omni_record(dict(row)))
        except ValueError:
            rejected += 1
    selected, holdout = deterministic_stratified_sample(
        canonical, fraction=args.fraction, seed=args.seed
    )
    _write_jsonl(output_dir / "omni_selected_20pct.jsonl", selected)
    _write_jsonl(output_dir / "omni_holdout_80pct.jsonl", holdout)

    manifest = {
        "dataset": args.repo_id,
        "source_split": args.split,
        "seed": args.seed,
        "fraction": args.fraction,
        "source_rows": len(dataset),
        "valid_rows": len(canonical),
        "rejected_source_rows": rejected,
        "selected_rows": len(selected),
        "holdout_rows": len(holdout),
        "warning": (
            "The selected partition is training data. Report Omni-MATH scores only on the "
            "recorded holdout partition, never on the original full test split."
        ),
    }
    _write_json(output_dir / "manifest.json", manifest)
    if args.sample_only:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return

    memory_index = None
    if args.memory_jsonl:
        memory_index = LexicalMemoryIndex(_read_jsonl(Path(args.memory_jsonl)))

    master_path = output_dir / "omni_teacher_master.jsonl"
    completed = {record["id"]: record for record in _read_jsonl(master_path)}
    pending = [record for record in selected if record["id"] not in completed]
    if args.limit is not None:
        pending = pending[: args.limit]

    config = ChatAPIConfig(
        model=str(args.api_model),
        base_url=str(args.api_base_url).rstrip("/"),
        api_key_env=args.api_key_env,
        timeout_seconds=300.0,
        max_retries=3,
    )
    local = threading.local()

    def client() -> OpenAICompatibleChatClient:
        if not hasattr(local, "client"):
            local.client = OpenAICompatibleChatClient(config)
        return local.client

    def annotate(record: Dict[str, Any]) -> Dict[str, Any]:
        memory = (
            memory_index.retrieve(record["problem"], args.memory_top_k)
            if memory_index is not None
            else []
        )
        messages = build_teacher_messages(record, memory, args.max_nodes)
        last_error: Exception | None = None
        previous_payload: Dict[str, Any] | None = None
        for _ in range(args.attempts):
            retry_messages = list(messages)
            if last_error is not None:
                if previous_payload is not None:
                    retry_messages.append(
                        {
                            "role": "assistant",
                            "content": json.dumps(previous_payload, ensure_ascii=False),
                        }
                    )
                retry_messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"The previous label failed validation: {last_error}. "
                            "Return a corrected JSON object only."
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
                return validate_teacher_annotation(
                    record, previous_payload, max_nodes=args.max_nodes
                )
            except Exception as exc:
                last_error = exc
        assert last_error is not None
        if previous_payload is not None:
            failed_payloads[record["id"]] = previous_payload
        raise last_error

    failures = []
    failed_payloads: Dict[str, Dict[str, Any]] = {}
    processed = 0
    if pending:
        for offset in range(0, len(pending), args.workers):
            batch = pending[offset : offset + args.workers]

            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {
                    executor.submit(annotate, record): record
                    for record in batch
                }

                for future in as_completed(futures):
                    source = futures[future]
                    try:
                        completed[source["id"]] = future.result()
                    except Exception as exc:
                        failure = {
                            "id": source["id"],
                            "error_type": type(exc).__name__,
                            "error": str(exc)[:2000],
                        }
                        if source["id"] in failed_payloads:
                            failure["teacher_payload"] = (
                                failed_payloads[source["id"]]
                            )
                        failures.append(failure)

                    processed += 1

            _write_jsonl(
                master_path,
                sorted(completed.values(), key=lambda row: row["id"]),
            )
            _write_jsonl(
                output_dir / "omni_teacher_failed.jsonl",
                failures,
            )
            print(
                f"annotated={processed}/{len(pending)} "
                f"success={len(completed)}"
            )

    _write_jsonl(master_path, sorted(completed.values(), key=lambda row: row["id"]))
    _write_jsonl(output_dir / "omni_teacher_failed.jsonl", failures)
    split_counts = _write_sft_partitions(
        output_dir,
        list(completed.values()),
        args.validation_fraction,
        args.seed,
    )
    manifest.update(
        {
            "teacher_model": args.api_model,
            "successfully_annotated": len(completed),
            "failed_this_run": len(failures),
            **split_counts,
        }
    )
    _write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def _write_sft_partitions(
    output_dir: Path,
    masters: List[Dict[str, Any]],
    validation_fraction: float,
    seed: int,
) -> Dict[str, int]:
    ranked = sorted(
        masters,
        key=lambda row: hashlib.sha256(f"{seed}:{row['id']}".encode()).hexdigest(),
    )
    validation_count = int(len(ranked) * validation_fraction + 0.5)
    validation_ids = {row["id"] for row in ranked[:validation_count]}
    partitions = {
        "graph_train": [],
        "graph_validation": [],
        "solve_train": [],
        "solve_validation": [],
    }
    for master in sorted(masters, key=lambda row: row["id"]):
        graph, solve = build_sft_records(master)
        suffix = "validation" if master["id"] in validation_ids else "train"
        partitions[f"graph_{suffix}"].append(graph)
        partitions[f"solve_{suffix}"].append(solve)
    for name, records in partitions.items():
        _write_jsonl(output_dir / f"omni_sft_{name}.jsonl", records)
    return {f"{name}_rows": len(records) for name, records in partitions.items()}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as stream:
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
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


if __name__ == "__main__":
    main()

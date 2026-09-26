#!/usr/bin/env python3
"""Add informal theorem metadata with an OpenAI-compatible teacher API."""

from __future__ import annotations

import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable

from fsg_rl.api_client import ChatAPIConfig, OpenAICompatibleChatClient
from fsg_rl.mathlib_memory import (
    build_enrichment_messages,
    merge_enrichment,
    normalize_traced_theorem,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--failed-output", required=True)
    parser.add_argument(
        "--api-model",
        default=os.environ.get("TEACHER_MODEL", "gpt-5.6-sol"),
    )
    parser.add_argument("--api-base-url", default=os.environ.get("TEACHER_BASE_URL"))
    parser.add_argument("--api-key-env", default="TEACHER_API_KEY")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument(
        "--source-revision",
        default=os.environ.get("MATHLIB_COMMIT", ""),
        help="Pinned mathlib commit, needed when input is the exact five-field export",
    )
    args = parser.parse_args()
    if not args.api_model or not args.api_base_url:
        parser.error("Set TEACHER_BASE_URL or pass --api-base-url")
    if args.workers < 1 or args.attempts < 1:
        parser.error("--workers and --attempts must be positive")

    input_records = []
    for raw_record in _read_jsonl(Path(args.input)):
        if not raw_record.get("source_revision") and args.source_revision:
            raw_record = {**raw_record, "source_revision": args.source_revision}
        if raw_record.get("memory_id"):
            # 已经标准化的 HF Mathlib 声明：保留 formal_declaration 类型。
            input_records.append(raw_record)
        else:
            # LeanDojo theorem trace 或五字段输入。
            input_records.append(normalize_traced_theorem(raw_record))
    output = Path(args.output).expanduser().resolve()
    failed_output = Path(args.failed_output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = {record["id"]: record for record in _read_jsonl(output)}
    pending = [record for record in input_records if record["id"] not in completed]
    if args.limit is not None:
        pending = pending[: args.limit]

    config = ChatAPIConfig(
        model=args.api_model,
        base_url=str(args.api_base_url).rstrip("/"),
        api_key_env=args.api_key_env,
        timeout_seconds=180.0,
        max_retries=3,
    )
    local = threading.local()

    def client() -> OpenAICompatibleChatClient:
        if not hasattr(local, "client"):
            local.client = OpenAICompatibleChatClient(config)
        return local.client

    def enrich(record: Dict[str, Any]) -> Dict[str, Any]:
        last_error: Exception | None = None
        for _ in range(args.attempts):
            try:
                payload = client().complete_json(
                    build_enrichment_messages(record),
                    temperature=0.0,
                    max_tokens=args.max_tokens,
                )
                return merge_enrichment(record, payload)
            except Exception as exc:  # API and validation errors are checkpointed alike.
                last_error = exc
        assert last_error is not None
        raise last_error

    failures = []
    if pending:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(enrich, record): record for record in pending}
            for index, future in enumerate(as_completed(futures), 1):
                source = futures[future]
                try:
                    completed[source["id"]] = future.result()
                except Exception as exc:
                    failures.append(
                        {
                            "id": source["id"],
                            "error_type": type(exc).__name__,
                            "error": str(exc)[:1000],
                        }
                    )
                if index % 25 == 0 or index == len(pending):
                    _write_jsonl(
                        output,
                        sorted(completed.values(), key=lambda row: row["id"]),
                    )
                    _write_jsonl(failed_output, failures)
                    print(f"processed={index}/{len(pending)} success={len(completed)}")

    _write_jsonl(output, sorted(completed.values(), key=lambda row: row["id"]))
    _write_jsonl(failed_output, failures)
    print(
        json.dumps(
            {
                "input": len(input_records),
                "enriched": len(completed),
                "failed_this_run": len(failures),
                "output": str(output),
            },
            indent=2,
        )
    )


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


if __name__ == "__main__":
    main()

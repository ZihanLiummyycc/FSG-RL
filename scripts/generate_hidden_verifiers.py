#!/usr/bin/env python3
"""Generate resumable Terra hidden-verifier bundles for executable candidates."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import threading
from typing import Any, Dict, Iterable, List

from fsg_rl.api_client import ChatAPIConfig, OpenAICompatibleChatClient
from fsg_rl.tool_execution import ToolExecutor
from fsg_rl.verifier_bundle import (
    VerifierBundleRejected,
    build_teacher_messages,
    compile_verifier_bundle,
)
from fsg_rl.verifier_execution import validate_compiled_verifier_record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--teacher-master", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--api-model", default=os.environ.get("TEACHER_MODEL", "gpt-5.6-terra"))
    parser.add_argument("--api-base-url", default=os.environ.get("TEACHER_BASE_URL"))
    parser.add_argument("--api-key-env", default="TEACHER_API_KEY")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--execution-backend",
        choices=["none", "local_limited"],
        default="none",
    )
    parser.add_argument("--execution-timeout-seconds", type=float, default=5.0)
    parser.add_argument("--memory-fraction", type=float, default=0.40)
    parser.add_argument("--local-memory-max", default="8g")
    args = parser.parse_args()
    if not args.api_base_url:
        parser.error("--api-base-url or TEACHER_BASE_URL is required")
    if args.execution_backend != "none" and args.workers != 1:
        parser.error(
            "Generation-time local execution requires --workers 1 so resource "
            "limits and symbolic timeouts run safely on the main thread"
        )

    candidates = _read_jsonl(Path(args.input))
    masters = {row["id"]: row for row in _read_jsonl(Path(args.teacher_master))}
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    accepted_path = output_dir / "hidden_verifier_teacher_accepted.jsonl"
    rejected_path = output_dir / "hidden_verifier_teacher_rejected.jsonl"
    failed_path = output_dir / "hidden_verifier_teacher_failed.jsonl"

    accepted = {row["id"]: row for row in _read_jsonl(accepted_path)}
    rejected = {row["id"]: row for row in _read_jsonl(rejected_path)}
    completed_ids = set(accepted) | set(rejected)
    pending = [row for row in candidates if row["id"] not in completed_ids]
    if args.limit is not None:
        pending = pending[: args.limit]
    missing = {row["id"] for row in pending} - set(masters)
    if missing:
        raise ValueError(f"Candidate IDs missing from teacher master: {sorted(missing)[:10]}")

    api_config = ChatAPIConfig(
        model=args.api_model,
        base_url=str(args.api_base_url).rstrip("/"),
        api_key_env=args.api_key_env,
        timeout_seconds=300.0,
        max_retries=2,
    )
    local = threading.local()
    execution_executor = None
    if args.execution_backend == "local_limited":
        execution_executor = ToolExecutor(
            {
                "sandbox": {
                    "backend": "local_limited",
                    "allow_unsafe_subprocess": True,
                    "timeout_seconds": args.execution_timeout_seconds,
                    "memory_fraction": args.memory_fraction,
                    "local_memory_max": args.local_memory_max,
                    "pids_limit": 32,
                }
            }
        )
        print(
            "GENERATION-TIME EXECUTION LIMITS:\n"
            + json.dumps(execution_executor.local_resource_summary(), indent=2),
            flush=True,
        )

    def client() -> OpenAICompatibleChatClient:
        if not hasattr(local, "client"):
            local.client = OpenAICompatibleChatClient(api_config)
        return local.client

    def annotate(candidate: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        messages = build_teacher_messages(candidate, masters[candidate["id"]])
        previous_payload: Dict[str, Any] | None = None
        last_error: Exception | None = None
        for _ in range(args.attempts):
            retry_messages = list(messages)
            if last_error is not None:
                if previous_payload is not None:
                    retry_messages.append(
                        {"role": "assistant", "content": json.dumps(previous_payload, ensure_ascii=False)}
                    )
                retry_messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"The previous verifier bundle failed validation: {last_error}. "
                            "Return a corrected complete JSON object only."
                        ),
                    }
                )
            try:
                previous_payload = client().complete_json(
                    retry_messages,
                    temperature=0.0,
                    max_tokens=args.max_tokens,
                )
                compiled = compile_verifier_bundle(
                    candidate,
                    previous_payload,
                    annotation_model=args.api_model,
                )
                if execution_executor is not None:
                    audit = validate_compiled_verifier_record(
                        compiled,
                        execution_executor,
                    )
                    compiled_metadata = dict(compiled.get("metadata", {}))
                    compiled_metadata["verifier_execution_audit"] = audit
                    compiled["metadata"] = compiled_metadata
                return "accepted", compiled
            except VerifierBundleRejected as exc:
                return "rejected", {
                    "id": candidate["id"],
                    "reason": str(exc),
                    "model": args.api_model,
                }
            except Exception as exc:
                last_error = exc
        assert last_error is not None
        raise last_error

    failures: List[Dict[str, Any]] = []
    processed = 0
    for offset in range(0, len(pending), max(1, args.workers)):
        batch = pending[offset : offset + max(1, args.workers)]
        if args.workers == 1:
            candidate = batch[0]
            try:
                status, value = annotate(candidate)
                if status == "accepted":
                    accepted[candidate["id"]] = value
                else:
                    rejected[candidate["id"]] = value
            except Exception as exc:
                failures.append(
                    {
                        "id": candidate["id"],
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:4000],
                    }
                )
            processed += 1
        else:
            with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
                futures = {executor.submit(annotate, row): row for row in batch}
                for future in as_completed(futures):
                    candidate = futures[future]
                    try:
                        status, value = future.result()
                        if status == "accepted":
                            accepted[candidate["id"]] = value
                        else:
                            rejected[candidate["id"]] = value
                    except Exception as exc:
                        failures.append(
                            {
                                "id": candidate["id"],
                                "error_type": type(exc).__name__,
                                "error": str(exc)[:4000],
                            }
                        )
                    processed += 1
        _write_jsonl(accepted_path, sorted(accepted.values(), key=lambda row: row["id"]))
        _write_jsonl(rejected_path, sorted(rejected.values(), key=lambda row: row["id"]))
        _write_jsonl(failed_path, failures)
        print(
            f"processed={processed}/{len(pending)} accepted={len(accepted)} "
            f"rejected={len(rejected)} failed={len(failures)}",
            flush=True,
        )

    manifest = {
        "input_rows": len(candidates),
        "pending_this_run": len(pending),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "failed_this_run": len(failures),
        "teacher_model": args.api_model,
        "accepted_path": str(accepted_path),
        "rejected_path": str(rejected_path),
        "failed_path": str(failed_path),
        "execution_validation": args.execution_backend,
        "note": (
            "Accepted records passed generation-time reference/mutant execution."
            if execution_executor is not None
            else "Teacher acceptance is provisional until execution validation passes."
        ),
    }
    (output_dir / "teacher_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


if __name__ == "__main__":
    main()

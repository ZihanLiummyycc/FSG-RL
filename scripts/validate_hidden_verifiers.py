#!/usr/bin/env python3
"""Execute reference and mutant implementations for hidden verifier bundles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from fsg_rl.tool_execution import ToolExecutor
from fsg_rl.verifier_bundle import strip_private_implementations
from fsg_rl.verifier_execution import validate_compiled_verifier_record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--backend",
        choices=["docker", "unshare", "local_limited"],
        default="docker",
    )
    parser.add_argument("--docker-image", default="python:3.11-slim")
    parser.add_argument("--timeout-seconds", type=float, default=2.0)
    parser.add_argument("--memory-fraction", type=float, default=0.40)
    parser.add_argument("--local-memory-max", default="8g")
    args = parser.parse_args()

    rows = _read_jsonl(Path(args.input))
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    executor = ToolExecutor(
        {
            "sandbox": {
                "backend": args.backend,
                "docker_image": args.docker_image,
                "timeout_seconds": args.timeout_seconds,
                "memory_limit": "512m",
                "memory_fraction": args.memory_fraction,
                "local_memory_max": args.local_memory_max,
                "cpu_limit": "1.0",
                "pids_limit": 32,
                "allow_unsafe_subprocess": args.backend == "local_limited",
            }
        }
    )
    if args.backend == "local_limited":
        print(
            "LOCAL EXECUTION RESOURCE LIMITS:\n"
            + json.dumps(executor.local_resource_summary(), indent=2),
            flush=True,
        )

    accepted = []
    rejected = []
    for index, row in enumerate(rows, 1):
        try:
            audit = validate_compiled_verifier_record(row, executor)
            clean = strip_private_implementations(row)
            metadata = dict(clean.get("metadata", {}))
            metadata["verifier_execution_audit"] = audit
            clean["metadata"] = metadata
            accepted.append(clean)
        except Exception as exc:
            rejected.append(
                {
                    "id": row.get("id"),
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:4000],
                }
            )
        print(
            f"validated={index}/{len(rows)} accepted={len(accepted)} rejected={len(rejected)}",
            flush=True,
        )

    accepted_path = output_dir / "hidden_verifier_validated.jsonl"
    rejected_path = output_dir / "hidden_verifier_execution_rejected.jsonl"
    _write_jsonl(accepted_path, accepted)
    _write_jsonl(rejected_path, rejected)
    manifest = {
        "teacher_accepted_input": len(rows),
        "execution_validated": len(accepted),
        "execution_rejected": len(rejected),
        "validation_rate": round(len(accepted) / len(rows), 6) if rows else 0.0,
        "validated_output": str(accepted_path),
        "rejected_output": str(rejected_path),
    }
    (output_dir / "execution_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
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

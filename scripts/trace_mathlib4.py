#!/usr/bin/env python3
"""Trace a pinned mathlib4 commit with LeanDojo-v2 and export stable JSONL."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", required=True, help="Pinned full mathlib4 commit SHA")
    parser.add_argument(
        "--url",
        default="https://github.com/leanprover-community/mathlib4",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--memory-output",
        help="Optional richer JSONL retaining proof-premise dependencies",
    )
    parser.add_argument("--database-json", required=True)
    parser.add_argument("--raid-dir", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--include-prefix",
        action="append",
        default=[],
        help="Optional Lean namespace prefix; repeat for multiple prefixes",
    )
    args = parser.parse_args()

    if len(args.commit) < 12 or args.commit in {"main", "master"}:
        raise SystemExit("--commit must be a pinned commit SHA, not a moving branch")
    os.environ["RAID_DIR"] = str(Path(args.raid_dir).expanduser().resolve())

    try:
        from lean_dojo_v2.database import DynamicDatabase
    except ImportError as exc:
        raise SystemExit(
            "Install lean-dojo-v2 in a separate Python 3.11 environment first"
        ) from exc

    # Import after RAID_DIR is set because LeanDojo reads its storage constants
    # during module initialization.
    from fsg_rl.mathlib_memory import formalize_traced_theorem, normalize_traced_theorem

    database_path = Path(args.database_json).expanduser().resolve()
    database_path.parent.mkdir(parents=True, exist_ok=True)
    database = DynamicDatabase(json_path=str(database_path))
    repository = database.trace_repository(
        url=args.url,
        commit=args.commit,
        build_deps=False,
    )
    if repository is None:
        raise SystemExit("LeanDojo failed to trace the pinned mathlib4 repository")

    formal_records = []
    memory_records = []
    prefixes = tuple(args.include_prefix)
    theorems = sorted(repository.get_all_theorems, key=lambda value: value.full_name)
    for theorem in theorems:
        if prefixes and not theorem.full_name.startswith(prefixes):
            continue
        raw = theorem.to_dict()
        raw["source_url"] = args.url
        raw["source_revision"] = args.commit
        formal_records.append(formalize_traced_theorem(raw))
        if args.memory_output:
            memory_records.append(normalize_traced_theorem(raw))
        if args.limit is not None and len(formal_records) >= args.limit:
            break

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in formal_records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(output)
    manifest = {
        "source_url": args.url,
        "source_revision": args.commit,
        "theorems": len(formal_records),
        "schema": [
            "lean_name",
            "namespace",
            "source_file",
            "formal_statement",
            "docstring",
        ],
    }
    if args.memory_output:
        memory_output = Path(args.memory_output).expanduser().resolve()
        memory_output.parent.mkdir(parents=True, exist_ok=True)
        memory_temporary = memory_output.with_suffix(memory_output.suffix + ".tmp")
        with memory_temporary.open("w", encoding="utf-8") as stream:
            for record in memory_records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        memory_temporary.replace(memory_output)
        manifest["memory_output"] = str(memory_output)
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), **manifest}, indent=2))


if __name__ == "__main__":
    main()

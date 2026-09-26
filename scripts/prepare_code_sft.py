#!/usr/bin/env python3
"""Create code-aware SFT records and a 50/25/25 replay mixture."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from fsg_rl.code_sft import build_code_sft_record, build_replay_mix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-train", required=True)
    parser.add_argument("--private-validation", required=True)
    parser.add_argument("--teacher-master", required=True)
    parser.add_argument("--graph-train", required=True)
    parser.add_argument("--solve-train", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--code-repeats", type=int, default=2)
    parser.add_argument("--replay-per-task", type=int)
    args = parser.parse_args()

    private_train = _read_jsonl(Path(args.private_train))
    private_validation = _read_jsonl(Path(args.private_validation))
    masters = {
        row["id"]: row for row in _read_jsonl(Path(args.teacher_master))
    }
    missing = {
        row["id"] for row in private_train + private_validation
    } - set(masters)
    if missing:
        raise ValueError(f"Teacher masters missing IDs: {sorted(missing)[:10]}")

    code_train = [
        build_code_sft_record(row, masters[row["id"]])
        for row in private_train
    ]
    code_validation = [
        build_code_sft_record(row, masters[row["id"]])
        for row in private_validation
    ]
    graph_train = _read_jsonl(Path(args.graph_train))
    solve_train = _read_jsonl(Path(args.solve_train))
    replay_per_task = (
        len(code_train) if args.replay_per_task is None else args.replay_per_task
    )
    mixed_train = build_replay_mix(
        code_train,
        graph_train,
        solve_train,
        seed=args.seed,
        code_repeats=args.code_repeats,
        replay_per_task=replay_per_task,
    )

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "code_train": output_dir / "code_sft_train_158.jsonl",
        "code_validation": output_dir / "code_sft_validation_17.jsonl",
        "mixed_train": output_dir / "code_sft_mixed_train.jsonl",
    }
    _write_jsonl(paths["code_train"], code_train)
    _write_jsonl(paths["code_validation"], code_validation)
    _write_jsonl(paths["mixed_train"], mixed_train)

    task_counts: Dict[str, int] = {}
    for record in mixed_train:
        task = str(record.get("task", "unknown"))
        task_counts[task] = task_counts.get(task, 0) + 1
    train_ids = {row["metadata"]["source_problem_id"] for row in code_train}
    validation_ids = {
        row["metadata"]["source_problem_id"] for row in code_validation
    }
    manifest = {
        "code_train_rows": len(code_train),
        "code_validation_rows": len(code_validation),
        "mixed_train_rows": len(mixed_train),
        "mixed_task_counts": task_counts,
        "code_train_validation_overlap": len(train_ids & validation_ids),
        "code_repeats": args.code_repeats,
        "replay_per_task": replay_per_task,
        "seed": args.seed,
        "mixed_id_sha256": _id_digest(mixed_train),
        "self_contained_code_prompt_rows": sum(
            _has_self_contained_prompt(record)
            for record in mixed_train
            if record.get("task") == "solve_with_executable_function_graph"
        ),
        "outputs": {key: str(value) for key, value in paths.items()},
        "privacy": (
            "Prompts contain only problem and public graph. Hidden tests, target_call, "
            "mutants, and private verifier implementations are excluded."
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _id_digest(records: Iterable[Dict[str, Any]]) -> str:
    value = "\n".join(sorted(str(record["id"]) for record in records))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _has_self_contained_prompt(record: Dict[str, Any]) -> bool:
    messages = record.get("messages", [])
    if not messages:
        return False
    content = messages[0].get("content", "")
    if isinstance(content, list):
        content = " ".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict)
        )
    normalized = " ".join(str(content).split())
    return "executed alone in a fresh Python 3 interpreter" in normalized


if __name__ == "__main__":
    main()

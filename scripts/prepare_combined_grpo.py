#!/usr/bin/env python3
"""Build leakage-audited medium-plus-Omni GRPO train and validation files."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

from fsg_rl.decomposition import validate_function_graph
from fsg_rl.schemas import FunctionGraph
from fsg_rl.verifier_bundle import strip_private_implementations


PRIVATE_KEYS = {
    "reference_code",
    "mutant_code",
    "reference_solution",
    "tagged_solution",
    "verifier_implementations",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--medium-accepted", required=True)
    parser.add_argument("--medium-stage", required=True)
    parser.add_argument("--medium-pool-validation", required=True)
    parser.add_argument("--medium-pool-test", required=True)
    parser.add_argument("--omni-train", required=True)
    parser.add_argument("--omni-validation", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--medium-validation-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    medium = _read_jsonl(Path(args.medium_accepted))
    medium_stage = _read_jsonl(Path(args.medium_stage))
    pool_validation = _read_jsonl(Path(args.medium_pool_validation))
    pool_test = _read_jsonl(Path(args.medium_pool_test))
    omni_train = _read_jsonl(Path(args.omni_train))
    omni_validation = _read_jsonl(Path(args.omni_validation))

    result = build_combined_splits(
        medium,
        medium_stage,
        pool_validation,
        pool_test,
        omni_train,
        omni_validation,
        validation_fraction=args.medium_validation_fraction,
        seed=args.seed,
    )

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "medium_private_train": output_dir / "medium_private_train.jsonl",
        "medium_private_validation": output_dir / "medium_private_validation.jsonl",
        "medium_grpo_train": output_dir / "medium_grpo_train.jsonl",
        "medium_grpo_validation": output_dir / "medium_grpo_validation.jsonl",
        "combined_grpo_train": output_dir / "combined_grpo_train.jsonl",
        "combined_grpo_validation": output_dir / "combined_grpo_validation.jsonl",
    }
    for key, path in paths.items():
        _write_jsonl(path, result[key])

    manifest = dict(result["audit"])
    manifest["outputs"] = {key: str(path) for key, path in paths.items()}
    _write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def build_combined_splits(
    medium_records: Sequence[Dict[str, Any]],
    medium_stage: Sequence[Dict[str, Any]],
    pool_validation: Sequence[Dict[str, Any]],
    pool_test: Sequence[Dict[str, Any]],
    omni_train: Sequence[Dict[str, Any]],
    omni_validation: Sequence[Dict[str, Any]],
    *,
    validation_fraction: float,
    seed: int,
) -> Dict[str, Any]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")

    medium = _unique_by_id(medium_records, "medium accepted")
    stage = _unique_by_id(medium_stage, "medium stage")
    pool_val = _unique_by_id(pool_validation, "medium pool validation")
    pool_test_rows = _unique_by_id(pool_test, "medium pool test")
    omni_train_by_id = _unique_by_id(omni_train, "Omni train")
    omni_validation_by_id = _unique_by_id(omni_validation, "Omni validation")

    if not set(medium) <= set(stage):
        raise ValueError("Medium accepted IDs are not a subset of the Stage-800 pool")
    if any(str(row.get("split")) != "train" for row in stage.values()):
        raise ValueError("Medium Stage-800 contains a non-train record")

    _require_disjoint("medium accepted", set(medium), "pool validation", set(pool_val))
    _require_disjoint("medium accepted", set(medium), "pool test", set(pool_test_rows))
    _require_disjoint("Omni train", set(omni_train_by_id), "Omni validation", set(omni_validation_by_id))
    _require_disjoint("medium", set(medium), "Omni train", set(omni_train_by_id))
    _require_disjoint("medium", set(medium), "Omni validation", set(omni_validation_by_id))

    for row in medium.values():
        _validate_verifier_record(row, require_private=True)
    for row in list(omni_train_by_id.values()) + list(omni_validation_by_id.values()):
        _validate_verifier_record(row, require_private=False)

    medium_train_rows, medium_validation_rows = stratified_split(
        list(medium.values()),
        validation_fraction=validation_fraction,
        seed=seed,
    )
    medium_private_train = [_with_split(row, "train") for row in medium_train_rows]
    medium_private_validation = [
        _with_split(row, "validation") for row in medium_validation_rows
    ]
    medium_grpo_train = [
        strip_private_implementations(row) for row in medium_private_train
    ]
    medium_grpo_validation = [
        strip_private_implementations(row) for row in medium_private_validation
    ]
    omni_train_rows = [_with_split(row, "train") for row in omni_train_by_id.values()]
    omni_validation_rows = [
        _with_split(row, "validation") for row in omni_validation_by_id.values()
    ]

    combined_train = sorted(
        medium_grpo_train + omni_train_rows,
        key=lambda row: str(row["id"]),
    )
    combined_validation = sorted(
        medium_grpo_validation + omni_validation_rows,
        key=lambda row: str(row["id"]),
    )

    train_ids = {str(row["id"]) for row in combined_train}
    validation_ids = {str(row["id"]) for row in combined_validation}
    _require_disjoint("combined train", train_ids, "combined validation", validation_ids)
    if _contains_forbidden_key(combined_train, PRIVATE_KEYS):
        raise ValueError("Combined GRPO train leaks private teacher implementations")
    if _contains_forbidden_key(combined_validation, PRIVATE_KEYS):
        raise ValueError("Combined validation leaks private teacher implementations")

    audit = {
        "seed": seed,
        "medium_validation_fraction": validation_fraction,
        "medium_accepted_rows": len(medium),
        "medium_train_rows": len(medium_grpo_train),
        "medium_validation_rows": len(medium_grpo_validation),
        "omni_train_rows": len(omni_train_rows),
        "omni_validation_rows": len(omni_validation_rows),
        "combined_train_rows": len(combined_train),
        "combined_validation_rows": len(combined_validation),
        "combined_train_family_counts": dict(Counter(_family(row) for row in combined_train)),
        "combined_validation_family_counts": dict(
            Counter(_family(row) for row in combined_validation)
        ),
        "medium_train_source_counts": dict(
            Counter(_source(row) for row in medium_grpo_train)
        ),
        "medium_validation_source_counts": dict(
            Counter(_source(row) for row in medium_grpo_validation)
        ),
        "medium_train_difficulty_counts": dict(
            Counter(str(_difficulty(row)) for row in medium_grpo_train)
        ),
        "medium_validation_difficulty_counts": dict(
            Counter(str(_difficulty(row)) for row in medium_grpo_validation)
        ),
        "train_validation_overlap": len(train_ids & validation_ids),
        "medium_pool_validation_overlap": len(set(medium) & set(pool_val)),
        "medium_pool_test_overlap": len(set(medium) & set(pool_test_rows)),
        "private_implementation_leakage": False,
        "evaluation_policy": (
            "Omni-17 remains the primary apples-to-apples development evaluation; "
            "the stratified medium holdout is a secondary evaluation. Official and "
            "internal source test sets remain untouched."
        ),
    }
    return {
        "medium_private_train": medium_private_train,
        "medium_private_validation": medium_private_validation,
        "medium_grpo_train": medium_grpo_train,
        "medium_grpo_validation": medium_grpo_validation,
        "combined_grpo_train": combined_train,
        "combined_grpo_validation": combined_validation,
        "audit": audit,
    }


def stratified_split(
    rows: Sequence[Dict[str, Any]],
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    target = int(math.floor(len(rows) * validation_fraction + 0.5))
    groups: Dict[tuple[str, str], list[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(_source(row), str(_difficulty(row)))].append(row)
    for values in groups.values():
        values.sort(key=lambda row: _stable_hash(str(row["id"]), seed))

    exact = {key: target * len(values) / len(rows) for key, values in groups.items()}
    quotas = {key: int(math.floor(value)) for key, value in exact.items()}
    remaining = target - sum(quotas.values())
    for key in sorted(groups, key=lambda value: (-(exact[value] - quotas[value]), str(value))):
        if remaining == 0:
            break
        if quotas[key] < len(groups[key]):
            quotas[key] += 1
            remaining -= 1
    if remaining:
        raise AssertionError(f"Could not allocate {remaining} validation rows")

    train = []
    validation = []
    for key, values in groups.items():
        validation.extend(values[: quotas[key]])
        train.extend(values[quotas[key] :])
    train.sort(key=lambda row: str(row["id"]))
    validation.sort(key=lambda row: str(row["id"]))
    if len(validation) != target or len(train) + len(validation) != len(rows):
        raise AssertionError("Stratified split produced inconsistent counts")
    return train, validation


def _validate_verifier_record(row: Mapping[str, Any], *, require_private: bool) -> None:
    row_id = str(row.get("id", "")).strip()
    text = str(row.get("text", row.get("problem", ""))).strip()
    answer = str(row.get("gold_answer", "")).strip()
    metadata = row.get("metadata")
    if not row_id or not text or not answer or not isinstance(metadata, dict):
        raise ValueError(f"Invalid verifier record {row_id!r}")
    public_data = metadata.get("function_graph")
    hidden_data = metadata.get("verification_graph")
    if not isinstance(public_data, dict) or not isinstance(hidden_data, dict):
        raise ValueError(f"Verifier record {row_id!r} lacks public/hidden graph")
    public_graph = FunctionGraph.from_dict(public_data)
    hidden_graph = FunctionGraph.from_dict(hidden_data)
    validate_function_graph(public_graph)
    validate_function_graph(hidden_graph)
    if public_graph.problem_id != row_id or hidden_graph.problem_id != row_id:
        raise ValueError(f"Verifier graph ID mismatch for {row_id!r}")
    if [node.id for node in public_graph.nodes] != [node.id for node in hidden_graph.nodes]:
        raise ValueError(f"Public/hidden node mismatch for {row_id!r}")
    if require_private and not isinstance(metadata.get("verifier_implementations"), dict):
        raise ValueError(f"Private medium record {row_id!r} lacks implementations")


def _with_split(row: Mapping[str, Any], split: str) -> Dict[str, Any]:
    result = deepcopy(dict(row))
    result["split"] = split
    return result


def _source(row: Mapping[str, Any]) -> str:
    metadata = row.get("metadata", {})
    return str(row.get("source") or (metadata.get("source") if isinstance(metadata, dict) else "unknown"))


def _difficulty(row: Mapping[str, Any]) -> Any:
    metadata = row.get("metadata", {})
    return row.get("difficulty", metadata.get("difficulty") if isinstance(metadata, dict) else None)


def _family(row: Mapping[str, Any]) -> str:
    return "medium" if str(row.get("id", "")).startswith("medium_") else "omni"


def _stable_hash(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def _unique_by_id(rows: Sequence[Dict[str, Any]], label: str) -> Dict[str, Dict[str, Any]]:
    result = {}
    for row in rows:
        row_id = str(row.get("id", "")).strip()
        if not row_id or row_id in result:
            raise ValueError(f"Missing or duplicate ID in {label}: {row_id!r}")
        result[row_id] = row
    return result


def _require_disjoint(left_name: str, left: set[str], right_name: str, right: set[str]) -> None:
    overlap = left & right
    if overlap:
        raise ValueError(
            f"{left_name}/{right_name} leakage: count={len(overlap)} "
            f"examples={sorted(overlap)[:10]}"
        )


def _contains_forbidden_key(value: Any, forbidden: set[str]) -> bool:
    if isinstance(value, dict):
        return bool(set(value) & forbidden) or any(
            _contains_forbidden_key(child, forbidden) for child in value.values()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_key(child, forbidden) for child in value)
    return False


def _read_jsonl(path: Path) -> list[Dict[str, Any]]:
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
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


if __name__ == "__main__":
    main()

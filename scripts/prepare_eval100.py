#!/usr/bin/env python3
"""Build a leakage-audited 100-problem executable evaluation set."""

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
    "mutant_code",
    "reference_code",
    "reference_solution",
    "tagged_solution",
    "verifier_implementations",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    select = subparsers.add_parser("select")
    select.add_argument("--pool-test", required=True)
    select.add_argument("--combined-train", required=True)
    select.add_argument("--combined-validation", required=True)
    select.add_argument("--output", required=True)
    select.add_argument("--candidate-count", type=int, default=12)
    select.add_argument("--seed", type=int, default=3030)

    candidates = subparsers.add_parser("build-candidates")
    candidates.add_argument("--teacher-master", required=True)
    candidates.add_argument("--output", required=True)

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--base-validation", required=True)
    finalize.add_argument("--extra-accepted", required=True)
    finalize.add_argument("--combined-train", required=True)
    finalize.add_argument("--output", required=True)
    finalize.add_argument("--audit-output", required=True)
    finalize.add_argument("--target-size", type=int, default=100)
    finalize.add_argument("--expected-omni", type=int, default=17)
    finalize.add_argument("--expected-existing-medium", type=int, default=79)
    finalize.add_argument("--seed", type=int, default=3030)

    args = parser.parse_args()
    if args.command == "select":
        rows = select_extra_candidates(
            _read_jsonl(Path(args.pool_test)),
            _read_jsonl(Path(args.combined_train)),
            _read_jsonl(Path(args.combined_validation)),
            count=args.candidate_count,
            seed=args.seed,
        )
        _write_jsonl(Path(args.output), rows)
        print(json.dumps(_selection_summary(rows), ensure_ascii=False, indent=2))
    elif args.command == "build-candidates":
        rows = build_verifier_candidates(
            _read_jsonl(Path(args.teacher_master))
        )
        _write_jsonl(Path(args.output), rows)
        print(json.dumps(_selection_summary(rows), ensure_ascii=False, indent=2))
    else:
        rows, audit = finalize_eval_set(
            _read_jsonl(Path(args.base_validation)),
            _read_jsonl(Path(args.extra_accepted)),
            _read_jsonl(Path(args.combined_train)),
            target_size=args.target_size,
            expected_omni=args.expected_omni,
            expected_existing_medium=args.expected_existing_medium,
            seed=args.seed,
        )
        _write_jsonl(Path(args.output), rows)
        audit["output"] = str(Path(args.output).expanduser().resolve())
        _write_json(Path(args.audit_output), audit)
        print(json.dumps(audit, ensure_ascii=False, indent=2))


def select_extra_candidates(
    pool_test: Sequence[Dict[str, Any]],
    combined_train: Sequence[Dict[str, Any]],
    combined_validation: Sequence[Dict[str, Any]],
    *,
    count: int,
    seed: int,
) -> list[Dict[str, Any]]:
    if count < 1:
        raise ValueError("candidate count must be positive")
    train_ids = _ids(combined_train, "combined train")
    validation_ids = _ids(combined_validation, "combined validation")
    _require_disjoint("combined train", train_ids, "combined validation", validation_ids)
    eligible = []
    seen = set()
    for row in pool_test:
        row_id = str(row.get("id", "")).strip()
        if not row_id or row_id in seen:
            raise ValueError(f"Missing or duplicate pool-test ID: {row_id!r}")
        seen.add(row_id)
        if row_id in train_ids or row_id in validation_ids:
            continue
        if str(row.get("split", "")) != "test":
            raise ValueError(f"Pool-test record {row_id!r} is not split=test")
        eligible.append(deepcopy(row))
    if len(eligible) < count:
        raise ValueError(f"Only {len(eligible)} unseen pool-test rows for {count} candidates")
    return _stratified_select(eligible, count=count, seed=seed)


def build_verifier_candidates(
    masters: Sequence[Dict[str, Any]],
) -> list[Dict[str, Any]]:
    result = []
    for master in masters:
        row_id = str(master.get("id", "")).strip()
        graph_data = master.get("function_graph")
        if not row_id or not isinstance(graph_data, dict):
            raise ValueError(f"Invalid teacher master {row_id!r}")
        graph = FunctionGraph.from_dict(graph_data)
        validate_function_graph(graph)
        if graph.problem_id != row_id:
            raise ValueError(f"Graph ID mismatch for {row_id!r}")
        text = str(master.get("problem", "")).strip()
        answer = str(master.get("gold_answer", "")).strip()
        if not text or not answer:
            raise ValueError(f"Teacher master {row_id!r} lacks problem/gold answer")
        candidate = {
            "id": row_id,
            "text": text,
            "gold_answer": answer,
            "split": "test",
            "source": master.get("source"),
            "difficulty": master.get("difficulty"),
            "metadata": {
                "source": master.get("source"),
                "difficulty": master.get("difficulty"),
                "domain": list(master.get("domain", [])),
                "function_graph": graph.to_dict(),
                "evaluation_origin": "medium_internal_test",
            },
        }
        if _contains_forbidden_key(candidate, PRIVATE_KEYS):
            raise ValueError(f"Verifier candidate {row_id!r} leaks private labels")
        result.append(candidate)
    return sorted(result, key=lambda row: str(row["id"]))


def finalize_eval_set(
    base_validation: Sequence[Dict[str, Any]],
    extra_accepted: Sequence[Dict[str, Any]],
    combined_train: Sequence[Dict[str, Any]],
    *,
    target_size: int,
    expected_omni: int,
    expected_existing_medium: int,
    seed: int,
) -> tuple[list[Dict[str, Any]], Dict[str, Any]]:
    base = _unique(base_validation, "base validation")
    accepted = _unique(extra_accepted, "extra accepted")
    train = _unique(combined_train, "combined train")
    base_omni = sum(_family(row) == "omni" for row in base.values())
    base_medium = sum(_family(row) == "medium" for row in base.values())
    if base_omni != expected_omni or base_medium != expected_existing_medium:
        raise ValueError(
            "Unexpected base validation composition: "
            f"omni={base_omni}, medium={base_medium}"
        )
    _require_disjoint("base validation", set(base), "combined train", set(train))
    _require_disjoint("extra accepted", set(accepted), "combined train", set(train))
    _require_disjoint("extra accepted", set(accepted), "base validation", set(base))

    extra_count = target_size - len(base)
    if extra_count < 0:
        raise ValueError("target size is smaller than base validation")
    if len(accepted) < extra_count:
        raise ValueError(
            f"Need {extra_count} accepted extras, only {len(accepted)} are available"
        )
    selected = _stratified_select(
        list(accepted.values()),
        count=extra_count,
        seed=seed,
    )
    extras = []
    for row in selected:
        _validate_executable_record(row)
        public = strip_private_implementations(row)
        public["split"] = "test"
        metadata = dict(public.get("metadata", {}))
        metadata["evaluation_origin"] = "medium_internal_test"
        public["metadata"] = metadata
        extras.append(public)
    rows = sorted(
        [deepcopy(row) for row in base.values()] + extras,
        key=lambda row: str(row["id"]),
    )
    if len(rows) != target_size:
        raise AssertionError(f"Expected {target_size} rows, got {len(rows)}")
    if _contains_forbidden_key(rows, PRIVATE_KEYS):
        raise ValueError("Final evaluation set leaks private teacher implementations")
    output_ids = {str(row["id"]) for row in rows}
    overlap = output_ids & set(train)
    if overlap:
        raise ValueError(f"Final evaluation/train leakage: {sorted(overlap)[:10]}")
    for row in rows:
        _validate_executable_record(row)

    audit = {
        "target_rows": target_size,
        "actual_rows": len(rows),
        "existing_validation_rows": len(base),
        "new_internal_test_rows": len(extras),
        "family_counts": dict(Counter(_family(row) for row in rows)),
        "source_counts": dict(Counter(_source(row) for row in rows)),
        "difficulty_counts": dict(Counter(str(_difficulty(row)) for row in rows)),
        "selected_extra_ids": [str(row["id"]) for row in extras],
        "train_overlap": len(overlap),
        "private_implementation_leakage": False,
        "selection_seed": seed,
        "policy": (
            "17 Omni executable validation + 79 existing Medium verifier validation + "
            "4 newly annotated Medium internal-test records. No official evaluation split "
            "or training record is used to construct model updates."
        ),
    }
    return rows, audit


def _stratified_select(
    rows: Sequence[Dict[str, Any]], *, count: int, seed: int
) -> list[Dict[str, Any]]:
    if count == 0:
        return []
    groups: Dict[str, list[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[_source(row)].append(row)
    for source_rows in groups.values():
        source_rows.sort(
            key=lambda row: (
                _stable_hash(f"{row.get('id')}:{_difficulty(row)}", seed),
                str(row.get("id")),
            )
        )
    exact = {key: count * len(values) / len(rows) for key, values in groups.items()}
    quotas = {key: min(len(groups[key]), int(math.floor(value))) for key, value in exact.items()}
    remaining = count - sum(quotas.values())
    for key in sorted(groups, key=lambda item: (-(exact[item] - quotas[item]), item)):
        if remaining == 0:
            break
        if quotas[key] < len(groups[key]):
            quotas[key] += 1
            remaining -= 1
    if remaining:
        raise ValueError(f"Could not allocate {remaining} stratified rows")
    selected = [row for key, values in groups.items() for row in values[: quotas[key]]]
    return sorted(selected, key=lambda row: _stable_hash(str(row.get("id")), seed))


def _validate_executable_record(row: Mapping[str, Any]) -> None:
    row_id = str(row.get("id", "")).strip()
    metadata = row.get("metadata")
    if not row_id or not isinstance(metadata, dict):
        raise ValueError(f"Invalid executable record {row_id!r}")
    public_data = metadata.get("function_graph")
    hidden_data = metadata.get("verification_graph")
    if not isinstance(public_data, dict) or not isinstance(hidden_data, dict):
        raise ValueError(f"Executable record {row_id!r} lacks public/hidden graph")
    public = FunctionGraph.from_dict(public_data)
    hidden = FunctionGraph.from_dict(hidden_data)
    validate_function_graph(public)
    validate_function_graph(hidden)
    if public.problem_id != row_id or hidden.problem_id != row_id:
        raise ValueError(f"Executable graph ID mismatch for {row_id!r}")
    if [node.id for node in public.nodes] != [node.id for node in hidden.nodes]:
        raise ValueError(f"Public/hidden nodes differ for {row_id!r}")


def _selection_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        "rows": len(rows),
        "source_counts": dict(Counter(_source(row) for row in rows)),
        "difficulty_counts": dict(Counter(str(_difficulty(row)) for row in rows)),
        "ids": [str(row.get("id")) for row in rows],
    }


def _family(row: Mapping[str, Any]) -> str:
    return "medium" if str(row.get("id", "")).startswith("medium_") else "omni"


def _source(row: Mapping[str, Any]) -> str:
    metadata = row.get("metadata", {})
    return str(
        row.get("source")
        or (metadata.get("source") if isinstance(metadata, dict) else None)
        or "unknown"
    )


def _difficulty(row: Mapping[str, Any]) -> Any:
    metadata = row.get("metadata", {})
    return row.get(
        "difficulty",
        metadata.get("difficulty") if isinstance(metadata, dict) else None,
    )


def _stable_hash(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def _ids(rows: Sequence[Mapping[str, Any]], label: str) -> set[str]:
    return set(_unique(rows, label))


def _unique(
    rows: Sequence[Dict[str, Any]], label: str
) -> Dict[str, Dict[str, Any]]:
    result = {}
    for row in rows:
        row_id = str(row.get("id", "")).strip()
        if not row_id or row_id in result:
            raise ValueError(f"Missing or duplicate ID in {label}: {row_id!r}")
        result[row_id] = row
    return result


def _require_disjoint(
    left_name: str, left: set[str], right_name: str, right: set[str]
) -> None:
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


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


if __name__ == "__main__":
    main()

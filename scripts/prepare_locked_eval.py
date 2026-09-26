#!/usr/bin/env python3
"""Select, finalize, and cryptographically lock an executable evaluation set."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import unicodedata
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
    select.add_argument("--medium-pool", action="append", required=True)
    select.add_argument("--omni-holdout", required=True)
    select.add_argument("--exclude-jsonl", action="append", default=[])
    select.add_argument("--existing-eval", action="append", default=[])
    select.add_argument("--output", required=True)
    select.add_argument("--audit-output", required=True)
    select.add_argument("--medium-candidates", type=int, default=370)
    select.add_argument("--omni-candidates", type=int, default=220)
    select.add_argument("--seed", type=int, default=60906)

    candidates = subparsers.add_parser("build-candidates")
    candidates.add_argument("--teacher-master", required=True)
    candidates.add_argument("--output", required=True)

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--selected-raw", required=True)
    finalize.add_argument("--accepted", required=True)
    finalize.add_argument("--exclude-jsonl", action="append", default=[])
    finalize.add_argument("--existing-eval", action="append", default=[])
    finalize.add_argument("--output", required=True)
    finalize.add_argument("--audit-output", required=True)
    finalize.add_argument("--medium-target", type=int, default=320)
    finalize.add_argument("--omni-target", type=int, default=80)
    finalize.add_argument("--seed", type=int, default=60906)

    args = parser.parse_args()
    if args.command == "select":
        medium = [
            row
            for path in args.medium_pool
            for row in _read_jsonl(Path(path))
        ]
        rows, audit = select_candidates(
            medium,
            _read_jsonl(Path(args.omni_holdout)),
            _read_many(args.exclude_jsonl),
            _read_many(args.existing_eval),
            medium_count=args.medium_candidates,
            omni_count=args.omni_candidates,
            seed=args.seed,
        )
        output = Path(args.output)
        _write_jsonl(output, rows)
        audit["candidate_jsonl"] = str(output.expanduser().resolve())
        audit["candidate_sha256"] = _sha256_file(output)
        _write_json(Path(args.audit_output), audit)
        print(json.dumps(audit, ensure_ascii=False, indent=2))
    elif args.command == "build-candidates":
        rows = build_verifier_candidates(
            _read_jsonl(Path(args.teacher_master))
        )
        _write_jsonl(Path(args.output), rows)
        print(json.dumps(_row_summary(rows), ensure_ascii=False, indent=2))
    else:
        rows, audit = finalize_locked_eval(
            _read_jsonl(Path(args.selected_raw)),
            _read_jsonl(Path(args.accepted)),
            _read_many(args.exclude_jsonl),
            _read_many(args.existing_eval),
            medium_target=args.medium_target,
            omni_target=args.omni_target,
            seed=args.seed,
        )
        output = Path(args.output)
        _write_jsonl(output, rows)
        audit.update(
            {
                "locked_jsonl": str(output.expanduser().resolve()),
                "locked_jsonl_sha256": _sha256_file(output),
                "locked_ids_sha256": _sha256_text(
                    "\n".join(str(row["id"]) for row in rows) + "\n"
                ),
            }
        )
        _write_json(Path(args.audit_output), audit)
        print(json.dumps(audit, ensure_ascii=False, indent=2))


def select_candidates(
    medium_rows: Sequence[Dict[str, Any]],
    omni_rows: Sequence[Dict[str, Any]],
    training_rows: Sequence[Dict[str, Any]],
    existing_eval_rows: Sequence[Dict[str, Any]],
    *,
    medium_count: int,
    omni_count: int,
    seed: int,
) -> tuple[list[Dict[str, Any]], Dict[str, Any]]:
    if medium_count < 1 or omni_count < 1:
        raise ValueError("Candidate counts must be positive")
    medium, medium_duplicate_rows, medium_conflicting_ids = _unique_pool(
        medium_rows, "medium pools"
    )
    omni, omni_duplicate_rows, omni_conflicting_ids = _unique_pool(
        omni_rows, "Omni holdout"
    )
    training_ids, training_texts = _identity_sets(training_rows)
    prior_eval_ids, prior_eval_texts = _identity_sets(existing_eval_rows)
    forbidden_ids = training_ids | prior_eval_ids
    forbidden_texts = training_texts | prior_eval_texts

    eligible_medium = []
    for row in medium.values():
        if _family(row) != "medium":
            raise ValueError(f"Non-medium row in medium pool: {row.get('id')!r}")
        split = str(row.get("split", ""))
        if split not in {"validation", "test"}:
            raise ValueError(
                f"Medium candidate {row['id']!r} is not an untouched validation/test row"
            )
        if not _is_forbidden(row, forbidden_ids, forbidden_texts):
            eligible_medium.append(row)

    eligible_omni = []
    for row in omni.values():
        if _family(row) != "omni":
            raise ValueError(f"Non-Omni row in Omni holdout: {row.get('id')!r}")
        if not _is_forbidden(row, forbidden_ids, forbidden_texts):
            eligible_omni.append(row)

    if len(eligible_medium) < medium_count:
        raise ValueError(
            f"Only {len(eligible_medium)} eligible medium rows for {medium_count} candidates"
        )
    if len(eligible_omni) < omni_count:
        raise ValueError(
            f"Only {len(eligible_omni)} eligible Omni rows for {omni_count} candidates"
        )

    selected_medium = _stratified_select(
        eligible_medium,
        count=medium_count,
        seed=seed,
        group_fields=("source", "difficulty"),
    )
    selected_omni = _stratified_select(
        eligible_omni,
        count=omni_count,
        seed=seed + 1,
        group_fields=("difficulty", "domain"),
    )
    selected = []
    for row in selected_medium + selected_omni:
        value = deepcopy(row)
        value["evaluation_candidate_origin"] = (
            "medium_internal_holdout"
            if _family(value) == "medium"
            else "omni_math_80pct_holdout"
        )
        selected.append(value)
    selected.sort(key=lambda row: str(row["id"]))

    selected_ids, selected_texts = _identity_sets(selected)
    if selected_ids & forbidden_ids or selected_texts & forbidden_texts:
        raise AssertionError("Candidate selection leaked a forbidden training/eval row")
    audit = {
        "stage": "candidate_selection",
        "selection_seed": seed,
        "medium_pool_rows": len(medium),
        "omni_holdout_rows": len(omni),
        "medium_duplicate_rows_removed": medium_duplicate_rows,
        "omni_duplicate_rows_removed": omni_duplicate_rows,
        "medium_conflicting_ids_removed": medium_conflicting_ids,
        "omni_conflicting_ids_removed": omni_conflicting_ids,
        "eligible_medium_rows": len(eligible_medium),
        "eligible_omni_rows": len(eligible_omni),
        "selected_medium_candidates": len(selected_medium),
        "selected_omni_candidates": len(selected_omni),
        "selected_total": len(selected),
        "excluded_training_ids": len(training_ids),
        "excluded_existing_eval_ids": len(prior_eval_ids),
        "id_overlap_with_training_or_prior_eval": 0,
        "text_overlap_with_training_or_prior_eval": 0,
        "source_counts": dict(Counter(_source(row) for row in selected)),
        "difficulty_counts": dict(
            Counter(str(_difficulty(row)) for row in selected)
        ),
        "policy": (
            "Candidates come only from untouched medium internal validation/test and "
            "the recorded Omni-MATH 80% holdout. Existing Eval100 and every supplied "
            "training JSONL are excluded by both problem ID and normalized problem text."
        ),
    }
    return selected, audit


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
        text = _problem_text(master)
        answer = str(master.get("gold_answer", "")).strip()
        if not text or not answer:
            raise ValueError(f"Teacher master {row_id!r} lacks problem/gold answer")
        candidate = {
            "id": row_id,
            "text": text,
            "gold_answer": answer,
            "split": "locked_test_candidate",
            "source": master.get("source"),
            "difficulty": master.get("difficulty"),
            "metadata": {
                "source": master.get("source"),
                "difficulty": master.get("difficulty"),
                "domain": list(master.get("domain", [])),
                "function_graph": graph.to_dict(),
                "evaluation_origin": master.get(
                    "evaluation_candidate_origin", "locked_holdout"
                ),
            },
        }
        if _contains_forbidden_key(candidate, PRIVATE_KEYS):
            raise ValueError(f"Verifier candidate {row_id!r} leaks private labels")
        result.append(candidate)
    return sorted(result, key=lambda row: str(row["id"]))


def finalize_locked_eval(
    selected_raw: Sequence[Dict[str, Any]],
    accepted_rows: Sequence[Dict[str, Any]],
    training_rows: Sequence[Dict[str, Any]],
    existing_eval_rows: Sequence[Dict[str, Any]],
    *,
    medium_target: int,
    omni_target: int,
    seed: int,
) -> tuple[list[Dict[str, Any]], Dict[str, Any]]:
    if medium_target < 1 or omni_target < 1:
        raise ValueError("Locked evaluation family targets must be positive")
    raw = _unique(selected_raw, "selected raw candidates")
    accepted = _unique(accepted_rows, "accepted verifier rows")
    unexpected = set(accepted) - set(raw)
    if unexpected:
        raise ValueError(
            f"Accepted verifier rows were not selected candidates: {sorted(unexpected)[:10]}"
        )
    training_ids, training_texts = _identity_sets(training_rows)
    prior_eval_ids, prior_eval_texts = _identity_sets(existing_eval_rows)
    forbidden_ids = training_ids | prior_eval_ids
    forbidden_texts = training_texts | prior_eval_texts

    eligible = []
    for row_id, row in accepted.items():
        _validate_executable_record(row)
        if _family(raw[row_id]) != _family(row):
            raise ValueError(f"Family changed during annotation for {row_id!r}")
        if _is_forbidden(row, forbidden_ids, forbidden_texts):
            raise ValueError(f"Accepted row overlaps training/prior evaluation: {row_id!r}")
        eligible.append(row)

    medium = [row for row in eligible if _family(row) == "medium"]
    omni = [row for row in eligible if _family(row) == "omni"]
    if len(medium) < medium_target or len(omni) < omni_target:
        raise ValueError(
            "Not enough execution-validated rows to lock Eval set: "
            f"medium={len(medium)}/{medium_target}, omni={len(omni)}/{omni_target}. "
            "Generate more candidates and rerun finalize."
        )
    selected = _stratified_select(
        medium,
        count=medium_target,
        seed=seed + 2,
        group_fields=("source", "difficulty"),
    ) + _stratified_select(
        omni,
        count=omni_target,
        seed=seed + 3,
        group_fields=("difficulty", "domain"),
    )

    rows = []
    for row in selected:
        public = strip_private_implementations(row)
        public.pop("reference_solution", None)
        public.pop("tagged_solution", None)
        public["split"] = "locked_test"
        metadata = dict(public.get("metadata", {}))
        metadata["locked_evaluation"] = True
        metadata["do_not_train"] = True
        public["metadata"] = metadata
        if _contains_forbidden_key(public, PRIVATE_KEYS):
            raise ValueError(f"Locked row {public.get('id')!r} leaks private code/labels")
        _validate_executable_record(public)
        rows.append(public)
    rows.sort(key=lambda row: str(row["id"]))

    output_ids, output_texts = _identity_sets(rows)
    id_overlap = output_ids & forbidden_ids
    text_overlap = output_texts & forbidden_texts
    if id_overlap or text_overlap:
        raise ValueError(
            "Locked evaluation leakage detected: "
            f"id_overlap={len(id_overlap)} text_overlap={len(text_overlap)}"
        )
    expected = medium_target + omni_target
    if len(rows) != expected or len(output_ids) != expected:
        raise AssertionError(f"Expected {expected} unique rows, got {len(rows)}")

    audit = {
        "stage": "locked_final",
        "locked": True,
        "do_not_train": True,
        "selection_seed": seed,
        "target_rows": expected,
        "actual_rows": len(rows),
        "medium_target": medium_target,
        "omni_target": omni_target,
        "accepted_available": len(eligible),
        "accepted_medium_available": len(medium),
        "accepted_omni_available": len(omni),
        "family_counts": dict(Counter(_family(row) for row in rows)),
        "source_counts": dict(Counter(_source(row) for row in rows)),
        "difficulty_counts": dict(
            Counter(str(_difficulty(row)) for row in rows)
        ),
        "train_id_overlap": len(output_ids & training_ids),
        "train_text_overlap": len(output_texts & training_texts),
        "prior_eval_id_overlap": len(output_ids & prior_eval_ids),
        "prior_eval_text_overlap": len(output_texts & prior_eval_texts),
        "private_implementation_leakage": False,
        "evaluation_protocol": (
            "Teacher-free deterministic greedy decoding. Public function_graph is visible "
            "to the policy; verification_graph, hidden tests, target_call, gold answer, and "
            "private implementations are unavailable during generation."
        ),
        "reporting_scope": (
            "This is a custom executable FSG evaluation set, not an official Omni-MATH, "
            "GSM8K, MATH, or MathQA benchmark score."
        ),
    }
    return rows, audit


def _stratified_select(
    rows: Sequence[Dict[str, Any]],
    *,
    count: int,
    seed: int,
    group_fields: Sequence[str],
) -> list[Dict[str, Any]]:
    if count < 0 or count > len(rows):
        raise ValueError(f"Cannot select {count} rows from {len(rows)}")
    groups: Dict[tuple[str, ...], list[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        values = []
        for field in group_fields:
            if field == "source":
                value: Any = _source(row)
            elif field == "difficulty":
                value = _difficulty(row)
            elif field == "domain":
                value = _domain_key(row)
            else:
                value = row.get(field)
            values.append(str(value))
        groups[tuple(values)].append(row)
    for values in groups.values():
        values.sort(key=lambda row: (_stable_hash(str(row["id"]), seed), str(row["id"])))
    exact = {key: count * len(values) / len(rows) for key, values in groups.items()}
    quotas = {key: min(len(groups[key]), int(value)) for key, value in exact.items()}
    remaining = count - sum(quotas.values())
    while remaining:
        candidates = [key for key in groups if quotas[key] < len(groups[key])]
        if not candidates:
            raise AssertionError(f"Could not allocate {remaining} stratified rows")
        key = min(
            candidates,
            key=lambda item: (-(exact[item] - quotas[item]), str(item)),
        )
        quotas[key] += 1
        remaining -= 1
    selected = [row for key, values in groups.items() for row in values[: quotas[key]]]
    return sorted(selected, key=lambda row: _stable_hash(str(row["id"]), seed + 17))


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


def _identity_sets(rows: Sequence[Mapping[str, Any]]) -> tuple[set[str], set[str]]:
    ids = set()
    texts = set()
    for row in rows:
        ids.update(_possible_problem_ids(row))
        text = _problem_text(row)
        if text:
            texts.add(_normalized_problem_hash(text))
    return ids, texts


def _possible_problem_ids(row: Mapping[str, Any]) -> set[str]:
    values = set()
    for value in (
        row.get("id"),
        row.get("problem_id"),
        row.get("source_problem_id"),
    ):
        text = str(value or "").strip()
        if text:
            values.add(text)
            values.add(text.split(":", 1)[0])
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        for key in ("id", "problem_id", "source_problem_id"):
            text = str(metadata.get(key, "")).strip()
            if text:
                values.add(text)
                values.add(text.split(":", 1)[0])
    return values


def _problem_text(row: Mapping[str, Any]) -> str:
    for key in ("problem", "text"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        for key in ("problem", "text"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    messages = row.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict) or message.get("role") != "user":
                continue
            content = message.get("content")
            if not isinstance(content, str):
                continue
            try:
                payload = json.loads(content)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                for key in ("problem", "text"):
                    value = payload.get(key)
                    if isinstance(value, str) and value.strip():
                        return value.strip()
    return ""


def _is_forbidden(
    row: Mapping[str, Any], forbidden_ids: set[str], forbidden_texts: set[str]
) -> bool:
    if _possible_problem_ids(row) & forbidden_ids:
        return True
    text = _problem_text(row)
    return bool(text and _normalized_problem_hash(text) in forbidden_texts)


def _normalized_problem_hash(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).lower()
    normalized = re.sub(r"\W+", "", normalized, flags=re.UNICODE)
    return _sha256_text(normalized)


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


def _unique_pool(
    rows: Sequence[Dict[str, Any]], label: str
) -> tuple[Dict[str, Dict[str, Any]], int, int]:
    """Deduplicate consistent rows and remove every ambiguous duplicate ID."""

    result: Dict[str, Dict[str, Any]] = {}
    duplicates = 0
    conflicting_ids = set()

    for row in rows:
        row_id = str(row.get("id", "")).strip()

        if not row_id:
            raise ValueError(f"Missing ID in {label}")

        if row_id in conflicting_ids:
            continue

        if row_id not in result:
            result[row_id] = row
            continue

        previous = result[row_id]

        same_problem = (
            _normalized_problem_hash(_problem_text(previous))
            == _normalized_problem_hash(_problem_text(row))
        )
        same_answer = (
            str(previous.get("gold_answer", "")).strip()
            == str(row.get("gold_answer", "")).strip()
        )

        if not same_problem or not same_answer:
            result.pop(row_id, None)
            conflicting_ids.add(row_id)
            continue

        duplicates += 1

    return result, duplicates, len(conflicting_ids)


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


def _domain_key(row: Mapping[str, Any]) -> str:
    metadata = row.get("metadata", {})
    value = row.get(
        "domain", metadata.get("domain", []) if isinstance(metadata, dict) else []
    )
    if isinstance(value, list):
        return "/".join(str(item) for item in value[:3])
    return str(value)


def _row_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        "rows": len(rows),
        "family_counts": dict(Counter(_family(row) for row in rows)),
        "source_counts": dict(Counter(_source(row) for row in rows)),
        "difficulty_counts": dict(Counter(str(_difficulty(row)) for row in rows)),
    }


def _contains_forbidden_key(value: Any, forbidden: set[str]) -> bool:
    if isinstance(value, dict):
        return bool(set(value) & forbidden) or any(
            _contains_forbidden_key(child, forbidden) for child in value.values()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_key(child, forbidden) for child in value)
    return False


def _read_many(paths: Sequence[str]) -> list[Dict[str, Any]]:
    return [row for path in paths for row in _read_jsonl(Path(path))]


def _read_jsonl(path: Path) -> list[Dict[str, Any]]:
    path = path.expanduser().resolve()
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
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _stable_hash(value: str, seed: int) -> str:
    return _sha256_text(f"{seed}:{value}")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.expanduser().resolve().open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()

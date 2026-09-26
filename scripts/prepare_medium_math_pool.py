#!/usr/bin/env python3
"""Build a leakage-audited medium-difficulty math candidate pool."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Callable, Dict, Iterable, List, Sequence


MATH_CONFIGS = (
    "algebra",
    "counting_and_probability",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gsm8k-count", type=int, default=600)
    parser.add_argument("--math-count", type=int, default=800)
    parser.add_argument("--mathqa-count", type=int, default=600)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--test-fraction", type=float, default=0.10)
    parser.add_argument("--smoke-count", type=int, default=80)
    args = parser.parse_args()
    _validate_fractions(args.validation_fraction, args.test_fraction)

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install the datasets package before preparing data") from exc

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    gsm_rows = load_dataset("openai/gsm8k", "main", split="train")
    gsm_records = [
        value
        for row in gsm_rows
        if (value := canonicalize_gsm8k(dict(row))) is not None
    ]

    math_records: List[Dict[str, Any]] = []
    math_source_rows = 0
    for config_name in MATH_CONFIGS:
        rows = load_dataset(
            "EleutherAI/hendrycks_math",
            config_name,
            split="train",
        )
        math_source_rows += len(rows)
        math_records.extend(
            value
            for row in rows
            if (
                value := canonicalize_math(dict(row), config_name=config_name)
            )
            is not None
        )

    endpoint = os.environ.get(
        "HF_ENDPOINT",
        "https://huggingface.co",
    ).rstrip("/")
    mathqa_revision = (
        "795ca7da22406a2d62cc8874d0f2b427386b68db"
    )
    mathqa_url = (
        f"{endpoint}/datasets/allenai/math_qa/resolve/"
        f"{mathqa_revision}/default/math_qa-train.parquet"
    )
    mathqa_rows = load_dataset(
        "parquet",
        data_files={"train": mathqa_url},
        split="train",
    )
    mathqa_records = [
        value
        for row in mathqa_rows
        if (value := canonicalize_mathqa(dict(row))) is not None
    ]

    selected_by_source = {
        "gsm8k": balanced_select(
            gsm_records,
            args.gsm8k_count,
            seed=args.seed,
            group_key=lambda row: (row["difficulty"],),
        ),
        "math": balanced_select(
            math_records,
            args.math_count,
            seed=args.seed,
            group_key=lambda row: (tuple(row["domain"]), row["difficulty"]),
        ),
        "mathqa": balanced_select(
            mathqa_records,
            args.mathqa_count,
            seed=args.seed,
            group_key=lambda row: (tuple(row["domain"]), row["difficulty"]),
        ),
    }

    selected = deduplicate_records(
        row
        for source_rows in selected_by_source.values()
        for row in source_rows
    )
    selected = assign_internal_splits(
        selected,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    selected.sort(key=lambda row: row["id"])

    partitions = {
        split: [row for row in selected if row["split"] == split]
        for split in ("train", "validation", "test")
    }
    smoke = balanced_select(
        partitions["train"],
        min(args.smoke_count, len(partitions["train"])),
        seed=args.seed + 1,
        group_key=lambda row: (row["source"], tuple(row["domain"])),
    )

    _write_jsonl(output_dir / "medium_math_all.jsonl", selected)
    for split, rows in partitions.items():
        _write_jsonl(output_dir / f"medium_math_{split}.jsonl", rows)
    _write_jsonl(output_dir / "medium_math_smoke.jsonl", smoke)

    manifest = {
        "seed": args.seed,
        "source_policy": "Only official train splits are used; official evaluation splits remain untouched.",
        "sources": {
            "openai/gsm8k": {
                "license": "MIT",
                "source_split": "train",
                "source_rows": len(gsm_rows),
                "eligible_rows": len(gsm_records),
                "selected_rows": len(selected_by_source["gsm8k"]),
                "filter": "3-8 annotated calculation steps and an explicit #### numeric answer",
            },
            "EleutherAI/hendrycks_math": {
                "license": "MIT",
                "source_split": "train",
                "configs": list(MATH_CONFIGS),
                "source_rows": math_source_rows,
                "eligible_rows": len(math_records),
                "selected_rows": len(selected_by_source["math"]),
                "filter": "Level 1-3 with an extractable boxed final answer",
            },
            "allenai/math_qa": {
                "license": "Apache-2.0",
                "source_split": "train",
                "source_rows": len(mathqa_rows),
                "eligible_rows": len(mathqa_records),
                "selected_rows": len(selected_by_source["mathqa"]),
                "filter": "2-5 operation program with an explicit numeric correct option",
            },
        },
        "requested_rows": args.gsm8k_count + args.math_count + args.mathqa_count,
        "deduplicated_rows": len(selected),
        "split_counts": {key: len(value) for key, value in partitions.items()},
        "smoke_rows": len(smoke),
        "source_counts": dict(Counter(row["source"] for row in selected)),
        "difficulty_counts": dict(
            Counter(str(row["difficulty"]) for row in selected)
        ),
        "warning": (
            "Internal validation/test rows must never enter SFT or GRPO training. "
            "Official GSM8K, MATH, and MathQA evaluation splits were not loaded."
        ),
    }
    _write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def canonicalize_gsm8k(row: Dict[str, Any]) -> Dict[str, Any] | None:
    problem = str(row.get("question", "")).strip()
    solution = str(row.get("answer", "")).strip()
    final_match = re.search(r"####\s*(.+?)\s*$", solution)
    step_count = len(re.findall(r"<<.*?>>", solution))
    if not problem or not final_match or not 3 <= step_count <= 8:
        return None
    answer = final_match.group(1).strip().replace(",", "")
    if not _is_explicit_numeric_answer(answer):
        return None
    return _canonical_record(
        source="openai/gsm8k",
        problem=problem,
        reference_solution=solution,
        gold_answer=answer,
        domain=["Arithmetic", "Multi-step Word Problems"],
        difficulty=min(5, max(2, step_count)),
        source_metadata={"calculation_steps": step_count, "source_split": "train"},
    )


def canonicalize_math(
    row: Dict[str, Any], *, config_name: str
) -> Dict[str, Any] | None:
    problem = str(row.get("problem", "")).strip()
    solution = str(row.get("solution", "")).strip()
    level_match = re.search(r"(\d+)", str(row.get("level", "")))
    if not problem or not solution or not level_match:
        return None
    level = int(level_match.group(1))
    if level not in {1, 2, 3}:
        return None
    answer = extract_last_boxed_value(solution)
    if not answer or len(answer) > 160:
        return None
    return _canonical_record(
        source="EleutherAI/hendrycks_math",
        problem=problem,
        reference_solution=solution,
        gold_answer=answer,
        domain=[config_name, str(row.get("type", config_name))],
        difficulty=level,
        source_metadata={
            "config": config_name,
            "level": str(row.get("level", "")),
            "type": str(row.get("type", "")),
            "source_split": "train",
        },
    )


def canonicalize_mathqa(row: Dict[str, Any]) -> Dict[str, Any] | None:
    problem = str(row.get("Problem", "")).strip()
    rationale = str(row.get("Rationale", "")).strip()
    options = str(row.get("options", "")).strip()
    correct = str(row.get("correct", "")).strip().lower()
    formula = str(row.get("linear_formula", "")).strip()
    operations = [part for part in formula.split("|") if part.strip()]
    if not problem or not rationale or not 2 <= len(operations) <= 5:
        return None
    parsed_options = parse_mathqa_options(options)
    answer = parsed_options.get(correct, "").strip()
    if not _is_explicit_numeric_answer(answer) or len(answer) > 96:
        return None
    category = str(row.get("category", "general")).strip() or "general"
    reference_solution = rationale + "\nOperation program: " + formula
    return _canonical_record(
        source="allenai/math_qa",
        problem=problem,
        reference_solution=reference_solution,
        gold_answer=answer,
        domain=["Math Word Problems", category],
        difficulty=min(5, max(2, len(operations))),
        source_metadata={
            "category": category,
            "options": options,
            "correct_option": correct,
            "annotated_formula": str(row.get("annotated_formula", "")),
            "linear_formula": formula,
            "operation_count": len(operations),
            "source_split": "train",
        },
    )


def extract_last_boxed_value(text: str) -> str | None:
    values = []
    for marker in (r"\boxed{", r"\fbox{"):
        cursor = 0
        while True:
            start = text.find(marker, cursor)
            if start < 0:
                break
            value_start = start + len(marker)
            depth = 1
            index = value_start
            while index < len(text) and depth:
                if text[index] == "{":
                    depth += 1
                elif text[index] == "}":
                    depth -= 1
                index += 1
            if depth:
                break
            values.append((start, text[value_start : index - 1].strip()))
            cursor = index
    if not values:
        return None
    return max(values, key=lambda item: item[0])[1]


def parse_mathqa_options(text: str) -> Dict[str, str]:
    pattern = re.compile(r"(?:^|,\s*)([a-eA-E])\s*\)\s*")
    markers = list(pattern.finditer(text))
    result = {}
    for index, marker in enumerate(markers):
        end = markers[index + 1].start() if index + 1 < len(markers) else len(text)
        result[marker.group(1).lower()] = text[marker.end() : end].strip(" ,")
    return result


def balanced_select(
    records: Sequence[Dict[str, Any]],
    count: int,
    *,
    seed: int,
    group_key: Callable[[Dict[str, Any]], Any],
) -> List[Dict[str, Any]]:
    if count < 0:
        raise ValueError("selection count must be non-negative")
    if count > len(records):
        raise ValueError(f"Requested {count} rows from only {len(records)} eligible rows")
    groups: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[group_key(row)].append(row)
    for rows in groups.values():
        rows.sort(key=lambda row: _stable_hash(row["id"], seed))
    positions = {key: 0 for key in groups}
    selected = []
    while len(selected) < count:
        progressed = False
        for key in sorted(groups, key=str):
            position = positions[key]
            if position >= len(groups[key]):
                continue
            selected.append(groups[key][position])
            positions[key] += 1
            progressed = True
            if len(selected) == count:
                break
        if not progressed:
            break
    if len(selected) != count:
        raise AssertionError(f"Selected {len(selected)} rows instead of {count}")
    return selected


def assign_internal_splits(
    records: Sequence[Dict[str, Any]],
    *,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> List[Dict[str, Any]]:
    _validate_fractions(validation_fraction, test_fraction)
    by_source: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_source[row["source"]].append(dict(row))
    result = []
    for source, rows in by_source.items():
        rows.sort(key=lambda row: _stable_hash(row["id"], seed))
        test_count = int(len(rows) * test_fraction + 0.5)
        validation_count = int(len(rows) * validation_fraction + 0.5)
        for index, row in enumerate(rows):
            if index < test_count:
                row["split"] = "test"
            elif index < test_count + validation_count:
                row["split"] = "validation"
            else:
                row["split"] = "train"
            result.append(row)
    return result


def deduplicate_records(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    result = []
    seen = set()
    for row in records:
        normalized = re.sub(r"\W+", "", row["problem"].lower())
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        result.append(dict(row))
    return result


def _canonical_record(
    *,
    source: str,
    problem: str,
    reference_solution: str,
    gold_answer: str,
    domain: List[str],
    difficulty: int,
    source_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    problem_hash = hashlib.sha256(problem.encode("utf-8")).hexdigest()
    source_slug = re.sub(r"[^a-z0-9]+", "_", source.lower()).strip("_")
    return {
        "id": f"medium_{source_slug}_{problem_hash[:16]}",
        "problem": problem,
        "reference_solution": reference_solution,
        "gold_answer": gold_answer,
        "domain": domain,
        "difficulty": difficulty,
        "source": source,
        "source_split": "train",
        "problem_sha256": problem_hash,
        "source_metadata": source_metadata,
    }


def _is_explicit_numeric_answer(value: str) -> bool:
    text = value.strip()
    if not text or not re.search(r"\d", text):
        return False
    lowered = text.lower()
    return not any(
        phrase in lowered
        for phrase in ("none", "cannot", "insufficient", "all of the above")
    )


def _validate_fractions(validation_fraction: float, test_fraction: float) -> None:
    if validation_fraction < 0 or test_fraction < 0:
        raise ValueError("split fractions must be non-negative")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation_fraction + test_fraction must be below one")


def _stable_hash(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def _write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _write_json(path: Path, value: Dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

"""Normalize LeanDojo theorem traces and enrich them for retrieval."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List


def formalize_traced_theorem(record: Dict[str, Any]) -> Dict[str, Any]:
    """Return the exact five-field auditable representation requested by the pipeline."""

    lean_name = str(record.get("lean_name", record.get("full_name", ""))).strip()
    source_file = str(record.get("source_file", record.get("file_path", ""))).strip()
    formal_statement = str(
        record.get("formal_statement", record.get("theorem_statement", ""))
    ).strip()
    if not lean_name:
        raise ValueError("Lean theorem record has no lean_name/full_name")
    if not source_file.endswith(".lean"):
        raise ValueError(f"Lean theorem {lean_name!r} has invalid source_file")
    if not formal_statement:
        raise ValueError(f"Lean theorem {lean_name!r} has no formal statement")
    namespace = str(record.get("namespace", "")).strip()
    if not namespace and "." in lean_name:
        namespace = lean_name.rsplit(".", 1)[0]
    return {
        "lean_name": lean_name,
        "namespace": namespace,
        "source_file": source_file,
        "formal_statement": formal_statement,
        "docstring": record.get("docstring"),
    }


def normalize_traced_theorem(record: Dict[str, Any]) -> Dict[str, Any]:
    """Convert one LeanDojo theorem record into the stable memory schema.

    LeanDojo is the source of truth for names, source positions, statements, and
    proof dependencies.  No language model is involved in this normalization.
    """

    formal = formalize_traced_theorem(record)
    lean_name = formal["lean_name"]
    namespace = formal["namespace"]
    source_file = formal["source_file"]
    formal_statement = formal["formal_statement"]

    premise_ids = sorted(
        premise
        for premise in _premise_names(record)
        if premise and premise != lean_name
    )
    revision = str(
        record.get("source_revision", record.get("commit", ""))
    ).strip()
    memory_id = f"mathlib4:{lean_name}"
    return {
        "memory_id": memory_id,
        "id": memory_id,
        "item_type": "theorem",
        "lean_name": lean_name,
        "namespace": namespace,
        "source_file": source_file,
        "formal_statement": formal_statement,
        "docstring": formal["docstring"],
        "premise_ids": premise_ids,
        "source": "mathlib4",
        "source_url": str(
            record.get(
                "source_url",
                record.get("url", "https://github.com/leanprover-community/mathlib4"),
            )
        ),
        "source_revision": revision,
        "license": "Apache-2.0",
        "keywords": [],
        "text": formal_statement,
        "metadata": {
            "lean_name": lean_name,
            "namespace": namespace,
            "source_file": source_file,
            "source_revision": revision,
            "premise_ids": premise_ids,
        },
    }


def build_enrichment_messages(record: Dict[str, Any]) -> List[Dict[str, str]]:
    payload = {
        "lean_name": record["lean_name"],
        "namespace": record.get("namespace", ""),
        "source_file": record["source_file"],
        "formal_statement": record["formal_statement"],
        "docstring": record.get("docstring"),
    }
    return [
        {"role": "system", "content": _ENRICHMENT_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def merge_enrichment(
    record: Dict[str, Any], enrichment: Dict[str, Any]
) -> Dict[str, Any]:
    """Validate API-produced semantic fields and merge them with Lean truth."""

    title = str(enrichment.get("title", "")).strip()
    informal = str(enrichment.get("informal_statement", "")).strip()
    if not title or not informal:
        raise ValueError("Theorem enrichment needs title and informal_statement")

    domain_path = _string_list(enrichment.get("domain_path"), "domain_path")
    keywords = _string_list(enrichment.get("keywords"), "keywords")
    preconditions = enrichment.get("preconditions", [])
    if not isinstance(preconditions, list):
        raise ValueError("Theorem enrichment preconditions must be a list")

    merged = dict(record)
    merged.update(
        {
            "title": title,
            "informal_statement": informal,
            "domain_path": domain_path,
            "keywords": keywords,
            "preconditions": preconditions,
        }
    )
    merged["text"] = "\n".join(
        part
        for part in (
            title,
            informal,
            "Formal Lean statement: " + str(record["formal_statement"]),
        )
        if part
    )
    metadata = dict(record.get("metadata", {}))
    metadata.update(
        {
            "title": title,
            "informal_statement": informal,
            "domain_path": domain_path,
            "preconditions": preconditions,
        }
    )
    merged["metadata"] = metadata
    return merged


def _premise_names(record: Dict[str, Any]) -> set[str]:
    result = {
        str(value).strip()
        for value in record.get("premise_ids", [])
        if str(value).strip()
    }
    for tactic in record.get("traced_tactics", []) or []:
        if not isinstance(tactic, dict):
            continue
        annotated = tactic.get("annotated_tactic")
        if not isinstance(annotated, (list, tuple)) or len(annotated) < 2:
            continue
        annotations: Iterable[Any] = annotated[1] or []
        for annotation in annotations:
            if isinstance(annotation, dict) and annotation.get("full_name"):
                result.add(str(annotation["full_name"]).strip())
    return result


def _string_list(value: Any, field: str) -> List[str]:
    if not isinstance(value, list):
        raise ValueError(f"Theorem enrichment {field} must be a list")
    result = [str(item).strip() for item in value if str(item).strip()]
    if not result:
        raise ValueError(f"Theorem enrichment {field} cannot be empty")
    return result


_ENRICHMENT_SYSTEM_PROMPT = """You translate a Lean 4 theorem declaration into retrieval
metadata for a mathematical reasoning system. Return one JSON object only with:
- title: a short English theorem title
- informal_statement: a faithful natural-language restatement
- domain_path: a list from broad to specific mathematical domains
- keywords: a list of useful English retrieval terms
- preconditions: a list of explicit assumptions, each with formal and informal fields

Never invent, remove, strengthen, or weaken assumptions. Preserve quantifiers and mathematical
types. Do not produce a proof. Do not repeat credentials, Markdown fences, or prose outside JSON.
"""

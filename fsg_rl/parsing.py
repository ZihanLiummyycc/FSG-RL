"""Parsing helpers for tagged function spans."""

from __future__ import annotations

import re
from typing import List, Optional

from .schemas import FunctionGraph, ParsedFunctionSpan


CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
VALUE_RE = re.compile(r"(dp\[\d+\])\s*=\s*(-?\d+)")
PARAM_VALUE_RE = re.compile(r"\b([a-zA-Z][a-zA-Z0-9_]*)\s*=\s*(-?\d+)\b")
RECURRENCE_RE = re.compile(r"recurrence\s*:\s*(.+)", re.IGNORECASE)


def parse_tagged_function_spans(raw_text: str, graph: FunctionGraph) -> List[ParsedFunctionSpan]:
    spans = []
    for index, node in enumerate(graph.nodes, start=1):
        aliases = [f"a_{node.id}", f"a_{index}"]
        if node.id == "main":
            aliases.extend(["a_main", "main"])

        content, start_char, end_char = _extract_first_tag(raw_text, aliases)
        code_blocks = [block.strip() for block in CODE_BLOCK_RE.findall(content)]
        extracted_answer = _extract_answer(content)
        recurrence_match = RECURRENCE_RE.search(content)
        extracted_formula = recurrence_match.group(1).strip() if recurrence_match else None
        extracted_values = {key: value for key, value in VALUE_RE.findall(content)}
        for key, value in PARAM_VALUE_RE.findall(content):
            if key not in {"dp"}:
                extracted_values.setdefault(key, value)

        spans.append(
            ParsedFunctionSpan(
                node_id=node.id,
                raw_text=content.strip(),
                code_blocks=code_blocks,
                extracted_answer=extracted_answer,
                extracted_formula=extracted_formula,
                extracted_values=extracted_values,
                start_char=start_char,
                end_char=end_char,
            )
        )
    return spans


def span_by_node(spans: List[ParsedFunctionSpan], node_id: str) -> Optional[ParsedFunctionSpan]:
    for span in spans:
        if span.node_id == node_id:
            return span
    return None


def _extract_first_tag(raw_text: str, aliases: List[str]) -> tuple[str, int, int]:
    for alias in aliases:
        pattern = re.compile(rf"<{re.escape(alias)}>(.*?)</{re.escape(alias)}>", re.DOTALL)
        match = pattern.search(raw_text)
        if match:
            return match.group(1), match.start(1), match.end(1)
    return "", -1, -1


def _extract_answer(content: str) -> Optional[str]:
    boxed = _extract_boxed_values(content)
    if boxed:
        return boxed[-1].strip()
    answer_match = re.search(r"final answer is\s+(-?\d+)", content, flags=re.IGNORECASE)
    if answer_match:
        return answer_match.group(1)
    return None


def _extract_boxed_values(content: str) -> List[str]:
    values = []
    cursor = 0
    marker = "\\boxed{"
    while True:
        start = content.find(marker, cursor)
        if start < 0:
            break
        value_start = start + len(marker)
        depth = 1
        index = value_start
        while index < len(content) and depth:
            if content[index] == "{":
                depth += 1
            elif content[index] == "}":
                depth -= 1
            index += 1
        if depth == 0:
            values.append(content[value_start : index - 1])
            cursor = index
        else:
            break
    return values

"""Build code-aware SFT records from execution-validated verifier bundles."""

from __future__ import annotations

from copy import deepcopy
import json
import random
import re
from typing import Any, Dict, Mapping, Sequence

from .parsing import parse_tagged_function_spans
from .rollout import build_policy_messages
from .schemas import FunctionGraph, MemoryContext, Problem


CODE_SFT_TASK = "solve_with_executable_function_graph"
PRIVATE_PROMPT_KEYS = {
    "expected",
    "expected_value",
    "expected_values",
    "formula",
    "values",
    "tests",
    "target_call",
    "reference_code",
    "mutant_code",
    "verification_graph",
    "verifier_implementations",
}


def build_code_sft_record(
    private_record: Mapping[str, Any],
    teacher_master: Mapping[str, Any],
) -> Dict[str, Any]:
    problem_id = str(private_record["id"])
    if str(teacher_master.get("id")) != problem_id:
        raise ValueError("Private verifier and teacher master IDs do not match")

    metadata = private_record.get("metadata", {})
    public_graph_data = metadata.get("function_graph")
    hidden_graph_data = metadata.get("verification_graph")
    implementations = metadata.get("verifier_implementations")
    if not isinstance(public_graph_data, dict):
        raise ValueError(f"{problem_id}: missing public function_graph")
    if not isinstance(hidden_graph_data, dict):
        raise ValueError(f"{problem_id}: missing verification_graph")
    if not isinstance(implementations, dict) or not implementations:
        raise ValueError(f"{problem_id}: missing verifier_implementations")

    public_graph = FunctionGraph.from_dict(dict(public_graph_data))
    hidden_graph = FunctionGraph.from_dict(dict(hidden_graph_data))
    if [node.id for node in public_graph.nodes] != [node.id for node in hidden_graph.nodes]:
        raise ValueError(f"{problem_id}: public/hidden graph node order differs")

    master_solution = str(teacher_master.get("tagged_solution", "")).strip()
    if not master_solution:
        raise ValueError(f"{problem_id}: teacher master has no tagged_solution")
    master_spans = {
        span.node_id: span
        for span in parse_tagged_function_spans(master_solution, public_graph)
    }

    target_parts = []
    executable_ids = []
    for node in public_graph.nodes:
        implementation = implementations.get(node.id)
        if implementation is not None:
            reference_code = str(implementation.get("reference_code", "")).strip()
            if not reference_code:
                raise ValueError(f"{problem_id}: empty reference code for {node.id}")
            content = f"```python\n{reference_code}\n```"
            executable_ids.append(node.id)
        else:
            span = master_spans.get(node.id)
            content = span.raw_text.strip() if span else ""
            if not content:
                raise ValueError(f"{problem_id}: empty reasoning span for {node.id}")
        target_parts.append(f"<a_{node.id}>\n{content}\n</a_{node.id}>")
    target = "\n".join(target_parts)

    problem_text = str(
        private_record.get("text", teacher_master.get("problem", ""))
    ).strip()
    if not problem_text:
        raise ValueError(f"{problem_id}: missing problem text")
    messages = build_policy_messages(
        Problem(
            id=problem_id,
            text=problem_text,
            gold_answer=str(private_record.get("gold_answer", "")),
        ),
        public_graph,
        MemoryContext(),
    )
    messages.append({"role": "assistant", "content": target})

    result = {
        "id": problem_id + ":code_solve",
        "task": CODE_SFT_TASK,
        "messages": messages,
        "metadata": {
            "source": "KbsdJames/Omni-MATH",
            "source_problem_id": problem_id,
            "executable_node_ids": executable_ids,
            "code_sft": True,
        },
    }
    validate_code_sft_record(result, private_record)
    return result


def validate_code_sft_record(
    record: Mapping[str, Any],
    private_record: Mapping[str, Any],
) -> None:
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) != 3:
        raise ValueError("Code SFT record must contain system, user, assistant messages")
    if [message.get("role") for message in messages] != ["system", "user", "assistant"]:
        raise ValueError("Code SFT message roles are invalid")

    metadata = private_record["metadata"]
    public_graph = FunctionGraph.from_dict(metadata["function_graph"])
    implementations = metadata["verifier_implementations"]
    target = str(messages[-1].get("content", ""))
    expected_ids = [node.id for node in public_graph.nodes]
    open_ids = re.findall(r"<a_([A-Za-z][A-Za-z0-9_]*)>", target)
    close_ids = re.findall(r"</a_([A-Za-z][A-Za-z0-9_]*)>", target)
    if open_ids != expected_ids or close_ids != expected_ids:
        raise ValueError(
            f"Code SFT tag order differs: expected={expected_ids} "
            f"open={open_ids} close={close_ids}"
        )

    spans = {
        span.node_id: span
        for span in parse_tagged_function_spans(target, public_graph)
    }
    for node_id, implementation in implementations.items():
        span = spans.get(node_id)
        if span is None or len(span.code_blocks) != 1:
            raise ValueError(f"Executable node {node_id!r} needs exactly one code block")
        expected_code = str(implementation["reference_code"]).strip()
        if span.code_blocks[0].strip() != expected_code:
            raise ValueError(f"Executable node {node_id!r} code differs from reference")

    user_payload = _message_text(messages[1])
    try:
        prompt_data = json.loads(user_payload)
    except json.JSONDecodeError as exc:
        raise ValueError("Code SFT user message must be one JSON object") from exc
    leaked = _find_private_keys(prompt_data)
    if leaked:
        raise ValueError(f"Code SFT prompt leaks private verifier keys: {sorted(leaked)}")

    assistant_text = target.lower()
    if "mutant_code" in assistant_text or "target_call" in assistant_text:
        raise ValueError("Code SFT target contains private verifier field names")
    main = spans.get("main")
    if main is None or main.extracted_answer is None:
        raise ValueError("Code SFT main span needs a boxed final answer")


def build_replay_mix(
    code_records: Sequence[Dict[str, Any]],
    graph_records: Sequence[Dict[str, Any]],
    solve_records: Sequence[Dict[str, Any]],
    *,
    seed: int,
    code_repeats: int = 2,
    replay_per_task: int | None = None,
) -> list[Dict[str, Any]]:
    if code_repeats < 1:
        raise ValueError("code_repeats must be positive")
    replay_count = len(code_records) if replay_per_task is None else replay_per_task
    if replay_count < 0:
        raise ValueError("replay_per_task cannot be negative")
    if replay_count > len(graph_records) or replay_count > len(solve_records):
        raise ValueError("Not enough graph/solve records for requested replay sample")

    randomizer = random.Random(seed)
    graph_sample = randomizer.sample(list(graph_records), replay_count)
    solve_sample = randomizer.sample(list(solve_records), replay_count)
    mixed = []
    for repeat in range(code_repeats):
        for record in code_records:
            clone = deepcopy(record)
            clone["id"] = f"{record['id']}:repeat{repeat + 1}"
            clone.setdefault("metadata", {})["replay_repeat"] = repeat + 1
            mixed.append(clone)
    mixed.extend(deepcopy(graph_sample))
    mixed.extend(deepcopy(solve_sample))
    randomizer.shuffle(mixed)
    return mixed


def _message_text(message: Mapping[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return str(content)


def _find_private_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        found = set(value) & PRIVATE_PROMPT_KEYS
        for child in value.values():
            found.update(_find_private_keys(child))
        return found
    if isinstance(value, list):
        found = set()
        for child in value:
            found.update(_find_private_keys(child))
        return found
    return set()

"""Compile and validate teacher-generated hidden verifier bundles."""

from __future__ import annotations

import ast
from copy import deepcopy
from dataclasses import replace
import re
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from .decomposition import validate_function_graph
from .sandbox_safety import (
    SandboxSafetyError,
    validate_python_code,
    validate_python_expression,
)
from .schemas import FunctionGraph, FunctionNode


CANONICAL_OUTPUT_TYPES = {
    "formula",
    "values",
    "python_function",
    "proof_claim",
    "final_answer",
}
PUBLIC_FORBIDDEN_KEYS = {
    "expected",
    "expected_value",
    "expected_values",
    "formula",
    "values",
    "tests",
    "target_call",
    "reference_code",
    "mutant_code",
}


class VerifierBundleError(ValueError):
    """Raised when a teacher verifier bundle cannot be trusted."""


class VerifierBundleRejected(VerifierBundleError):
    """Raised when the teacher determines that a problem is not executable."""


def compile_verifier_bundle(
    source_record: Mapping[str, Any],
    teacher_payload: Mapping[str, Any],
    *,
    annotation_model: str,
) -> Dict[str, Any]:
    """Validate teacher output and build separate public/hidden graphs."""

    if teacher_payload.get("eligible") is not True:
        reason = str(teacher_payload.get("rejection_reason", "not executable")).strip()
        raise VerifierBundleRejected(reason or "not executable")

    metadata = source_record.get("metadata", {})
    source_graph_data = metadata.get("function_graph") if isinstance(metadata, dict) else None
    if not isinstance(source_graph_data, dict):
        raise VerifierBundleError("Source record has no metadata.function_graph")
    source_graph = FunctionGraph.from_dict(dict(source_graph_data))
    validate_function_graph(source_graph)

    annotations = teacher_payload.get("node_verifiers")
    if not isinstance(annotations, list):
        raise VerifierBundleError("Teacher output needs node_verifiers array")
    by_id: Dict[str, Dict[str, Any]] = {}
    for index, value in enumerate(annotations):
        if not isinstance(value, dict):
            raise VerifierBundleError(f"node_verifiers[{index}] must be an object")
        node_id = str(value.get("node_id", "")).strip()
        if not node_id or node_id in by_id:
            raise VerifierBundleError(f"Invalid or duplicate node_id {node_id!r}")
        by_id[node_id] = value

    expected_ids = {node.id for node in source_graph.nodes}
    if set(by_id) != expected_ids:
        raise VerifierBundleError(
            "node_verifiers IDs do not match source graph: "
            f"expected={sorted(expected_ids)} actual={sorted(by_id)}"
        )

    public_nodes: List[FunctionNode] = []
    hidden_nodes: List[FunctionNode] = []
    implementations: Dict[str, Dict[str, Any]] = {}
    executable_ids: List[str] = []

    for source_node in source_graph.nodes:
        annotation = by_id[source_node.id]
        output_type = str(annotation.get("canonical_output_type", "")).strip()
        if output_type not in CANONICAL_OUTPUT_TYPES:
            raise VerifierBundleError(
                f"Node {source_node.id!r} has unsupported canonical_output_type "
                f"{output_type!r}"
            )
        if source_node.id == "main" and output_type != "final_answer":
            raise VerifierBundleError("Node 'main' must have canonical_output_type=final_answer")
        if source_node.id != "main" and output_type == "final_answer":
            raise VerifierBundleError("Only node 'main' may use final_answer")

        check_types = _string_list(annotation.get("check_types", []), "check_types")
        public_spec = {
            "check_types": check_types,
            "semantic_requirements": _semantic_requirements(
                source_node.verification_spec
            ),
        }
        if _contains_forbidden_key(public_spec, PUBLIC_FORBIDDEN_KEYS):
            raise VerifierBundleError(f"Node {source_node.id!r} public spec leaks hidden data")

        hidden_spec: Dict[str, Any] = {}
        if output_type == "python_function":
            if source_node.id == "main":
                raise VerifierBundleError("The main node cannot be an executable helper")
            tests = _validate_tests(source_node.id, annotation.get("tests"))
            hidden_spec["tests"] = tests
            reference_code = _bundle_safe_code(
                str(annotation.get("reference_code", "")),
                label=f"{source_node.id}.reference_code",
            )
            mutant_code = _bundle_safe_code(
                str(annotation.get("mutant_code", "")),
                label=f"{source_node.id}.mutant_code",
            )
            _validate_implementation_contract(
                source_node,
                tests,
                reference_code,
                label=f"{source_node.id}.reference_code",
            )
            _validate_implementation_contract(
                source_node,
                tests,
                mutant_code,
                label=f"{source_node.id}.mutant_code",
            )
            implementations[source_node.id] = {
                "reference_code": reference_code,
                "mutant_code": mutant_code,
                "oracle_strategy": str(annotation.get("oracle_strategy", "")).strip(),
                "estimated_runtime_seconds": _bounded_runtime(
                    annotation.get("estimated_runtime_seconds", 0.0)
                ),
            }
            executable_ids.append(source_node.id)
        elif output_type == "formula":
            formula = str(annotation.get("formula", "")).strip()
            if not formula:
                raise VerifierBundleError(f"Node {source_node.id!r} formula is empty")
            hidden_spec["formula"] = formula
        elif output_type == "values":
            values = annotation.get("values")
            if not isinstance(values, dict) or not values:
                raise VerifierBundleError(f"Node {source_node.id!r} values must be non-empty")
            hidden_spec["values"] = dict(values)
        elif output_type == "final_answer":
            target_call = str(annotation.get("target_call", "")).strip()
            if target_call:
                _bundle_safe_expression(
                    target_call, label=f"{source_node.id}.target_call"
                )
                hidden_spec["target_call"] = target_call

        public_nodes.append(
            replace(
                source_node,
                expected_output_type=output_type,
                verification_spec=public_spec,
            )
        )
        hidden_nodes.append(
            replace(
                source_node,
                expected_output_type=output_type,
                verification_spec=hidden_spec,
            )
        )

    if not executable_ids:
        raise VerifierBundleError("At least one non-main python_function node is required")

    public_graph = FunctionGraph(
        problem_id=source_graph.problem_id,
        nodes=public_nodes,
        edges=deepcopy(source_graph.edges),
    )
    hidden_graph = FunctionGraph(
        problem_id=source_graph.problem_id,
        nodes=hidden_nodes,
        edges=deepcopy(source_graph.edges),
    )
    validate_function_graph(public_graph)
    validate_function_graph(hidden_graph)

    main = hidden_graph.node_by_id("main")
    target_call = str(main.verification_spec.get("target_call", "")) if main else ""
    if not target_call:
        raise VerifierBundleError("The main hidden verifier needs target_call")
    test_expressions = {
        str(test.get("call", test.get("expression", ""))).strip()
        for node in hidden_graph.nodes
        for test in node.verification_spec.get("tests", [])
    }
    if target_call not in test_expressions:
        raise VerifierBundleError(
            f"main.target_call {target_call!r} is not present in executable tests"
        )

    confidence = float(teacher_payload.get("confidence", 0.0))
    if not 0.0 <= confidence <= 1.0:
        raise VerifierBundleError("confidence must be between 0 and 1")

    result = deepcopy(dict(source_record))
    result_metadata = dict(result.get("metadata", {}))
    result_metadata["function_graph"] = public_graph.to_dict()
    result_metadata["verification_graph"] = hidden_graph.to_dict()
    result_metadata["verifier_implementations"] = implementations
    result_metadata["verifier_annotation"] = {
        "model": annotation_model,
        "confidence": confidence,
        "reason": str(teacher_payload.get("reason", "")).strip(),
        "executable_node_ids": executable_ids,
    }
    result["metadata"] = result_metadata
    if _contains_forbidden_key(
        result,
        {"reference_solution", "tagged_solution"},
    ):
        raise VerifierBundleError("Compiled verifier record contains teacher-solution leakage")
    return result


def build_teacher_messages(
    candidate: Mapping[str, Any],
    master: Mapping[str, Any],
) -> List[Dict[str, str]]:
    """Build a private teacher request; reference material never enters output records."""

    payload = {
        "problem_id": candidate["id"],
        "problem": candidate["text"],
        "gold_answer": candidate["gold_answer"],
        "reference_solution": master.get("reference_solution", ""),
        "tagged_solution": master.get("tagged_solution", ""),
        "source_function_graph": candidate.get("metadata", {}).get("function_graph"),
    }
    return [
        {"role": "system", "content": _TEACHER_SYSTEM_PROMPT},
        {"role": "user", "content": _json_dumps(payload)},
    ]


def strip_private_implementations(record: Mapping[str, Any]) -> Dict[str, Any]:
    result = deepcopy(dict(record))
    metadata = dict(result.get("metadata", {}))
    metadata.pop("verifier_implementations", None)
    result["metadata"] = metadata
    return result


def _validate_tests(node_id: str, raw_tests: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw_tests, list):
        raise VerifierBundleError(f"Node {node_id!r} tests must be an array")
    if len(raw_tests) < 4:
        raise VerifierBundleError(f"Node {node_id!r} needs at least four hidden tests")
    tests = []
    unit_count = 0
    property_count = 0
    for index, raw in enumerate(raw_tests):
        if not isinstance(raw, dict):
            raise VerifierBundleError(f"Node {node_id!r} test {index} must be an object")
        kind = str(raw.get("kind", "")).strip()
        if kind not in {"unit", "property"}:
            raise VerifierBundleError(
                f"Node {node_id!r} test {index} kind must be unit/property"
            )
        expression = str(raw.get("call", raw.get("expression", ""))).strip()
        _bundle_safe_expression(expression, label=f"{node_id}.tests[{index}]")
        if "expected" not in raw:
            raise VerifierBundleError(f"Node {node_id!r} test {index} lacks expected")
        test = {"kind": kind, "expected": deepcopy(raw["expected"])}
        test["call" if kind == "unit" else "expression"] = expression
        source = str(raw.get("source", "")).strip()
        if source:
            test["source"] = source
        tests.append(test)
        unit_count += int(kind == "unit")
        property_count += int(kind == "property")
    if unit_count < 3 or property_count < 1:
        raise VerifierBundleError(
            f"Node {node_id!r} needs >=3 unit and >=1 property tests; "
            f"got unit={unit_count} property={property_count}"
        )
    return tests


def _bundle_safe_code(code: str, *, label: str) -> str:
    try:
        return validate_python_code(code, label=label)
    except SandboxSafetyError as exc:
        raise VerifierBundleError(str(exc)) from exc


def _bundle_safe_expression(expression: str, *, label: str) -> str:
    try:
        return validate_python_expression(expression, label=label)
    except SandboxSafetyError as exc:
        raise VerifierBundleError(str(exc)) from exc


_SAFE_TEST_CALLS = {
    "abs",
    "all",
    "any",
    "bool",
    "dict",
    "enumerate",
    "filter",
    "float",
    "int",
    "len",
    "list",
    "map",
    "max",
    "min",
    "pow",
    "range",
    "reversed",
    "round",
    "set",
    "sorted",
    "str",
    "sum",
    "tuple",
    "zip",
}


def _validate_implementation_contract(
    source_node: FunctionNode,
    tests: Sequence[Mapping[str, Any]],
    code: str,
    *,
    label: str,
) -> None:
    tree = ast.parse(code, mode="exec")
    defined = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.asname or alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.update(alias.asname or alias.name for alias in node.names)

    signature_match = re.match(r"\s*([A-Za-z_]\w*)\s*\(", source_node.signature)
    if signature_match and signature_match.group(1) not in defined:
        raise VerifierBundleError(
            f"{label} must define signature function {signature_match.group(1)!r}"
        )

    available_calls = defined | imported | _SAFE_TEST_CALLS
    for index, test in enumerate(tests):
        expression = str(test.get("call", test.get("expression", "")))
        expression_tree = ast.parse(expression, mode="eval")
        missing = sorted(
            {
                call.func.id
                for call in ast.walk(expression_tree)
                if isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id not in available_calls
            }
        )
        if missing:
            raise VerifierBundleError(
                f"{label} test {index} calls undefined functions {missing}; "
                "use the exact function names defined in the implementation"
            )


def _semantic_requirements(spec: Mapping[str, Any]) -> List[str]:
    keys = (
        "requirements",
        "must_establish",
        "must_include",
        "must_show",
        "must_state",
        "required_claims",
        "required_conclusion",
        "reason",
        "method",
        "check_method",
    )
    values: List[str] = []
    for key in keys:
        raw = spec.get(key)
        if isinstance(raw, list):
            values.extend(str(item).strip() for item in raw if str(item).strip())
        elif raw is not None and str(raw).strip():
            values.append(str(raw).strip())
    return values[:12]


def _string_list(value: Any, label: str) -> List[str]:
    if not isinstance(value, list):
        raise VerifierBundleError(f"{label} must be an array")
    result = [str(item).strip() for item in value if str(item).strip()]
    return result


def _bounded_runtime(value: Any) -> float:
    runtime = float(value)
    if not 0.0 < runtime <= 2.0:
        raise VerifierBundleError(
            f"estimated_runtime_seconds must be in (0,2], got {runtime}"
        )
    return runtime


def _contains_forbidden_key(value: Any, forbidden: Iterable[str]) -> bool:
    forbidden_set = set(forbidden)
    if isinstance(value, dict):
        return bool(set(value) & forbidden_set) or any(
            _contains_forbidden_key(child, forbidden_set) for child in value.values()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_key(child, forbidden_set) for child in value)
    return False


def _json_dumps(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False)


_TEACHER_SYSTEM_PROMPT = r"""You compile an existing mathematical function graph into a
secure hidden verifier bundle. Return one JSON object only. You may use the private reference
solution to design tests, but never copy reference_solution or tagged_solution into output.

Return:
{
  "eligible": true or false,
  "confidence": number from 0 to 1,
  "reason": "short explanation",
  "rejection_reason": null or "reason",
  "node_verifiers": [one object for every original node]
}

Every node object has:
- node_id: exactly an original node ID; do not add, remove, rename, or reorder graph logic.
- canonical_output_type: formula, values, python_function, proof_claim, or final_answer.
- check_types: public names such as unit, property, small_bruteforce, symbolic, backward.
- tests: [] unless python_function. For python_function provide >=3 unit tests and >=1
  property test. A unit test uses {"kind":"unit","call":"f(...) ","expected":...}.
  A property test uses {"kind":"property","expression":"...","expected":true,
  "source":"invariant" or "small_bruteforce"}.
- reference_code and mutant_code: non-empty self-contained Python only for python_function.
  Both define the exact function named by the node signature. The mutant must be plausible but
  wrong and must fail at least one hidden test.
- oracle_strategy and estimated_runtime_seconds (0,2] for python_function.
- formula for formula nodes; values object for values nodes.
- target_call only for main/final_answer. It must exactly equal a unit-test call from one
  executable node and evaluate to the problem's final answer.

Reject the whole problem with eligible=false when no non-main node supports an independent,
deterministic Python verifier. Do not create tests that merely return or compare a hard-coded
gold answer. Prefer boundary tests plus small exhaustive checks. Inline the brute-force oracle
inside property expressions using standard Python, so candidate code need not define a hidden
oracle helper. No files, network, subprocesses, eval, exec, randomness, time, APIs, or third-party
packages. Allowed imports: math, fractions, itertools, functools, collections, heapq, operator,
statistics. Code and test expressions must finish in at most two seconds. main must be
final_answer and every original node must appear exactly once. Do not include Markdown fences.

Before returning eligible=true, perform this mandatory self-check for every executable node:
1. List the functions actually defined by reference_code and mutant_code. Every direct function
   call in every test must use one of those exact names or a standard Python builtin. Never use
   placeholder names such as f, solve, helper, or oracle unless that exact function is defined.
2. Manually evaluate every unit-test call against reference_code and make expected exactly match
   the result. Distinguish a quantity from its square, signed determinants from absolute values,
   ordered results from unordered results, and list/tuple representations.
3. Check the property expression can run using only names defined in reference_code plus Python
   builtins/imports. It must call the candidate function and independently recompute or verify the
   property over several small inputs.
4. Mentally run mutant_code on all tests and confirm at least one test fails for the mathematical
   defect, not merely because of a missing name or syntax error.
5. Copy target_call character-for-character from one unit-test call and verify its result is
   mathematically equivalent to gold_answer, allowing only presentation differences such as a
   currency symbol or physical unit.

If any self-check cannot be completed reliably, return eligible=false instead of guessing.
"""

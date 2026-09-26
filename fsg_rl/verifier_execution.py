"""Execute and audit compiled hidden-verifier bundles."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping

from .answer_equivalence import answers_equivalent
from .schemas import FunctionGraph, ParsedFunctionSpan
from .tool_execution import ToolExecutor
from .verifier_bundle import VerifierBundleError


def validate_compiled_verifier_record(
    row: Mapping[str, Any],
    executor: ToolExecutor,
) -> Dict[str, Any]:
    """Require reference success, backward agreement, and mutant detection."""

    metadata = row.get("metadata", {})
    graph = FunctionGraph.from_dict(metadata["verification_graph"])
    implementations = metadata.get("verifier_implementations", {})
    if not isinstance(implementations, dict) or not implementations:
        raise VerifierBundleError("Missing verifier_implementations")

    reference_execution = executor.execute(
        implementation_spans(graph, implementations, use_mutant=None),
        graph,
    )
    if not reference_execution.executable:
        raise VerifierBundleError(
            f"Reference code is not executable: {reference_execution.stderr}"
        )
    for node_id in implementations:
        result = reference_execution.node_results.get(node_id)
        if result is None or not result.test_results:
            raise VerifierBundleError(f"Reference node {node_id!r} produced no tests")
        failed = [test for test in result.test_results if not test.get("passed")]
        if failed:
            raise VerifierBundleError(
                f"Reference node {node_id!r} failed tests: {failed[:3]}"
            )

    main = graph.node_by_id("main")
    target_call = str(main.verification_spec.get("target_call", "")) if main else ""
    actual = None
    for result in reference_execution.node_results.values():
        if target_call in result.outputs:
            actual = result.outputs[target_call]
            break
    if actual is None or not answers_equivalent(actual, row.get("gold_answer")):
        raise VerifierBundleError(
            f"Backward target mismatch: call={target_call!r} actual={actual!r} "
            f"gold={row.get('gold_answer')!r}"
        )

    mutant_audits = {}
    for node_id in implementations:
        mutant_execution = executor.execute(
            implementation_spans(graph, implementations, use_mutant=node_id),
            graph,
        )
        result = mutant_execution.node_results.get(node_id)
        caught = bool(
            result is None
            or not result.executable
            or any(not test.get("passed") for test in result.test_results)
        )
        if not caught:
            raise VerifierBundleError(
                f"Hidden tests did not catch mutant for node {node_id!r}"
            )
        mutant_audits[node_id] = {
            "caught": True,
            "executable": bool(result and result.executable),
            "failed_tests": sum(
                not test.get("passed")
                for test in (result.test_results if result else [])
            ),
        }

    return {
        "reference_all_tests_passed": True,
        "backward_target_passed": True,
        "mutants": mutant_audits,
        "reference_runtime_seconds": reference_execution.runtime_seconds,
    }


def implementation_spans(
    graph: FunctionGraph,
    implementations: Mapping[str, Mapping[str, Any]],
    *,
    use_mutant: str | None,
) -> List[ParsedFunctionSpan]:
    spans = []
    for node in graph.nodes:
        implementation = implementations.get(node.id)
        if implementation:
            key = "mutant_code" if node.id == use_mutant else "reference_code"
            code = str(implementation[key])
            spans.append(
                ParsedFunctionSpan(
                    node_id=node.id,
                    raw_text=code,
                    code_blocks=[code],
                    start_char=0,
                    end_char=len(code),
                )
            )
        else:
            spans.append(ParsedFunctionSpan(node_id=node.id, raw_text="not executable"))
    return spans

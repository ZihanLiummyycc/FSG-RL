"""Teacher-guided policy repair for training and inference."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from .rollout import PolicyBackend
from .schemas import (
    ExecutionResult,
    FunctionGraph,
    MemoryContext,
    PolicyRollout,
    Problem,
    VerificationResult,
)
from .teacher import TeacherClient


class RepairModule:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        section = config.get("repair", {})
        self.enabled = bool(section.get("enabled", True))
        self.repair_budget = int(section.get("repair_budget", 1))
        raw_problem_budget = section.get("max_teacher_calls_per_problem")
        self.max_teacher_calls_per_problem = (
            int(raw_problem_budget) if raw_problem_budget is not None else None
        )
        self.group_mode = str(section.get("group_mode", "immediate"))
        self.fail_open = bool(section.get("fail_open", False))
        self.teacher = TeacherClient(config) if self.enabled else None

    def classify_failure(
        self,
        verification: VerificationResult,
        execution: ExecutionResult,
    ) -> str:
        for node_result in execution.node_results.values():
            if node_result.timeout or (node_result.stderr and not node_result.executable):
                return "execution_failure"
        for failure in verification.failures:
            if failure.get("type") == "dependency_failure":
                return "dependency_failure"
            if failure.get("type") in {
                "test_failure",
                "property_failure",
                "final_answer_failure",
                "node_verification_failure",
            }:
                return "verification_failure"
        return "unknown_failure"

    def repair_once(
        self,
        problem: Problem,
        graph: FunctionGraph,
        memory_context: MemoryContext,
        rollout: PolicyRollout,
        execution: ExecutionResult,
        verification: VerificationResult,
        policy: PolicyBackend,
        rollout_index: int,
    ) -> Tuple[PolicyRollout, FunctionGraph, Dict[str, Any]]:
        if not self.enabled or self.repair_budget <= 0 or self.teacher is None:
            return rollout, graph, {"status": "skipped", "repair_trace": []}

        repaired_graph, feedback, repair_record = self.prepare_repair(
            problem,
            graph,
            memory_context,
            rollout,
            execution,
            verification,
        )
        repaired_rollout = policy.rollout(
            problem,
            repaired_graph,
            memory_context,
            rollout_index=rollout_index,
            repair_feedback=feedback,
        )
        return repaired_rollout, repaired_graph, repair_record

    def prepare_repair(
        self,
        problem: Problem,
        graph: FunctionGraph,
        memory_context: MemoryContext,
        rollout: PolicyRollout,
        execution: ExecutionResult,
        verification: VerificationResult,
    ) -> Tuple[FunctionGraph, Dict[str, Any], Dict[str, Any]]:
        if not self.enabled or self.repair_budget <= 0 or self.teacher is None:
            raise RuntimeError("Teacher repair is disabled")

        failure_type = self.classify_failure(verification, execution)
        teacher_plan = self.teacher.diagnose_and_plan(
            problem,
            graph,
            rollout,
            execution,
            verification,
            failure_type,
            memory_context=memory_context,
        )
        repaired_graph = graph
        if teacher_plan.get("repaired_graph"):
            repaired_graph = FunctionGraph.from_dict(teacher_plan["repaired_graph"])
            if {node.id for node in repaired_graph.nodes} != {node.id for node in graph.nodes}:
                raise ValueError(
                    "Teacher graph repair may change edges/specifications but not node IDs "
                    "inside an existing GRPO rollout group"
                )

        feedback = {
            "failure_type": failure_type,
            "diagnosis": teacher_plan["diagnosis"],
            "repair_instructions": teacher_plan["repair_instructions"],
            "failed_nodes": teacher_plan.get("failed_nodes", []),
            "failed_edges": teacher_plan.get("failed_edges", []),
            "failed_checks": _public_failure_summaries(verification),
        }
        return repaired_graph, feedback, {
            "status": "repaired_by_policy",
            "failure_type": failure_type,
            "teacher_diagnosis": teacher_plan["diagnosis"],
            "teacher_repair_instructions": teacher_plan["repair_instructions"],
            "memory_item": teacher_plan.get("memory_item", ""),
            "retrieved_theory_ids": [
                item.id for item in memory_context.theorem_items[:3]
            ],
            "graph_repaired": repaired_graph.to_dict() != graph.to_dict(),
            "repair_trace": [teacher_plan],
        }

    def judge_group(
        self,
        problem: Problem,
        graph: FunctionGraph,
        memory_context: MemoryContext,
        rollouts: list[PolicyRollout],
        executions: list[ExecutionResult],
    ) -> Dict[str, Any]:
        if not self.enabled or self.teacher is None:
            raise RuntimeError("Teacher group judging is disabled")
        return self.teacher.judge_rollout_group(
            problem,
            graph,
            rollouts,
            executions,
            memory_context=memory_context,
        )


def _public_failure_summaries(
    verification: VerificationResult,
) -> list[Dict[str, Any]]:
    """Keep hidden expected values and gold answers out of the policy prompt."""

    allowed_keys = {
        "type",
        "node",
        "edge",
        "relation_type",
        "equivalence_method",
        "equivalence_error",
        "error_type",
    }
    return [
        {
            key: value
            for key, value in dict(failure).items()
            if key in allowed_keys
        }
        for failure in verification.failures
    ]

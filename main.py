#!/usr/bin/env python3
"""Run FSG-RL training or inference with real model backends."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from fsg_rl.configuration import resolve_path, validate_config
from fsg_rl.datasets import load_dataset
from fsg_rl.decomposition import build_decomposer
from fsg_rl.distributed_training import (
    DistributedContext,
    configure_distributed_paths,
    finalize_distributed,
    initialize_distributed,
    merge_distributed_summaries,
    merge_rank_jsonl,
    shard_indexed,
)
from fsg_rl.grpo import GRPOTrainer
from fsg_rl.memory import MemoryBank
from fsg_rl.metrics import summarize_metrics
from fsg_rl.parsing import parse_tagged_function_spans
from fsg_rl.repair import RepairModule
from fsg_rl.reward import RewardAssigner
from fsg_rl.rollout import TransformersPolicy, build_policy
from fsg_rl.schemas import (
    ExecutionResult,
    FunctionGraph,
    PolicyRollout,
    Problem,
    VerificationResult,
)
from fsg_rl.tool_execution import ToolExecutor
from fsg_rl.training_graphs import resolve_verification_graph
from fsg_rl.verifier import Verifier


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    _resolve_config_paths(config, path.parent.resolve())
    return config


def run(
    config: Dict[str, Any],
    mode: str,
    distributed: DistributedContext | None = None,
) -> Dict[str, Any]:
    distributed = distributed or DistributedContext()
    validate_config(config, mode)
    started = time.monotonic()
    all_problems = load_dataset(config)
    indexed_problems, dropped_problems = shard_indexed(all_problems, distributed)
    memory = MemoryBank(config)
    executor = ToolExecutor(config)
    verifier = Verifier(config)
    policy = build_policy(config, trainable=mode == "train")
    decomposer = build_decomposer(config, policy)
    repair_module = (
        RepairModule(config)
        if config.get("repair", {}).get("enabled", False)
        else None
    )
    reward_assigner = RewardAssigner(config)
    trainer = None
    if mode == "train":
        if not isinstance(policy, TransformersPolicy):
            raise TypeError("Training requires TransformersPolicy")
        trainer = GRPOTrainer(config, policy)

    num_rollouts = int(config.get("rollout", {}).get("num_rollouts", 1))
    problem_summaries: List[Dict[str, Any]] = []
    metric_summaries: List[Dict[str, Any]] = []
    _initialize_trajectory_file(config)

    for local_problem_index, (problem_index, problem) in enumerate(indexed_problems):
        problem_started = time.monotonic()
        memory_context = memory.retrieve(problem)
        if config.get("memory", {}).get("log_retrieval", False):
            theorem_ids = [item.id for item in memory_context.theorem_items]
            print(
                f"problem={problem.id} retrieved_theory={theorem_ids}",
                flush=True,
            )
        base_graph = decomposer.construct(problem, memory_context)
        print(
            f"rank={distributed.rank} local_problem={local_problem_index + 1}/"
            f"{len(indexed_problems)} global_index={problem_index + 1} id={problem.id} "
            f"rollout_group={num_rollouts} "
            f"reference_policy={getattr(policy, 'reference_policy_description', 'n/a')}",
            flush=True,
        )
        rollouts: List[PolicyRollout] = []
        graphs: List[FunctionGraph] = []
        executions: List[ExecutionResult] = []
        verifications: List[VerificationResult] = []
        verification_graphs: List[FunctionGraph] = []
        repair_records: List[Dict[str, Any]] = []
        teacher_calls_used = 0
        max_teacher_calls = (
            repair_module.max_teacher_calls_per_problem
            if repair_module
            and repair_module.max_teacher_calls_per_problem is not None
            else num_rollouts * (repair_module.repair_budget if repair_module else 0)
        )
        shared_feedback_mode = bool(
            mode == "train"
            and repair_module
            and repair_module.enabled
            and repair_module.group_mode == "shared_feedback"
        )
        group_judge_mode = bool(
            mode == "train"
            and repair_module
            and repair_module.enabled
            and repair_module.group_mode == "group_judge"
        )
        shared_graph = base_graph
        shared_feedback: Dict[str, Any] | None = None
        teacher_probe: Dict[str, Any] | None = None
        teacher_group_judgment: Dict[str, Any] | None = None
        shared_repair_record: Dict[str, Any] = {"status": "not_needed"}

        if shared_feedback_mode:
            probe_rollout = policy.rollout(
                problem,
                base_graph,
                memory_context,
                rollout_index=1_000_000 + problem_index,
            )
            probe_verification_graph = resolve_verification_graph(problem, base_graph)
            probe_execution, probe_verification = _evaluate_rollout(
                problem,
                base_graph,
                probe_verification_graph,
                probe_rollout,
                executor,
                verifier,
            )
            teacher_probe = {
                "rollout": probe_rollout.to_dict(),
                "execution": probe_execution.to_dict(),
                "verification": probe_verification.to_dict(),
            }
            if probe_verification.has_failures() and teacher_calls_used < max_teacher_calls:
                teacher_calls_used += 1
                try:
                    (
                        shared_graph,
                        shared_feedback,
                        shared_repair_record,
                    ) = repair_module.prepare_repair(
                        problem,
                        base_graph,
                        memory_context,
                        probe_rollout,
                        probe_execution,
                        probe_verification,
                    )
                    resolve_verification_graph(problem, shared_graph)
                    shared_repair_record["status"] = "shared_feedback"
                except Exception as exc:
                    if not repair_module.fail_open:
                        raise
                    shared_graph = base_graph
                    shared_feedback = None
                    shared_repair_record = {
                        "status": "teacher_error",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
            elif probe_verification.has_failures():
                shared_repair_record = {"status": "teacher_budget_exhausted"}
            else:
                shared_repair_record = {"status": "probe_passed"}

        for rollout_index in range(num_rollouts):
            graph = shared_graph if shared_feedback_mode else base_graph
            rollout = policy.rollout(
                problem,
                graph,
                memory_context,
                rollout_index=rollout_index,
                repair_feedback=shared_feedback if shared_feedback_mode else None,
            )
            verification_graph = resolve_verification_graph(problem, graph)
            execution, verification = _evaluate_rollout(
                problem,
                graph,
                verification_graph,
                rollout,
                executor,
                verifier,
            )
            repair_record: Dict[str, Any] = {"status": "not_needed", "attempts": []}

            if group_judge_mode:
                repair_record["status"] = "teacher_free_group_pending"
            elif shared_feedback_mode:
                repair_record.update(shared_repair_record)
            elif (
                repair_module
                and repair_module.enabled
                and verification.has_failures()
                and teacher_calls_used >= max_teacher_calls
            ):
                repair_record["status"] = "teacher_budget_exhausted"
            elif repair_module and repair_module.enabled and verification.has_failures():
                repair_record["status"] = "attempted"
                if config.get("repair", {}).get("record_before_after", False):
                    repair_record.update(
                        {
                            "rollout_before_repair": rollout.to_dict(),
                            "execution_before_repair": execution.to_dict(),
                            "verification_before_repair": verification.to_dict(),
                        }
                    )
                for attempt in range(repair_module.repair_budget):
                    if teacher_calls_used >= max_teacher_calls:
                        repair_record["status"] = "teacher_budget_exhausted"
                        break
                    teacher_calls_used += 1
                    try:
                        rollout, graph, attempt_record = repair_module.repair_once(
                            problem,
                            graph,
                            memory_context,
                            rollout,
                            execution,
                            verification,
                            policy,
                            rollout_index=(
                                rollout_index * repair_module.repair_budget + attempt + 1
                            ),
                        )
                    except Exception as exc:
                        if not repair_module.fail_open:
                            raise
                        repair_record["status"] = "teacher_error"
                        repair_record["attempts"].append(
                            {
                                "status": "teacher_error",
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                            }
                        )
                        break
                    verification_graph = resolve_verification_graph(problem, graph)
                    execution, verification = _evaluate_rollout(
                        problem,
                        graph,
                        verification_graph,
                        rollout,
                        executor,
                        verifier,
                    )
                    attempt_record["verification_after_repair"] = verification.to_dict()
                    repair_record["attempts"].append(attempt_record)
                    repair_record.update(
                        {
                            key: value
                            for key, value in attempt_record.items()
                            if key
                            in {
                                "failure_type",
                                "teacher_diagnosis",
                                "teacher_repair_instructions",
                                "memory_item",
                                "graph_repaired",
                            }
                        }
                    )
                    if not verification.has_failures():
                        repair_record["status"] = "succeeded"
                        break
                else:
                    repair_record["status"] = "failed"
                    _append_mistake_notebook(
                        config,
                        problem,
                        graph,
                        rollout,
                        execution,
                        verification,
                        repair_record,
                    )

            rollouts.append(rollout)
            graphs.append(graph)
            executions.append(execution)
            verifications.append(verification)
            verification_graphs.append(verification_graph)
            repair_records.append(repair_record)

        teacher_scores = [0.0 for _ in rollouts]
        teacher_node_scores: List[Dict[str, float]] = [dict() for _ in rollouts]
        if group_judge_mode:
            if teacher_calls_used >= max_teacher_calls:
                for repair_record in repair_records:
                    repair_record["status"] = "teacher_budget_exhausted"
            else:
                teacher_calls_used += 1
                try:
                    teacher_group_judgment = repair_module.judge_group(
                        problem,
                        base_graph,
                        memory_context,
                        rollouts,
                        executions,
                    )
                    for judgment in teacher_group_judgment["rollout_judgments"]:
                        index = int(judgment["index"])
                        teacher_scores[index] = float(judgment["overall_score"])
                        teacher_node_scores[index] = dict(judgment["node_scores"])
                        repair_records[index].update(
                            {
                                "status": "group_judged",
                                "teacher_score": teacher_scores[index],
                                "teacher_node_scores": teacher_node_scores[index],
                                "teacher_diagnosis": judgment["diagnosis"],
                            }
                        )
                except Exception as exc:
                    if not repair_module.fail_open:
                        raise
                    teacher_group_judgment = {
                        "status": "teacher_error",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                    for repair_record in repair_records:
                        repair_record.update(
                            {
                                "status": "teacher_error",
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                            }
                        )

        verifier.apply_group_consensus(
            [rollout.parsed_spans for rollout in rollouts],
            verifications,
            base_graph,
        )

        rewards = []
        rollout_records = []
        for (
            rollout,
            graph,
            verification_graph,
            execution,
            verification,
            repair_record,
            teacher_score,
            node_teacher_scores,
        ) in zip(
            rollouts,
            graphs,
            verification_graphs,
            executions,
            verifications,
            repair_records,
            teacher_scores,
            teacher_node_scores,
        ):
            reward = reward_assigner.assign(
                problem,
                verification_graph,
                verification,
                execution,
                rollout.parsed_spans,
                repaired=rollout.repaired,
                rollout=rollout,
                teacher_score=teacher_score,
                teacher_node_scores=node_teacher_scores,
            )
            rewards.append(reward)
            print(
                f"problem={problem.id} rollout={len(rewards)}/{num_rollouts} "
                f"reward={reward.total_reward:.6f} format={reward.format_gate:.3f} "
                f"signature={reward.signature_reward:.3f} "
                f"execution={reward.execution_reward:.3f} "
                f"unit={reward.unit_test_reward:.3f} "
                f"property={reward.property_test_reward:.3f} "
                f"teacher={reward.teacher_reward:.3f} "
                f"answer_gate={reward.answer_gate:.3f} "
                f"backward={reward.backward_reward:.3f} "
                f"final={reward.final_reward:.3f}",
                flush=True,
            )
            rollout_records.append(
                {
                    "rollout": rollout.to_dict(),
                    "function_graph": graph.to_dict(),
                    "execution": execution.to_dict(),
                    "verification": verification.to_dict(),
                    "reward": reward.to_dict(),
                    "repair": repair_record,
                }
            )

        update_summary = trainer.update(rollouts, rewards) if trainer else None
        if update_summary:
            print(
                f"problem={problem.id} update={update_summary['update_count']} "
                f"loss={update_summary['loss']:.6f} "
                f"kl={update_summary['approx_kl']:.6f} "
                f"grad_norm={update_summary['grad_norm']:.6f}",
                flush=True,
            )
        if mode == "train":
            memory.update(problem, base_graph, rollout_records)

        problem_summary = {
            "problem": problem.to_dict(),
            "problem_index": problem_index,
            "wall_time_seconds": time.monotonic() - problem_started,
            "memory_context_sizes": {
                key: len(value) for key, value in memory_context.to_dict().items()
            },
            "function_graph": base_graph.to_dict(),
            "rollout_records": rollout_records,
            "teacher_calls_used": teacher_calls_used,
            "teacher_probe": teacher_probe,
            "teacher_group_judgment": teacher_group_judgment,
            "grpo_update": update_summary,
        }
        if config.get("output", {}).get("include_memory_context", False):
            problem_summary["memory_context"] = memory_context.to_dict()
        _append_trajectory(config, problem_summary)
        metric_summaries.append(_metric_problem_view(problem_summary))
        if config.get("output", {}).get("include_problem_records_in_summary", False):
            problem_summaries.append(problem_summary)
        else:
            problem_summaries.append(
                {
                    "problem_id": problem.id,
                    "problem_index": problem_index,
                    "wall_time_seconds": problem_summary["wall_time_seconds"],
                    "num_rollouts": len(rollout_records),
                    "grpo_update": update_summary,
                }
            )

    final_checkpoint = str(trainer.save_final()) if trainer else None
    return {
        "mode": mode,
        "policy_backend": config.get("policy", {}).get("backend"),
        "policy_model": config.get("policy", {}).get("model_name_or_path"),
        "reference_policy": getattr(policy, "reference_policy_description", None),
        "num_problems": len(indexed_problems),
        "wall_time_seconds": time.monotonic() - started,
        "problems": problem_summaries,
        "metrics": summarize_metrics(metric_summaries),
        "memory_sizes": memory.size_summary(),
        "final_checkpoint": final_checkpoint,
        "distributed_rank": distributed.rank,
        "distributed_world_size": distributed.world_size,
        "global_dataset_rows": len(all_problems),
        "dropped_problem_ids": [problem.id for problem in dropped_problems],
    }


def _evaluate_rollout(
    problem: Problem,
    public_graph: FunctionGraph,
    verification_graph: FunctionGraph,
    rollout: PolicyRollout,
    executor: ToolExecutor,
    verifier: Verifier,
) -> Tuple[ExecutionResult, VerificationResult]:
    rollout.parsed_spans = parse_tagged_function_spans(rollout.raw_text, public_graph)
    execution = executor.execute(rollout.parsed_spans, verification_graph)
    verification = verifier.verify(
        problem,
        verification_graph,
        rollout.parsed_spans,
        execution,
    )
    return execution, verification


def _append_mistake_notebook(
    config: Dict[str, Any],
    problem: Problem,
    graph: FunctionGraph,
    rollout: PolicyRollout,
    execution: ExecutionResult,
    verification: VerificationResult,
    repair: Dict[str, Any],
) -> None:
    raw_path = config.get("output", {}).get("mistake_notebook_path")
    if not raw_path:
        return
    path = Path(str(raw_path)).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "problem": problem.to_dict(),
        "function_graph": graph.to_dict(),
        "rollout": rollout.to_dict(),
        "execution": execution.to_dict(),
        "verification": verification.to_dict(),
        "repair": repair,
    }
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def _initialize_trajectory_file(config: Dict[str, Any]) -> None:
    raw_path = config.get("output", {}).get("trajectory_path")
    if not raw_path:
        return
    path = Path(str(raw_path)).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def _append_trajectory(config: Dict[str, Any], problem_summary: Dict[str, Any]) -> None:
    raw_path = config.get("output", {}).get("trajectory_path")
    if not raw_path:
        return
    path = Path(str(raw_path)).expanduser()
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(problem_summary, ensure_ascii=False) + "\n")


def _metric_problem_view(problem_summary: Dict[str, Any]) -> Dict[str, Any]:
    records = []
    for record in problem_summary["rollout_records"]:
        records.append(
            {
                "rollout": {
                    "completion_token_count": record["rollout"].get(
                        "completion_token_count", 0
                    ),
                    "tool_call_count": sum(
                        len(span.get("code_blocks", []))
                        for span in record["rollout"].get("parsed_spans", [])
                    ),
                },
                "execution": {
                    "runtime_seconds": record["execution"].get("runtime_seconds", 0.0),
                    "executable": record["execution"].get("executable", False),
                    "timeout": record["execution"].get("timeout", False),
                    "node_results": {
                        node_id: {
                            "executable": result.get("executable", False),
                            "test_results": result.get("test_results", []),
                        }
                        for node_id, result in record["execution"].get(
                            "node_results", {}
                        ).items()
                    },
                },
                "verification": record["verification"],
                "reward": record["reward"],
                "repair": {"status": record["repair"].get("status", "not_needed")},
            }
        )
    return {
        "wall_time_seconds": problem_summary["wall_time_seconds"],
        "rollout_records": records,
    }


def _resolve_config_paths(config: Dict[str, Any], config_dir: Path) -> None:
    path_fields = [
        ("dataset", "algorithm_problem_dataset_path"),
        ("dataset", "theorem_dataset_path"),
        ("dataset", "algorithm_dataset_path"),
        ("memory", "storage_path"),
        ("training", "output_dir"),
        ("training", "resume_optimizer_path"),
        ("output", "summary_path"),
        ("output", "trajectory_path"),
        ("output", "mistake_notebook_path"),
    ]
    for section_name, key in path_fields:
        section = config.get(section_name, {})
        if section.get(key):
            section[key] = str(resolve_path(str(section[key]), config_dir))

    policy = config.get("policy", {})
    model_path = str(policy.get("model_name_or_path", ""))
    if model_path.startswith((".", "/", "~")):
        policy["model_name_or_path"] = str(resolve_path(model_path, config_dir))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Function-Structured Graph RL.")
    parser.add_argument("--config", required=True, help="Path to JSON config")
    parser.add_argument(
        "--mode",
        default="inference",
        choices=["train", "inference"],
        help="Train Qwen with GRPO or run inference with optional memory and teacher repair",
    )
    args = parser.parse_args()

    distributed = initialize_distributed(args.mode)
    try:
        config_path = Path(args.config).expanduser().resolve()
        config = load_config(config_path)
        base_paths = configure_distributed_paths(config, distributed)
        summary = run(config, args.mode, distributed)

        output_path = config.get("output", {}).get("summary_path")
        if output_path:
            path = Path(str(output_path)).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(summary, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        gathered = distributed.all_gather_object(summary)
        distributed.barrier()
        if distributed.is_primary:
            merged = merge_distributed_summaries(gathered)
            if distributed.enabled:
                for label in ("trajectory", "mistakes"):
                    raw_path = base_paths.get(label)
                    if raw_path:
                        merge_rank_jsonl(Path(raw_path), distributed.world_size)
                base_summary = base_paths.get("summary")
                if base_summary:
                    path = Path(base_summary)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(
                        json.dumps(merged, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
            print(json.dumps(merged, indent=2, ensure_ascii=False))
        distributed.barrier()
    finally:
        finalize_distributed(distributed)


if __name__ == "__main__":
    main()

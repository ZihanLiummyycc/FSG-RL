"""Configuration validation and path helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


class ConfigurationError(ValueError):
    """Raised before expensive model loading when configuration is invalid."""


def resolve_path(value: str, config_dir: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (config_dir / path).resolve()


def validate_config(config: Dict[str, Any], mode: str) -> None:
    dataset = config.get("dataset", {})
    dataset_path = dataset.get("algorithm_problem_dataset_path")
    if not dataset_path:
        raise ConfigurationError("dataset.algorithm_problem_dataset_path is required")
    if int(dataset.get("offset", 0)) < 0:
        raise ConfigurationError("dataset.offset must be non-negative")

    policy = config.get("policy", {})
    backend = policy.get("backend")
    if backend not in {"transformers", "vllm"}:
        raise ConfigurationError("policy.backend must be 'transformers' or 'vllm'")
    if not policy.get("model_name_or_path"):
        raise ConfigurationError("policy.model_name_or_path is required")
    if mode == "train" and backend != "transformers":
        raise ConfigurationError(
            "In-process GRPO updates require policy.backend='transformers'; "
            "use vLLM for inference or collection"
        )
    rollout = config.get("rollout", {})
    if mode == "train" and int(rollout.get("num_rollouts", 0)) < 2:
        raise ConfigurationError("GRPO training requires rollout.num_rollouts >= 2")
    if "enable_thinking" in rollout and not isinstance(
        rollout["enable_thinking"], bool
    ):
        raise ConfigurationError("rollout.enable_thinking must be a boolean")
    training = config.get("training", {})
    initial_update_count = int(training.get("initial_update_count", 0))
    resume_optimizer_path = str(training.get("resume_optimizer_path", "")).strip()
    if initial_update_count < 0:
        raise ConfigurationError("training.initial_update_count must be non-negative")
    if mode == "train" and bool(initial_update_count) != bool(resume_optimizer_path):
        raise ConfigurationError(
            "Resumed GRPO requires both training.initial_update_count and "
            "training.resume_optimizer_path"
        )

    reward = config.get("reward", {})
    answer_gate_floor = float(reward.get("answer_gate_floor", 1.0))
    if not 0.0 <= answer_gate_floor <= 1.0:
        raise ConfigurationError("reward.answer_gate_floor must be between 0 and 1")
    if float(reward.get("span_answer_credit", 0.0)) < 0.0:
        raise ConfigurationError("reward.span_answer_credit must be non-negative")
    if float(reward.get("lambda_teacher", 0.0)) < 0.0:
        raise ConfigurationError("reward.lambda_teacher must be non-negative")

    decomposition = config.get("decomposition", {})
    if decomposition.get("backend") not in {"policy", "dataset"}:
        raise ConfigurationError("decomposition.backend must be 'policy' or 'dataset'")

    repair = config.get("repair", {})
    teacher = config.get("teacher", {})
    repair_budget = int(repair.get("repair_budget", 1))
    if repair_budget < 0:
        raise ConfigurationError("repair.repair_budget must be non-negative")
    raw_teacher_call_budget = repair.get("max_teacher_calls_per_problem")
    if raw_teacher_call_budget is not None and int(raw_teacher_call_budget) < 0:
        raise ConfigurationError(
            "repair.max_teacher_calls_per_problem must be non-negative"
        )
    group_mode = str(repair.get("group_mode", "immediate"))
    if group_mode not in {"immediate", "shared_feedback", "group_judge"}:
        raise ConfigurationError(
            "repair.group_mode must be 'immediate', 'shared_feedback', or "
            "'group_judge'"
        )
    if mode != "train" and group_mode in {"shared_feedback", "group_judge"}:
        raise ConfigurationError(
            f"repair.group_mode={group_mode!r} is supported only in training"
        )
    if group_mode == "group_judge":
        if int(raw_teacher_call_budget or 0) != 1:
            raise ConfigurationError(
                "repair.group_mode='group_judge' requires exactly one teacher call "
                "per problem"
            )
        if float(reward.get("lambda_teacher", 0.0)) <= 0.0:
            raise ConfigurationError(
                "repair.group_mode='group_judge' requires reward.lambda_teacher > 0"
            )
        if str(teacher.get("feedback_visibility", "aggregate")) != "aggregate":
            raise ConfigurationError(
                "repair.group_mode='group_judge' requires aggregate teacher visibility"
            )
    if repair.get("enabled", False) and not teacher.get("enabled", False):
        raise ConfigurationError(
            "Teacher-guided repair requires teacher.enabled=true; disable repair for the "
            "w/o-teacher ablation"
        )
    if repair.get("enabled", False):
        _validate_api_section(teacher.get("api", {}), "teacher.api")
        feedback_visibility = str(
            teacher.get("feedback_visibility", "aggregate")
        )
        if feedback_visibility not in {"aggregate", "full"}:
            raise ConfigurationError(
                "teacher.feedback_visibility must be 'aggregate' or 'full'"
            )

    sandbox = config.get("sandbox", {})
    sandbox_backend = sandbox.get("backend", "docker")
    if sandbox_backend not in {"docker", "unshare", "local_limited", "subprocess"}:
        raise ConfigurationError(
            "sandbox.backend must be 'docker', 'unshare', 'local_limited', or "
            "'subprocess'"
        )
    if sandbox_backend in {"local_limited", "subprocess"} and not sandbox.get(
        "allow_unsafe_subprocess", False
    ):
        raise ConfigurationError(
            "Local execution runs generated code on the host; set "
            "sandbox.allow_unsafe_subprocess=true only for trusted smoke tests"
        )


def _validate_api_section(section: Dict[str, Any], name: str) -> None:
    for key in ("model", "base_url", "api_key_env"):
        if not str(section.get(key, "")).strip():
            raise ConfigurationError(f"{name}.{key} is required")

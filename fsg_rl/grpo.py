"""In-process span-aware GRPO updates for the local Transformers policy."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterator, List, Sequence, Tuple

from .rollout import TransformersPolicy
from .schemas import PolicyRollout, RewardRecord


class GRPOTrainer:
    """Runs clipped policy updates with function-span token advantages.

    The rollout-wide reward is assigned to untagged completion tokens. Tokens
    inside a function tag receive that node's group-normalized span reward.
    Prompt and padding tokens are always masked from the loss.
    """

    def __init__(self, config: Dict[str, Any], policy: TransformersPolicy):
        if not policy.trainable:
            raise ValueError("GRPOTrainer requires a trainable TransformersPolicy")
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("GRPO training requires PyTorch") from exc

        self.torch = torch
        self.policy = policy
        self.model = policy.model
        self.tokenizer = policy.tokenizer
        self.config = config.get("grpo", {})
        self.training_config = config.get("training", {})
        self.epsilon = float(self.config.get("epsilon", 0.2))
        self.beta_kl = float(self.config.get("beta_kl", 0.01))
        self.optimization_epochs = int(self.config.get("optimization_epochs", 1))
        self.logprob_chunk_size = int(self.config.get("logprob_chunk_size", 128))
        self.max_grad_norm = float(self.training_config.get("max_grad_norm", 1.0))
        self.update_count = 0
        if float(self.training_config.get("lora_dropout", 0.0)) != 0.0:
            raise ValueError(
                "GRPO requires training.lora_dropout=0 so old/new policy ratios are deterministic"
            )

        parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        if not parameters:
            raise ValueError("Policy has no trainable parameters")
        self.optimizer = torch.optim.AdamW(
            parameters,
            lr=float(self.training_config.get("learning_rate", 1e-5)),
            weight_decay=float(self.training_config.get("weight_decay", 0.0)),
        )
        self.update_count = int(self.training_config.get("initial_update_count", 0))
        resume_optimizer_path = str(
            self.training_config.get("resume_optimizer_path", "")
        ).strip()
        if resume_optimizer_path:
            optimizer_path = Path(resume_optimizer_path).expanduser()
            if not optimizer_path.is_file():
                raise FileNotFoundError(
                    f"GRPO optimizer checkpoint does not exist: {optimizer_path}"
                )
            optimizer_state = torch.load(optimizer_path, map_location="cpu")
            self.optimizer.load_state_dict(optimizer_state)
        self.distributed = bool(
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
        )
        self.distributed_rank = (
            torch.distributed.get_rank() if self.distributed else 0
        )
        self.distributed_world_size = (
            torch.distributed.get_world_size() if self.distributed else 1
        )
        if self.beta_kl > 0 and policy.reference_policy_description == "unavailable":
            raise ValueError(
                "beta_kl > 0 requires a fixed reference adapter or base reference policy"
            )

    @staticmethod
    def group_normalize(values: Sequence[float]) -> List[float]:
        if not values:
            return []
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        std = math.sqrt(variance)
        if std < 1e-8:
            return [0.0 for _ in values]
        return [(value - mean) / std for value in values]

    def update(
        self,
        rollouts: List[PolicyRollout],
        reward_records: List[RewardRecord],
    ) -> Dict[str, Any]:
        if len(rollouts) != len(reward_records) or len(rollouts) < 2:
            raise ValueError("GRPO requires at least two aligned rollouts and reward records")
        for rollout in rollouts:
            if not rollout.prompt_token_ids or not rollout.completion_token_ids:
                raise ValueError("GRPO rollouts must come from the local Transformers backend")

        advantages = self._prepare_advantages(rollouts, reward_records)
        total_tokens = sum(len(rollout.completion_token_ids) for rollout in rollouts)

        self.model.eval()
        with self.torch.no_grad():
            old_log_probs = [self._all_completion_log_probs(rollout).cpu() for rollout in rollouts]
            reference_log_probs = None
            if self.beta_kl > 0:
                with self.policy.reference_adapter_context():
                    reference_log_probs = [
                        self._all_completion_log_probs(rollout).cpu() for rollout in rollouts
                    ]

        losses: List[float] = []
        policy_objectives: List[float] = []
        kl_values: List[float] = []
        for _ in range(self.optimization_epochs):
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            epoch_loss = 0.0
            epoch_policy_objective = 0.0
            epoch_kl = 0.0
            for row, rollout in enumerate(rollouts):
                for start, end, new_log_probs in self._iter_completion_log_prob_chunks(rollout):
                    device = new_log_probs.device
                    old_chunk = old_log_probs[row][start:end].to(device)
                    advantage_chunk = advantages[row][start:end].to(device)
                    log_ratio = (new_log_probs - old_chunk).clamp(min=-20.0, max=20.0)
                    ratio = log_ratio.exp()
                    unclipped = ratio * advantage_chunk
                    clipped = (
                        ratio.clamp(1.0 - self.epsilon, 1.0 + self.epsilon)
                        * advantage_chunk
                    )
                    policy_objective = self.torch.minimum(unclipped, clipped)

                    if reference_log_probs is None:
                        kl = self.torch.zeros_like(policy_objective)
                    else:
                        reference_chunk = reference_log_probs[row][start:end].to(device)
                        ref_delta = (reference_chunk - new_log_probs).clamp(
                            min=-20.0,
                            max=20.0,
                        )
                        kl = ref_delta.exp() - ref_delta - 1.0

                    token_objective = policy_objective - self.beta_kl * kl
                    chunk_loss = -token_objective.sum() / max(1, total_tokens)
                    chunk_loss.backward()
                    epoch_loss += float(chunk_loss.detach().cpu())
                    epoch_policy_objective += float(policy_objective.detach().sum().cpu())
                    epoch_kl += float(kl.detach().sum().cpu())

            self._synchronize_gradients()

            grad_norm = self.torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in self.model.parameters() if parameter.requires_grad],
                self.max_grad_norm,
            )
            self.optimizer.step()

            losses.append(epoch_loss)
            policy_objectives.append(epoch_policy_objective / max(1, total_tokens))
            kl_values.append(epoch_kl / max(1, total_tokens))

        self.update_count += 1
        checkpoint = self._maybe_save_checkpoint()
        return {
            "update_count": self.update_count,
            "num_rollouts": len(rollouts),
            "group_advantages": self.group_normalize(
                [record.total_reward for record in reward_records]
            ),
            "loss": sum(losses) / len(losses),
            "policy_objective": sum(policy_objectives) / len(policy_objectives),
            "approx_kl": sum(kl_values) / len(kl_values),
            "epsilon": self.epsilon,
            "beta_kl": self.beta_kl,
            "grad_norm": float(grad_norm.detach().cpu()) if hasattr(grad_norm, "detach") else float(grad_norm),
            "checkpoint": str(checkpoint) if checkpoint else None,
        }

    def save_final(self) -> Path:
        output_dir = Path(str(self.training_config.get("output_dir", "outputs/policy"))).expanduser()
        final_dir = output_dir / "final"
        if self.distributed_rank == 0:
            self._save(final_dir)
        if self.distributed:
            self.torch.distributed.barrier()
        return final_dir

    def _synchronize_gradients(self) -> None:
        if not self.distributed:
            return
        for parameter in self.model.parameters():
            if not parameter.requires_grad:
                continue
            if parameter.grad is None:
                parameter.grad = self.torch.zeros_like(parameter)
            self.torch.distributed.all_reduce(
                parameter.grad,
                op=self.torch.distributed.ReduceOp.SUM,
            )
            parameter.grad.div_(self.distributed_world_size)

    def _prepare_advantages(
        self,
        rollouts: List[PolicyRollout],
        records: List[RewardRecord],
    ) -> List[Any]:
        torch = self.torch
        total_advantages = self.group_normalize([record.total_reward for record in records])
        span_keys = sorted({key for record in records for key in record.span_rewards})
        span_advantages = {
            key: self.group_normalize([record.span_rewards.get(key, 0.0) for record in records])
            for key in span_keys
        }

        advantages = []
        for row, rollout in enumerate(rollouts):
            completion_length = len(rollout.completion_token_ids)
            vector = torch.full(
                (completion_length,),
                float(total_advantages[row]),
                dtype=torch.float32,
            )

            for node_id, token_range in rollout.span_token_ranges.items():
                if node_id not in span_advantages or len(token_range) != 2:
                    continue
                start, end = token_range
                start = max(0, min(int(start), completion_length))
                end = max(start, min(int(end), completion_length))
                vector[start:end] = float(span_advantages[node_id][row])
            advantages.append(vector)
        return advantages

    def _all_completion_log_probs(self, rollout: PolicyRollout) -> Any:
        chunks = [
            log_probs
            for _, _, log_probs in self._iter_completion_log_prob_chunks(rollout)
        ]
        if not chunks:
            return self.torch.empty(0, dtype=self.torch.float32)
        return self.torch.cat(chunks, dim=0)

    def _iter_completion_log_prob_chunks(
        self,
        rollout: PolicyRollout,
    ) -> Iterator[Tuple[int, int, Any]]:
        """Score only a small completion slice at a time.

        Qwen3.5 has a roughly 248k vocabulary. Passing logits_to_keep avoids
        materializing prompt logits, while chunking bounds completion-logit
        memory independently of rollout group size.
        """

        torch = self.torch
        device = _model_input_device(self.model)
        prompt_ids = rollout.prompt_token_ids
        completion_ids = rollout.completion_token_ids
        for start in range(0, len(completion_ids), self.logprob_chunk_size):
            end = min(start + self.logprob_chunk_size, len(completion_ids))
            prefix = prompt_ids + completion_ids[:end]
            input_ids = torch.tensor(prefix, dtype=torch.long, device=device).unsqueeze(0)
            attention_mask = torch.ones_like(input_ids)
            count = end - start
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                logits_to_keep=count + 1,
            )
            logits = outputs.logits[:, -(count + 1) : -1, :]
            targets = torch.tensor(
                completion_ids[start:end],
                dtype=torch.long,
                device=device,
            ).view(1, -1, 1)
            selected = torch.log_softmax(logits, dim=-1).gather(
                dim=-1,
                index=targets,
            ).squeeze(0).squeeze(-1).float()
            yield start, end, selected
            del outputs, logits, input_ids, attention_mask, targets, selected

    def _maybe_save_checkpoint(self) -> Path | None:
        interval = int(self.training_config.get("save_every_updates", 0))
        if interval <= 0 or self.update_count % interval:
            return None
        output_dir = Path(str(self.training_config.get("output_dir", "outputs/policy"))).expanduser()
        checkpoint = output_dir / f"checkpoint-{self.update_count}"
        if self.distributed_rank == 0:
            self._save(checkpoint)
        if self.distributed:
            self.torch.distributed.barrier()
        return checkpoint

    def _save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        save_kwargs = {}
        if self.policy.reference_adapter_name and self.policy.policy_adapter_name:
            save_kwargs["selected_adapters"] = [self.policy.policy_adapter_name]
        self.model.save_pretrained(path, **save_kwargs)
        self.policy.processor.save_pretrained(path)
        self.torch.save(self.optimizer.state_dict(), path / "optimizer.pt")


def _model_input_device(model: Any) -> Any:
    try:
        return next(
            parameter.device
            for parameter in model.parameters()
            if parameter.device.type != "meta"
        )
    except StopIteration as exc:
        raise RuntimeError("Model has no materialized parameters") from exc

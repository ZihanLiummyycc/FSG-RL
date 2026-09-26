"""Small torchrun helpers for synchronous data-parallel GRPO."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence, TypeVar
import os


T = TypeVar("T")


@dataclass(frozen=True)
class DistributedContext:
    enabled: bool = False
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1

    @property
    def is_primary(self) -> bool:
        return self.rank == 0

    def barrier(self) -> None:
        if self.enabled:
            import torch.distributed as dist

            dist.barrier()

    def all_gather_object(self, value: Any) -> list[Any]:
        if not self.enabled:
            return [value]
        import torch.distributed as dist

        values: list[Any] = [None for _ in range(self.world_size)]
        dist.all_gather_object(values, value)
        return values


def initialize_distributed(mode: str) -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return DistributedContext()
    if mode != "train":
        raise ValueError("torchrun multi-GPU mode is supported only for training")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    import torch
    import torch.distributed as dist

    if not torch.cuda.is_available():
        raise RuntimeError("Distributed GRPO requires CUDA")
    torch.cuda.set_device(local_rank)
    timeout_seconds = int(
        os.environ.get("FSG_DISTRIBUTED_TIMEOUT_SECONDS", "180")
    )
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(seconds=timeout_seconds),
    )
    probe = torch.ones(1, device=torch.device("cuda", local_rank))
    dist.all_reduce(probe, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize(local_rank)
    if float(probe.item()) != float(world_size):
        raise RuntimeError(
            "NCCL connectivity probe returned an unexpected result: "
            f"{probe.item()} != {world_size}"
        )
    if rank == 0:
        print(
            f"NCCL connectivity probe passed for {world_size} ranks",
            flush=True,
        )
    return DistributedContext(
        enabled=True,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
    )


def finalize_distributed(context: DistributedContext) -> None:
    if not context.enabled:
        return
    import torch.distributed as dist

    dist.destroy_process_group()


def configure_distributed_paths(
    config: Dict[str, Any], context: DistributedContext
) -> Dict[str, str]:
    if not context.enabled:
        return {}
    config.setdefault("policy", {})["device_map"] = {"": context.local_rank}
    base_paths: Dict[str, str] = {}
    for section_name, key, label in (
        ("memory", "storage_path", "memory"),
        ("output", "summary_path", "summary"),
        ("output", "trajectory_path", "trajectory"),
        ("output", "mistake_notebook_path", "mistakes"),
    ):
        section = config.get(section_name, {})
        raw_path = section.get(key)
        if not raw_path:
            continue
        base_paths[label] = str(raw_path)
        section[key] = str(ranked_path(Path(str(raw_path)), context.rank))
    return base_paths


def shard_indexed(
    values: Sequence[T], context: DistributedContext
) -> tuple[list[tuple[int, T]], list[T]]:
    if not context.enabled:
        return list(enumerate(values)), []
    usable = len(values) - len(values) % context.world_size
    indexed = list(enumerate(values[:usable]))
    return indexed[context.rank:usable:context.world_size], list(values[usable:])


def ranked_path(path: Path, rank: int) -> Path:
    return path.with_name(f"{path.stem}.rank-{rank}{path.suffix}")


def merge_rank_jsonl(base_path: Path, world_size: int) -> None:
    temporary = base_path.with_suffix(base_path.suffix + ".tmp")
    base_path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8") as output:
        for rank in range(world_size):
            source = ranked_path(base_path, rank)
            if not source.is_file():
                continue
            with source.open(encoding="utf-8") as stream:
                for line in stream:
                    if line.strip():
                        output.write(line)
    temporary.replace(base_path)


def merge_distributed_summaries(values: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    summaries = list(values)
    if not summaries:
        return {}
    weights = [int(value.get("num_problems", 0)) for value in summaries]
    total_weight = sum(weights)
    metrics = {}
    metric_keys = {
        key
        for value in summaries
        for key in value.get("metrics", {})
    }
    for key in metric_keys:
        numerator = sum(
            float(value.get("metrics", {}).get(key, 0.0)) * weight
            for value, weight in zip(summaries, weights)
        )
        metrics[key] = numerator / total_weight if total_weight else 0.0

    primary = dict(summaries[0])
    primary.update(
        {
            "num_problems": total_weight,
            "wall_time_seconds": max(
                float(value.get("wall_time_seconds", 0.0)) for value in summaries
            ),
            "problems": sorted(
                [problem for value in summaries for problem in value.get("problems", [])],
                key=lambda problem: int(problem.get("problem_index", 0)),
            ),
            "metrics": metrics,
            "memory_sizes": {
                key: sum(
                    int(value.get("memory_sizes", {}).get(key, 0))
                    for value in summaries
                )
                for key in {
                    item
                    for value in summaries
                    for item in value.get("memory_sizes", {})
                }
            },
            "distributed": {
                "world_size": len(summaries),
                "local_problem_counts": weights,
                "gradient_synchronization": "mean_all_reduce",
            },
        }
    )
    return primary

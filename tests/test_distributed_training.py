from __future__ import annotations

from pathlib import Path
import unittest

from fsg_rl.distributed_training import (
    DistributedContext,
    merge_distributed_summaries,
    ranked_path,
    shard_indexed,
)
from fsg_rl.grpo import GRPOTrainer


class DistributedTrainingTests(unittest.TestCase):
    def test_two_way_shard_uses_all_866_rows(self):
        values = list(range(866))
        shards = []
        for rank in range(2):
            local, dropped = shard_indexed(
                values,
                DistributedContext(True, rank, rank, 2),
            )
            shards.extend(index for index, _ in local)
            self.assertEqual(len(local), 433)
            self.assertEqual(dropped, [])
        self.assertEqual(sorted(shards), list(range(866)))

    def test_six_way_shard_drops_only_remainder(self):
        values = list(range(866))
        shards = []
        dropped = None
        for rank in range(6):
            local, local_dropped = shard_indexed(
                values,
                DistributedContext(True, rank, rank, 6),
            )
            shards.extend(index for index, _ in local)
            dropped = local_dropped
            self.assertEqual(len(local), 144)
        self.assertEqual(sorted(shards), list(range(864)))
        self.assertEqual(dropped, [864, 865])

    def test_ranked_path_preserves_suffix(self):
        self.assertEqual(
            ranked_path(Path("/tmp/trajectories.jsonl"), 3),
            Path("/tmp/trajectories.rank-3.jsonl"),
        )

    def test_summary_merge_is_problem_weighted(self):
        merged = merge_distributed_summaries(
            [
                {
                    "num_problems": 2,
                    "wall_time_seconds": 10,
                    "problems": [{"problem_index": 0}],
                    "metrics": {"rollout_accuracy": 0.5},
                    "memory_sizes": {"failure_cases": 2},
                },
                {
                    "num_problems": 1,
                    "wall_time_seconds": 12,
                    "problems": [{"problem_index": 1}],
                    "metrics": {"rollout_accuracy": 1.0},
                    "memory_sizes": {"failure_cases": 1},
                },
            ]
        )
        self.assertEqual(merged["num_problems"], 3)
        self.assertAlmostEqual(merged["metrics"]["rollout_accuracy"], 2 / 3)
        self.assertEqual(merged["memory_sizes"]["failure_cases"], 3)
        self.assertEqual(merged["wall_time_seconds"], 12)

    def test_grpo_gradient_synchronization_averages_all_ranks(self):
        class FakeGradient:
            def __init__(self, value):
                self.value = value

            def div_(self, divisor):
                self.value /= divisor

        class FakeParameter:
            requires_grad = True

            def __init__(self, value):
                self.grad = FakeGradient(value)

        class FakeModel:
            def __init__(self, parameters):
                self._parameters = parameters

            def parameters(self):
                return self._parameters

        class FakeReduceOp:
            SUM = "sum"

        class FakeDistributed:
            ReduceOp = FakeReduceOp

            def __init__(self):
                self.calls = []

            def all_reduce(self, gradient, *, op):
                self.calls.append((gradient, op))

        class FakeTorch:
            def __init__(self):
                self.distributed = FakeDistributed()

        trainer = object.__new__(GRPOTrainer)
        trainer.distributed = True
        trainer.distributed_world_size = 6
        trainer.torch = FakeTorch()
        parameters = [FakeParameter(12.0), FakeParameter(6.0)]
        trainer.model = FakeModel(parameters)

        trainer._synchronize_gradients()

        self.assertEqual([parameter.grad.value for parameter in parameters], [2.0, 1.0])
        self.assertEqual(
            [op for _, op in trainer.torch.distributed.calls],
            ["sum", "sum"],
        )


if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import rollout


def make_dataset(num_trajectories: int, trajectory_length: int, episode_id_start: int) -> dict[str, np.ndarray]:
    size = num_trajectories * trajectory_length
    observations = np.arange(size, dtype=np.float32).reshape(-1, 1)
    episode_ids = np.repeat(
        np.arange(episode_id_start, episode_id_start + num_trajectories), trajectory_length
    )
    timeouts = np.zeros(size, dtype=bool)
    timeouts[trajectory_length - 1 :: trajectory_length] = True
    return {
        "observations": observations,
        "actions": np.zeros((size, 1), dtype=np.float32),
        "next_observations": observations + 1,
        "rewards": np.ones(size, dtype=np.float32),
        "terminals": np.zeros(size, dtype=bool),
        "timeouts": timeouts,
        "episode_ids": episode_ids,
    }


class GeneratedDatasetTests(unittest.TestCase):
    def test_expert_loader_rejects_unsupported_task(self):
        with self.assertRaisesRegex(ValueError, "Unsupported generated-data task 'Ant-v5'"):
            rollout.load_expert_policy("Ant-v5", "expert.zip")

    def test_collection_helpers_reject_unsupported_task_before_creating_env(self):
        with patch.object(rollout, "_collect_source") as collect_source:
            with self.assertRaisesRegex(ValueError, "Unsupported task 'Ant-v5'"):
                rollout.collect_suboptimal(
                    "Ant-v5", "expert.zip", num_trajectories=1,
                    max_timesteps=1,
                )
            with self.assertRaisesRegex(ValueError, "Unsupported task 'Ant-v5'"):
                rollout.collect_dataset(
                    "Ant-v5", "expert.zip", num_samples=1,
                    prop_clean_expert=0.0,
                )
        collect_source.assert_not_called()

    def test_allocates_complete_trajectories_by_source(self):
        calls = []

        def collect_expert(**kwargs):
            calls.append((
                "expert", kwargs["num_trajectories"], kwargs["noise_scale"],
                kwargs["episode_id_start"],
            ))
            return make_dataset(kwargs["num_trajectories"], 10, kwargs["episode_id_start"])

        def collect_random(**kwargs):
            calls.append(("random", kwargs["num_trajectories"], kwargs["episode_id_start"]))
            return make_dataset(kwargs["num_trajectories"], 10, kwargs["episode_id_start"])

        with patch.object(rollout, "collect_expert", side_effect=collect_expert), patch.object(
            rollout, "collect_suboptimal", side_effect=collect_random
        ):
            dataset, metadata = rollout.collect_dataset(
                "Reacher-v5", "expert.zip", max_timesteps=10, num_samples=100, noise_scale=0.3,
                prop_clean_expert=0.2, prop_noisy_expert=0.5, seed=0,
            )

        self.assertEqual(
            calls,
            [("expert", 2, 0.0, 0), ("expert", 5, 0.3, 2), ("random", 3, 7)],
        )
        np.testing.assert_array_equal(np.unique(dataset["episode_ids"]), np.arange(10))
        np.testing.assert_array_equal(
            np.bincount(dataset["episode_ids"]), np.full(10, 10)
        )
        self.assertEqual(metadata["requested_num_samples"], 100)
        self.assertEqual(metadata["num_clean_expert_trajectories"], 2)
        self.assertEqual(metadata["num_noisy_expert_trajectories"], 5)
        self.assertEqual(metadata["num_random_trajectories"], 3)
        self.assertAlmostEqual(metadata["actual_prop_clean_expert_trajectories"], 0.2)
        self.assertAlmostEqual(metadata["actual_prop_noisy_expert_trajectories"], 0.5)
        self.assertAlmostEqual(metadata["actual_prop_random_trajectories"], 0.3)

    def test_short_episodes_add_whole_trajectories(self):
        calls = []

        def collect_expert(**kwargs):
            calls.append((kwargs["num_trajectories"], kwargs["episode_id_start"]))
            return make_dataset(kwargs["num_trajectories"], 3, kwargs["episode_id_start"])

        with patch.object(rollout, "collect_expert", side_effect=collect_expert):
            dataset, metadata = rollout.collect_dataset(
                "Reacher-v5", "expert.zip", max_timesteps=5, num_samples=10,
                prop_clean_expert=1.0, seed=0,
            )

        self.assertEqual(calls, [(2, 0), (2, 2)])
        self.assertEqual(len(dataset["rewards"]), 12)
        np.testing.assert_array_equal(np.bincount(dataset["episode_ids"]), np.full(4, 3))
        self.assertEqual(metadata["num_trajectories"], 4)
        self.assertEqual(metadata["num_transitions"], 12)

    def test_rejects_composition_above_one(self):
        with self.assertRaisesRegex(ValueError, "cannot sum above 1"):
            rollout.collect_dataset(
                "Reacher-v5", "expert.zip",
                prop_clean_expert=0.6, prop_noisy_expert=0.5,
            )

    def test_scalar_dataset_array_is_rejected_as_corrupt(self):
        dataset = make_dataset(2, 3, 0)
        dataset["rewards"] = np.asarray(1.0, dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "transition axis"):
            rollout.validate_dataset(dataset)

    def test_failed_atomic_save_preserves_existing_dataset(self):
        dataset = make_dataset(2, 3, 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.npz"
            path.write_bytes(b"existing")

            with patch.object(
                rollout.np, "savez_compressed", side_effect=RuntimeError("write failed")
            ), self.assertRaisesRegex(RuntimeError, "write failed"):
                rollout.save_dataset(dataset, path)

            self.assertEqual(path.read_bytes(), b"existing")
            self.assertEqual(list(path.parent.glob(".dataset.npz.*.tmp")), [])

    def test_atomic_save_round_trip(self):
        dataset = make_dataset(2, 3, 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.npz"
            rollout.save_dataset(dataset, path)
            loaded = rollout.load_dataset(path)

        for key in rollout.DATASET_KEYS:
            np.testing.assert_array_equal(loaded[key], dataset[key])


if __name__ == "__main__":
    unittest.main()

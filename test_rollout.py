import unittest
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
                "TestEnv-v0", "expert.zip", max_timesteps=10, num_samples=100, noise_scale=0.3,
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
                "TestEnv-v0", "expert.zip", max_timesteps=5, num_samples=10,
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
                "TestEnv-v0", "expert.zip",
                prop_clean_expert=0.6, prop_noisy_expert=0.5,
            )


if __name__ == "__main__":
    unittest.main()

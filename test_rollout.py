import unittest
from unittest.mock import patch

import numpy as np

import rollout


def make_dataset(size: int, episode_id: int) -> dict[str, np.ndarray]:
    observations = np.arange(size, dtype=np.float32).reshape(-1, 1)
    return {
        "observations": observations,
        "actions": np.zeros((size, 1), dtype=np.float32),
        "next_observations": observations + 1,
        "rewards": np.ones(size, dtype=np.float32),
        "terminals": np.zeros(size, dtype=bool),
        "timeouts": np.zeros(size, dtype=bool),
        "episode_ids": np.full(size, episode_id, dtype=np.int64),
    }


class GeneratedDatasetTests(unittest.TestCase):
    def test_collects_clean_noisy_and_random_components(self):
        calls = []

        def collect_expert(**kwargs):
            calls.append(("expert", kwargs["num_samples"], kwargs["noise_scale"], kwargs["episode_id_start"]))
            return make_dataset(kwargs["num_samples"], kwargs["episode_id_start"])

        def collect_random(**kwargs):
            calls.append(("random", kwargs["num_samples"], kwargs["episode_id_start"]))
            return make_dataset(kwargs["num_samples"], kwargs["episode_id_start"])

        with patch.object(rollout, "collect_expert", side_effect=collect_expert), patch.object(
            rollout, "collect_suboptimal", side_effect=collect_random
        ):
            dataset, metadata = rollout.collect_dataset(
                "TestEnv-v0", "expert.zip", num_samples=10, noise_scale=0.3,
                prop_clean_expert=0.2, prop_noisy_expert=0.5, seed=0,
            )

        self.assertEqual(calls, [("expert", 2, 0.0, 0), ("expert", 5, 0.3, 1), ("random", 3, 2)])
        np.testing.assert_array_equal(dataset["episode_ids"], [0, 0, 1, 1, 1, 1, 1, 2, 2, 2])
        self.assertEqual(metadata["requested_num_clean_expert"], 2)
        self.assertEqual(metadata["requested_num_noisy_expert"], 5)
        self.assertEqual(metadata["requested_num_random"], 3)
        self.assertAlmostEqual(metadata["actual_prop_clean_expert"], 0.2)
        self.assertAlmostEqual(metadata["actual_prop_noisy_expert"], 0.5)
        self.assertAlmostEqual(metadata["actual_prop_random"], 0.3)

    def test_rejects_composition_above_one(self):
        with self.assertRaisesRegex(ValueError, "cannot sum above 1"):
            rollout.collect_dataset(
                "TestEnv-v0", "expert.zip",
                prop_clean_expert=0.6, prop_noisy_expert=0.5,
            )


if __name__ == "__main__":
    unittest.main()

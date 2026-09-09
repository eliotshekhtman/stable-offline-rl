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

    def test_allocates_transition_quotas_by_source(self):
        calls = []

        def collect_expert(**kwargs):
            calls.append((
                "expert", kwargs["min_transitions"], kwargs["noise_scale"],
                kwargs["episode_id_start"],
            ))
            count = int(np.ceil(kwargs["min_transitions"] / 10))
            return make_dataset(count, 10, kwargs["episode_id_start"])

        def collect_random(**kwargs):
            calls.append(("random", kwargs["min_transitions"], kwargs["episode_id_start"]))
            count = int(np.ceil(kwargs["min_transitions"] / 10))
            return make_dataset(count, 10, kwargs["episode_id_start"])

        with patch.object(rollout, "collect_expert", side_effect=collect_expert), patch.object(
            rollout, "collect_suboptimal", side_effect=collect_random
        ):
            dataset, metadata = rollout.collect_dataset(
                "Reacher-v5", "expert.zip", max_timesteps=10, num_samples=100, noise_scale=0.3,
                prop_clean_expert=0.2, prop_noisy_expert=0.5, seed=0,
            )

        self.assertEqual(
            calls,
            [("expert", 20, 0.0, 0), ("expert", 50, 0.3, 2), ("random", 30, 7)],
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

    def test_short_episodes_add_whole_trajectories_without_reopening_env(self):
        def trajectory(env, action_fn, max_timesteps, episode_id, seed):
            return make_dataset(1, 3, episode_id)

        with patch.object(rollout.gym, "make") as make_env, patch.object(
            rollout, "collect_traj", side_effect=trajectory
        ) as collect_traj:
            dataset = rollout._collect_source(
                "Reacher-v5", lambda env: object(), None, 5,
                np.random.default_rng(0), episode_id_start=7, min_transitions=10,
            )
        make_env.assert_called_once_with("Reacher-v5")
        make_env.return_value.close.assert_called_once()
        self.assertEqual(collect_traj.call_count, 4)
        self.assertEqual(len(dataset["rewards"]), 12)
        np.testing.assert_array_equal(np.unique(dataset["episode_ids"]), [7, 8, 9, 10])

    def test_unequal_source_lengths_meet_each_quota_and_skip_zero_sources(self):
        for noisy_fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
            calls = []

            def collect_expert(**kwargs):
                length = 3 if kwargs["noise_scale"] else 10
                target = kwargs["min_transitions"]
                calls.append((kwargs["noise_scale"], target))
                return make_dataset(
                    int(np.ceil(target / length)), length, kwargs["episode_id_start"]
                )

            with self.subTest(noisy_fraction=noisy_fraction), patch.object(
                rollout, "collect_expert", side_effect=collect_expert
            ), patch.object(rollout, "collect_suboptimal") as random_source:
                dataset, metadata = rollout.collect_dataset(
                    "Walker2d-v5", "expert.zip", max_timesteps=10, num_samples=100,
                    noise_scale=0.5, prop_clean_expert=1.0 - noisy_fraction,
                    prop_noisy_expert=noisy_fraction, seed=10000,
                )
                random_source.assert_not_called()
                self.assertGreaterEqual(len(dataset["rewards"]), 100)
                for name, share, length in (
                    ("clean_expert", 1.0 - noisy_fraction, 10),
                    ("noisy_expert", noisy_fraction, 3),
                ):
                    quota = int(np.ceil(100 * share))
                    actual = metadata[f"num_{name}_transitions"]
                    self.assertGreaterEqual(actual, quota)
                    self.assertLess(actual, quota + length)
                self.assertEqual(len(calls), 1 if noisy_fraction in (0.0, 1.0) else 2)
                ids = np.unique(dataset["episode_ids"])
                np.testing.assert_array_equal(ids, np.arange(len(ids)))

    def test_variable_episode_lengths_reproducible_with_bounded_overshoot(self):
        def collect_once():
            def trajectory(env, action_fn, max_timesteps, episode_id, seed):
                length = (2, 7, 3, 5)[episode_id]
                data = make_dataset(1, length, episode_id)
                data["observations"][:] = seed
                data["timeouts"][-1] = False
                data["terminals"][-1] = True
                return data
            with patch.object(rollout.gym, "make"), patch.object(
                rollout, "collect_traj", side_effect=trajectory
            ) as collect_traj:
                data = rollout._collect_source(
                    "Walker2d-v5", lambda env: object(), None, 10,
                    np.random.default_rng(13), min_transitions=10,
                )
                self.assertEqual(collect_traj.call_count, 3)
                return data
        first, second = collect_once(), collect_once()
        for key in rollout.DATASET_KEYS:
            np.testing.assert_array_equal(first[key], second[key])
        self.assertEqual(len(first["rewards"]), 12)
        self.assertEqual(first["terminals"].sum(), 3)
        self.assertFalse(first["timeouts"].any())

    def test_fixed_count_helper_and_quota_helper_preserve_noise_and_seed_sequence(self):
        # Reacher episodes have fixed length, so the two helper stopping rules
        # must produce identical actions and observations for identical RNGs.
        class Expert:
            def predict(self, obs, deterministic):
                assert deterministic
                return np.array([0.95, -0.95], dtype=np.float32), None

        with patch.object(rollout, "load_expert_policy", return_value=Expert()):
            fixed = rollout.collect_expert(
                "Reacher-v5", "expert.zip", 2, 50, noise_scale=0.5,
                rng=np.random.default_rng(10000),
            )
            quota = rollout.collect_expert(
                "Reacher-v5", "expert.zip", None, 50, noise_scale=0.5,
                rng=np.random.default_rng(10000), min_transitions=100,
            )
        for key in rollout.DATASET_KEYS:
            np.testing.assert_array_equal(fixed[key], quota[key])
        self.assertTrue(np.all(np.abs(quota["actions"]) <= 1.0))

    def test_collection_target_validation_precedes_environment_creation(self):
        with patch.object(rollout.gym, "make") as make_env:
            for count, minimum in ((None, None), (1, 1), (None, 0), (None, -1), (None, 1.5)):
                with self.subTest(count=count, minimum=minimum), self.assertRaises(ValueError):
                    rollout._collect_source(
                        "Walker2d-v5", lambda env: object(), count, 10,
                        np.random.default_rng(0), min_transitions=minimum,
                    )
            make_env.assert_not_called()

    def test_quota_rounding_and_zero_random_share(self):
        np.testing.assert_array_equal(rollout.transition_quotas(100, [0.2, 0.5, 0.3]), [20, 50, 30])
        np.testing.assert_array_equal(rollout.transition_quotas(101, [0.5, 0.5]), [51, 51])
        with patch.object(rollout, "collect_expert") as expert, patch.object(
            rollout, "collect_suboptimal"
        ) as random_source:
            expert.side_effect = lambda **kw: make_dataset(1, kw["min_transitions"], kw["episode_id_start"])
            _, metadata = rollout.collect_dataset(
                "Reacher-v5", "expert.zip", num_samples=100,
                prop_clean_expert=0.7, prop_noisy_expert=0.3,
            )
        random_source.assert_not_called()
        self.assertEqual(metadata["requested_transition_quotas"], {"clean_expert": 70, "noisy_expert": 30, "random": 0})

    def test_random_only_collection_meets_quota_without_loading_an_expert(self):
        with patch.object(rollout, "load_expert_policy") as expert:
            first, metadata = rollout.collect_dataset(
                "Reacher-v5", "unused.zip", max_timesteps=7, num_samples=20,
                prop_clean_expert=0.0, prop_noisy_expert=0.0, seed=10000,
            )
            second, _ = rollout.collect_dataset(
                "Reacher-v5", "unused.zip", max_timesteps=7, num_samples=20,
                prop_clean_expert=0.0, prop_noisy_expert=0.0, seed=10000,
            )
        expert.assert_not_called()
        self.assertEqual(metadata["num_random_transitions"], 21)
        np.testing.assert_array_equal(np.bincount(first["episode_ids"]), [7, 7, 7])
        for key in rollout.DATASET_KEYS:
            np.testing.assert_array_equal(first[key], second[key])

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

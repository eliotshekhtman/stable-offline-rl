import unittest
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np

import load_offline


class RobomimicDatasetTests(unittest.TestCase):
    def test_only_lift_and_can_are_supported(self):
        self.assertEqual(set(load_offline.ROBOMIMIC_LOW_DIM_DATASETS), {"Lift", "Can"})
        with self.assertRaisesRegex(ValueError, "Unsupported task 'Square'"):
            load_offline.list_robomimic_dataset_specs("Square")
        with self.assertRaisesRegex(ValueError, "Unsupported task 'Square'"):
            load_offline.make_robomimic_dataset_tag({
                "task": "Square", "dataset_type": "ph",
            })

    def test_selects_one_robomimic_dataset(self):
        self.assertEqual(len(load_offline.list_robomimic_dataset_specs("Can")), 5)
        specs = load_offline.list_robomimic_dataset_specs("Can", "mh")
        self.assertEqual([spec["dataset_type"] for spec in specs], ["mh"])
        self.assertNotIn("task_semantics", specs[0])
        lift_spec = load_offline.list_robomimic_dataset_specs("Lift", "mg_dense")[0]
        self.assertEqual(lift_spec["task_semantics"], "continuing")
        self.assertEqual(
            load_offline.make_robomimic_dataset_tag(lift_spec),
            "robomimic_lift_mg_dense_continuing",
        )
        with self.assertRaisesRegex(ValueError, "available: ph, mh, mg_sparse, mg_dense, paired"):
            load_offline.list_robomimic_dataset_specs("Can", "unknown")

    def test_lift_discards_success_terminals_but_other_tasks_retain_them(self):
        with h5py.File("robomimic-test.hdf5", "w", driver="core", backing_store=False) as file:
            demo = file.create_group("demo")
            obs = demo.create_group("obs")
            next_obs = demo.create_group("next_obs")
            for key in load_offline.ROBOMIMIC_OBS_KEYS:
                values = np.arange(4, dtype=np.float32).reshape(-1, 1)
                obs.create_dataset(key, data=values)
                next_obs.create_dataset(key, data=values + 1)
            demo.create_dataset("actions", data=np.zeros((4, 2), dtype=np.float32))
            demo.create_dataset("rewards", data=np.asarray([0, 1, 2, 3], dtype=np.float32))
            demo.create_dataset("dones", data=np.asarray([False, True, False, False]))

            can = load_offline.robomimic_demo_to_transitions(demo, episode_id=7, task="Can")
            lift = load_offline.robomimic_demo_to_transitions(demo, episode_id=7, task="Lift")

        np.testing.assert_array_equal(can["rewards"], [0, 1, 2, 3])
        np.testing.assert_array_equal(can["terminals"], [False, True, False, False])
        np.testing.assert_array_equal(can["timeouts"], [False, False, False, True])
        np.testing.assert_array_equal(lift["terminals"], [False, False, False, False])
        np.testing.assert_array_equal(lift["timeouts"], [False, False, False, True])
        np.testing.assert_array_equal(lift["episode_ids"], [7, 7, 7, 7])


class MinariDatasetTests(unittest.TestCase):
    def test_only_supported_mujoco_tasks_have_minari_prefixes(self):
        self.assertEqual(
            set(load_offline.MINARI_PREFIXES),
            {"Reacher-v5", "HalfCheetah-v5", "Walker2d-v5"},
        )
        with self.assertRaisesRegex(ValueError, "Unsupported task 'Ant-v5'"):
            load_offline.list_minari_dataset_ids("Ant-v5")
        with self.assertRaisesRegex(ValueError, "Unsupported Minari dataset"):
            load_offline.make_minari_dataset_tag("mujoco/ant/expert-v0")

    @patch("minari.load_dataset")
    def test_direct_minari_loader_rejects_unsupported_prefix(self, load_dataset):
        with self.assertRaisesRegex(ValueError, "Unsupported Minari dataset"):
            load_offline.load_minari_dataset("mujoco/ant/expert-v0")
        load_dataset.assert_not_called()

    @patch("minari.list_remote_datasets")
    def test_selects_one_minari_dataset_by_leaf_or_full_id(self, list_remote_datasets):
        list_remote_datasets.return_value = {
            "mujoco/halfcheetah/simple-v0": {},
            "mujoco/halfcheetah/medium-v0": {},
            "mujoco/halfcheetah/expert-v0": {},
        }
        all_ids = load_offline.list_minari_dataset_ids("HalfCheetah-v5")
        self.assertEqual(len(all_ids), 3)
        expected = ["mujoco/halfcheetah/medium-v0"]
        self.assertEqual(load_offline.list_minari_dataset_ids("HalfCheetah-v5", "medium-v0"), expected)
        self.assertEqual(load_offline.list_minari_dataset_ids("HalfCheetah-v5", expected[0]), expected)
        with self.assertRaisesRegex(ValueError, "available: expert-v0, medium-v0, simple-v0"):
            load_offline.list_minari_dataset_ids("HalfCheetah-v5", "unknown-v0")

    @patch("minari.load_dataset")
    def test_loads_whole_episodes_until_transition_quota_without_replacement(self, load_dataset):
        episodes = []
        lengths = [3, 1, 5, 2]
        for episode_id, length in enumerate(lengths):
            episodes.append(SimpleNamespace(
                id=episode_id,
                observations=(10 * episode_id + np.arange(length + 1, dtype=np.float32))[:, None],
                actions=np.zeros((length, 1), dtype=np.float32),
                rewards=np.full(length, episode_id, dtype=np.float32),
                terminations=np.asarray([False] * (length - 1) + [episode_id % 2 == 0]),
                truncations=np.asarray([False] * (length - 1) + [episode_id % 2 == 1]),
            ))

        minari_dataset = SimpleNamespace(
            total_episodes=4,
            total_steps=sum(lengths),
            episode_indices=np.arange(4),
            env_spec=SimpleNamespace(id="Reacher-v5"),
        )
        yielded_indices = []

        def iterate_episodes(indices):
            for index in indices:
                yielded_indices.append(index)
                yield episodes[index]

        minari_dataset.iterate_episodes = iterate_episodes
        load_dataset.return_value = minari_dataset

        dataset, metadata = load_offline.load_minari_transition_subset(
            "mujoco/reacher/medium-v0", num_transitions=7, seed=7, episode_id_start=10
        )
        load_dataset.assert_called_once_with("mujoco/reacher/medium-v0", download=True)
        permutation = np.random.default_rng(7).permutation(4)
        count = np.searchsorted(np.cumsum(np.asarray(lengths)[permutation]), 7) + 1
        expected_indices = permutation[:count]
        expected_lengths = np.asarray(lengths)[expected_indices]
        np.testing.assert_array_equal(yielded_indices, expected_indices)
        self.assertEqual(len(set(yielded_indices)), len(yielded_indices))
        np.testing.assert_array_equal(
            dataset["observations"],
            np.concatenate([episodes[index].observations[:-1] for index in expected_indices]),
        )
        np.testing.assert_array_equal(
            dataset["next_observations"],
            np.concatenate([episodes[index].observations[1:] for index in expected_indices]),
        )
        np.testing.assert_array_equal(
            dataset["episode_ids"], np.repeat(np.arange(10, 10 + count), expected_lengths)
        )
        for key, attribute in (("terminals", "terminations"), ("timeouts", "truncations")):
            np.testing.assert_array_equal(
                dataset[key],
                np.concatenate([getattr(episodes[index], attribute) for index in expected_indices]),
            )
        self.assertEqual(metadata, {
            "dataset_id": "mujoco/reacher/medium-v0",
            "env_id": "Reacher-v5",
            "available_num_episodes": 4,
            "available_num_transitions": sum(lengths),
            "num_episodes": count,
            "num_transitions": sum(expected_lengths),
            "seed": 7,
        })
        self.assertGreaterEqual(metadata["num_transitions"], 7)
        self.assertLess(metadata["num_transitions"] - 7, expected_lengths[-1])

        repeated, repeated_metadata = load_offline.load_minari_transition_subset(
            "mujoco/reacher/medium-v0", num_transitions=7, seed=7, episode_id_start=10
        )
        self.assertEqual(repeated_metadata, metadata)
        for key in dataset:
            np.testing.assert_array_equal(repeated[key], dataset[key])

        complete, complete_metadata = load_offline.load_minari_transition_subset(
            "mujoco/reacher/medium-v0", num_transitions=sum(lengths), seed=7
        )
        self.assertEqual(complete_metadata["num_transitions"], sum(lengths))
        self.assertEqual(complete_metadata["num_episodes"], len(episodes))
        np.testing.assert_array_equal(
            complete["episode_ids"], np.repeat(np.arange(4), np.asarray(lengths)[permutation])
        )

        yielded_indices.clear()
        with self.assertRaisesRegex(ValueError, "contains only 11 transitions"):
            load_offline.load_minari_transition_subset(
                "mujoco/reacher/medium-v0", num_transitions=12, seed=7
            )
        self.assertEqual(yielded_indices, [])

        minari_dataset.total_steps = 20
        with self.assertRaisesRegex(ValueError, "exhausted its episodes after 11 transitions"):
            load_offline.load_minari_transition_subset(
                "mujoco/reacher/medium-v0", num_transitions=12, seed=7
            )
        self.assertEqual(len(yielded_indices), len(episodes))
        self.assertEqual(len(set(yielded_indices)), len(episodes))

    @patch("minari.load_dataset")
    def test_transition_quota_must_be_positive_before_loading(self, load_dataset):
        for quota in (0, -1):
            with self.subTest(quota=quota):
                with self.assertRaisesRegex(ValueError, "num_transitions must be positive"):
                    load_offline.load_minari_transition_subset(
                        "mujoco/reacher/medium-v0", num_transitions=quota, seed=7
                    )
        load_dataset.assert_not_called()

    @patch("minari.list_remote_datasets")
    def test_walker_uses_its_own_minari_prefix(self, list_remote_datasets):
        list_remote_datasets.return_value = {
            "mujoco/walker2d/simple-v0": {},
            "mujoco/walker2d/medium-v0": {},
            "mujoco/walker2d/expert-v0": {},
        }
        self.assertEqual(
            load_offline.list_minari_dataset_ids("Walker2d-v5", "medium-v0"),
            ["mujoco/walker2d/medium-v0"],
        )
        list_remote_datasets.assert_called_once_with(prefix="mujoco/walker2d")


if __name__ == "__main__":
    unittest.main()

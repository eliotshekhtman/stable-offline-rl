import unittest
from unittest.mock import patch

import h5py
import numpy as np

import load_offline


class RobomimicDatasetTests(unittest.TestCase):
    def test_selects_one_robomimic_dataset(self):
        self.assertEqual(len(load_offline.list_robomimic_dataset_specs("Can")), 5)
        specs = load_offline.list_robomimic_dataset_specs("Can", "mh")
        self.assertEqual([spec["dataset_type"] for spec in specs], ["mh"])
        with self.assertRaisesRegex(ValueError, "available: ph, mh, mg_sparse, mg_dense, paired"):
            load_offline.list_robomimic_dataset_specs("Can", "unknown")

    def test_internal_done_does_not_truncate_demo(self):
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

            dataset = load_offline.robomimic_demo_to_transitions(demo, episode_id=7)

        np.testing.assert_array_equal(dataset["rewards"], [0, 1, 2, 3])
        np.testing.assert_array_equal(dataset["terminals"], [False, True, False, False])
        np.testing.assert_array_equal(dataset["timeouts"], [False, False, False, True])
        np.testing.assert_array_equal(dataset["episode_ids"], [7, 7, 7, 7])


class MinariDatasetTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()

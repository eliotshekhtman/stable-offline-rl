import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import chunking
from policies import termination_fn_never
import sweep


class MobileShiftArgumentTests(unittest.TestCase):
    def parse(self, *extra):
        with patch.object(sys, "argv", ["sweep.py", "--env", "Reacher-v5", *extra]):
            return sweep.parse_args()

    def test_default_and_override(self):
        defaults = self.parse()
        self.assertEqual(defaults.mobile_return_shift, 30.0)
        self.assertEqual(defaults.mopo_penalty_coef, 0.5)
        self.assertEqual(defaults.mobile_penalty_coef, 1.5)
        self.assertIsNone(defaults.model_actor_learning_rate)
        self.assertEqual(defaults.model_critic_learning_rate, 3e-4)
        self.assertFalse(defaults.model_manipulation_settings)
        self.assertEqual(
            self.parse("--mobile-return-shift", "17.5").mobile_return_shift,
            17.5,
        )

    def test_dynamics_chunk_mode_default_and_override(self):
        self.assertEqual(self.parse().dynamics_chunk_mode, "direct")
        self.assertEqual(
            self.parse("--dynamics-chunk-mode", "recursive").dynamics_chunk_mode,
            "recursive",
        )

    def test_recursive_dynamics_is_schema_equivalent_at_chunk_length_one(self):
        direct_args = self.parse()
        direct_args.seed = 0
        recursive_args = self.parse("--dynamics-chunk-mode", "recursive")
        recursive_args.seed = 0

        direct_schema = sweep.make_training_schema(
            "mopo", "Lift", {"source": "test"}, 1, direct_args
        )
        recursive_schema = sweep.make_training_schema(
            "mopo", "Lift", {"source": "test"}, 1, recursive_args
        )

        self.assertEqual(recursive_schema, direct_schema)
        self.assertNotIn("chunk_dynamics", recursive_schema["model_based"])

    def test_recursive_dynamics_only_changes_higher_chunk_schema(self):
        direct_args = self.parse()
        direct_args.seed = 0
        recursive_args = self.parse("--dynamics-chunk-mode", "recursive")
        recursive_args.seed = 0

        direct_schema = sweep.make_training_schema(
            "mopo", "Lift", {"source": "test"}, 4, direct_args
        )
        recursive_schema = sweep.make_training_schema(
            "mopo", "Lift", {"source": "test"}, 4, recursive_args
        )

        recursive_block = recursive_schema["model_based"].pop("chunk_dynamics")
        self.assertEqual(recursive_block, {"version": 1, "mode": "recursive"})
        self.assertEqual(recursive_schema, direct_schema)

    def test_old_chunk_one_schema_remains_discoverable(self):
        direct_args = self.parse()
        direct_args.seed = 0
        recursive_args = self.parse("--dynamics-chunk-mode", "recursive")
        recursive_args.seed = 0
        old_schema = sweep.make_training_schema(
            "mopo", "Lift", {"source": "test"}, 1, direct_args
        )
        requested_schema = sweep.make_training_schema(
            "mopo", "Lift", {"source": "test"}, 1, recursive_args
        )

        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "old-run"
            run_dir.mkdir()
            (run_dir / "run_manifest.json").write_text(
                json.dumps({"training_schema": old_schema}), encoding="utf-8"
            )
            with patch("sweep.run_is_complete", return_value=True):
                self.assertEqual(
                    sweep.find_trained_run(Path(directory), requested_schema),
                    run_dir,
                )

    def test_output_and_evaluation_defaults(self):
        args = self.parse()
        self.assertFalse(args.quiet)
        self.assertEqual(args.checkpoint_eval_episodes, 20)
        self.assertEqual(args.final_eval_episodes, 100)
        self.assertEqual(args.minari_fractions, [0.0, 0.25, 0.5, 0.75, 1.0])
        self.assertTrue(self.parse("--quiet").quiet)

    def test_clean_minari_arguments(self):
        args = self.parse(
            "--dataset-source", "clean-minari",
            "--dataset", "medium-v0",
            "--minari-fraction", "0", "0.5", "1",
        )
        self.assertEqual(args.dataset, "medium-v0")
        self.assertEqual(args.minari_fractions, [0.0, 0.5, 1.0])

        with self.assertRaises(SystemExit):
            self.parse("--dataset-source", "clean-minari")
        with self.assertRaises(SystemExit):
            self.parse(
                "--dataset-source", "clean-minari",
                "--dataset", "medium-v0",
                "--minari-fraction", "1.1",
            )

    def test_negative_shift_is_rejected(self):
        with self.assertRaises(SystemExit):
            self.parse("--mobile-return-shift", "-1")

    def test_negative_model_penalty_is_rejected(self):
        with self.assertRaises(SystemExit):
            self.parse("--mopo-penalty-coef", "-1")
        with self.assertRaises(SystemExit):
            self.parse("--mobile-penalty-coef", "-1")
        with self.assertRaises(SystemExit):
            self.parse("--model-actor-learning-rate", "0")
        with self.assertRaises(SystemExit):
            self.parse("--model-critic-learning-rate", "0")

    def test_model_actor_learning_rate_is_recorded(self):
        args = self.parse("--model-actor-learning-rate", "3e-5")
        args.seed = 0
        schema = sweep.make_training_schema(
            "mopo", "Lift", {"source": "test"}, 2, args
        )

        self.assertEqual(schema["model_based"]["actor_learning_rate"], 3e-5)

    def test_model_critic_learning_rate_is_recorded(self):
        args = self.parse("--model-critic-learning-rate", "1e-4")
        args.seed = 0
        schema = sweep.make_training_schema(
            "mopo", "Lift", {"source": "test"}, 2, args
        )

        self.assertEqual(schema["model_based"]["critic_learning_rate"], 1e-4)

    def test_reacher_mobile_schema_records_shifted_clamp(self):
        args = self.parse("--mobile-return-shift", "20")
        args.seed = 0
        schema = sweep.make_training_schema(
            "mobile", "Reacher-v5", {"source": "test"}, 1, args
        )

        self.assertEqual(
            schema["mobile"],
            {"return_shift": 20.0, "clamp_target_q": True},
        )

    def test_non_reacher_schema_is_unchanged(self):
        args = self.parse()
        args.seed = 0
        schema = sweep.make_training_schema(
            "mobile", "HalfCheetah-v5", {"source": "test"}, 1, args
        )

        self.assertNotIn("mobile", schema)

    def test_lift_model_based_schema_records_synthetic_termination(self):
        args = self.parse()
        args.seed = 0
        schema = sweep.make_training_schema(
            "mopo", "Lift", {"source": "test"}, 2, args
        )

        self.assertEqual(
            schema["model_based"]["synthetic_termination"],
            "never",
        )

    def test_manipulation_model_schema_records_changed_training(self):
        args = self.parse(
            "--model-manipulation-settings",
            "--mobile-penalty-coef", "1.0",
        )
        args.seed = 0
        schema = sweep.make_training_schema(
            "mobile", "Lift", {"source": "test"}, 2, args
        )

        self.assertEqual(schema["mobile"]["penalty_coef"], 1.0)
        self.assertEqual(schema["mobile"]["num_critics"], 10)
        self.assertTrue(schema["mobile"]["max_q_backup"])
        self.assertEqual(
            schema["model_based"]["manipulation_settings"]["reward_normalization"],
            "zscore",
        )

    def test_recursive_manipulation_rewards_match_normalized_macro_rewards(self):
        observations = np.arange(7, dtype=np.float32).reshape(-1, 1)
        primitive_dataset = {
            "observations": observations,
            "actions": np.arange(7, dtype=np.float32).reshape(-1, 1),
            "next_observations": observations + 1.0,
            "rewards": np.arange(1, 8, dtype=np.float32),
            "terminals": np.asarray([False] * 6 + [True]),
            "timeouts": np.zeros(7, dtype=bool),
            "episode_ids": np.zeros(7, dtype=np.int64),
        }
        chunk_dataset = chunking.make_action_chunk_dataset(
            primitive_dataset, chunk_length=3, discount=0.99
        )

        policy_dataset, dynamics_dataset = sweep.prepare_model_based_datasets(
            "mobile",
            primitive_dataset,
            chunk_dataset,
            chunk_length=3,
            base_discount=0.99,
            dynamics_chunk_mode="recursive",
            model_manipulation_settings=True,
        )
        transformed_primitive = {
            key: value.copy() for key, value in primitive_dataset.items()
        }
        transformed_primitive["rewards"] = dynamics_dataset["rewards"].reshape(-1)
        transformed_chunks = chunking.make_action_chunk_dataset(
            transformed_primitive, chunk_length=3, discount=0.99
        )

        expected = (
            chunk_dataset["rewards"] - chunk_dataset["rewards"].mean()
        ) / (chunk_dataset["rewards"].std() + 1e-3)
        np.testing.assert_allclose(policy_dataset["rewards"], expected, rtol=1e-6)
        np.testing.assert_allclose(
            transformed_chunks["rewards"],
            policy_dataset["rewards"],
            rtol=1e-5,
            atol=2e-6,
        )
        self.assertEqual(dynamics_dataset["actions"].shape, (7, 1))
        self.assertEqual(dynamics_dataset["rewards"].shape, (7, 1))

    def test_direct_dynamics_preserves_existing_real_buffer_training_path(self):
        dataset = {
            "observations": np.zeros((2, 1), dtype=np.float32),
            "actions": np.zeros((2, 1), dtype=np.float32),
            "next_observations": np.ones((2, 1), dtype=np.float32),
            "rewards": np.asarray([1.0, 2.0], dtype=np.float32),
            "terminals": np.asarray([False, True]),
            "timeouts": np.asarray([False, False]),
            "episode_ids": np.asarray([0, 0]),
        }

        policy_dataset, dynamics_dataset = sweep.prepare_model_based_datasets(
            "mopo",
            dataset,
            dataset,
            chunk_length=1,
            base_discount=0.99,
            dynamics_chunk_mode="direct",
            model_manipulation_settings=False,
        )

        self.assertIs(policy_dataset, dataset)
        self.assertIsNone(dynamics_dataset)

        requested_recursive_policy, requested_recursive_dynamics = (
            sweep.prepare_model_based_datasets(
                "mopo",
                dataset,
                dataset,
                chunk_length=1,
                base_discount=0.99,
                dynamics_chunk_mode="recursive",
                model_manipulation_settings=False,
            )
        )
        self.assertIs(requested_recursive_policy, dataset)
        self.assertIsNone(requested_recursive_dynamics)

    def test_lift_synthetic_transitions_never_terminate(self):
        next_obs = np.zeros((3, 19), dtype=np.float32)
        next_obs[:, 11] = [0.83, 0.84, 0.85]

        np.testing.assert_array_equal(
            termination_fn_never(np.zeros((3, 19)), None, next_obs),
            [[False], [False], [False]],
        )

    def test_generated_clean_lookup_reuses_noise_independent_schema(self):
        args = self.parse()
        args.seed = 0
        expert_path = Path("/tmp/Reacher-v5.zip")
        schema = {
            "version": sweep.DATASET_SCHEMA_VERSION,
            "source": "generated",
            "env_name": "Reacher-v5",
            "expert_path": str(expert_path),
            "max_timesteps": args.max_timesteps,
            "num_samples": 500,
            "noise_scale": 0.5,
            "prop_clean_expert": 1.0,
            "prop_noisy_expert": 0.0,
            "prop_random": 0.0,
            "prop_expert": 1.0,
            "deterministic": True,
            "seed": 0,
            "test_fraction": args.test_fraction,
        }

        with tempfile.TemporaryDirectory() as directory:
            dataset_root = Path(directory)
            dataset_dir = (
                dataset_root
                / "samples500_clean1_noisy0_noise0.5_seed0"
                / "20260827"
            )
            dataset_dir.mkdir(parents=True)
            (dataset_dir / "metadata.json").write_text(json.dumps({
                "dataset_schema": schema,
            }))
            for filename in ("full.npz", "train.npz", "test.npz"):
                (dataset_dir / filename).touch()
            newer_dir = (
                dataset_root
                / "samples500_clean1_noisy0_noise0_seed0"
                / "20260828"
            )
            newer_dir.mkdir(parents=True)
            (newer_dir / "metadata.json").write_text(json.dumps({
                "dataset_schema": {**schema, "noise_scale": 0.0},
            }))
            for filename in ("full.npz", "train.npz", "test.npz"):
                (newer_dir / filename).touch()

            with patch("sweep.find_trained_run") as find_run:
                find_run.side_effect = lambda parent, _: (
                    Path("/existing") if "noise0.5" in str(parent) else None
                )
                found = sweep.find_generated_clean_dataset(
                    dataset_root, Path("/trained"), "Reacher-v5", expert_path, 500, args
                )
            self.assertEqual(found, (dataset_dir.parent, dataset_dir.parent.name, schema))
            self.assertIsNone(sweep.find_generated_clean_dataset(
                dataset_root, Path("/trained"), "Reacher-v5", expert_path, 501, args
            ))

    @patch("sweep.train_algos", return_value=[])
    @patch("sweep.find_generated_clean_dataset")
    @patch("sweep.load_offline.list_minari_dataset_ids")
    def test_zero_minari_fraction_uses_stored_generated_identity(
        self, list_minari_ids, find_clean, train_algos
    ):
        args = self.parse(
            "--dataset-source", "clean-minari",
            "--dataset", "medium-v0",
            "--num-samples", "500",
            "--minari-fraction", "0",
        )
        args.seed = 0
        schema = {"source": "generated", "noise_scale": 0.5}
        list_minari_ids.return_value = ["mujoco/reacher/medium-v0"]
        find_clean.return_value = (Path("/datasets/clean"), "old_clean_tag", schema)

        sweep.run_clean_minari_sweep(
            "Reacher-v5", Path("/experts/Reacher-v5.zip"),
            Path("/datasets"), Path("/trained"), Path("/evals"), args,
        )

        self.assertEqual(train_algos.call_args.args[4], "old_clean_tag")
        self.assertIs(train_algos.call_args.args[5], schema)
        self.assertEqual(train_algos.call_args.args[6].dataset_source, "generated")

    @patch("sweep.load_offline.load_minari_episode_subset")
    @patch("sweep.rollout.collect_expert")
    @patch("sweep.gym.make")
    def test_clean_minari_collection_allocates_complete_trajectories(
        self, make_env, collect_expert, load_minari_subset
    ):
        make_env.return_value = SimpleNamespace(
            spec=SimpleNamespace(max_episode_steps=50),
            close=lambda: None,
        )

        def dataset(num_episodes, episode_id_start):
            size = num_episodes * 50
            episode_ids = np.repeat(
                np.arange(episode_id_start, episode_id_start + num_episodes), 50
            )
            return {
                "observations": np.zeros((size, 2), dtype=np.float32),
                "actions": np.zeros((size, 1), dtype=np.float32),
                "next_observations": np.zeros((size, 2), dtype=np.float32),
                "rewards": np.zeros(size, dtype=np.float32),
                "terminals": np.zeros(size, dtype=bool),
                "timeouts": np.isin(np.arange(size), np.arange(49, size, 50)),
                "episode_ids": episode_ids,
            }

        collect_expert.return_value = dataset(7, 0)
        load_minari_subset.return_value = (
            dataset(3, 7),
            {
                "env_id": "Reacher-v5",
                "available_num_episodes": 10000,
                "available_num_transitions": 500000,
            },
        )
        args = self.parse(
            "--dataset-source", "clean-minari",
            "--dataset", "medium-v0",
        )
        args.seed = 0

        with tempfile.NamedTemporaryFile(suffix=".zip") as expert:
            combined, metadata = sweep.collect_clean_minari_dataset(
                "Reacher-v5", Path(expert.name), "mujoco/reacher/medium-v0",
                num_samples=500, minari_fraction=0.3, args=args,
            )

        self.assertEqual(len(combined["rewards"]), 500)
        self.assertEqual(len(np.unique(combined["episode_ids"])), 10)
        self.assertEqual(metadata["num_clean_expert_trajectories"], 7)
        self.assertEqual(metadata["num_minari_trajectories"], 3)
        self.assertEqual(metadata["actual_minari_trajectory_fraction"], 0.3)
        self.assertEqual(
            load_minari_subset.call_args.kwargs["episode_id_start"], 7
        )


if __name__ == "__main__":
    unittest.main()

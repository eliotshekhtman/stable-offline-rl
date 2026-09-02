import concurrent.futures
import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

import chunking
import rollout
from policies import termination_fn_never
import sweep


class StorageRootTests(unittest.TestCase):
    def parse(self, *extra, env="Reacher-v5"):
        with patch.object(sys, "argv", ["sweep.py", "--env", env, *extra]):
            return sweep.parse_args()

    def test_default_and_override_are_normalized_without_creating_directories(self):
        self.assertEqual(
            self.parse().storage_root,
            sweep.DEFAULT_STORAGE_ROOT.resolve(),
        )

        with tempfile.TemporaryDirectory() as directory:
            storage_root = Path(directory) / "not-created-by-parsing"
            args = self.parse("--storage-root", str(storage_root))

            self.assertEqual(args.storage_root, storage_root.resolve())
            self.assertFalse(storage_root.exists())

    def test_prepare_storage_root_creates_and_checks_all_artifact_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            storage_root = Path(directory) / "storage"

            sweep.prepare_storage_root(storage_root)

            for name in ("datasets", "trained", "evals"):
                self.assertTrue((storage_root / name).is_dir())
                self.assertEqual(list((storage_root / name).iterdir()), [])

    def test_prepare_storage_root_reports_an_unusable_child(self):
        with tempfile.TemporaryDirectory() as directory:
            storage_root = Path(directory) / "storage"
            storage_root.mkdir()
            (storage_root / "datasets").write_text("not a directory", encoding="utf-8")

            with self.assertRaisesRegex(OSError, str(storage_root / "datasets")):
                sweep.prepare_storage_root(storage_root)

    def test_prepare_storage_root_reports_a_write_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            storage_root = Path(directory) / "storage"
            with (
                patch(
                    "sweep.tempfile.NamedTemporaryFile",
                    side_effect=PermissionError("permission denied"),
                ),
                self.assertRaisesRegex(OSError, str(storage_root / "datasets")),
            ):
                sweep.prepare_storage_root(storage_root)

    def test_prepare_storage_root_is_safe_under_concurrent_startup(self):
        with tempfile.TemporaryDirectory() as directory:
            storage_root = Path(directory) / "storage"
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                list(executor.map(sweep.prepare_storage_root, [storage_root] * 4))

            for name in ("datasets", "trained", "evals"):
                self.assertTrue((storage_root / name).is_dir())
                self.assertEqual(list((storage_root / name).iterdir()), [])

    def test_storage_root_does_not_change_training_identity_or_expert_path(self):
        default_args = self.parse()
        external_args = self.parse("--storage-root", "/tmp/other-storage")
        default_args.seed = external_args.seed = 7

        self.assertEqual(
            sweep.make_training_schema(
                "mopo", "Reacher-v5", {"source": "test"}, 4, default_args
            ),
            sweep.make_training_schema(
                "mopo", "Reacher-v5", {"source": "test"}, 4, external_args
            ),
        )
        self.assertEqual(
            sweep.resolve_expert_path(default_args.expert, "Reacher-v5"),
            sweep.resolve_expert_path(external_args.expert, "Reacher-v5"),
        )

    def test_all_dataset_sources_receive_only_the_selected_storage_paths(self):
        cases = (
            ("Reacher-v5", (), "run_generated_configuration", (2, 3, 4)),
            (
                "Reacher-v5",
                ("--dataset-source", "minari"),
                "run_minari_sweep",
                (1, 2, 3),
            ),
            (
                "Reacher-v5",
                (
                    "--dataset-source", "clean-minari",
                    "--dataset", "medium-v0",
                ),
                "run_clean_minari_sweep",
                (2, 3, 4),
            ),
            (
                "Lift",
                ("--dataset-source", "robomimic"),
                "run_robomimic_sweep",
                (1, 2, 3),
            ),
        )

        for env_name, extra, function_name, root_indexes in cases:
            with self.subTest(source=function_name), tempfile.TemporaryDirectory() as directory:
                storage_root = Path(directory) / "storage"
                args = self.parse(*extra, env=env_name)
                args.seed = 0
                with patch(f"sweep.{function_name}", return_value=[]) as runner:
                    sweep.run_sweep(
                        env_name,
                        Path("/experts") / f"{env_name}.zip",
                        storage_root,
                        args,
                    )

                call_args = runner.call_args.args
                actual_roots = tuple(call_args[index] for index in root_indexes)
                self.assertEqual(
                    actual_roots,
                    (
                        storage_root / "datasets" / env_name,
                        storage_root / "trained" / env_name,
                        storage_root / "evals" / env_name,
                    ),
                )

    def test_main_uses_the_selected_root_for_every_seed_and_automatic_plotting(self):
        with tempfile.TemporaryDirectory() as directory:
            storage_root = (Path(directory) / "storage").resolve()
            args = self.parse(
                "--storage-root", str(storage_root),
                "--seed", "3", "4",
                "--eval",
            )
            eval_dirs = [
                storage_root / "evals" / args.env / "run-3",
                storage_root / "evals" / args.env / "run-4",
            ]

            with (
                patch("sweep.parse_args", return_value=args),
                patch("sweep.prepare_storage_root") as prepare_storage_root,
                patch("sweep.run_sweep", side_effect=[[eval_dirs[0]], [eval_dirs[1]]]) as run_sweep,
                patch("sweep.maybe_plot") as maybe_plot,
            ):
                sweep.main()

            prepare_storage_root.assert_called_once_with(storage_root)
            self.assertEqual(run_sweep.call_count, 2)
            self.assertEqual(
                [call.kwargs["storage_root"] for call in run_sweep.call_args_list],
                [storage_root, storage_root],
            )
            self.assertEqual(
                [call.kwargs["args"].seed for call in run_sweep.call_args_list],
                [3, 4],
            )
            maybe_plot.assert_called_once_with(
                storage_root / "evals" / args.env,
                eval_dirs,
                args,
            )


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

    def test_rejects_unsupported_tasks_and_task_source_pairs(self):
        with (
            patch.object(sys, "argv", ["sweep.py", "--env", "Ant-v5"]),
            self.assertRaises(SystemExit),
        ):
            sweep.parse_args()
        with (
            patch.object(sys, "argv", ["sweep.py", "--env", "Lift"]),
            self.assertRaises(SystemExit),
        ):
            sweep.parse_args()
        with patch.object(
            sys, "argv",
            ["sweep.py", "--env", "Lift", "--dataset-source", "robomimic"],
        ):
            self.assertEqual(sweep.parse_args().env, "Lift")

    def test_dynamics_chunk_mode_default_and_override(self):
        self.assertEqual(self.parse().dynamics_chunk_mode, "direct")
        self.assertEqual(
            self.parse("--dynamics-chunk-mode", "recursive").dynamics_chunk_mode,
            "recursive",
        )

    def test_zero_checkpoint_evaluation_is_allowed(self):
        self.assertEqual(
            self.parse("--checkpoint-eval-episodes", "0").checkpoint_eval_episodes,
            0,
        )
        with self.assertRaises(SystemExit):
            self.parse("--checkpoint-eval-episodes", "-1")

    def test_validation_only_runs_with_eval(self):
        args = SimpleNamespace(eval=False, device="cpu")
        with patch("validation.validate_run") as validate_run:
            sweep.maybe_validate(Path("run"), args)
            validate_run.assert_not_called()

            args.eval = True
            sweep.maybe_validate(Path("run"), args)
            validate_run.assert_called_once_with(Path("run"), "cpu")

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

    def test_programmatic_sweep_rejects_unsupported_task_before_creating_dirs(self):
        args = self.parse()
        args.seed = 0
        with patch.object(Path, "mkdir") as mkdir, self.assertRaisesRegex(
            ValueError, "Unsupported task 'Ant-v5'"
        ):
            sweep.run_sweep(
                "Ant-v5", Path("/unused/expert.zip"), Path("/unused/storage"), args
            )
        mkdir.assert_not_called()

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
        for key in ("observations", "actions", "next_observations"):
            self.assertTrue(
                np.shares_memory(dynamics_dataset[key], primitive_dataset[key])
            )

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

    def test_clean_minari_short_episodes_top_up_without_repeating_subset(self):
        args = self.parse(
            "--dataset-source", "clean-minari",
            "--dataset", "medium-v0",
        )
        args.seed = 13

        def make_dataset(num_episodes, episode_id_start, episode_length, marker):
            size = num_episodes * episode_length
            episode_ids = np.repeat(
                np.arange(episode_id_start, episode_id_start + num_episodes),
                episode_length,
            )
            observations = (
                marker + np.arange(size, dtype=np.float32)
            ).reshape(-1, 1)
            timeouts = np.zeros(size, dtype=bool)
            timeouts[episode_length - 1::episode_length] = True
            return {
                "observations": observations,
                "actions": np.zeros((size, 1), dtype=np.float32),
                "next_observations": observations + 1.0,
                "rewards": np.zeros(size, dtype=np.float32),
                "terminals": np.zeros(size, dtype=bool),
                "timeouts": timeouts,
                "episode_ids": episode_ids,
            }

        def collect_once():
            clean_call = 0
            minari_call = 0
            minari_requests = []
            clean_rngs = []

            def collect_expert(**kwargs):
                nonlocal clean_call
                clean_rngs.append(kwargs["rng"])
                lengths = (20, 5, 30)
                length = lengths[min(clean_call, len(lengths) - 1)]
                clean_call += 1
                return make_dataset(
                    kwargs["num_trajectories"], kwargs["episode_id_start"],
                    length, marker=1000 * clean_call,
                )

            def load_subset(**kwargs):
                nonlocal minari_call
                lengths = (20, 5, 30)
                length = lengths[min(minari_call, len(lengths) - 1)]
                minari_call += 1
                offset = kwargs["episode_offset"]
                count = kwargs["num_episodes"]
                minari_requests.append((offset, count))
                return (
                    make_dataset(
                        count, kwargs["episode_id_start"], length,
                        marker=10000 + 1000 * offset,
                    ),
                    {
                        "env_id": "Reacher-v5",
                        "available_num_episodes": 100,
                        "available_num_transitions": 5000,
                    },
                )

            env = SimpleNamespace(
                spec=SimpleNamespace(max_episode_steps=50),
                close=lambda: None,
            )
            with patch("sweep.gym.make", return_value=env), patch(
                "sweep.rollout.collect_expert", side_effect=collect_expert
            ), patch(
                "sweep.load_offline.load_minari_episode_subset",
                side_effect=load_subset,
            ), tempfile.NamedTemporaryFile(suffix=".zip") as expert:
                dataset, metadata = sweep.collect_clean_minari_dataset(
                    "Reacher-v5", Path(expert.name),
                    "mujoco/reacher/medium-v0", num_samples=500,
                    minari_fraction=0.5, args=args,
                )
            return dataset, metadata, minari_requests, clean_rngs

        first_dataset, first_metadata, first_requests, first_rngs = collect_once()
        second_dataset, second_metadata, second_requests, second_rngs = collect_once()

        self.assertGreaterEqual(len(first_requests), 3)
        self.assertEqual(first_requests, second_requests)
        for (offset, count), (next_offset, _) in zip(
            first_requests, first_requests[1:]
        ):
            self.assertEqual(next_offset, offset + count)
        selected = [
            episode
            for offset, count in first_requests
            for episode in range(offset, offset + count)
        ]
        self.assertEqual(len(selected), len(set(selected)))
        self.assertTrue(all(rng is first_rngs[0] for rng in first_rngs))
        self.assertTrue(all(rng is second_rngs[0] for rng in second_rngs))
        self.assertGreaterEqual(len(first_dataset["rewards"]), 500)
        self.assertEqual(
            {key: value for key, value in first_metadata.items() if key != "policy_path"},
            {key: value for key, value in second_metadata.items() if key != "policy_path"},
        )
        for key in first_dataset:
            np.testing.assert_array_equal(first_dataset[key], second_dataset[key])

    def test_new_dataset_cache_omits_full_copy_and_reuses_without_it(self):
        observations = np.arange(4, dtype=np.float32).reshape(-1, 1)
        dataset = {
            "observations": observations,
            "actions": np.zeros((4, 1), dtype=np.float32),
            "next_observations": observations + 1.0,
            "rewards": np.ones(4, dtype=np.float32),
            "terminals": np.asarray([False, True, False, True]),
            "timeouts": np.zeros(4, dtype=bool),
            "episode_ids": np.asarray([0, 0, 1, 1]),
        }
        args = SimpleNamespace(test_fraction=0.5, seed=0)
        schema = {"source": "test", "seed": 0}

        with tempfile.TemporaryDirectory() as directory:
            dataset_parent = Path(directory) / "tag"
            with patch("sweep.timestamp_name", return_value="variant"):
                first_dataset, first_paths = sweep.get_or_create_dataset(
                    dataset_parent, schema,
                    lambda: (dataset, {"source": "test"}), args,
                )
            dataset_dir = dataset_parent / "variant"
            self.assertFalse((dataset_dir / "full.npz").exists())
            self.assertNotIn("full_dataset_path", first_paths)
            self.assertTrue(sweep.dataset_cache_is_complete(dataset_dir))

            create = Mock(side_effect=AssertionError("cache was not reused"))
            second_dataset, second_paths = sweep.get_or_create_dataset(
                dataset_parent, schema, create, args,
            )
            create.assert_not_called()
            self.assertEqual(first_paths, second_paths)
            for key in rollout.DATASET_KEYS:
                np.testing.assert_array_equal(first_dataset[key], second_dataset[key])

    def test_dataset_cache_rejects_corrupt_heldout_split(self):
        observations = np.arange(4, dtype=np.float32).reshape(-1, 1)
        dataset = {
            "observations": observations,
            "actions": np.zeros((4, 1), dtype=np.float32),
            "next_observations": observations + 1.0,
            "rewards": np.ones(4, dtype=np.float32),
            "terminals": np.asarray([False, True, False, True]),
            "timeouts": np.zeros(4, dtype=bool),
            "episode_ids": np.asarray([0, 0, 1, 1]),
        }
        args = SimpleNamespace(test_fraction=0.5, seed=0)
        schema = {"source": "test", "seed": 0}

        with tempfile.TemporaryDirectory() as directory:
            dataset_parent = Path(directory) / "tag"
            with patch("sweep.timestamp_name", return_value="variant"):
                sweep.get_or_create_dataset(
                    dataset_parent, schema,
                    lambda: (dataset, {"source": "test"}), args,
                )
            (dataset_parent / "variant" / "test.npz").write_bytes(b"corrupt")

            self.assertIsNone(
                sweep.find_cached_dataset(dataset_parent, schema)
            )

    def test_dataset_creation_lock_prevents_duplicate_concurrent_cache(self):
        observations = np.arange(4, dtype=np.float32).reshape(-1, 1)
        dataset = {
            "observations": observations,
            "actions": np.zeros((4, 1), dtype=np.float32),
            "next_observations": observations + 1.0,
            "rewards": np.ones(4, dtype=np.float32),
            "terminals": np.asarray([False, True, False, True]),
            "timeouts": np.zeros(4, dtype=bool),
            "episode_ids": np.asarray([0, 0, 1, 1]),
        }
        args = SimpleNamespace(test_fraction=0.5, seed=0)
        schema = {"source": "concurrent-test"}
        call_count = 0
        count_lock = threading.Lock()

        def create_dataset():
            nonlocal call_count
            with count_lock:
                call_count += 1
            time.sleep(0.05)
            return dataset, {"source": "concurrent-test"}

        with tempfile.TemporaryDirectory() as directory, patch(
            "sweep.timestamp_name", return_value="variant"
        ):
            parent = Path(directory) / "tag"
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                results = list(executor.map(
                    lambda _: sweep.get_or_create_dataset(
                        parent, schema, create_dataset, args
                    ),
                    range(2),
                ))

            self.assertEqual(call_count, 1)
            self.assertEqual(results[0][1], results[1][1])
            self.assertEqual(
                len(list(parent.glob("*/metadata.json"))), 1
            )

    def test_legacy_manifest_remains_complete_without_unused_full_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            required = [
                root / "model" / "policy.pth",
                root / "train.npz",
                root / "test.npz",
                root / "metadata.json",
                root / "checkpoint" / "step_0" / "policy.pth",
            ]
            for path in required:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            manifest = {
                "algo": "bc",
                "model_dir": str(root / "model"),
                "full_dataset_path": str(root / "missing-full.npz"),
                "train_dataset_path": str(root / "train.npz"),
                "test_dataset_path": str(root / "test.npz"),
                "dataset_metadata_path": str(root / "metadata.json"),
                "checkpoints": [{
                    "policy_path": str(root / "checkpoint" / "step_0" / "policy.pth")
                }],
            }
            self.assertTrue(sweep.run_is_complete(manifest))

    def test_successful_run_artifacts_compact_without_removing_manifest_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            final_dir = run_dir / "checkpoint" / "step_10"
            initial_dir = run_dir / "checkpoint" / "step_0"
            model_dir = run_dir / "model"
            for path in (final_dir, initial_dir, model_dir):
                path.mkdir(parents=True, exist_ok=True)

            final_policy = final_dir / "policy.pth"
            model_policy = model_dir / "policy.pth"
            rolling_policy = run_dir / "checkpoint" / "policy.pth"
            for path in (final_policy, model_policy, rolling_policy):
                path.write_bytes(b"same policy")
            for filename in ("dynamics.pth", "mu.npy", "std.npy"):
                (initial_dir / filename).write_bytes(b"same " + filename.encode())
                (model_dir / filename).write_bytes(b"same " + filename.encode())

            sweep.compact_run_artifacts(
                run_dir, "mopo", epochs=10, steps_per_epoch=1
            )

            self.assertFalse(rolling_policy.exists())
            self.assertTrue(final_policy.samefile(model_policy))
            for filename in ("dynamics.pth", "mu.npy", "std.npy"):
                self.assertTrue(
                    (initial_dir / filename).samefile(model_dir / filename)
                )

    def test_artifact_compaction_preserves_nonidentical_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.pth"
            duplicate = root / "duplicate.pth"
            source.write_bytes(b"source")
            duplicate.write_bytes(b"different")

            self.assertFalse(
                sweep.replace_identical_file_with_hardlink(source, duplicate)
            )
            self.assertEqual(duplicate.read_bytes(), b"different")

    def test_train_algo_closes_environment_when_logger_creation_fails(self):
        env = Mock()
        env.action_space = Mock()
        args = SimpleNamespace(seed=0)
        with tempfile.TemporaryDirectory() as directory, patch(
            "sweep.make_env", return_value=env
        ), patch(
            "sweep.chunking.ActionChunkWrapper", side_effect=lambda item, _: item
        ), patch(
            "sweep.build_logger", side_effect=RuntimeError("logger failed")
        ), self.assertRaisesRegex(RuntimeError, "logger failed"):
            sweep.train_algo(
                algo="bc",
                env_name="Reacher-v5",
                primitive_dataset={},
                chunk_dataset={},
                chunk_length=1,
                run_dir=Path(directory) / "run",
                eval_dir=Path(directory) / "eval",
                split_paths={},
                training_schema={"macro_discount": 0.99},
                args=args,
            )
        env.close.assert_called_once_with()

    def test_build_logger_closes_partial_logger_on_hyperparameter_failure(self):
        logger = Mock()
        logger.log_hyperparameters.side_effect = RuntimeError("log failed")
        args = SimpleNamespace(
            seed=0,
            device="cpu",
            epoch=1,
            step_per_epoch=1,
            batch_size=2,
        )
        with patch("sweep.Logger", return_value=logger), self.assertRaisesRegex(
            RuntimeError, "log failed"
        ):
            sweep.build_logger(
                Path("/unused"), args, "bc", "Reacher-v5", 1, 0.99
            )
        logger.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()

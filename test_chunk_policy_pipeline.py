import argparse
import copy
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import gymnasium as gym
import numpy as np
import torch

import chunking
import eval as evaluation
import plot
import policies
import sweep
import validation
from chunk_policy import NormalizedObservationBackbone, fit_observation_stats


SUPPORTED_ALGOS = ("td3bc", "iql", "mopo", "mobile")


def parse_args(*extra):
    with patch.object(sys, "argv", ["sweep.py", "--env", "Reacher-v5", *extra]):
        args = sweep.parse_args()
    args.seed = args.seeds[0]
    return args


def policy_args(algo, rho=None):
    flags = [] if rho is None else ["--chunk-policy", "decay-sn", "--chunk-policy-rho", str(rho)]
    return parse_args(
        "--algos", algo, "--device", "cpu", "--epoch", "2",
        "--step-per-epoch", "1", "--iql-hidden-dims", "8", "8",
        "--td3bc-hidden-dims", "8", "8", *flags,
    )


def training_schema(algo, chunk_length=4, rho=None):
    return sweep.make_training_schema(
        algo, "Reacher-v5", {"source": "minari", "seed": 0},
        chunk_length, policy_args(algo, rho),
    )


class TinyEnv(gym.Env):
    def __init__(self, action_dim=2):
        self.observation_space = gym.spaces.Box(-10.0, 10.0, shape=(3,))
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(action_dim,))
        self.spec = SimpleNamespace(id="Reacher-v5")

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(3, dtype=np.float32), {}


class ChunkPolicyCliAndSchemaTests(unittest.TestCase):
    def test_default_and_supported_explicit_modes(self):
        args = parse_args()
        self.assertEqual(args.chunk_policy, "standard")
        self.assertIsNone(args.chunk_policy_rho)
        for algo in SUPPORTED_ALGOS:
            for rho in (0.8, 1.0):
                with self.subTest(algo=algo, rho=rho):
                    args = policy_args(algo, rho)
                    self.assertEqual(args.chunk_policy, "decay-sn")
                    self.assertEqual(args.chunk_policy_rho, rho)

    def test_invalid_cli_settings_fail(self):
        cases = [
            ["--chunk-policy", "unknown"],
            ["--chunk-policy", "decay-sn"],
            ["--chunk-policy-rho", "0.8"],
            ["--chunk-policy", "standard", "--chunk-policy-rho", "0.8"],
        ]
        cases.extend(
            ["--chunk-policy", "decay-sn", f"--chunk-policy-rho={rho}"]
            for rho in ("0", "-0.5", "1.1", "nan", "inf", "-inf")
        )
        for flags in cases:
            with self.subTest(flags=flags), patch("sys.stderr", new=io.StringIO()), self.assertRaises(SystemExit):
                parse_args("--algos", "td3bc", *flags)

    def test_enabled_mode_rejects_unsupported_and_mixed_algorithms(self):
        for algo in ("none", "bc", "cql", "edac", "combo"):
            for algos in ([algo], ["td3bc", algo]):
                with self.subTest(algos=algos), patch("sys.stderr", new=io.StringIO()), self.assertRaises(SystemExit):
                    parse_args(
                        "--algos", *algos, "--chunk-policy", "decay-sn",
                        "--chunk-policy-rho", "0.8",
                    )

    def test_legacy_namespaces_and_explicit_standard_have_identical_schemas(self):
        for algo in (*policies.MODEL_FREE_ALGOS, *policies.MODEL_BASED_ALGOS):
            with self.subTest(algo=algo):
                default = policy_args(algo)
                legacy = argparse.Namespace(**vars(default))
                del legacy.chunk_policy
                del legacy.chunk_policy_rho
                explicit = argparse.Namespace(**vars(default))
                explicit.chunk_policy = "standard"
                schemas = [
                    sweep.make_training_schema(algo, "Reacher-v5", {}, 4, args)
                    for args in (default, legacy, explicit)
                ]
                self.assertEqual(schemas[0], schemas[1])
                self.assertEqual(schemas[0], schemas[2])
                self.assertNotIn("policy_architecture", schemas[0])
                self.assertEqual(schemas[0]["version"], 3)

    def test_enabled_schema_changes_only_actor_metadata_including_horizon_one(self):
        for algo in SUPPORTED_ALGOS:
            for chunk_length in (1, 4):
                with self.subTest(algo=algo, chunk_length=chunk_length):
                    legacy = training_schema(algo, chunk_length)
                    enabled = training_schema(algo, chunk_length, 0.8)
                    config = enabled.pop("policy_architecture")
                    self.assertEqual(config, {
                        "version": 1 if algo == "td3bc" else 2, "type": "decay-sn", "rho": 0.8,
                    })
                    self.assertEqual(enabled, legacy)

    def test_reuse_keeps_legacy_and_each_rho_separate(self):
        schemas = [training_schema("iql", rho=rho) for rho in (None, 0.8, 1.0)]
        old_decay = copy.deepcopy(schemas[1])
        old_decay["policy_architecture"]["version"] = 1
        schemas.append(old_decay)
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            paths = []
            for index, schema in enumerate(schemas):
                path = parent / str(index)
                sweep.write_json_atomic(path / "run_manifest.json", {"training_schema": schema})
                paths.append(path)
            with patch.object(sweep, "run_is_complete", return_value=True):
                for schema, path in zip(schemas, paths):
                    self.assertEqual(sweep.find_trained_run(parent, schema), path)
                self.assertIsNone(sweep.find_trained_run(parent, training_schema("iql", rho=0.5)))


class ChunkPolicyBuilderForwardingTests(unittest.TestCase):
    def test_training_passes_actual_horizon_only_to_enabled_or_cql_builders(self):
        cases = [(algo, None) for algo in policies.MODEL_FREE_ALGOS]
        cases.extend((algo, 0.8) for algo in ("td3bc", "iql"))
        for algo, rho in cases:
            with self.subTest(algo=algo, rho=rho), tempfile.TemporaryDirectory() as directory:
                args = policy_args(algo, rho)
                schema = training_schema(algo, rho=rho)
                observations = np.asarray([[1, 4, -2], [3, 2, 0], [-1, 3, 1]], dtype=np.float32)
                with (
                    patch.object(sweep, "make_env", return_value=TinyEnv()),
                    patch.object(sweep, "build_logger", return_value=Mock()),
                    patch.object(sweep, "build_buffer", return_value=Mock()),
                    patch.object(sweep, "build_model_free_policy", side_effect=RuntimeError("builder reached")) as builder,
                    self.assertRaisesRegex(RuntimeError, "builder reached"),
                ):
                    sweep.train_algo(
                        algo, "Reacher-v5", {}, {"observations": observations}, 4,
                        Path(directory) / "run", Path(directory) / "eval", {}, schema, args,
                    )
                expected = {"discount": 0.99**4}
                if algo == "cql" or rho is not None:
                    expected["chunk_length"] = 4
                actual = dict(builder.call_args.kwargs)
                if algo == "iql" and rho is not None:
                    stats = actual.pop("observation_stats")
                    for found, wanted in zip(stats, fit_observation_stats(observations)):
                        np.testing.assert_array_equal(found, wanted)
                self.assertEqual(actual, expected)

    def test_model_based_training_fits_real_chunk_starts_only(self):
        starts = np.asarray([[1, 4, -2], [3, 2, 0], [-1, 3, 1]], dtype=np.float32)
        chunk_dataset = {"observations": starts, "next_observations": starts + 100}
        primitive_dataset = {"observations": starts * 1000}
        for algo in ("mopo", "mobile"):
            for rho in (None, 0.8):
                with self.subTest(algo=algo, rho=rho), tempfile.TemporaryDirectory() as directory:
                    args = policy_args(algo, rho)
                    schema = training_schema(algo, rho=rho)
                    with (
                        patch.object(sweep, "make_env", return_value=TinyEnv()),
                        patch.object(sweep, "build_logger", return_value=Mock()),
                        patch.object(sweep, "build_buffer", return_value=Mock()),
                        patch.object(sweep, "ReplayBuffer"),
                        patch.object(sweep, "prepare_model_based_datasets", return_value=(chunk_dataset, None)),
                        patch.object(sweep, "fit_observation_stats", wraps=fit_observation_stats) as fit,
                        patch.object(sweep, "build_model_based_policy", side_effect=RuntimeError("builder reached")) as builder,
                        self.assertRaisesRegex(RuntimeError, "builder reached"),
                    ):
                        sweep.train_algo(
                            algo, "Reacher-v5", primitive_dataset, chunk_dataset, 4,
                            Path(directory) / "run", Path(directory) / "eval", {}, schema, args,
                        )
                    if rho is None:
                        fit.assert_not_called()
                        self.assertNotIn("observation_stats", builder.call_args.kwargs)
                    else:
                        fit.assert_called_once()
                        self.assertIs(fit.call_args.args[0], starts)
                        for found, wanted in zip(builder.call_args.kwargs["observation_stats"], fit_observation_stats(starts)):
                            np.testing.assert_array_equal(found, wanted)


class ChunkPolicyLoaderTests(unittest.TestCase):
    @staticmethod
    def manifest(algo, chunk_length=4, rho=0.8):
        schema = training_schema(algo, chunk_length, rho)
        return {
            "algo": algo, "epoch": schema["epoch"],
            "step_per_epoch": schema["step_per_epoch"],
            "chunk_length": chunk_length, "base_discount": 0.99,
            "macro_discount": 0.99**chunk_length, "training_schema": schema,
        }

    def test_malformed_architecture_fails_before_builder_or_weight_read(self):
        valid = {"version": 1, "type": "decay-sn", "rho": 0.8}
        invalid = [
            None, "decay-sn", [], {},
            {**valid, "version": 3}, {**valid, "version": True},
            {**valid, "type": "unknown"}, {**valid, "type": "standard"},
            {**valid, "chunk_length": 4}, {**valid, "extra": 1},
            {key: value for key, value in valid.items() if key != "rho"},
            *({**valid, "rho": rho} for rho in (None, True, "0.8", 0, -1, 1.1, float("nan"), float("inf"))),
        ]
        for config in invalid:
            manifest = self.manifest("iql")
            manifest["training_schema"]["policy_architecture"] = config
            with (
                self.subTest(config=config),
                patch.object(evaluation, "build_model_free_policy") as builder,
                patch.object(evaluation.torch, "load") as load,
                self.assertRaises(ValueError),
            ):
                evaluation.load_policy_and_dynamics(manifest, "cpu", Path("missing.pth"), None, {}, TinyEnv(8))
            builder.assert_not_called()
            load.assert_not_called()

    def test_architecture_rejects_unsupported_manifest_algorithm(self):
        for algo in ("none", "bc", "cql", "edac", "combo"):
            manifest = self.manifest("iql")
            manifest["algo"] = algo
            with self.subTest(algo=algo), patch.object(evaluation.torch, "load") as load, self.assertRaises(ValueError):
                evaluation.load_policy_and_dynamics(manifest, "cpu", Path("missing.pth"), None, {}, TinyEnv(8))
            load.assert_not_called()

    def test_legacy_loader_keeps_namespace_and_builder_interface(self):
        for algo in ("iql", "td3bc"):
            manifest = self.manifest(algo, rho=None)
            with (
                self.subTest(algo=algo),
                patch.object(evaluation, "build_model_free_policy", return_value=(Mock(), None)) as builder,
                patch.object(evaluation.torch, "load", return_value={}),
            ):
                evaluation.load_policy_and_dynamics(
                    manifest, "cpu", Path("legacy.pth"), None,
                    {"observations": np.zeros((2, 3), dtype=np.float32)}, TinyEnv(8),
                )
            self.assertEqual(builder.call_args.kwargs, {"discount": 0.99**4})
            self.assertFalse(hasattr(builder.call_args.args[3], "chunk_policy"))
            self.assertFalse(hasattr(builder.call_args.args[3], "chunk_policy_rho"))
            self.assertFalse(hasattr(builder.call_args.args[3], "chunk_policy_config"))

    def test_actual_checkpoint_roundtrip_all_supported_algorithms(self):
        observations = np.asarray([[1, 4, -2], [3, 2, 0], [-1, 3, 1]], dtype=np.float32)
        for algo in SUPPORTED_ALGOS:
            for chunk_length in (1, 4):
                for version in ((1,) if algo == "td3bc" else (1, 2)):
                    with self.subTest(algo=algo, chunk_length=chunk_length, version=version), tempfile.TemporaryDirectory() as directory:
                        args = policy_args(algo, 0.8)
                        env = TinyEnv(2 * chunk_length)
                        manifest = self.manifest(algo, chunk_length)
                        manifest["training_schema"]["policy_architecture"]["version"] = version
                        args.chunk_policy_config = manifest["training_schema"]["policy_architecture"]
                        dataset = {
                            "observations": observations.copy(),
                            "next_observations": observations.copy() + 0.1,
                            "actions": np.zeros((3, 2 * chunk_length), dtype=np.float32),
                            "rewards": np.zeros(3, dtype=np.float32),
                            "terminals": np.zeros(3, dtype=np.float32),
                        }
                        normalization = (
                            {"observation_stats": fit_observation_stats(observations)} if version == 2 else {}
                        )
                        torch.manual_seed(17)
                        if algo in policies.MODEL_FREE_ALGOS:
                            buffer = sweep.build_buffer(dataset, env, "cpu")
                            policy, _ = policies.build_model_free_policy(
                                algo, env, buffer, args, discount=0.99**chunk_length,
                                chunk_length=chunk_length, **normalization,
                            )
                        else:
                            policy, _, _ = policies.build_model_based_policy(
                                algo, env, args, discount=0.99**chunk_length,
                                chunk_length=chunk_length, build_dynamics_model=False,
                                **normalization,
                            )
                        policy.eval()
                        before = policy.select_action(observations, deterministic=True)
                        path = Path(directory) / "policy.pth"
                        torch.save(policy.state_dict(), path)
                        loader_dataset = dict(dataset)
                        if version == 2:
                            loader_dataset["observations"] = observations * 100 - 50
                        restored, dynamics = evaluation.load_policy_and_dynamics(
                            manifest, "cpu", path, None, loader_dataset, env,
                        )
                        self.assertIsNone(dynamics)
                        self.assertEqual(before.shape, (3, 2 * chunk_length))
                        np.testing.assert_array_equal(restored.select_action(observations, deterministic=True), before)
                        for key, value in policy.state_dict().items():
                            torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
                        if algo == "td3bc":
                            np.testing.assert_array_equal(restored.scaler.mu, policy.scaler.mu)
                            np.testing.assert_array_equal(restored.scaler.std, policy.scaler.std)
                            np.testing.assert_array_equal(restored.scaler.mu, observations.mean(axis=0, keepdims=True))
                            np.testing.assert_array_equal(restored.scaler.std, observations.std(axis=0, keepdims=True) + 1e-3)
                        if version == 2:
                            incomplete = {key: value for key, value in policy.state_dict().items() if not key.endswith(".mean")}
                            self.assertLess(len(incomplete), len(policy.state_dict()))
                            torch.save(incomplete, path)
                            with self.assertRaisesRegex(RuntimeError, "Missing key"):
                                evaluation.load_policy_and_dynamics(manifest, "cpu", path, None, loader_dataset, env)


class ChunkPolicyEndToEndTests(unittest.TestCase):
    def test_real_training_checkpoint_loading_and_validation(self):
        rng = np.random.default_rng(73)
        observations = rng.normal(size=(48, 3)).astype(np.float32) * [0.2, 2, 1] + [1, -1, 3]
        observations = observations.astype(np.float32)
        primitive = {
            "observations": observations,
            "next_observations": np.roll(observations, -1, axis=0),
            "actions": rng.uniform(-0.8, 0.8, size=(48, 2)).astype(np.float32),
            "rewards": rng.uniform(-1, 1, size=48).astype(np.float32),
            "terminals": np.zeros(48, dtype=bool),
            "timeouts": np.tile([False] * 7 + [True], 6),
            "episode_ids": np.repeat(np.arange(6), 8),
        }
        primitive_before = {key: value.copy() for key, value in primitive.items()}
        for algo in ("iql", "mopo", "mobile"):
            for length in (1, 4):
                modes = ("direct", "recursive") if algo != "iql" and length == 4 else ("direct",)
                for mode in modes:
                    with self.subTest(algo=algo, length=length, mode=mode), tempfile.TemporaryDirectory() as directory:
                        args = policy_args(algo, 0.8)
                        args.batch_size = 8
                        args.dynamics_max_epochs = 1
                        args.rollout_batch_size = 8
                        args.rollout_length = 1
                        args.rollout_freq = 1
                        args.model_retain_epochs = 1
                        args.real_ratio = 0.5
                        args.dynamics_chunk_mode = mode
                        args.quiet = True
                        chunk_dataset = chunking.make_action_chunk_dataset(primitive, length, 0.99)
                        chunk_before = {key: value.copy() for key, value in chunk_dataset.items()}
                        expected_mean, expected_std = fit_observation_stats(chunk_dataset["observations"])
                        schema = sweep.make_training_schema(algo, "Reacher-v5", {}, length, args)
                        run_dir = Path(directory) / "run"
                        with patch.object(sweep, "make_env", side_effect=lambda *_: TinyEnv()):
                            sweep.train_algo(
                                algo, "Reacher-v5", primitive, chunk_dataset, length,
                                run_dir, Path(directory) / "eval", {}, schema, args,
                            )
                        manifest = json.loads((run_dir / "run_manifest.json").read_text())
                        self.assertEqual([item["step"] for item in manifest["checkpoints"]], [0, 1, 2])
                        env = TinyEnv(2 * length)
                        for checkpoint in manifest["checkpoints"]:
                            policy_path = Path(checkpoint["policy_path"])
                            policy, _ = evaluation.load_policy_and_dynamics(
                                manifest, "cpu", policy_path, None, {}, env,
                            )
                            normalizers = [m for m in policy.modules() if isinstance(m, NormalizedObservationBackbone)]
                            self.assertGreater(len(normalizers), 1)
                            for module in normalizers:
                                np.testing.assert_array_equal(module.mean.numpy(), expected_mean)
                                np.testing.assert_array_equal(module.std.numpy(), expected_std)
                            actions = policy.select_action(observations, deterministic=True)
                            self.assertTrue(np.isfinite(actions).all())
                            heldout = validation.prepare_policy_dataset(manifest, policy, chunk_dataset, chunk_dataset)
                            metrics = validation.validate_policy(algo, policy, heldout, batch_size=8)
                            self.assertTrue(all(np.isfinite(value) for value in metrics.values()))
                            if algo != "iql":
                                with_dynamics, dynamics = evaluation.load_policy_and_dynamics(
                                    manifest, "cpu", policy_path, Path(checkpoint["dynamics_path"]), {}, env,
                                )
                                self.assertIsNotNone(dynamics)
                                np.testing.assert_array_equal(
                                    with_dynamics.select_action(observations, deterministic=True), actions,
                                )
                        for key in primitive:
                            np.testing.assert_array_equal(primitive[key], primitive_before[key])
                            np.testing.assert_array_equal(chunk_dataset[key], chunk_before[key])


class ChunkPolicyPlotTests(unittest.TestCase):
    @staticmethod
    def row(chunk_length=4, rho=None):
        return {
            "algo": "iql", "chunk_length": chunk_length,
            "training_schema": training_schema("iql", chunk_length, rho),
        }

    @staticmethod
    def cohort(*series):
        return plot.validate_plot_cohort({
            "version": plot.PLOT_COHORT_VERSION, "seeds": [0],
            "ablation": {"name": "chunk_length", "values": [1, 4]},
            "match": {}, "series": list(series),
        })

    def test_saved_plot_rows_average_seeds_only_within_one_architecture(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata_path = root / "metadata.json"
            sweep.write_json_atomic(metadata_path, {
                "source": "minari", "dataset_id": "mujoco/reacher/expert-v0",
            })
            eval_dirs = []
            for index, (rho, version) in enumerate(((None, None), (0.8, 1), (0.8, 2), (1.0, 2))):
                for seed in (0, 1):
                    row = self.row(rho=rho)
                    if version is not None:
                        row["training_schema"]["policy_architecture"]["version"] = version
                    row["training_schema"]["seed"] = seed
                    row["training_schema"]["dataset"]["seed"] = seed
                    row.update({
                        "env_name": "Reacher-v5", "dataset_source": "minari",
                        "dataset_tag": "minari-reacher", "dataset_metadata_path": str(metadata_path),
                    })
                    eval_dir = root / f"variant-{index}" / f"seed-{seed}"
                    manifest_path = eval_dir / "run_manifest.json"
                    sweep.write_json_atomic(manifest_path, row)
                    sweep.write_json_atomic(eval_dir / "results.json", {
                        "run_manifest_path": str(manifest_path),
                        "evaluation_config": {"schema_version": plot.EVALUATION_SCHEMA_VERSION},
                        "last_policy_performance_mean": float(index + seed),
                        "expert_performance_mean": 10.0,
                    })
                    eval_dirs.append(eval_dir)
            self.assertEqual(plot.latest_eval_dirs(root), sorted(eval_dirs))
            rows = plot.load_rows(eval_dirs)
            self.assertEqual(len({row["seed_group"] for row in rows}), 4)
            averaged = plot.average_seed_rows(rows)
            self.assertEqual(len(averaged), 4)
            self.assertTrue(all(len(row["seed_rows"]) == 2 for row in averaged))
            self.assertEqual([row["last_policy_performance_mean"] for row in averaged], [0.5, 1.5, 2.5, 3.5])

    def test_version_selectors_keep_normalized_and_old_decay_separate(self):
        old = self.row(rho=0.8)
        old["training_schema"]["policy_architecture"]["version"] = 1
        new = self.row(rho=0.8)
        with self.assertRaisesRegex(ValueError, "multiple seed-averaged configurations"):
            plot.algorithm_groups([old, new], "chunk_length")
        cohort = self.cohort(
            {"algo": "iql", "match": {"policy_architecture.version": 1}},
            {"algo": "iql", "match": {"policy_architecture.version": 2}},
        )
        groups = plot.algorithm_groups(plot.select_plot_cohort([old, new], cohort), "chunk_length")
        self.assertEqual(len(groups), 2)
        self.assertEqual([
            group[1][0]["training_schema"]["policy_architecture"]["version"] for group in groups
        ], [1, 2])

    def test_nested_selectors_separate_rhos_and_null_selects_legacy(self):
        rows = [self.row(rho=rho) for rho in (None, 0.8, 1.0)]
        cohort = self.cohort(
            {"algo": "iql", "match": {"policy_architecture": None}},
            {"algo": "iql", "match": {"policy_architecture.type": "decay-sn", "policy_architecture.rho": 0.8}},
            {"algo": "iql", "match": {"policy_architecture.type": "decay-sn", "policy_architecture.rho": 1.0}},
        )
        selected = plot.select_plot_cohort(rows, cohort)
        groups = plot.algorithm_groups(selected, "chunk_length")
        self.assertEqual(len(groups), 3)
        self.assertNotIn("policy_architecture", groups[0][1][0]["training_schema"])
        self.assertEqual([group[1][0]["training_schema"]["policy_architecture"]["rho"] for group in groups[1:]], [0.8, 1.0])

    def test_unselected_mixed_architectures_are_not_combined(self):
        with self.assertRaisesRegex(ValueError, "multiple seed-averaged configurations"):
            plot.algorithm_groups([self.row(), self.row(rho=0.8)], "chunk_length")

    def test_enabled_chunk_curve_is_coherent_but_rho_change_is_rejected(self):
        rows = [self.row(chunk_length, 0.8) for chunk_length in (1, 4)]
        self.assertEqual(len(plot.algorithm_groups(rows, "chunk_length")), 1)
        varied = copy.deepcopy(rows)
        varied[1]["training_schema"]["policy_architecture"]["rho"] = 0.5
        with self.assertRaisesRegex(ValueError, "policy_architecture.rho"):
            plot.algorithm_groups(varied, "chunk_length")


if __name__ == "__main__":
    unittest.main()

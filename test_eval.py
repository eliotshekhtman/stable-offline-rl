import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import gymnasium as gym
import numpy as np
import torch

import chunking
import eval as evaluation


class PositionEnv(gym.Env):
    def __init__(self, terminate_at: int = 5, success_at: int | None = None):
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32)
        self.model = SimpleNamespace(nq=1, nv=1)
        positions = np.zeros((2, 3), dtype=np.float64)
        self.data = SimpleNamespace(
            qpos=np.zeros(1, dtype=np.float64),
            qvel=np.zeros(1, dtype=np.float64),
            xpos=positions,
            body_xpos=positions,
        )
        self.sim = SimpleNamespace(data=self.data, forward=lambda: None)
        self.terminate_at = terminate_at
        self.success_at = success_at
        self.steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.set_state(np.zeros(1), np.zeros(1))
        return self._get_obs(), {"x_position": 0.0}

    def step(self, action):
        self.steps += 1
        self.data.qvel[:] = action
        self.data.qpos += action
        self.data.xpos[1, 0] = self.data.qpos[0]
        info = {"x_position": float(self.data.qpos[0])}
        return self._get_obs(), 1.0, self.steps == self.terminate_at, False, info

    def set_state(self, qpos, qvel):
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        self.data.xpos[1, 0] = qpos[0]

    def _get_obs(self):
        return self.data.qpos.astype(np.float32).copy()

    def _get_observations(self, force_update=False):
        return {"obs": self._get_obs()}

    def _flatten_obs(self, observations):
        return observations["obs"]

    def _check_success(self):
        return self.success_at is not None and self.steps >= self.success_at


class CountingChunkPolicy:
    def __init__(self, chunk_length: int):
        self.chunk_length = chunk_length
        self.calls = 0

    def select_action(self, obs, deterministic=False):
        self.calls += 1
        return np.ones((len(obs), self.chunk_length), dtype=np.float32)


def manifest(chunk_length: int = 3) -> dict:
    return {
        "env_name": "HalfCheetah-v5",
        "dataset_source": "minari",
        "chunk_length": chunk_length,
        "training_schema": {"dataset": {"source": "minari"}},
    }


class EvaluationTests(unittest.TestCase):
    def test_expert_baseline_is_shared_and_legacy_per_run_cache_is_reused(self):
        manifest_data = {
            "env_name": "HalfCheetah-v5",
            "dataset_source": "minari",
            "expert": "/experts/HalfCheetah-v5.zip",
        }
        args = SimpleNamespace(final_eval_episodes=2, reuse_eval=False)
        expert_info = {
            "returns": np.asarray([1.0, 3.0], dtype=np.float32),
            "return_mean": 2.0, "return_std": 1.0,
            "performance": np.asarray([2.0, 4.0], dtype=np.float32),
            "performance_mean": 3.0, "performance_std": 1.0,
        }
        with tempfile.TemporaryDirectory() as directory:
            env_root = Path(directory) / "evals" / "HalfCheetah-v5"
            first_eval = env_root / "run_a" / "variant"
            second_eval = env_root / "run_b" / "variant"
            first_eval.mkdir(parents=True)
            second_eval.mkdir(parents=True)
            with patch.object(
                evaluation, "evaluate_expert", return_value=expert_info
            ) as evaluate_expert:
                first = evaluation.load_or_evaluate_expert(
                    first_eval, manifest_data, args, rollout_seed=100
                )
                second = evaluation.load_or_evaluate_expert(
                    second_eval,
                    {**manifest_data, "dataset_source": "generated"},
                    args,
                    rollout_seed=100,
                )

            evaluate_expert.assert_called_once()
            np.testing.assert_array_equal(first["returns"], second["returns"])
            self.assertFalse((first_eval / "expert.npz").exists())
            self.assertEqual(
                len(list((env_root / "_expert_cache").glob("*.npz"))), 1
            )

            legacy_eval = env_root / "run_c" / "variant"
            legacy_eval.mkdir(parents=True)
            legacy_seed = 101
            np.savez_compressed(
                legacy_eval / "expert.npz",
                returns=expert_info["returns"],
                performance=expert_info["performance"],
            )
            (legacy_eval / "expert.json").write_text(json.dumps({
                "version": evaluation.EVALUATION_SCHEMA_VERSION,
                "env_name": "HalfCheetah-v5",
                "expert": manifest_data["expert"],
                "episodes": 2,
                "seed": legacy_seed,
            }), encoding="utf-8")
            args.reuse_eval = True
            with patch.object(
                evaluation, "evaluate_expert",
                side_effect=AssertionError("matching legacy cache must be reused"),
            ):
                legacy = evaluation.load_or_evaluate_expert(
                    legacy_eval, manifest_data, args, rollout_seed=legacy_seed
                )
            np.testing.assert_array_equal(legacy["returns"], expert_info["returns"])
            self.assertEqual(
                len(list((env_root / "_expert_cache").glob("*.npz"))), 2
            )

    def test_lift_expert_metadata_finds_continuing_ph_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            task_root = Path(directory) / "datasets" / "Lift"
            training_path = (
                task_root / "robomimic_lift_mg_dense_continuing"
                / "training" / "metadata.json"
            )
            ph_path = (
                task_root / "robomimic_lift_ph_continuing"
                / "cached" / "metadata.json"
            )
            training_path.parent.mkdir(parents=True)
            ph_path.parent.mkdir(parents=True)
            training_path.write_text(json.dumps({
                "task": "Lift", "dataset_type": "mg_dense",
            }), encoding="utf-8")
            ph_path.write_text(json.dumps({
                "task": "Lift", "dataset_type": "ph", "marker": "cached-ph",
            }), encoding="utf-8")

            with patch.object(
                evaluation.load_offline, "load_robomimic_dataset",
                side_effect=AssertionError("local PH metadata should be used"),
            ):
                metadata = evaluation.robomimic_expert_metadata({
                    "dataset_metadata_path": str(training_path),
                })
            self.assertEqual(metadata["marker"], "cached-ph")

    def test_compact_rollout_cache_omits_unused_arrays_and_reads_legacy_cache(self):
        env = PositionEnv(terminate_at=2)
        policy = CountingChunkPolicy(chunk_length=1)
        compact = evaluation.evaluate_policy_rollouts(
            policy, env, manifest(chunk_length=1), episodes=1, seed=0,
            body_ids=np.asarray([1]), include_contraction_state=False,
        )
        self.assertEqual(
            set(compact),
            {
                "returns", "performance", "performance_metric",
                "performance_label", "performance_higher_is_better",
                "decision_observations", "action_chunks",
            },
        )

        checkpoint = {"step": 10, "policy_path": "/policy.pth"}
        args = SimpleNamespace(reuse_eval=True)
        with tempfile.TemporaryDirectory() as directory:
            eval_dir = Path(directory)
            evaluation.save_cached_rollout(
                eval_dir, manifest(chunk_length=1), checkpoint, 1, 0, compact,
                include_contraction_state=False,
            )
            cache_path = eval_dir / "rollouts" / "step_10.npz"
            with np.load(cache_path) as data:
                self.assertEqual(
                    set(data.files),
                    {"returns", "performance", "decision_observations", "action_chunks"},
                )
            loaded = evaluation.load_cached_rollout(
                eval_dir, manifest(chunk_length=1), checkpoint, args, 1, 0,
                include_contraction_state=False,
            )
            np.testing.assert_array_equal(loaded["returns"], compact["returns"])

            config_path = cache_path.with_suffix(".json")
            legacy_config = json.loads(config_path.read_text(encoding="utf-8"))
            legacy_config.pop("contraction_state")
            config_path.write_text(json.dumps(legacy_config), encoding="utf-8")
            self.assertIsNotNone(evaluation.load_cached_rollout(
                eval_dir, manifest(chunk_length=1), checkpoint, args, 1, 0,
                include_contraction_state=False,
            ))

    def test_rollout_cache_marker_is_written_after_payload(self):
        rollout_info = {
            "returns": np.asarray([1.0], dtype=np.float32),
            "performance": np.asarray([2.0], dtype=np.float32),
            "decision_observations": np.asarray([[3.0]], dtype=np.float32),
            "action_chunks": np.asarray([[4.0]], dtype=np.float32),
        }
        checkpoint = {"step": 10, "policy_path": "/policy.pth"}
        args = SimpleNamespace(reuse_eval=True)
        with tempfile.TemporaryDirectory() as directory:
            eval_dir = Path(directory)
            evaluation.save_cached_rollout(
                eval_dir, manifest(chunk_length=1), checkpoint, 2, 99,
                rollout_info, include_contraction_state=False,
            )
            with patch.object(
                evaluation, "atomic_write_json", side_effect=OSError("interrupted")
            ):
                with self.assertRaisesRegex(OSError, "interrupted"):
                    evaluation.save_cached_rollout(
                        eval_dir, manifest(chunk_length=1), checkpoint, 1, 0,
                        rollout_info, include_contraction_state=False,
                    )

            cache_path = eval_dir / "rollouts" / "step_10.npz"
            self.assertTrue(cache_path.exists())
            self.assertFalse(cache_path.with_suffix(".json").exists())
            self.assertIsNone(evaluation.load_cached_rollout(
                eval_dir, manifest(chunk_length=1), checkpoint, args, 1, 0,
                include_contraction_state=False,
            ))

    def test_contraction_cache_invalidates_old_marker_before_overwrite(self):
        checkpoint = {"step": 10, "policy_path": "/policy.pth"}
        manifest_data = manifest(chunk_length=1)
        old_args = SimpleNamespace(
            contraction_trajectories=1,
            contraction_horizon=2,
            perturbation_scale=0.01,
        )
        new_args = SimpleNamespace(
            reuse_eval=True,
            contraction_trajectories=1,
            contraction_horizon=3,
            perturbation_scale=0.01,
        )
        contraction = {"distance_curves": np.zeros((1, 2), dtype=np.float32)}
        with tempfile.TemporaryDirectory() as directory:
            eval_dir = Path(directory)
            evaluation.save_cached_contraction(
                eval_dir, manifest_data, checkpoint, old_args, 0, contraction
            )
            with patch.object(
                evaluation, "atomic_write_json", side_effect=OSError("interrupted")
            ), self.assertRaisesRegex(OSError, "interrupted"):
                evaluation.save_cached_contraction(
                    eval_dir, manifest_data, checkpoint, new_args, 0, contraction
                )

            self.assertTrue((eval_dir / "contraction_last.npz").exists())
            self.assertFalse((eval_dir / "contraction_last.json").exists())
            self.assertIsNone(evaluation.load_cached_contraction(
                eval_dir, manifest_data, checkpoint, new_args, 0
            ))

    def test_corrupt_evaluation_rollout_and_contraction_caches_are_misses(self):
        checkpoint = {"step": 10, "policy_path": "/policy.pth"}
        manifest_data = manifest(chunk_length=1)
        args = SimpleNamespace(
            reuse_eval=True,
            contraction_trajectories=1,
            contraction_horizon=2,
            perturbation_scale=0.01,
        )
        with tempfile.TemporaryDirectory() as directory:
            eval_dir = Path(directory)
            for name in (
                "history.json", "history.npz", "returns_last.npz",
                "contraction_last.npz", "conservativity.npz",
            ):
                (eval_dir / name).touch()
            (eval_dir / "results.json").write_text("{", encoding="utf-8")
            self.assertFalse(evaluation.evaluation_is_complete(
                eval_dir, {"expected": "config"}
            ))

            rollout_path = eval_dir / "rollouts" / "step_10.npz"
            rollout_path.parent.mkdir()
            rollout_path.write_bytes(b"not an npz")
            evaluation.atomic_write_json(
                rollout_path.with_suffix(".json"),
                evaluation.rollout_cache_config(
                    manifest_data, checkpoint, 1, 0,
                    include_contraction_state=False,
                ),
            )
            self.assertIsNone(evaluation.load_cached_rollout(
                eval_dir, manifest_data, checkpoint, args, 1, 0,
                include_contraction_state=False,
            ))

            contraction_path = eval_dir / "contraction_last.npz"
            contraction_path.write_bytes(b"not an npz")
            evaluation.atomic_write_json(
                eval_dir / "contraction_last.json",
                evaluation.contraction_cache_config(
                    manifest_data, checkpoint, args, 0
                ),
            )
            self.assertIsNone(evaluation.load_cached_contraction(
                eval_dir, manifest_data, checkpoint, args, 0
            ))

            rollout_path.with_suffix(".json").write_text(
                "not-json", encoding="utf-8"
            )
            self.assertIsNone(evaluation.load_cached_rollout(
                eval_dir, manifest_data, checkpoint, args, 1, 0,
                include_contraction_state=False,
            ))

    def test_final_only_evaluation_skips_nonfinal_rollouts_and_reuses_final_ood(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            eval_dir = root / "evals" / "HalfCheetah-v5" / "run" / "variant"
            run_dir = root / "run"
            run_dir.mkdir()
            dataset = {
                "observations": np.asarray([[0.0]], dtype=np.float32),
                "actions": np.asarray([[0.0]], dtype=np.float32),
                "next_observations": np.asarray([[0.0]], dtype=np.float32),
                "rewards": np.asarray([0.0], dtype=np.float32),
                "terminals": np.asarray([True]),
                "timeouts": np.asarray([False]),
                "episode_ids": np.asarray([0]),
            }
            training_manifest = {
                "env_name": "HalfCheetah-v5",
                "dataset_source": "minari",
                "eval_dir": str(eval_dir),
                "train_dataset_path": str(root / "train.npz"),
                "test_dataset_path": str(root / "test.npz"),
                "chunk_length": 1,
                "base_discount": 0.99,
                "macro_discount": 0.99,
                "algo": "bc",
                "dataset_tag": "test",
                "expert": "/expert.zip",
                "epoch": 1,
                "step_per_epoch": 1,
                "training_schema": {"dataset": {"source": "minari"}},
                "checkpoints": [
                    {"step": 0, "requested_percent": 0, "actual_percent": 0.0,
                     "policy_path": "/step0.pth"},
                    {"step": 10, "requested_percent": 100, "actual_percent": 100.0,
                     "policy_path": "/step10.pth"},
                ],
            }
            (run_dir / "run_manifest.json").write_text(
                json.dumps(training_manifest), encoding="utf-8"
            )
            eval_dir.mkdir(parents=True)
            (eval_dir / "results.json").write_text(
                json.dumps({"evaluation_config": {"stale": True}}),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                device="cpu", checkpoint_eval_episodes=0, final_eval_episodes=1,
                expert=None, seed=4, contraction_trajectories=1,
                contraction_horizon=1, perturbation_scale=0.0, ood_samples=1,
                reuse_eval=True,
            )
            rollout_info = {
                "returns": np.asarray([1.0], dtype=np.float32),
                "performance": np.asarray([2.0], dtype=np.float32),
                "performance_metric": "forward_displacement",
                "performance_label": "forward displacement (m)",
                "performance_higher_is_better": True,
                "decision_observations": np.asarray([[0.0]], dtype=np.float32),
                "action_chunks": np.asarray([[0.0]], dtype=np.float32),
                "position_trajectories": np.zeros((1, 2, 3), dtype=np.float32),
                "position_lengths": np.asarray([2]),
                "initial_qpos": np.zeros((1, 1)),
                "initial_qvel": np.zeros((1, 1)),
            }
            conservativity = {
                "state_ood_ratio": np.asarray(1.0, dtype=np.float32),
                "state_action_ood_ratio": np.asarray(1.0, dtype=np.float32),
            }
            expert_info = {
                "returns": np.asarray([3.0], dtype=np.float32),
                "return_mean": 3.0, "return_std": 0.0,
                "performance": np.asarray([4.0], dtype=np.float32),
                "performance_mean": 4.0, "performance_std": 0.0,
            }
            env = PositionEnv(terminate_at=1)
            env.close = MagicMock()
            policy = MagicMock()

            def load_dataset(_):
                self.assertFalse((eval_dir / "results.json").exists())
                return dataset

            with (
                patch.object(evaluation.rollout, "load_dataset", side_effect=load_dataset),
                patch.object(evaluation, "make_eval_env", return_value=env),
                patch.object(evaluation, "load_policy_and_dynamics", return_value=(policy, None)),
                patch.object(evaluation, "agent_position_bodies", return_value=(np.asarray([1]), ["agent"])),
                patch.object(evaluation, "prepare_conservativity", return_value={}),
                patch.object(evaluation, "evaluate_policy_rollouts", return_value=rollout_info) as evaluate_rollouts,
                patch.object(evaluation, "save_cached_rollout"),
                patch.object(evaluation, "evaluate_conservativity", return_value=conservativity) as evaluate_ood,
                patch.object(evaluation, "evaluate_contraction", return_value={"distance_curves": np.zeros((1, 1))}),
                patch.object(evaluation, "save_cached_contraction"),
                patch.object(evaluation, "load_or_evaluate_expert", return_value=expert_info),
            ):
                evaluation.evaluate_run(run_dir, args)

            evaluate_rollouts.assert_called_once()
            self.assertEqual(evaluate_rollouts.call_args.args[3], 1)
            self.assertTrue(evaluate_rollouts.call_args.kwargs["include_contraction_state"])
            evaluate_ood.assert_called_once()
            env.close.assert_called_once()
            history = json.loads((eval_dir / "history.json").read_text(encoding="utf-8"))
            self.assertEqual([record["step"] for record in history["records"]], [10])

    def test_rollout_executes_chunks_open_loop_without_stopping_on_positive_gym_reward(self):
        env = PositionEnv(terminate_at=5)
        policy = CountingChunkPolicy(chunk_length=3)

        result = evaluation.rollout_policy_episode(
            env, policy, manifest(), reset_seed=0, body_ids=np.asarray([1])
        )

        self.assertEqual(policy.calls, 2)
        self.assertEqual(len(result["decision_observations"]), 2)
        self.assertEqual(len(result["positions"]), 6)
        self.assertEqual(result["return"], 5.0)
        self.assertEqual(result["performance"], 5.0)

    def test_robomimic_rollout_uses_task_success_instead_of_positive_reward(self):
        env = PositionEnv(terminate_at=5, success_at=4)
        policy = CountingChunkPolicy(chunk_length=3)
        robomimic_manifest = {
            "env_name": "Lift",
            "dataset_source": "robomimic",
            "chunk_length": 3,
        }

        result = evaluation.rollout_policy_episode(
            env, policy, robomimic_manifest, reset_seed=0, body_ids=np.asarray([1])
        )

        self.assertEqual(policy.calls, 2)
        self.assertEqual(len(result["positions"]), 5)
        self.assertEqual(result["return"], 4.0)
        self.assertEqual(result["performance"], 1.0)

    def test_checkpoint_record_tracks_evaluation_budget_and_seed(self):
        rollout_info = {
            "returns": np.asarray([1.0, 3.0]),
            "performance": np.asarray([0.0, 1.0]),
        }
        conservativity = {
            "state_ood_ratio": np.asarray(1.2),
            "state_action_ood_ratio": np.asarray(1.4),
        }
        checkpoint = {"requested_percent": 100, "actual_percent": 100.0, "step": 200}

        record = evaluation.checkpoint_record(
            checkpoint, rollout_info, conservativity, episodes=2, rollout_seed=1_000_000
        )

        self.assertEqual(record["evaluation_episodes"], 2)
        self.assertEqual(record["evaluation_seed"], 1_000_000)
        self.assertEqual(record["policy_return_mean"], 2.0)
        self.assertEqual(record["policy_performance_mean"], 0.5)

    def test_performance_dispatch_rejects_tasks_outside_supported_scope(self):
        with self.assertRaisesRegex(ValueError, "Unsupported task 'Ant-v5'"):
            evaluation.performance_definition("Ant-v5")
        with self.assertRaisesRegex(ValueError, "Unsupported task 'Square'"):
            evaluation.episode_performance(
                "Square", MagicMock(), 0.0, 1, False, {}, {}
            )

    def test_zero_perturbation_matches_reused_base_trajectory(self):
        env = PositionEnv(terminate_at=5)
        policy = CountingChunkPolicy(chunk_length=3)
        base = evaluation.evaluate_policy_rollouts(
            policy, env, manifest(), episodes=1, seed=0, body_ids=np.asarray([1])
        )

        contraction = evaluation.evaluate_contraction(
            policy, env, manifest(), base, trajectory_count=1, horizon=5,
            perturbation_scale=0.0, seed=0, body_ids=np.asarray([1]),
            body_names=["agent"],
        )

        np.testing.assert_allclose(contraction["distance_curves"], 0.0)

    def test_continuing_lift_contraction_runs_past_success(self):
        env = PositionEnv(terminate_at=5, success_at=2)
        policy = CountingChunkPolicy(chunk_length=1)
        continuing_manifest = {
            "env_name": "Lift",
            "dataset_source": "robomimic",
            "chunk_length": 1,
            "training_schema": {"dataset": {"task_semantics": "continuing"}},
        }
        base = evaluation.evaluate_policy_rollouts(
            policy, env, continuing_manifest, episodes=1, seed=0,
            body_ids=np.asarray([1]),
        )

        self.assertEqual(base["position_lengths"].tolist(), [3])
        with patch.object(
            evaluation,
            "controlled_agent_indices",
            return_value=(np.asarray([0]), np.asarray([0])),
        ):
            contraction = evaluation.evaluate_contraction(
                policy, env, continuing_manifest, base, trajectory_count=1, horizon=5,
                perturbation_scale=0.0, seed=0, body_ids=np.asarray([1]),
                body_names=["agent"],
            )

        self.assertEqual(contraction["distance_curves"].shape, (1, 6))
        np.testing.assert_allclose(contraction["distance_curves"], 0.0)

    def test_predict_next_obs_dispatches_recursive_dynamics_through_adapter(self):
        class RecursiveDynamics:
            def __init__(self):
                self.calls = []

            def mean_next_obss(self, observations, action_chunks):
                self.calls.append((observations.copy(), action_chunks.copy()))
                return observations + action_chunks[:, :1]

        dynamics = RecursiveDynamics()
        observations = np.asarray([[1.0], [3.0]], dtype=np.float32)
        action_chunks = np.asarray([[0.5, 2.0], [-1.0, 4.0]], dtype=np.float32)

        predictions = evaluation.predict_next_obs(
            dynamics, observations, action_chunks
        )

        np.testing.assert_allclose(predictions, [[1.5], [2.0]])
        self.assertEqual(len(dynamics.calls), 1)
        np.testing.assert_array_equal(dynamics.calls[0][0], observations)
        np.testing.assert_array_equal(dynamics.calls[0][1], action_chunks)

    def test_predict_next_obs_preserves_legacy_direct_elite_mean(self):
        class IdentityScaler:
            def transform(self, values):
                return torch.as_tensor(values, dtype=torch.float32)

        class DirectModel:
            def __init__(self):
                self.elites = torch.as_tensor([0, 2], dtype=torch.long)

            def __call__(self, model_input):
                batch_size = len(model_input)
                means = torch.zeros((3, batch_size, 2), dtype=torch.float32)
                means[0, :, 0] = 1.0
                means[1, :, 0] = 100.0
                means[2, :, 0] = 3.0
                return means, torch.zeros_like(means)

        dynamics = SimpleNamespace(scaler=IdentityScaler(), model=DirectModel())
        observations = np.asarray([[5.0], [7.0]], dtype=np.float32)
        action_chunks = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)

        predictions = evaluation.predict_next_obs(
            dynamics, observations, action_chunks
        )

        np.testing.assert_allclose(predictions, [[7.0], [9.0]])

    def test_loader_passes_explicit_recursive_chunk_configuration(self):
        manifest_data = self._model_based_manifest(
            chunk_length=4,
            chunk_dynamics={"version": 1, "mode": "recursive"},
        )

        build_call = self._load_with_mocked_model_builder(manifest_data)

        self.assertEqual(build_call.kwargs["dynamics_chunk_mode"], "recursive")
        self.assertEqual(build_call.kwargs["chunk_length"], 4)
        self.assertEqual(build_call.kwargs["base_discount"], 0.97)
        self.assertEqual(build_call.kwargs["primitive_action_dim"], 1)
        self.assertTrue(build_call.kwargs["build_dynamics_model"])

    def test_loader_defaults_field_absent_legacy_manifest_to_direct(self):
        manifest_data = self._model_based_manifest(chunk_length=4)

        build_call = self._load_with_mocked_model_builder(manifest_data)

        self.assertEqual(build_call.kwargs["dynamics_chunk_mode"], "direct")
        self.assertEqual(build_call.kwargs["chunk_length"], 4)
        self.assertEqual(build_call.kwargs["primitive_action_dim"], 1)

    def test_model_based_loader_does_not_require_training_data(self):
        manifest_data = self._model_based_manifest(chunk_length=1)
        env = chunking.ActionChunkWrapper(PositionEnv(), 1)
        policy = MagicMock()
        try:
            with (
                patch.object(
                    evaluation,
                    "build_model_based_policy",
                    return_value=(policy, None, None),
                ) as builder,
                patch.object(evaluation.torch, "load", return_value={}),
            ):
                loaded_policy, loaded_dynamics = evaluation.load_policy_and_dynamics(
                    manifest_data,
                    "cpu",
                    Path("policy.pth"),
                    None,
                    {},
                    env,
                )
        finally:
            env.close()

        self.assertIs(loaded_policy, policy)
        self.assertIsNone(loaded_dynamics)
        self.assertFalse(builder.call_args.kwargs["build_dynamics_model"])

    def test_model_based_loader_accepts_in_memory_legacy_builder_signature(self):
        manifest_data = self._model_based_manifest(chunk_length=1)
        env = chunking.ActionChunkWrapper(PositionEnv(), 1)
        policy = MagicMock()
        dynamics = MagicMock()

        def legacy_builder(
            algo, env, args, discount, chunk_length=1, base_discount=0.99,
            dynamics_chunk_mode="direct", primitive_action_dim=None,
        ):
            return policy, dynamics, None

        try:
            with (
                patch.object(
                    evaluation, "build_model_based_policy", new=legacy_builder
                ),
                patch.object(evaluation.torch, "load", return_value={}),
            ):
                loaded_policy, loaded_dynamics = evaluation.load_policy_and_dynamics(
                    manifest_data, "cpu", Path("policy.pth"), None, {}, env
                )
        finally:
            env.close()

        self.assertIs(loaded_policy, policy)
        self.assertIs(loaded_dynamics, dynamics)

    def test_model_free_loader_only_supplies_statistics_to_td3bc(self):
        env = chunking.ActionChunkWrapper(PositionEnv(), 1)
        dataset = {"observations": np.asarray([[1.0], [3.0]], dtype=np.float32)}
        policy = MagicMock()
        common = {
            "epoch": 3,
            "step_per_epoch": 4,
            "chunk_length": 1,
            "macro_discount": 0.99,
        }
        try:
            iql_manifest = {
                **common,
                "algo": "iql",
                "training_schema": {"iql": {
                    "temperature": 3.0, "expectile": 0.7,
                    "learning_rate": 3e-4, "lr_schedule": "constant",
                    "hidden_dims": [256, 256],
                }},
            }
            with (
                patch.object(
                    evaluation, "build_model_free_policy",
                    return_value=(policy, None),
                ) as builder,
                patch.object(evaluation.torch, "load", return_value={}),
            ):
                evaluation.load_policy_and_dynamics(
                    iql_manifest, "cpu", Path("policy.pth"), None, dataset, env
                )
            self.assertIsNone(builder.call_args.args[2])

            td3bc_manifest = {
                **common,
                "algo": "td3bc",
                "training_schema": {"td3bc": {
                    "learning_rate": 3e-4, "alpha": 2.5,
                    "hidden_dims": [256, 256],
                }},
            }
            with (
                patch.object(
                    evaluation, "build_model_free_policy",
                    return_value=(policy, None),
                ) as builder,
                patch.object(evaluation.torch, "load", return_value={}),
            ):
                evaluation.load_policy_and_dynamics(
                    td3bc_manifest, "cpu", Path("policy.pth"), None, dataset, env
                )
            obs_mean, obs_std = builder.call_args.args[2].normalize_obs()
            np.testing.assert_array_equal(obs_mean, [[2.0]])
            np.testing.assert_allclose(obs_std, [[1.001]])
        finally:
            env.close()

    def test_cql_loader_reconstructs_legacy_defaults_and_overrides(self):
        env = chunking.ActionChunkWrapper(PositionEnv(), 1)
        policy = MagicMock()
        common = {
            "algo": "cql",
            "epoch": 3,
            "step_per_epoch": 4,
            "chunk_length": 1,
            "macro_discount": 0.99,
        }
        try:
            for cql_schema, expected_lr, expected_max, expected_mode in (
                ({}, 1e-4, 1.0, "fixed"),
                (
                    {
                        "entropy_learning_rate": 3e-4,
                        "entropy_alpha_max": None,
                        "lagrange_target_mode": "action-volume",
                    },
                    3e-4,
                    None,
                    "action-volume",
                ),
            ):
                training_schema = {"implementation_version": 2}
                if cql_schema:
                    training_schema["cql"] = cql_schema
                manifest = {**common, "training_schema": training_schema}
                with (
                    patch.object(
                        evaluation,
                        "build_model_free_policy",
                        return_value=(policy, None),
                    ) as builder,
                    patch.object(evaluation.torch, "load", return_value={}),
                ):
                    evaluation.load_policy_and_dynamics(
                        manifest, "cpu", Path("policy.pth"), None, {}, env
                    )
                build_args = builder.call_args.args[3]
                self.assertEqual(
                    build_args.cql_entropy_learning_rate, expected_lr
                )
                self.assertEqual(build_args.cql_entropy_alpha_max, expected_max)
                self.assertEqual(
                    build_args.cql_lagrange_target_mode, expected_mode
                )
                self.assertEqual(builder.call_args.kwargs["chunk_length"], 1)
        finally:
            env.close()

    def test_loader_rejects_unsupported_manifest_algorithm(self):
        manifest_data = {
            "algo": "unknown",
            "epoch": 3,
            "step_per_epoch": 4,
            "training_schema": {},
        }
        env = chunking.ActionChunkWrapper(PositionEnv(), 1)
        try:
            with self.assertRaisesRegex(
                ValueError, "Unsupported algorithm: unknown"
            ):
                evaluation.load_policy_and_dynamics(
                    manifest_data,
                    "cpu",
                    Path("policy.pth"),
                    None,
                    {},
                    env,
                )
        finally:
            env.close()

    @staticmethod
    def _model_based_manifest(
        chunk_length: int,
        chunk_dynamics: dict | None = None,
    ) -> dict:
        model_based = {}
        if chunk_dynamics is not None:
            model_based["chunk_dynamics"] = chunk_dynamics
        return {
            "algo": "mopo",
            "epoch": 3,
            "step_per_epoch": 4,
            "chunk_length": chunk_length,
            "base_discount": 0.97,
            "macro_discount": 0.97 ** chunk_length,
            "training_schema": {
                "model_based": model_based,
                "mopo": {"penalty_coef": 0.5},
            },
        }

    @staticmethod
    def _load_with_mocked_model_builder(manifest_data: dict):
        base_env = PositionEnv()
        env = chunking.ActionChunkWrapper(base_env, manifest_data["chunk_length"])
        dataset = {
            "observations": np.zeros((2, 1), dtype=np.float32),
            "actions": np.zeros(
                (2, manifest_data["chunk_length"]), dtype=np.float32
            ),
            "next_observations": np.ones((2, 1), dtype=np.float32),
            "rewards": np.zeros((2,), dtype=np.float32),
            "terminals": np.zeros((2,), dtype=np.float32),
        }
        policy = MagicMock()
        dynamics = MagicMock()
        try:
            with (
                patch.object(
                    evaluation,
                    "build_model_based_policy",
                    return_value=(policy, dynamics, None),
                ) as builder,
                patch.object(evaluation.torch, "load", return_value={}),
            ):
                evaluation.load_policy_and_dynamics(
                    manifest_data,
                    "cpu",
                    Path("policy.pth"),
                    Path("dynamics.pth"),
                    dataset,
                    env,
                )
            return builder.call_args
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()

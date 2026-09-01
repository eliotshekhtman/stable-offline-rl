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

    def test_loader_defaults_field_absent_legacy_manifest_to_direct(self):
        manifest_data = self._model_based_manifest(chunk_length=4)

        build_call = self._load_with_mocked_model_builder(manifest_data)

        self.assertEqual(build_call.kwargs["dynamics_chunk_mode"], "direct")
        self.assertEqual(build_call.kwargs["chunk_length"], 4)
        self.assertEqual(build_call.kwargs["primitive_action_dim"], 1)

    def test_loader_retains_legacy_rambo_manifest_arguments(self):
        manifest_data = self._model_based_manifest(chunk_length=1)
        manifest_data.update(
            algo="rambo",
            adv_weight=3e-4,
            adv_batch_size=256,
            rollout_length=5,
        )

        build_call = self._load_with_mocked_model_builder(manifest_data)
        build_args = build_call.args[2]

        self.assertEqual(build_call.args[0], "rambo")
        self.assertEqual(build_args.adv_weight, 3e-4)
        self.assertEqual(build_args.adv_batch_size, 256)
        self.assertEqual(build_args.rollout_length, 5)
        self.assertEqual(build_call.kwargs["dynamics_chunk_mode"], "direct")

    def test_loader_retains_legacy_dql_manifest_configuration(self):
        manifest_data = {
            "algo": "dql",
            "epoch": 3,
            "step_per_epoch": 4,
            "chunk_length": 1,
            "base_discount": 0.99,
            "macro_discount": 0.99,
            "dql_config": {"legacy": True},
            "training_schema": {},
        }
        env = chunking.ActionChunkWrapper(PositionEnv(), 1)
        dataset = {
            "observations": np.zeros((2, 1), dtype=np.float32),
            "actions": np.zeros((2, 1), dtype=np.float32),
            "next_observations": np.ones((2, 1), dtype=np.float32),
            "rewards": np.zeros((2,), dtype=np.float32),
            "terminals": np.zeros((2,), dtype=np.float32),
        }
        policy = MagicMock()
        try:
            with (
                patch.object(
                    evaluation,
                    "build_model_free_policy",
                    return_value=(policy, None),
                ) as builder,
                patch.object(evaluation.torch, "load", return_value={}),
            ):
                evaluation.load_policy_and_dynamics(
                    manifest_data,
                    "cpu",
                    Path("policy.pth"),
                    None,
                    dataset,
                    env,
                )
        finally:
            env.close()

        self.assertEqual(builder.call_args.args[0], "dql")
        self.assertEqual(builder.call_args.kwargs["dql_config"], {"legacy": True})
        self.assertFalse(hasattr(builder.call_args.args[3], "adv_weight"))

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

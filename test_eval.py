import unittest
from types import SimpleNamespace

import gymnasium as gym
import numpy as np

import eval as evaluation


class PositionEnv(gym.Env):
    def __init__(self, terminate_at: int = 5):
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32)
        self.model = SimpleNamespace(nq=1, nv=1)
        self.data = SimpleNamespace(
            qpos=np.zeros(1, dtype=np.float64),
            qvel=np.zeros(1, dtype=np.float64),
            xpos=np.zeros((2, 3), dtype=np.float64),
        )
        self.terminate_at = terminate_at
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

    def test_only_robomimic_sparse_reward_stops_on_success(self):
        self.assertFalse(evaluation.stops_on_sparse_success(manifest(), 1.0))
        self.assertTrue(
            evaluation.stops_on_sparse_success({"dataset_source": "robomimic"}, 1.0)
        )

    def test_best_checkpoint_uses_metric_direction_and_latest_tie(self):
        records = [
            {"policy_performance_mean": 2.0, "actual_percent": 20.0},
            {"policy_performance_mean": 1.0, "actual_percent": 50.0},
            {"policy_performance_mean": 1.0, "actual_percent": 80.0},
        ]
        self.assertEqual(evaluation.select_best_record(records, True)["actual_percent"], 20.0)
        self.assertEqual(evaluation.select_best_record(records, False)["actual_percent"], 80.0)

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


if __name__ == "__main__":
    unittest.main()

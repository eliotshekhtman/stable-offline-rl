import unittest
from types import SimpleNamespace

import gymnasium as gym
import numpy as np

import eval as evaluation
import policies
import task_support
from offlinerlkit.dynamics import EnsembleDynamics
from offlinerlkit.utils.termination_fns import termination_fn_halfcheetah


class WalkerSupportTests(unittest.TestCase):
    def test_supported_sources_and_other_task_restrictions(self):
        for source in ("generated", "minari", "clean-minari"):
            with self.subTest(source=source):
                task_support.require_supported_task("Walker2d-v5", source)
        self.assertIn("Walker2d-v5", task_support.GENERATED_TASKS)
        self.assertNotIn("Walker2d-v5", task_support.ROBOMIMIC_TASKS)
        with self.assertRaisesRegex(ValueError, "does not support"):
            task_support.require_supported_task("Walker2d-v5", "robomimic")
        for task in ("Walker2d-v4", "Hopper-v5", "Ant-v5"):
            with self.subTest(task=task), self.assertRaisesRegex(
                ValueError, "Unsupported task"
            ):
                task_support.require_supported_task(task, "generated")

    def test_displacement_matches_halfcheetah_metric(self):
        for task in ("Walker2d-v5", "HalfCheetah-v5"):
            with self.subTest(task=task):
                self.assertEqual(
                    evaluation.performance_definition(task),
                    ("forward_displacement", "forward displacement (m)", True),
                )
                self.assertEqual(
                    evaluation.episode_performance(
                        task, None, episode_return=123.0, primitive_steps=5,
                        succeeded=False, reset_info={"x_position": 2.0},
                        final_info={"x_position": -1.0},
                    ),
                    -3.0,
                )

    def test_termination_matches_native_health_at_boundaries(self):
        env = gym.make("Walker2d-v5")
        try:
            env.reset(seed=0)
            heights = (0.79, 0.8, 0.800001, 1.2, 1.999999, 2.0, 2.01)
            angles = (-1.01, -1.0, -0.999999, 0.0, 0.999999, 1.0, 1.01)
            for height in heights:
                for angle in angles:
                    with self.subTest(height=height, angle=angle):
                        qpos = env.unwrapped.data.qpos.copy()
                        qpos[1:3] = height, angle
                        env.unwrapped.set_state(qpos, np.zeros_like(qpos))
                        obs = env.unwrapped._get_obs()[None, :]
                        terminal = policies.termination_fn_walker2d(obs, None, obs)
                        self.assertEqual(terminal.shape, (1, 1))
                        self.assertEqual(terminal.dtype, np.dtype(bool))
                        self.assertEqual(
                            bool(terminal[0, 0]), not env.unwrapped.is_healthy
                        )
        finally:
            env.close()

    def test_nonfinite_predictions_rejected_without_arbitrary_position_cutoff(self):
        obs = np.zeros((5, 17))
        obs[:, 0] = 1.2
        obs[1, 3] = 101.0
        obs[2, 3] = np.nan
        obs[3, 3] = np.inf
        obs[4, 3] = -np.inf
        np.testing.assert_array_equal(
            policies.termination_fn_walker2d(obs, None, obs),
            [[False], [False], [True], [True], [True]],
        )

    def test_random_rollouts_match_native_fall_termination(self):
        env = gym.make("Walker2d-v5")
        falls = 0
        try:
            for seed in range(8):
                obs, _ = env.reset(seed=seed)
                env.action_space.seed(seed)
                while True:
                    action = env.action_space.sample()
                    next_obs, _, terminated, truncated, _ = env.step(action)
                    predicted = policies.termination_fn_walker2d(
                        obs[None, :], action[None, :], next_obs[None, :]
                    )
                    self.assertEqual(bool(predicted[0, 0]), terminated)
                    if terminated or truncated:
                        falls += int(terminated)
                        break
                    obs = next_obs
        finally:
            env.close()
        self.assertGreater(falls, 0)

    def test_time_limit_is_not_a_model_terminal(self):
        env = gym.make("Walker2d-v5", max_episode_steps=1)
        try:
            obs, reset_info = env.reset(seed=0)
            action = np.zeros(env.action_space.shape)
            next_obs, reward, terminated, truncated, final_info = env.step(action)
            self.assertFalse(terminated)
            self.assertTrue(truncated)
            self.assertFalse(policies.termination_fn_walker2d(
                obs[None, :], action[None, :], next_obs[None, :]
            )[0, 0])
            self.assertEqual(
                evaluation.episode_performance(
                    "Walker2d-v5", env, reward, 1, False, reset_info, final_info
                ),
                final_info["x_position"] - reset_info["x_position"],
            )
        finally:
            env.close()

    def test_h1_dynamics_use_walker_predicate_in_both_requested_modes(self):
        for mode in ("direct", "recursive"):
            with self.subTest(mode=mode):
                dynamics = policies.build_dynamics(
                    obs_dim=17, action_dim=6, task="Walker2d-v5",
                    args=SimpleNamespace(device="cpu"), hidden_dims=[4] * 4,
                    penalty_coef=0.5, chunk_length=1, dynamics_chunk_mode=mode,
                )
                self.assertIs(type(dynamics), EnsembleDynamics)
                self.assertIs(dynamics.terminal_fn, policies.termination_fn_walker2d)

    def test_existing_task_termination_selection_is_unchanged(self):
        for task, expected in (
            ("HalfCheetah-v5", termination_fn_halfcheetah),
            ("Reacher-v5", policies.termination_fn_never),
            ("Lift", policies.termination_fn_never),
            ("Can", policies.termination_fn_never),
        ):
            with self.subTest(task=task):
                dynamics = policies.build_dynamics(
                    obs_dim=17, action_dim=6, task=task,
                    args=SimpleNamespace(device="cpu"), hidden_dims=[4] * 4,
                    penalty_coef=0.5,
                )
                self.assertIs(dynamics.terminal_fn, expected)


if __name__ == "__main__":
    unittest.main()

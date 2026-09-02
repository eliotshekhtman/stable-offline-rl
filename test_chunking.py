import unittest

import gymnasium as gym
import numpy as np

import chunking
from rollout import DATASET_KEYS


class CountingEnv(gym.Env):
    def __init__(self, terminate_at: int = 10):
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32)
        self.terminate_at = terminate_at
        self.state = 0.0
        self.steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = 0.0
        self.steps = 0
        return np.asarray([self.state], dtype=np.float32), {}

    def step(self, action):
        self.state += float(action[0])
        self.steps += 1
        observation = np.asarray([self.state], dtype=np.float32)
        return observation, self.state, self.steps == self.terminate_at, False, {"steps": self.steps}


class ActionChunkDatasetTests(unittest.TestCase):
    def setUp(self):
        observations = np.arange(7, dtype=np.float32).reshape(-1, 1)
        self.dataset = {
            "observations": observations,
            "actions": np.arange(7, dtype=np.float32).reshape(-1, 1),
            "next_observations": observations + 1,
            "rewards": np.arange(1, 8, dtype=np.float32),
            "terminals": np.asarray([False, False, False, True, False, False, True]),
            "timeouts": np.zeros(7, dtype=bool),
            "episode_ids": np.asarray([10, 10, 10, 10, 20, 20, 20]),
        }

    def test_length_one_is_identity(self):
        chunk_dataset = chunking.make_action_chunk_dataset(self.dataset, 1, 0.99)
        self.assertIs(chunk_dataset, self.dataset)
        for key in DATASET_KEYS:
            np.testing.assert_array_equal(chunk_dataset[key], self.dataset[key])

    def test_length_one_preserves_legacy_reward_coercion_for_noncanonical_input(self):
        dataset = {key: value.copy() for key, value in self.dataset.items()}
        dataset["rewards"] = dataset["rewards"].astype(np.float64)

        chunk_dataset = chunking.make_action_chunk_dataset(dataset, 1, 0.99)

        self.assertIsNot(chunk_dataset, dataset)
        self.assertEqual(chunk_dataset["rewards"].dtype, np.float32)
        np.testing.assert_array_equal(chunk_dataset["rewards"], dataset["rewards"])

    def test_stride_one_windows_stay_within_episodes(self):
        chunk_dataset = chunking.make_action_chunk_dataset(self.dataset, 2, 0.99)
        np.testing.assert_array_equal(chunk_dataset["episode_ids"], [10, 10, 10, 20, 20])
        np.testing.assert_array_equal(chunk_dataset["actions"][:, 0], [0, 1, 2, 4, 5])
        np.testing.assert_allclose(
            chunk_dataset["rewards"],
            [1 + 0.99 * 2, 2 + 0.99 * 3, 3 + 0.99 * 4, 5 + 0.99 * 6, 6 + 0.99 * 7],
        )
        np.testing.assert_array_equal(
            chunk_dataset["next_observations"],
            self.dataset["next_observations"][[1, 2, 3, 5, 6]],
        )

    def test_noncontiguous_episode_is_rejected(self):
        dataset = {key: value.copy() for key, value in self.dataset.items()}
        dataset["episode_ids"] = np.asarray([10, 10, 20, 20, 10, 10, 10])
        with self.assertRaisesRegex(ValueError, "multiple noncontiguous blocks"):
            chunking.make_action_chunk_dataset(dataset, 2, 0.99)


class ActionChunkExecutionTests(unittest.TestCase):
    def test_wrapper_matches_manual_open_loop_steps(self):
        actions = np.asarray([[0.2], [-0.1], [0.4]], dtype=np.float32)
        manual_env = CountingEnv()
        wrapped_env = chunking.ActionChunkWrapper(CountingEnv(), len(actions))
        manual_env.reset()
        wrapped_env.reset()

        rewards = []
        observations = []
        for action in actions:
            observation, reward, terminated, truncated, _ = manual_env.step(action)
            rewards.append(reward)
            observations.append(observation)

        chunk_observation, chunk_reward, chunk_terminated, chunk_truncated, info = wrapped_env.step(
            actions.reshape(-1)
        )
        np.testing.assert_allclose(chunk_observation, observation)
        self.assertAlmostEqual(chunk_reward, sum(rewards))
        self.assertEqual((chunk_terminated, chunk_truncated), (terminated, truncated))
        np.testing.assert_allclose(info["primitive_actions"], actions)
        np.testing.assert_allclose(info["primitive_rewards"], rewards)
        np.testing.assert_allclose(info["primitive_next_observations"], observations)

    def test_execution_stops_at_primitive_termination(self):
        env = chunking.ActionChunkWrapper(CountingEnv(terminate_at=2), chunk_length=4)
        env.reset()
        observation, reward, terminated, truncated, info = env.step(
            np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        )
        np.testing.assert_allclose(observation, [0.3])
        self.assertAlmostEqual(reward, 0.4, places=6)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["primitive_steps"], 2)
        np.testing.assert_allclose(info["primitive_actions"].reshape(-1), [0.1, 0.2])


if __name__ == "__main__":
    unittest.main()

import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn

from chunked_dynamics import (
    DIRECT_DYNAMICS_MODE,
    DYNAMICS_CHUNK_MODES,
    RECURSIVE_DYNAMICS_MODE,
    RecursiveChunkDynamics,
    resolve_dynamics_chunk_mode,
)
from offlinerlkit.dynamics import EnsembleDynamics
from offlinerlkit.utils.scaler import StandardScaler


class DeterministicEnsembleModel(nn.Module):
    """Two primitive models with distinct, easily checked state deltas."""

    def __init__(self):
        super().__init__()
        self.num_ensemble = 2
        self.num_elites = 2
        self.device = torch.device("cpu")
        self.register_buffer("elites", torch.tensor([0, 1]))

    def random_elite_idxs(self, batch_size):
        return np.zeros(batch_size, dtype=np.int64)

    def forward(self, model_input):
        model_input = torch.as_tensor(model_input, dtype=torch.float32)
        if model_input.ndim == 2:
            model_input = model_input[None].expand(self.num_ensemble, -1, -1)
        if model_input.shape[0] != self.num_ensemble:
            raise ValueError("expected a complete ensemble axis")

        actions = model_input[..., 1]
        ensemble_offsets = torch.arange(self.num_ensemble, dtype=torch.float32)[:, None] * 2.0
        mean = torch.zeros((*model_input.shape[:-1], 2), dtype=torch.float32)
        mean[..., 0] = actions + ensemble_offsets
        mean[..., 1] = 10.0 * actions
        return mean, torch.full_like(mean, -100.0)


def terminal_at_three(_observations, _actions, next_observations):
    return (next_observations[:, :1] >= 3.0)


def never_terminal(observations, _actions, _next_observations):
    return np.zeros((len(observations), 1), dtype=bool)


def make_dynamics(
    chunk_length=2,
    discount=0.5,
    penalty_coef=0.0,
    uncertainty_mode="aleatoric",
    terminal_fn=never_terminal,
):
    return RecursiveChunkDynamics(
        DeterministicEnsembleModel(),
        object(),
        StandardScaler(mu=np.zeros((1, 2)), std=np.ones((1, 2))),
        terminal_fn,
        chunk_length=chunk_length,
        primitive_action_dim=1,
        discount=discount,
        penalty_coef=penalty_coef,
        uncertainty_mode=uncertainty_mode,
    )


class DynamicsModeTests(unittest.TestCase):
    def test_public_modes_and_one_step_canonicalization(self):
        self.assertEqual(DYNAMICS_CHUNK_MODES, (DIRECT_DYNAMICS_MODE, RECURSIVE_DYNAMICS_MODE))
        self.assertEqual(resolve_dynamics_chunk_mode(RECURSIVE_DYNAMICS_MODE, 1), "direct")
        self.assertEqual(resolve_dynamics_chunk_mode(DIRECT_DYNAMICS_MODE, 4), "direct")
        self.assertEqual(resolve_dynamics_chunk_mode(RECURSIVE_DYNAMICS_MODE, 4), "recursive")

    def test_invalid_modes_and_lengths_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "dynamics chunk mode"):
            resolve_dynamics_chunk_mode("rolled", 4)
        for invalid_length in (0, -1, 1.5, True):
            with self.subTest(invalid_length=invalid_length):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    resolve_dynamics_chunk_mode("direct", invalid_length)

    def test_h_one_step_and_sampling_delegate_to_ensemble_dynamics(self):
        dynamics = make_dynamics(chunk_length=1)
        observations = np.zeros((2, 1), dtype=np.float32)
        actions = np.zeros((2, 1), dtype=np.float32)
        sentinel_step = object()
        with patch.object(EnsembleDynamics, "step", return_value=sentinel_step) as primitive_step:
            self.assertIs(dynamics.step(observations, actions), sentinel_step)
            primitive_step.assert_called_once_with(observations, actions)

        tensor_observations = torch.zeros((2, 1))
        tensor_actions = torch.zeros((2, 1))
        sentinel_samples = object()
        with patch.object(
            EnsembleDynamics, "sample_next_obss", return_value=sentinel_samples
        ) as primitive_sample:
            self.assertIs(
                dynamics.sample_next_obss(tensor_observations, tensor_actions, 3),
                sentinel_samples,
            )
            primitive_sample.assert_called_once_with(tensor_observations, tensor_actions, 3)


class RecursiveMacroStepTests(unittest.TestCase):
    def test_step_stops_rows_independently_and_aggregates_mopo_accounting(self):
        dynamics = make_dynamics(
            chunk_length=3,
            discount=0.5,
            penalty_coef=0.5,
            uncertainty_mode="pairwise-diff",
            terminal_fn=terminal_at_three,
        )
        observations = np.zeros((2, 1), dtype=np.float32)
        action_chunks = np.asarray([[1.0, 2.0, 9.0], [3.0, 9.0, 9.0]], dtype=np.float32)

        with patch("numpy.random.normal", return_value=0.0):
            next_observations, rewards, terminals, info = dynamics.step(
                observations, action_chunks
            )

        np.testing.assert_allclose(next_observations, [[3.0], [3.0]])
        np.testing.assert_array_equal(terminals, [[True], [True]])
        np.testing.assert_allclose(info["raw_reward"], [[20.0], [30.0]])
        np.testing.assert_allclose(info["penalty"], [[1.5], [1.0]])
        np.testing.assert_allclose(rewards, info["raw_reward"] - 0.5 * info["penalty"])

    def test_invalid_macro_action_width_is_rejected(self):
        dynamics = make_dynamics(chunk_length=3)
        with self.assertRaisesRegex(ValueError, "action must have shape"):
            dynamics.step(np.zeros((2, 1)), np.zeros((2, 2)))


class RecursiveEnsembleSamplingTests(unittest.TestCase):
    def test_sampling_preserves_ensemble_identity_and_freezes_terminal_particles(self):
        dynamics = make_dynamics(
            chunk_length=2,
            terminal_fn=terminal_at_three,
        )
        observations = torch.zeros((1, 1))
        action_chunks = torch.ones((1, 2))

        with patch("torch.randn_like", side_effect=lambda tensor: torch.zeros_like(tensor)):
            endpoints = dynamics.sample_next_obss(observations, action_chunks, num_samples=3)

        self.assertEqual(tuple(endpoints.shape), (3, 2, 1, 1))
        torch.testing.assert_close(endpoints[:, 0], torch.full((3, 1, 1), 2.0))
        # Ensemble 1 reaches the terminal state 3 on its first primitive step.
        # It must not advance to 6 using the second action.
        torch.testing.assert_close(endpoints[:, 1], torch.full((3, 1, 1), 3.0))

    def test_deterministic_endpoint_averages_coherent_elite_rollouts(self):
        dynamics = make_dynamics(
            chunk_length=2,
            terminal_fn=terminal_at_three,
        )
        endpoint = dynamics.mean_next_obss(
            np.zeros((1, 1), dtype=np.float32),
            np.ones((1, 2), dtype=np.float32),
        )

        np.testing.assert_allclose(endpoint, [[2.5]])


if __name__ == "__main__":
    unittest.main()

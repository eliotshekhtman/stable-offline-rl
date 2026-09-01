"""Learned-dynamics support for recursively executing action chunks.

The policy and critics still operate on flattened action chunks.  This adapter
keeps the learned model primitive: it predicts one transition at a time and
recursively applies those predictions to each action in a chunk.
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from offlinerlkit.dynamics import EnsembleDynamics
from offlinerlkit.utils.scaler import StandardScaler


DIRECT_DYNAMICS_MODE = "direct"
RECURSIVE_DYNAMICS_MODE = "recursive"
DYNAMICS_CHUNK_MODES = (DIRECT_DYNAMICS_MODE, RECURSIVE_DYNAMICS_MODE)


def resolve_dynamics_chunk_mode(requested: str, chunk_length: int) -> str:
    """Validate a requested mode and canonicalize every one-step run to direct."""
    if isinstance(chunk_length, bool) or not isinstance(chunk_length, int) or chunk_length < 1:
        raise ValueError("chunk_length must be a positive integer")
    if requested not in DYNAMICS_CHUNK_MODES:
        raise ValueError(
            f"dynamics chunk mode must be one of {DYNAMICS_CHUNK_MODES}, got {requested!r}"
        )
    return DIRECT_DYNAMICS_MODE if chunk_length == 1 else requested


class RecursiveChunkDynamics(EnsembleDynamics):
    """Apply a primitive ``EnsembleDynamics`` model across an action chunk.

    Training, persistence, and primitive prediction parameters are inherited
    unchanged.  ``step`` exposes one macro transition for MOPO-style rollouts;
    ``sample_next_obss`` preserves a coherent ensemble identity throughout each
    chunk for MOBILE's disagreement calculation.
    """

    def __init__(
        self,
        model: nn.Module,
        optim: torch.optim.Optimizer,
        scaler: StandardScaler,
        terminal_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
        *,
        chunk_length: int,
        primitive_action_dim: int,
        discount: float,
        penalty_coef: float = 0.0,
        uncertainty_mode: str = "aleatoric",
    ) -> None:
        if isinstance(chunk_length, bool) or not isinstance(chunk_length, int) or chunk_length < 1:
            raise ValueError("chunk_length must be a positive integer")
        if (
            isinstance(primitive_action_dim, bool)
            or not isinstance(primitive_action_dim, int)
            or primitive_action_dim < 1
        ):
            raise ValueError("primitive_action_dim must be a positive integer")
        if not np.isfinite(discount) or not 0.0 <= discount <= 1.0:
            raise ValueError("discount must be finite and in [0, 1]")

        super().__init__(
            model,
            optim,
            scaler,
            terminal_fn,
            penalty_coef=penalty_coef,
            uncertainty_mode=uncertainty_mode,
        )
        self.chunk_length = chunk_length
        self.primitive_action_dim = primitive_action_dim
        self.discount = float(discount)

    @torch.no_grad()
    def step(
        self,
        obs: np.ndarray,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """Predict one macro transition by recursively taking primitive steps."""
        # The integration canonicalizes H=1 to ordinary EnsembleDynamics, but
        # retaining this exact delegation makes accidental construction benign.
        if self.chunk_length == 1:
            return super().step(obs, action)

        observations = np.asarray(obs)
        action_chunks = self._reshape_numpy_actions(action, len(observations))
        next_observations = observations.astype(np.float32, copy=True)
        rewards = np.zeros((len(observations), 1), dtype=np.float32)
        raw_rewards = np.zeros_like(rewards)
        penalties = np.zeros_like(rewards)
        terminals = np.zeros((len(observations), 1), dtype=bool)
        alive = np.ones(len(observations), dtype=bool)
        saw_penalty = False

        for primitive_index in range(self.chunk_length):
            active_indices = np.flatnonzero(alive)
            if not len(active_indices):
                break

            primitive_actions = action_chunks[active_indices, primitive_index]
            (
                primitive_next_observations,
                primitive_rewards,
                primitive_terminals,
                primitive_info,
            ) = super().step(next_observations[active_indices], primitive_actions)

            primitive_next_observations = np.asarray(primitive_next_observations, dtype=np.float32)
            primitive_rewards = self._as_column(
                primitive_rewards, len(active_indices), "primitive rewards", dtype=np.float32
            )
            primitive_terminals = self._as_column(
                primitive_terminals, len(active_indices), "primitive terminals", dtype=bool
            )
            primitive_raw_rewards = self._as_column(
                primitive_info.get("raw_reward", primitive_rewards),
                len(active_indices),
                "primitive raw rewards",
                dtype=np.float32,
            )
            weight = self.discount**primitive_index

            next_observations[active_indices] = primitive_next_observations
            rewards[active_indices] += weight * primitive_rewards
            raw_rewards[active_indices] += weight * primitive_raw_rewards

            if "penalty" in primitive_info:
                primitive_penalties = self._as_column(
                    primitive_info["penalty"],
                    len(active_indices),
                    "primitive penalties",
                    dtype=np.float32,
                )
                penalties[active_indices] += weight * primitive_penalties
                saw_penalty = True

            newly_terminal = primitive_terminals[:, 0]
            terminals[active_indices, 0] |= newly_terminal
            alive[active_indices[newly_terminal]] = False

        info = {"raw_reward": raw_rewards}
        if saw_penalty:
            info["penalty"] = penalties
        return next_observations, rewards, terminals, info

    @torch.no_grad()
    def sample_next_obss(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        num_samples: int,
    ) -> torch.Tensor:
        """Sample recursive endpoints with fixed model identity per particle."""
        if self.chunk_length == 1:
            return super().sample_next_obss(obs, action, num_samples)
        if isinstance(num_samples, bool) or not isinstance(num_samples, int) or num_samples < 1:
            raise ValueError("num_samples must be a positive integer")

        action_chunks = self._reshape_tensor_actions(action, len(obs))
        num_ensembles = int(self.model.num_ensemble)
        states = obs[None, None].expand(num_samples, num_ensembles, -1, -1).clone()
        alive = torch.ones((*states.shape[:-1], 1), dtype=torch.bool, device=states.device)

        for primitive_index in range(self.chunk_length):
            primitive_actions = action_chunks[:, primitive_index]
            mean, logvar = self._ensemble_moments(states, primitive_actions)
            samples = mean + torch.randn_like(logvar) * torch.sqrt(torch.exp(logvar))
            proposed_states = samples[..., :-1]
            primitive_terminals = self._particle_terminal_mask(
                states, primitive_actions, proposed_states
            )
            states = torch.where(alive, proposed_states, states)
            alive &= ~primitive_terminals

        elite_indices = self.model.elites.detach().to(device=states.device, dtype=torch.long)
        return states.index_select(1, elite_indices)

    @torch.no_grad()
    def mean_next_obss(
        self,
        obs: np.ndarray,
        action: np.ndarray,
    ) -> np.ndarray:
        """Return the elite-averaged deterministic recursive endpoint."""
        observations = np.asarray(obs, dtype=np.float32)
        action_chunks = self._reshape_numpy_actions(action, len(observations))
        device = self.model.device
        tensor_observations = torch.as_tensor(observations, device=device)
        tensor_actions = torch.as_tensor(action_chunks, device=device)
        num_ensembles = int(self.model.num_ensemble)
        states = tensor_observations[None, None].expand(1, num_ensembles, -1, -1).clone()
        alive = torch.ones((*states.shape[:-1], 1), dtype=torch.bool, device=device)

        for primitive_index in range(self.chunk_length):
            primitive_actions = tensor_actions[:, primitive_index]
            mean, _ = self._ensemble_moments(states, primitive_actions)
            proposed_states = mean[..., :-1]
            primitive_terminals = self._particle_terminal_mask(
                states, primitive_actions, proposed_states
            )
            states = torch.where(alive, proposed_states, states)
            alive &= ~primitive_terminals

        elite_indices = self.model.elites.detach().to(device=device, dtype=torch.long)
        elite_states = states[0].index_select(0, elite_indices)
        return elite_states.mean(dim=0).cpu().numpy()

    def _reshape_numpy_actions(self, action: np.ndarray, batch_size: int) -> np.ndarray:
        actions = np.asarray(action)
        expected_shape = (batch_size, self.chunk_length * self.primitive_action_dim)
        if actions.shape != expected_shape:
            raise ValueError(f"action must have shape {expected_shape}, got {actions.shape}")
        return actions.reshape(batch_size, self.chunk_length, self.primitive_action_dim)

    def _reshape_tensor_actions(self, action: torch.Tensor, batch_size: int) -> torch.Tensor:
        expected_shape = (batch_size, self.chunk_length * self.primitive_action_dim)
        if tuple(action.shape) != expected_shape:
            raise ValueError(f"action must have shape {expected_shape}, got {tuple(action.shape)}")
        return action.reshape(batch_size, self.chunk_length, self.primitive_action_dim)

    def _ensemble_moments(
        self,
        states: torch.Tensor,
        primitive_actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_samples, num_ensembles, batch_size, _ = states.shape
        expanded_actions = primitive_actions[None, None].expand(
            num_samples, num_ensembles, -1, -1
        )
        model_inputs = torch.cat((states, expanded_actions), dim=-1)
        model_inputs = model_inputs.permute(1, 0, 2, 3).reshape(
            num_ensembles, num_samples * batch_size, -1
        )
        model_inputs = self.scaler.transform_tensor(model_inputs)
        mean, logvar = self.model(model_inputs)

        expected_prefix = (num_ensembles, num_samples * batch_size)
        if tuple(mean.shape[:2]) != expected_prefix or mean.shape != logvar.shape:
            raise ValueError(
                "ensemble model returned incompatible moments: "
                f"mean={tuple(mean.shape)}, logvar={tuple(logvar.shape)}"
            )

        output_dim = mean.shape[-1]
        mean = mean.reshape(num_ensembles, num_samples, batch_size, output_dim).permute(
            1, 0, 2, 3
        )
        logvar = logvar.reshape(num_ensembles, num_samples, batch_size, output_dim).permute(
            1, 0, 2, 3
        )
        mean[..., :-1] += states
        return mean, logvar

    def _particle_terminal_mask(
        self,
        states: torch.Tensor,
        primitive_actions: torch.Tensor,
        proposed_states: torch.Tensor,
    ) -> torch.Tensor:
        num_samples, num_ensembles, batch_size, obs_dim = states.shape
        expanded_actions = primitive_actions[None, None].expand(
            num_samples, num_ensembles, -1, -1
        )
        flat_size = num_samples * num_ensembles * batch_size
        terminals = self.terminal_fn(
            states.detach().cpu().numpy().reshape(flat_size, obs_dim),
            expanded_actions.detach().cpu().numpy().reshape(flat_size, self.primitive_action_dim),
            proposed_states.detach().cpu().numpy().reshape(flat_size, obs_dim),
        )
        terminals = np.asarray(terminals, dtype=bool)
        if terminals.size != flat_size:
            raise ValueError(
                f"terminal_fn returned {terminals.size} values for {flat_size} transitions"
            )
        return torch.as_tensor(
            terminals.reshape(num_samples, num_ensembles, batch_size, 1),
            device=states.device,
        )

    @staticmethod
    def _as_column(value, batch_size: int, name: str, dtype) -> np.ndarray:
        array = np.asarray(value, dtype=dtype)
        if array.size != batch_size:
            raise ValueError(f"{name} must contain {batch_size} values, got shape {array.shape}")
        return array.reshape(batch_size, 1)

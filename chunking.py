# Tasks:
# - Convert ordered primitive transitions into fixed-length macro-transitions.
# - Expose flattened action chunks through a Gymnasium wrapper.
# - Preserve primitive-step traces when chunks execute open-loop.

import gymnasium as gym
import numpy as np

from rollout import DATASET_KEYS, validate_dataset


def make_action_chunk_dataset(
    dataset: dict[str, np.ndarray],
    chunk_length: int,
    discount: float,
) -> dict[str, np.ndarray]:
    """Convert ordered primitive transitions into stride-one action chunks."""
    validate_dataset(dataset)
    chunk_length = _validate_positive_integer(chunk_length, "chunk_length")
    discount = float(discount)
    if not np.isfinite(discount):
        raise ValueError("discount must be finite.")

    window_offsets = np.arange(chunk_length)
    discount_powers = discount**window_offsets
    chunks = {key: [] for key in DATASET_KEYS}

    for start, stop in _ordered_episode_slices(dataset):
        chunk_count = stop - start - chunk_length + 1
        if chunk_count <= 0:
            continue

        starts = np.arange(start, start + chunk_count)
        window_indices = starts[:, None] + window_offsets
        final_indices = starts + chunk_length - 1
        chunks["observations"].append(dataset["observations"][starts])
        chunks["actions"].append(dataset["actions"][window_indices].reshape(chunk_count, -1))
        chunks["next_observations"].append(dataset["next_observations"][final_indices])
        chunks["rewards"].append(dataset["rewards"][window_indices] @ discount_powers)
        chunks["terminals"].append(np.any(dataset["terminals"][window_indices], axis=1))
        chunks["timeouts"].append(np.any(dataset["timeouts"][window_indices], axis=1))
        chunks["episode_ids"].append(dataset["episode_ids"][starts])

    if not chunks["rewards"]:
        raise ValueError(f"No episode contains {chunk_length} consecutive transitions.")

    chunk_dataset = {key: np.concatenate(values, axis=0) for key, values in chunks.items()}
    chunk_dataset["rewards"] = chunk_dataset["rewards"].astype(np.float32, copy=False)
    return chunk_dataset


def execute_action_chunk(
    env: gym.Env,
    action_chunk: np.ndarray,
    chunk_length: int,
    max_primitive_steps: int | None = None,
) -> tuple[object, float, bool, bool, dict]:
    """Execute a flattened action chunk open-loop in a Gymnasium environment."""
    chunk_length = _validate_positive_integer(chunk_length, "chunk_length")
    if not isinstance(env.action_space, gym.spaces.Box):
        raise TypeError("Action chunks require a Box action space.")

    expected_size = chunk_length * int(np.prod(env.action_space.shape, dtype=int))
    flat_chunk = np.asarray(action_chunk, dtype=env.action_space.dtype).reshape(-1)
    if flat_chunk.size != expected_size:
        raise ValueError(
            f"Expected an action chunk with {expected_size} values "
            f"({chunk_length} actions), got {flat_chunk.size}."
        )
    primitive_actions = flat_chunk.reshape((chunk_length, *env.action_space.shape))
    step_count = chunk_length
    if max_primitive_steps is not None:
        step_count = min(
            chunk_length,
            _validate_positive_integer(max_primitive_steps, "max_primitive_steps"),
        )

    executed_actions = []
    rewards = []
    next_observations = []
    total_reward = 0.0
    terminated = truncated = False
    last_env_info = {}
    final_observation = None

    for action in primitive_actions[:step_count]:
        final_observation, reward, terminated, truncated, last_env_info = env.step(action)
        executed_actions.append(action.copy())
        rewards.append(reward)
        next_observations.append(np.asarray(final_observation).copy())
        total_reward += float(reward)
        if terminated or truncated:
            break

    info = dict(last_env_info)
    info.update(
        primitive_actions=np.asarray(executed_actions, dtype=env.action_space.dtype),
        primitive_rewards=np.asarray(rewards, dtype=np.float32),
        primitive_next_observations=np.asarray(next_observations),
        primitive_steps=len(executed_actions),
    )
    return final_observation, total_reward, bool(terminated), bool(truncated), info


class ActionChunkWrapper(gym.Wrapper):
    """Expose ``chunk_length`` primitive Box actions as one flat action."""

    def __init__(self, env: gym.Env, chunk_length: int):
        super().__init__(env)
        self.chunk_length = _validate_positive_integer(chunk_length, "chunk_length")
        if not isinstance(env.action_space, gym.spaces.Box):
            raise TypeError("ActionChunkWrapper requires a Box action space.")

        low = np.tile(env.action_space.low.reshape(-1), self.chunk_length)
        high = np.tile(env.action_space.high.reshape(-1), self.chunk_length)
        self.action_space = gym.spaces.Box(low=low, high=high, dtype=env.action_space.dtype)

    @property
    def spec(self):
        """Preserve both Gymnasium EnvSpec and robomimic's lightweight spec."""
        return self.env.spec

    def step(self, action: np.ndarray):
        return execute_action_chunk(self.env, action, self.chunk_length)


def _ordered_episode_slices(dataset: dict[str, np.ndarray]) -> list[tuple[int, int]]:
    episode_ids = np.asarray(dataset["episode_ids"])
    if episode_ids.ndim != 1:
        raise ValueError("episode_ids must be one-dimensional.")
    if dataset["rewards"].ndim != 1:
        raise ValueError("rewards must be one-dimensional.")
    if dataset["terminals"].ndim != 1 or dataset["timeouts"].ndim != 1:
        raise ValueError("terminals and timeouts must be one-dimensional.")
    if dataset["observations"].shape != dataset["next_observations"].shape:
        raise ValueError("observations and next_observations must have matching shapes.")
    if not len(episode_ids):
        return []

    boundaries = np.flatnonzero(episode_ids[1:] != episode_ids[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    stops = np.concatenate((boundaries, [len(episode_ids)]))
    slices = []
    seen_episode_ids = set()

    for start, stop in zip(starts, stops):
        episode_id = episode_ids[start].item()
        if episode_id in seen_episode_ids:
            raise ValueError(f"Episode ID {episode_id!r} appears in multiple noncontiguous blocks.")
        seen_episode_ids.add(episode_id)
        if stop - start > 1 and not np.allclose(
            dataset["next_observations"][start : stop - 1],
            dataset["observations"][start + 1 : stop],
        ):
            raise ValueError(f"Episode ID {episode_id!r} contains non-adjacent or out-of-order transitions.")
        slices.append((int(start), int(stop)))

    return slices


def _validate_positive_integer(value: int, name: str) -> int:
    if (
        not isinstance(value, (int, np.integer))
        or isinstance(value, (bool, np.bool_))
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)

# Tasks:
# - Generate offline transition datasets by rolling out Gymnasium environments.
# - Implement clean expert, noise-injected expert, and random-action collection.
# - Save/load this project's canonical .npz transition schema and JSON metadata.
# - Preserve Gymnasium terminated vs truncated signals as terminals vs timeouts.

import os
import tempfile
from pathlib import Path
from typing import Callable

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from task_support import GENERATED_TASKS, require_supported_task


DATASET_KEYS = (
    "observations",
    "actions",
    "next_observations",
    "rewards",
    "terminals",
    "timeouts",
    "episode_ids",
)
MAX_SEED = np.iinfo(np.int32).max


def load_expert_policy(env_name: str, policy_path: str):
    """Load the SB3 expert policy convention used by this project."""
    if env_name in GENERATED_TASKS:
        return SAC.load(policy_path)
    raise ValueError(
        f"Unsupported generated-data task {env_name!r}; supported tasks are "
        "Reacher-v5 and HalfCheetah-v5."
    )


def collect_traj(
    env: gym.Env,
    action_fn: Callable[[np.ndarray], np.ndarray],
    max_timesteps: int,
    episode_id: int,
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    """Collect one Gymnasium trajectory as one-step transition arrays."""
    obs, _ = env.reset(seed=seed)
    transitions = {
        "observations": [],
        "actions": [],
        "next_observations": [],
        "rewards": [],
        "terminals": [],
        "timeouts": [],
        "episode_ids": [],
    }

    for _ in range(max_timesteps):
        action = np.asarray(action_fn(obs), dtype=np.float32)
        obs_before_step = np.asarray(obs, dtype=np.float32).copy()
        next_obs, reward, terminated, truncated, _ = env.step(action)

        transitions["observations"].append(obs_before_step)
        transitions["actions"].append(action.copy())
        transitions["next_observations"].append(np.asarray(next_obs, dtype=np.float32).copy())
        transitions["rewards"].append(np.float32(reward))
        transitions["terminals"].append(bool(terminated))
        transitions["timeouts"].append(bool(truncated))
        transitions["episode_ids"].append(episode_id)

        obs = next_obs
        if terminated or truncated:
            break
    else:
        # max_timesteps is a collector-imposed truncation even when it is
        # shorter than the environment's own TimeLimit.
        if transitions["timeouts"]:
            transitions["timeouts"][-1] = True

    return {
        "observations": np.asarray(transitions["observations"], dtype=np.float32),
        "actions": np.asarray(transitions["actions"], dtype=np.float32),
        "next_observations": np.asarray(transitions["next_observations"], dtype=np.float32),
        "rewards": np.asarray(transitions["rewards"], dtype=np.float32),
        "terminals": np.asarray(transitions["terminals"], dtype=bool),
        "timeouts": np.asarray(transitions["timeouts"], dtype=bool),
        "episode_ids": np.asarray(transitions["episode_ids"], dtype=np.int64),
    }


def collect_expert(
    env_name: str,
    policy_path: str,
    num_trajectories: int,
    max_timesteps: int,
    noise_scale: float = 0.0,
    deterministic: bool = True,
    rng: np.random.Generator | None = None,
    episode_id_start: int = 0,
) -> dict[str, np.ndarray]:
    """Collect an exact number of complete expert trajectories."""
    require_supported_task(env_name, "generated")
    rng = np.random.default_rng() if rng is None else rng

    def make_action_fn(env: gym.Env) -> Callable[[np.ndarray], np.ndarray]:
        policy = load_expert_policy(env_name, policy_path)
        action_dim = int(np.prod(env.action_space.shape))

        def action_fn(obs: np.ndarray) -> np.ndarray:
            action, _ = policy.predict(obs, deterministic=deterministic)
            action = np.asarray(action, dtype=np.float32)
            if noise_scale > 0.0:
                noise = rng.normal(
                    loc=0.0,
                    scale=noise_scale / np.sqrt(action_dim),
                    size=env.action_space.shape,
                ).astype(np.float32)
                action = action + noise
            return np.clip(action, env.action_space.low, env.action_space.high).astype(np.float32)

        return action_fn

    return _collect_source(
        env_name=env_name,
        make_action_fn=make_action_fn,
        num_trajectories=num_trajectories,
        max_timesteps=max_timesteps,
        rng=rng,
        episode_id_start=episode_id_start,
    )


def collect_suboptimal(
    env_name: str,
    policy_path: str,
    num_trajectories: int,
    max_timesteps: int,
    noise_scale: float = 0.0,
    deterministic: bool = True,
    rng: np.random.Generator | None = None,
    episode_id_start: int = 0,
) -> dict[str, np.ndarray]:
    """Collect an exact number of complete random-action trajectories."""
    require_supported_task(env_name, "generated")
    rng = np.random.default_rng() if rng is None else rng

    def make_action_fn(env: gym.Env) -> Callable[[np.ndarray], np.ndarray]:
        env.action_space.seed(int(rng.integers(0, MAX_SEED)))

        def action_fn(_: np.ndarray) -> np.ndarray:
            return np.asarray(env.action_space.sample(), dtype=np.float32)

        return action_fn

    return _collect_source(
        env_name=env_name,
        make_action_fn=make_action_fn,
        num_trajectories=num_trajectories,
        max_timesteps=max_timesteps,
        rng=rng,
        episode_id_start=episode_id_start,
    )


def collect_dataset(
    env_name: str,
    policy_path: str,
    max_timesteps: int = 300,
    num_samples: int = 10000,
    noise_scale: float = 0.0,
    prop_clean_expert: float = 1.0,
    prop_noisy_expert: float = 0.0,
    deterministic: bool = True,
    seed: int | None = None,
) -> tuple[dict[str, np.ndarray], dict]:
    """Collect an ordered offline-RL dataset containing complete episodes.

    The two expert proportions determine clean and noisy expert trajectory
    allocations; the remainder is random data. Whole trajectories are added
    until the dataset contains at least num_samples transitions.
    """
    require_supported_task(env_name, "generated")
    _validate_collection_args(
        max_timesteps=max_timesteps,
        num_samples=num_samples,
        noise_scale=noise_scale,
        prop_clean_expert=prop_clean_expert,
        prop_noisy_expert=prop_noisy_expert,
    )

    rng = np.random.default_rng(seed)
    proportions = np.asarray(
        [prop_clean_expert, prop_noisy_expert, 1.0 - prop_clean_expert - prop_noisy_expert]
    )
    target_num_trajectories = max(1, int(np.ceil(num_samples / max_timesteps)))
    datasets = []
    next_episode_id = 0
    trajectory_counts = np.zeros(3, dtype=np.int64)
    transition_counts = np.zeros(3, dtype=np.int64)

    while transition_counts.sum() < num_samples:
        target_counts = np.asarray(
            _allocate_trajectory_counts(target_num_trajectories, proportions), dtype=np.int64
        )
        for source, target_count in enumerate(target_counts):
            count = int(target_count - trajectory_counts[source])
            if count == 0:
                continue
            if source < 2:
                component = collect_expert(
                    env_name=env_name,
                    policy_path=policy_path,
                    num_trajectories=count,
                    max_timesteps=max_timesteps,
                    noise_scale=0.0 if source == 0 else noise_scale,
                    deterministic=deterministic,
                    rng=rng,
                    episode_id_start=next_episode_id,
                )
            else:
                component = collect_suboptimal(
                    env_name=env_name,
                    policy_path=policy_path,
                    num_trajectories=count,
                    max_timesteps=max_timesteps,
                    noise_scale=noise_scale,
                    deterministic=deterministic,
                    rng=rng,
                    episode_id_start=next_episode_id,
                )
            datasets.append(component)
            trajectory_counts[source] += count
            transition_counts[source] += len(component["rewards"])
            next_episode_id += count

        if transition_counts.sum() < num_samples:
            mean_length = transition_counts.sum() / trajectory_counts.sum()
            shortfall = num_samples - transition_counts.sum()
            target_num_trajectories += max(1, int(np.ceil(shortfall / mean_length)))

    dataset = _concat_datasets(datasets)
    metadata = make_metadata(
        env_name=env_name,
        policy_path=policy_path,
        max_timesteps=max_timesteps,
        num_samples=num_samples,
        noise_scale=noise_scale,
        prop_clean_expert=prop_clean_expert,
        prop_noisy_expert=prop_noisy_expert,
        deterministic=deterministic,
        seed=seed,
        trajectory_counts=trajectory_counts,
        transition_counts=transition_counts,
    )
    return dataset, metadata


def make_metadata(
    env_name: str,
    policy_path: str,
    max_timesteps: int,
    num_samples: int,
    noise_scale: float,
    prop_clean_expert: float,
    prop_noisy_expert: float,
    deterministic: bool,
    seed: int | None,
    trajectory_counts: np.ndarray,
    transition_counts: np.ndarray,
) -> dict:
    prop_random = 1.0 - prop_clean_expert - prop_noisy_expert
    num_trajectories = int(trajectory_counts.sum())
    num_transitions = int(transition_counts.sum())
    return {
        "env_name": env_name,
        "policy_path": str(policy_path),
        "max_timesteps": max_timesteps,
        "requested_num_samples": num_samples,
        "requested_prop_clean_expert": prop_clean_expert,
        "requested_prop_noisy_expert": prop_noisy_expert,
        "requested_prop_random": prop_random,
        "num_trajectories": num_trajectories,
        "num_clean_expert_trajectories": int(trajectory_counts[0]),
        "num_noisy_expert_trajectories": int(trajectory_counts[1]),
        "num_random_trajectories": int(trajectory_counts[2]),
        "actual_prop_clean_expert_trajectories": float(trajectory_counts[0] / num_trajectories),
        "actual_prop_noisy_expert_trajectories": float(trajectory_counts[1] / num_trajectories),
        "actual_prop_random_trajectories": float(trajectory_counts[2] / num_trajectories),
        "num_transitions": num_transitions,
        "num_clean_expert_transitions": int(transition_counts[0]),
        "num_noisy_expert_transitions": int(transition_counts[1]),
        "num_random_transitions": int(transition_counts[2]),
        "actual_prop_clean_expert_transitions": float(transition_counts[0] / num_transitions),
        "actual_prop_noisy_expert_transitions": float(transition_counts[1] / num_transitions),
        "actual_prop_random_transitions": float(transition_counts[2] / num_transitions),
        "noise_scale": noise_scale,
        "deterministic": deterministic,
        "seed": seed,
    }


def save_dataset(dataset: dict[str, np.ndarray], dataset_path: str | Path) -> None:
    """Save dataset arrays as compressed NumPy data."""
    dataset_path = Path(dataset_path)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    validate_dataset(dataset)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{dataset_path.name}.",
            suffix=".tmp",
            dir=dataset_path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            np.savez_compressed(temporary, **dataset)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, dataset_path)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def load_dataset(dataset_path: str | Path) -> dict[str, np.ndarray]:
    """Load the transition arrays saved by save_dataset."""
    with np.load(dataset_path) as data:
        dataset = {key: data[key] for key in DATASET_KEYS}
    validate_dataset(dataset)
    return dataset


def split_dataset(
    dataset: dict[str, np.ndarray],
    test_fraction: float,
    seed: int | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Split whole episodes while preserving transition order within each split."""
    validate_dataset(dataset)
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1.")

    rng = np.random.default_rng(seed)
    unique_episode_ids = np.unique(dataset["episode_ids"])
    if len(unique_episode_ids) < 2:
        raise ValueError("Episode splitting requires at least two episodes.")
    unique_episode_ids = rng.permutation(unique_episode_ids)
    requested_test_size = max(1, int(round(len(unique_episode_ids) * test_fraction)))
    test_size = min(len(unique_episode_ids) - 1, requested_test_size)
    test_episode_ids = unique_episode_ids[:test_size]
    test_mask = np.isin(dataset["episode_ids"], test_episode_ids)
    indices = np.arange(len(dataset["episode_ids"]))
    train_indices, test_indices = indices[~test_mask], indices[test_mask]

    return (
        {key: dataset[key][train_indices] for key in DATASET_KEYS},
        {key: dataset[key][test_indices] for key in DATASET_KEYS},
    )


def _collect_source(
    env_name: str,
    make_action_fn: Callable[[gym.Env], Callable[[np.ndarray], np.ndarray]],
    num_trajectories: int,
    max_timesteps: int,
    rng: np.random.Generator,
    episode_id_start: int = 0,
) -> dict[str, np.ndarray]:
    env = gym.make(env_name)
    try:
        action_fn = make_action_fn(env)
        datasets = []
        for trajectory in range(num_trajectories):
            traj = collect_traj(
                env,
                action_fn,
                max_timesteps=max_timesteps,
                episode_id=episode_id_start + trajectory,
                seed=int(rng.integers(0, MAX_SEED)),
            )
            if len(traj["rewards"]) == 0:
                raise RuntimeError("Collected an empty trajectory; check the environment and action function.")
            datasets.append(traj)

        dataset = _concat_datasets(datasets)
        return dataset
    finally:
        env.close()


def _concat_datasets(datasets: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not datasets:
        raise ValueError("Cannot concatenate an empty dataset list.")
    return {key: np.concatenate([dataset[key] for dataset in datasets], axis=0) for key in DATASET_KEYS}


def validate_dataset(dataset: dict[str, np.ndarray]) -> None:
    extra = [key for key in dataset if key not in DATASET_KEYS]
    if extra:
        raise ValueError(f"Dataset has unexpected keys: {extra}")
    missing = [key for key in DATASET_KEYS if key not in dataset]
    if missing:
        raise ValueError(f"Dataset is missing required keys: {missing}")

    scalar_keys = [key for key in DATASET_KEYS if np.asarray(dataset[key]).ndim == 0]
    if scalar_keys:
        raise ValueError(f"Dataset arrays must have a transition axis: {scalar_keys}")
    lengths = {key: len(dataset[key]) for key in DATASET_KEYS}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Dataset arrays have inconsistent lengths: {lengths}")


def _validate_collection_args(
    max_timesteps: int,
    num_samples: int,
    noise_scale: float,
    prop_clean_expert: float,
    prop_noisy_expert: float,
) -> None:
    if max_timesteps <= 0:
        raise ValueError("max_timesteps must be positive.")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")
    if noise_scale < 0.0:
        raise ValueError("noise_scale must be nonnegative.")
    if not 0.0 <= prop_clean_expert <= 1.0:
        raise ValueError("prop_clean_expert must be between 0 and 1.")
    if not 0.0 <= prop_noisy_expert <= 1.0:
        raise ValueError("prop_noisy_expert must be between 0 and 1.")
    if prop_clean_expert + prop_noisy_expert > 1.0:
        raise ValueError("clean and noisy expert proportions cannot sum above 1.")


def _allocate_trajectory_counts(
    num_trajectories: int,
    proportions: np.ndarray,
) -> tuple[int, int, int]:
    """Return a deterministic integer allocation that stays close to each proportion."""
    counts = np.zeros(3, dtype=np.int64)
    for total in range(1, num_trajectories + 1):
        deficits = total * proportions - counts
        counts[int(np.argmax(deficits))] += 1
    return tuple(int(count) for count in counts)

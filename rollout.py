import json
from pathlib import Path
from typing import Callable

import gymnasium as gym
import numpy as np
from sb3_contrib import TQC
from stable_baselines3 import PPO, SAC


DATASET_KEYS = (
    "observations",
    "actions",
    "next_observations",
    "rewards",
    "terminals",
    "timeouts",
)


def load_expert_policy(env_name: str, policy_path: str):
    """Load the SB3 expert policy convention used by this project."""
    if env_name == "Humanoid-v5":
        return TQC.load(policy_path)
    if env_name == "Swimmer-v5":
        return PPO.load(policy_path)
    if any(env_name.endswith(f"-v{version}") for version in (2, 3, 4, 5)):
        return SAC.load(policy_path)
    raise ValueError(f"Unknown environment name: {env_name}")


def collect_traj(
    env: gym.Env,
    action_fn: Callable[[np.ndarray], np.ndarray],
    max_timesteps: int,
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

        obs = next_obs
        if terminated or truncated:
            break

    return {
        "observations": np.asarray(transitions["observations"], dtype=np.float32),
        "actions": np.asarray(transitions["actions"], dtype=np.float32),
        "next_observations": np.asarray(transitions["next_observations"], dtype=np.float32),
        "rewards": np.asarray(transitions["rewards"], dtype=np.float32),
        "terminals": np.asarray(transitions["terminals"], dtype=bool),
        "timeouts": np.asarray(transitions["timeouts"], dtype=bool),
    }


def collect_expert(
    env_name: str,
    policy_path: str,
    num_samples: int,
    max_timesteps: int,
    noise_scale: float = 0.0,
    deterministic: bool = True,
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    """Collect transition samples from a clipped, noise-injected expert."""
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
        num_samples=num_samples,
        max_timesteps=max_timesteps,
        rng=rng,
    )


def collect_suboptimal(
    env_name: str,
    policy_path: str,
    num_samples: int,
    max_timesteps: int,
    noise_scale: float = 0.0,
    deterministic: bool = True,
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    """Collect transition samples from pure random actions.

    policy_path, noise_scale, and deterministic are accepted for symmetry with
    CollectExpert and CollectDataset, but random-action collection does not use
    them.
    """
    del policy_path, noise_scale, deterministic

    rng = np.random.default_rng() if rng is None else rng

    def make_action_fn(env: gym.Env) -> Callable[[np.ndarray], np.ndarray]:
        env.action_space.seed(_next_seed(rng))

        def action_fn(_: np.ndarray) -> np.ndarray:
            return np.asarray(env.action_space.sample(), dtype=np.float32)

        return action_fn

    return _collect_source(
        env_name=env_name,
        make_action_fn=make_action_fn,
        num_samples=num_samples,
        max_timesteps=max_timesteps,
        rng=rng,
    )


def collect_dataset(
    env_name: str,
    policy_path: str,
    max_timesteps: int = 300,
    num_samples: int = 10000,
    noise_scale: float = 0.0,
    prop_expert: float = 1.0,
    deterministic: bool = True,
    seed: int | None = None,
) -> tuple[dict[str, np.ndarray], dict]:
    """Collect a shuffled offline-RL transition dataset.

    prop_expert controls the fraction of samples drawn from the noise-injected
    expert. The remainder are collected from uniform random actions.
    """
    _validate_collection_args(
        max_timesteps=max_timesteps,
        num_samples=num_samples,
        noise_scale=noise_scale,
        prop_expert=prop_expert,
    )

    rng = np.random.default_rng(seed)
    num_expert, num_suboptimal = _split_sample_counts(num_samples, prop_expert)
    datasets = []

    if num_expert > 0:
        datasets.append(
            collect_expert(
                env_name=env_name,
                policy_path=policy_path,
                num_samples=num_expert,
                max_timesteps=max_timesteps,
                noise_scale=noise_scale,
                deterministic=deterministic,
                rng=rng,
            )
        )
    if num_suboptimal > 0:
        datasets.append(
            collect_suboptimal(
                env_name=env_name,
                policy_path=policy_path,
                num_samples=num_suboptimal,
                max_timesteps=max_timesteps,
                noise_scale=noise_scale,
                deterministic=deterministic,
                rng=rng,
            )
        )

    dataset = _shuffle_dataset(_concat_datasets(datasets), rng)
    metadata = make_metadata(
        env_name=env_name,
        policy_path=policy_path,
        max_timesteps=max_timesteps,
        num_samples=num_samples,
        noise_scale=noise_scale,
        prop_expert=prop_expert,
        deterministic=deterministic,
        seed=seed,
    )
    return dataset, metadata


def make_metadata(
    env_name: str,
    policy_path: str,
    max_timesteps: int,
    num_samples: int,
    noise_scale: float,
    prop_expert: float,
    deterministic: bool,
    seed: int | None,
) -> dict:
    num_expert, num_suboptimal = _split_sample_counts(num_samples, prop_expert)
    return {
        "env_name": env_name,
        "policy_path": str(policy_path),
        "max_timesteps": max_timesteps,
        "num_expert": num_expert,
        "num_suboptimal": num_suboptimal,
        "noise_scale": noise_scale,
        "deterministic": deterministic,
        "seed": seed,
    }


def save_dataset(dataset: dict[str, np.ndarray], dataset_path: str | Path, metadata: dict | None = None) -> None:
    """Save dataset arrays as compressed NumPy data and optional JSON metadata."""
    dataset_path = Path(dataset_path)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    _validate_dataset(dataset)
    np.savez_compressed(dataset_path, **dataset)

    if metadata is not None:
        with dataset_path.with_suffix(".json").open("w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2, sort_keys=True)


def load_dataset(dataset_path: str | Path) -> dict[str, np.ndarray]:
    """Load the transition arrays saved by save_dataset."""
    with np.load(dataset_path) as data:
        dataset = {key: data[key] for key in DATASET_KEYS}
    _validate_dataset(dataset)
    return dataset


def load_metadata(dataset_path: str | Path) -> dict:
    """Load JSON metadata next to a saved dataset."""
    with Path(dataset_path).with_suffix(".json").open("r", encoding="utf-8") as file:
        return json.load(file)


def _collect_source(
    env_name: str,
    make_action_fn: Callable[[gym.Env], Callable[[np.ndarray], np.ndarray]],
    num_samples: int,
    max_timesteps: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    env = gym.make(env_name)
    try:
        action_fn = make_action_fn(env)
        return _collect_and_sample(
            env=env,
            action_fn=action_fn,
            num_samples=num_samples,
            max_timesteps=max_timesteps,
            rng=rng,
        )
    finally:
        env.close()


def _collect_and_sample(
    env: gym.Env,
    action_fn: Callable[[np.ndarray], np.ndarray],
    num_samples: int,
    max_timesteps: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    target_samples = int(np.ceil(1.5 * num_samples))
    datasets = []
    collected = 0

    while collected < target_samples:
        traj_seed = _next_seed(rng)
        traj = collect_traj(env, action_fn, max_timesteps=max_timesteps, seed=traj_seed)
        if len(traj["rewards"]) == 0:
            raise RuntimeError("Collected an empty trajectory; check the environment and action function.")
        datasets.append(traj)
        collected += len(traj["rewards"])

    return _sample_dataset(_concat_datasets(datasets), num_samples=num_samples, rng=rng)


def _concat_datasets(datasets: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not datasets:
        raise ValueError("Cannot concatenate an empty dataset list.")
    return {key: np.concatenate([dataset[key] for dataset in datasets], axis=0) for key in DATASET_KEYS}


def _sample_dataset(dataset: dict[str, np.ndarray], num_samples: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    _validate_dataset(dataset)
    total = len(dataset["rewards"])
    if num_samples > total:
        raise ValueError(f"Requested {num_samples} samples from a dataset with only {total} samples.")
    indices = rng.choice(total, size=num_samples, replace=False)
    return {key: dataset[key][indices] for key in DATASET_KEYS}


def _shuffle_dataset(dataset: dict[str, np.ndarray], rng: np.random.Generator) -> dict[str, np.ndarray]:
    _validate_dataset(dataset)
    indices = rng.permutation(len(dataset["rewards"]))
    return {key: dataset[key][indices] for key in DATASET_KEYS}


def _validate_dataset(dataset: dict[str, np.ndarray]) -> None:
    extra = [key for key in dataset if key not in DATASET_KEYS]
    if extra:
        raise ValueError(f"Dataset has unexpected keys: {extra}")
    missing = [key for key in DATASET_KEYS if key not in dataset]
    if missing:
        raise ValueError(f"Dataset is missing required keys: {missing}")

    lengths = {key: len(dataset[key]) for key in DATASET_KEYS}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Dataset arrays have inconsistent lengths: {lengths}")


def _validate_collection_args(
    max_timesteps: int,
    num_samples: int,
    noise_scale: float,
    prop_expert: float,
) -> None:
    if max_timesteps <= 0:
        raise ValueError("max_timesteps must be positive.")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")
    if noise_scale < 0.0:
        raise ValueError("noise_scale must be nonnegative.")
    if not 0.0 <= prop_expert <= 1.0:
        raise ValueError("prop_expert must be between 0 and 1.")


def _split_sample_counts(num_samples: int, prop_expert: float) -> tuple[int, int]:
    num_expert = int(round(num_samples * prop_expert))
    num_expert = min(max(num_expert, 0), num_samples)
    return num_expert, num_samples - num_expert


def _next_seed(rng: np.random.Generator) -> int:
    return int(rng.integers(0, np.iinfo(np.int32).max))

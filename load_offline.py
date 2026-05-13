# Tasks:
# - Load premade offline datasets from external sources such as Minari.
# - Convert external episode formats into this project's canonical transition schema.
# - Discover all relevant Minari datasets for a requested Gymnasium environment.
# - Keep external dataset loading separate from generated rollout collection.

from typing import Any

import numpy as np

from rollout import DATASET_KEYS


MINARI_PREFIXES = {
    "Ant-v5": "mujoco/ant",
    "HalfCheetah-v5": "mujoco/halfcheetah",
    "Hopper-v5": "mujoco/hopper",
    "Humanoid-v5": "mujoco/humanoid",
    "InvertedDoublePendulum-v5": "mujoco/inverteddoublependulum",
    "InvertedPendulum-v5": "mujoco/invertedpendulum",
    "Pusher-v5": "mujoco/pusher",
    "Reacher-v5": "mujoco/reacher",
    "Swimmer-v5": "mujoco/swimmer",
    "Walker2d-v5": "mujoco/walker2d",
}


def list_minari_dataset_ids(env_name: str) -> list[str]:
    """Return all remote Minari datasets that belong to the requested env."""
    import minari

    prefix = minari_prefix_for_env(env_name)
    datasets = minari.list_remote_datasets(prefix=prefix)
    dataset_ids = sorted(datasets)
    if not dataset_ids:
        raise ValueError(f"No Minari datasets found for env {env_name!r} with prefix {prefix!r}.")
    return dataset_ids


def minari_prefix_for_env(env_name: str) -> str:
    if env_name not in MINARI_PREFIXES:
        raise ValueError(f"No Minari prefix is configured for environment: {env_name}")
    return MINARI_PREFIXES[env_name]


def load_minari_dataset(dataset_id: str, seed: int | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Download/load one Minari dataset and convert it to this project's transition schema."""
    import minari

    minari_dataset = minari.load_dataset(dataset_id, download=True)
    dataset = minari_to_transition_dataset(minari_dataset, seed=seed)
    metadata = make_minari_metadata(dataset_id, minari_dataset, dataset, seed)
    return dataset, metadata


def minari_to_transition_dataset(minari_dataset: Any, seed: int | None = None) -> dict[str, np.ndarray]:
    datasets = [episode_to_transitions(episode) for episode in minari_dataset.iterate_episodes()]
    dataset = concat_datasets(datasets)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(dataset["rewards"]))
    return {key: dataset[key][indices] for key in DATASET_KEYS}


def episode_to_transitions(episode: Any) -> dict[str, np.ndarray]:
    observations = np.asarray(episode.observations, dtype=np.float32)
    actions = np.asarray(episode.actions, dtype=np.float32)
    rewards = np.asarray(episode.rewards, dtype=np.float32)
    terminals = np.asarray(episode.terminations, dtype=bool)
    timeouts = np.asarray(episode.truncations, dtype=bool)

    transition_count = len(actions)
    if len(observations) != transition_count + 1:
        raise ValueError(
            f"Episode {episode.id} has {len(observations)} observations and {transition_count} actions; "
            "expected one more observation than action."
        )
    lengths = {
        "actions": len(actions),
        "rewards": len(rewards),
        "terminations": len(terminals),
        "truncations": len(timeouts),
    }
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Episode {episode.id} has inconsistent transition lengths: {lengths}")

    return {
        "observations": observations[:-1],
        "actions": actions,
        "next_observations": observations[1:],
        "rewards": rewards,
        "terminals": terminals,
        "timeouts": timeouts,
    }


def concat_datasets(datasets: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not datasets:
        raise ValueError("Cannot concatenate an empty dataset list.")
    return {key: np.concatenate([dataset[key] for dataset in datasets], axis=0) for key in DATASET_KEYS}


def make_minari_metadata(
    dataset_id: str,
    minari_dataset: Any,
    dataset: dict[str, np.ndarray],
    seed: int | None,
) -> dict[str, Any]:
    env_spec = getattr(minari_dataset, "env_spec", None)
    return {
        "source": "minari",
        "dataset_id": dataset_id,
        "env_id": getattr(env_spec, "id", None),
        "num_episodes": int(getattr(minari_dataset, "total_episodes")),
        "num_transitions": int(len(dataset["rewards"])),
        "seed": seed,
    }


def make_minari_dataset_tag(dataset_id: str) -> str:
    return "minari_" + dataset_id.replace("/", "_")

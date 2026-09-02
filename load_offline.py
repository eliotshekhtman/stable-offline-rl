# Tasks:
# - Load premade offline datasets from external sources such as Minari.
# - Convert external episode formats into this project's canonical transition schema.
# - Discover all relevant Minari datasets for a requested Gymnasium environment.
# - Discover and load low-dimensional robomimic robosuite datasets.
# - Keep external dataset loading separate from generated rollout collection.

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import h5py
import numpy as np

from rollout import DATASET_KEYS
from task_support import require_supported_task


MINARI_PREFIXES = {
    "HalfCheetah-v5": "mujoco/halfcheetah",
    "Reacher-v5": "mujoco/reacher",
}

ROBOMIMIC_HF_REPO_ID = "robomimic/robomimic_datasets"
ROBOMIMIC_OBS_KEYS = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object")
ROBOMIMIC_ENV_KEYS = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object-state")

ROBOMIMIC_LOW_DIM_DATASETS = {
    "Can": [
        ("ph", "v1.5/can/ph/low_dim_v15.hdf5", 400),
        ("mh", "v1.5/can/mh/low_dim_v15.hdf5", 500),
        ("mg_sparse", "v1.5/can/mg/low_dim_sparse_v15.hdf5", 400),
        ("mg_dense", "v1.5/can/mg/low_dim_dense_v15.hdf5", 400),
        ("paired", "v1.5/can/paired/low_dim_v15.hdf5", 400),
    ],
    "Lift": [
        ("ph", "v1.5/lift/ph/low_dim_v15.hdf5", 400),
        ("mh", "v1.5/lift/mh/low_dim_v15.hdf5", 500),
        ("mg_sparse", "v1.5/lift/mg/low_dim_sparse_v15.hdf5", 400),
        ("mg_dense", "v1.5/lift/mg/low_dim_dense_v15.hdf5", 400),
    ],
}

def list_minari_dataset_ids(env_name: str, dataset_name: str | None = None) -> list[str]:
    """Return all matching Minari datasets, or one requested dataset."""
    import minari

    require_supported_task(env_name, "minari")
    prefix = MINARI_PREFIXES[env_name]
    datasets = minari.list_remote_datasets(prefix=prefix)
    dataset_ids = sorted(dataset_id for dataset_id in datasets if dataset_id.startswith(prefix + "/"))
    if not dataset_ids:
        raise ValueError(f"No Minari datasets found for env {env_name!r} with prefix {prefix!r}.")
    if dataset_name is not None:
        matches = [
            dataset_id for dataset_id in dataset_ids
            if dataset_id == dataset_name or dataset_id.rsplit("/", 1)[-1] == dataset_name
        ]
        if not matches:
            available = ", ".join(dataset_id.rsplit("/", 1)[-1] for dataset_id in dataset_ids)
            raise ValueError(
                f"Unknown Minari dataset {dataset_name!r} for {env_name!r}; available: {available}"
            )
        return matches
    return dataset_ids


def load_minari_dataset(dataset_id: str, seed: int | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Download/load one Minari dataset and convert it to this project's transition schema."""
    require_supported_minari_dataset(dataset_id)
    import minari

    minari_dataset = minari.load_dataset(dataset_id, download=True)
    dataset = concat_datasets([episode_to_transitions(episode) for episode in minari_dataset.iterate_episodes()])
    env_spec = getattr(minari_dataset, "env_spec", None)
    return dataset, {
        "source": "minari",
        "dataset_id": dataset_id,
        "env_id": getattr(env_spec, "id", None),
        "num_episodes": int(getattr(minari_dataset, "total_episodes")),
        "num_transitions": int(len(dataset["rewards"])),
        "seed": seed,
    }


def load_minari_episode_subset(
    dataset_id: str,
    num_episodes: int,
    seed: int,
    episode_id_start: int,
    episode_offset: int = 0,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load one slice of a deterministic Minari episode permutation."""
    require_supported_minari_dataset(dataset_id)
    import minari

    minari_dataset = minari.load_dataset(dataset_id, download=True)
    available_episodes = int(minari_dataset.total_episodes)
    if num_episodes <= 0 or episode_offset < 0:
        raise ValueError("num_episodes must be positive and episode_offset nonnegative.")
    selection_stop = episode_offset + num_episodes
    if selection_stop > available_episodes:
        raise ValueError(
            f"Requested {selection_stop} unique episodes from {dataset_id}, but it contains "
            f"only {available_episodes} episodes "
            f"({int(minari_dataset.total_steps)} transitions); cannot top up without repetition."
        )

    rng = np.random.default_rng(seed)
    episode_indices = rng.permutation(minari_dataset.episode_indices)[
        episode_offset:selection_stop
    ]
    episodes = []
    for episode_id, episode in enumerate(
        minari_dataset.iterate_episodes(episode_indices), start=episode_id_start
    ):
        transitions = episode_to_transitions(episode)
        transitions["episode_ids"] = np.full(
            len(transitions["actions"]), episode_id, dtype=np.int64
        )
        episodes.append(transitions)

    dataset = concat_datasets(episodes)
    env_spec = getattr(minari_dataset, "env_spec", None)
    return dataset, {
        "dataset_id": dataset_id,
        "env_id": getattr(env_spec, "id", None),
        "available_num_episodes": available_episodes,
        "available_num_transitions": int(minari_dataset.total_steps),
        "num_episodes": num_episodes,
        "num_transitions": int(len(dataset["rewards"])),
        "episode_offset": episode_offset,
        "seed": seed,
    }


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
        "episode_ids": np.full(transition_count, episode.id, dtype=np.int64),
    }


def concat_datasets(datasets: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not datasets:
        raise ValueError("Cannot concatenate an empty dataset list.")
    return {key: np.concatenate([dataset[key] for dataset in datasets], axis=0) for key in DATASET_KEYS}


def make_minari_dataset_tag(dataset_id: str) -> str:
    require_supported_minari_dataset(dataset_id)
    return "minari_" + dataset_id.replace("/", "_")


def require_supported_minari_dataset(dataset_id: str) -> str:
    for env_name, prefix in MINARI_PREFIXES.items():
        if dataset_id.startswith(prefix + "/"):
            require_supported_task(env_name, "minari")
            return env_name
    prefixes = ", ".join(sorted(MINARI_PREFIXES.values()))
    raise ValueError(
        f"Unsupported Minari dataset {dataset_id!r}; supported prefixes are: "
        f"{prefixes}."
    )


def list_robomimic_dataset_specs(
    env_name: str,
    dataset_name: str | None = None,
) -> list[dict[str, Any]]:
    """Return all low-dimensional dataset specs, or one requested type."""
    require_supported_task(env_name, "robomimic")
    specs = ROBOMIMIC_LOW_DIM_DATASETS[env_name]
    if dataset_name is not None:
        matches = [spec for spec in specs if spec[0] == dataset_name]
        if not matches:
            available = ", ".join(spec[0] for spec in specs)
            raise ValueError(
                f"Unknown robomimic dataset {dataset_name!r} for {env_name!r}; available: {available}"
            )
        specs = matches
    dataset_specs = [
        {
            "source": "robomimic",
            "task": env_name,
            "dataset_type": dataset_type,
            "repo_path": repo_path,
            "horizon": horizon,
        }
        for dataset_type, repo_path, horizon in specs
    ]
    if env_name == "Lift":
        for spec in dataset_specs:
            spec["task_semantics"] = "continuing"
    return dataset_specs


def load_robomimic_dataset(spec: dict[str, Any], seed: int | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Download/load one robomimic low-dimensional HDF5 dataset."""
    require_supported_task(spec.get("task"), "robomimic")
    from huggingface_hub import hf_hub_download

    dataset_path = hf_hub_download(repo_id=ROBOMIMIC_HF_REPO_ID, filename=spec["repo_path"], repo_type="dataset")
    with h5py.File(dataset_path, "r") as file:
        demos = [
            robomimic_demo_to_transitions(file["data"][demo_key], episode_id, spec["task"])
            for episode_id, demo_key in enumerate(sorted(file["data"].keys()))
        ]
        dataset = concat_datasets(demos)
        env_args = json.loads(file["data"].attrs["env_args"])
        episode_returns = [float(demo["rewards"].sum()) for demo in demos]

    return dataset, {
        **spec,
        "dataset_id": f"{spec['task']}_{spec['dataset_type']}",
        "hdf5_path": dataset_path,
        "obs_keys": list(ROBOMIMIC_OBS_KEYS),
        "env_keys": list(ROBOMIMIC_ENV_KEYS),
        "env_args": env_args,
        "num_episodes": len(episode_returns),
        "num_transitions": int(len(dataset["rewards"])),
        "episode_return_mean": float(np.mean(episode_returns)),
        "episode_return_std": float(np.std(episode_returns)),
        "seed": seed,
    }


def robomimic_demo_to_transitions(
    demo: h5py.Group,
    episode_id: int,
    task: str,
) -> dict[str, np.ndarray]:
    require_supported_task(task, "robomimic")
    observations = np.concatenate([np.asarray(demo["obs"][key], dtype=np.float32) for key in ROBOMIMIC_OBS_KEYS], axis=1)
    next_observations = np.concatenate([
        np.asarray(demo["next_obs"][key], dtype=np.float32) for key in ROBOMIMIC_OBS_KEYS
    ], axis=1)
    actions = np.asarray(demo["actions"], dtype=np.float32)
    rewards = np.asarray(demo["rewards"], dtype=np.float32)
    terminals = np.asarray(demo["dones"], dtype=bool)
    timeouts = np.zeros(len(actions), dtype=bool)
    if task == "Lift":
        terminals = np.zeros_like(terminals)
        timeouts[-1] = True
    elif len(timeouts) and not terminals[-1]:
        timeouts[-1] = True

    return {
        "observations": observations,
        "actions": actions,
        "next_observations": next_observations,
        "rewards": rewards,
        "terminals": terminals,
        "timeouts": timeouts,
        "episode_ids": np.full(len(actions), episode_id, dtype=np.int64),
    }


def make_robomimic_dataset_tag(spec: dict[str, Any]) -> str:
    require_supported_task(spec.get("task"), "robomimic")
    tag = f"robomimic_{spec['task'].lower()}_{spec['dataset_type']}"
    return f"{tag}_continuing" if spec["task"] == "Lift" else tag


def load_metadata(metadata_path: str | Path) -> dict[str, Any]:
    with Path(metadata_path).open("r", encoding="utf-8") as file:
        return json.load(file)


def make_robomimic_env(metadata: dict[str, Any]):
    """Build a flat Gymnasium-compatible robosuite env matching a robomimic low-dim dataset."""
    require_supported_task(metadata.get("task"), "robomimic")
    import robosuite
    from robosuite.wrappers import GymWrapper

    env_args = copy.deepcopy(metadata["env_args"])
    env_name = env_args["env_name"]
    env_kwargs = env_args["env_kwargs"]
    env_kwargs.pop("env_name", None)
    env_kwargs.pop("env_lang", None)
    env_kwargs.update(
        has_renderer=False,
        has_offscreen_renderer=False,
        horizon=metadata["horizon"],
        ignore_done=False,
        use_object_obs=True,
        use_camera_obs=False,
    )
    env = GymWrapper(robosuite.make(env_name, **env_kwargs), keys=list(ROBOMIMIC_ENV_KEYS), flatten_obs=True)
    env.spec = SimpleNamespace(id=metadata["task"])
    return env

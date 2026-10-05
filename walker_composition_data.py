"""Collect one Walker composition through the isolated reference interpreter.

Only complete per-composition datasets are returned; there are no source pools.
The calling launcher owns caching, locking, publication, and episode splitting.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from walker_reference import TASK, array_hashes, convert_dataset, environment_info


PROTOCOL = "walker-composition-v1"
SETTINGS = ("clean", "noise0.5", "noise1.0", "clean-medium-v2")
RAW_KEYS = ("observations", "actions", "next_observations", "rewards", "terminals", "timeouts")
MAX_SEED = np.iinfo(np.int32).max
# Verified official expert used by the existing isolated reference experiments.
EXPERT_SHA256 = "2985dc3d436a8baa10e8f43f946fe1d7c6337a4e4fbc65def70618093b18c59a"
EXPERT_ACTOR_SHA256 = "7a95af73a6ebe915e5bdf664bf9e9b70064cc82b961fde5157de6ae98f468ad0"
EXPERT_URL = "https://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/walker2d_expert-v2.hdf5"


def requested_quotas(spec):
    """Mirror rollout.transition_quotas without importing the modern runtime."""
    if spec.get("protocol") != PROTOCOL or spec.get("setting") not in SETTINGS:
        raise ValueError("Unsupported Walker composition protocol or setting")
    for key in ("seed", "num_samples"):
        value = spec.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{key} must be an integer")
    if not 0 <= spec["seed"] < 2**32 or spec["num_samples"] <= 0:
        raise ValueError("Use a nonnegative 32-bit seed and a positive transition budget")
    proportions = np.asarray([spec["clean_fraction"], spec["other_fraction"]], dtype=np.float64)
    if (not np.isfinite(proportions).all() or (proportions < 0).any()
            or not np.isclose(proportions.sum(), 1., rtol=0., atol=1e-12)):
        raise ValueError("Composition fractions must be finite, nonnegative, and sum to one")
    if spec["setting"] == "clean" and spec["other_fraction"] != 0:
        raise ValueError("The clean endpoint cannot request another source")
    fraction = spec["test_fraction"]
    if not np.isfinite(fraction) or not 0 <= fraction < 1:
        raise ValueError("test_fraction must lie in [0, 1)")
    quotas = np.ceil(np.nextafter(spec["num_samples"] * proportions, -np.inf)).astype(np.int64)
    return dict(zip(("clean", "other"), map(int, quotas)))


def medium_selection(ranges, quota, seed):
    """Ordinary clean/Minari selection: a seeded permutation, no replacement."""
    lengths = ranges[:, 1] - ranges[:, 0]
    if quota <= 0 or quota > int(lengths.sum()):
        raise ValueError(f"Medium quota {quota} exceeds {int(lengths.sum())} complete transitions, or is zero")
    order = np.random.default_rng(seed).permutation(len(ranges))
    count = int(np.searchsorted(np.cumsum(lengths[order]), quota)) + 1
    return order[:count].tolist()


def collect_source(env, actor, quota, noise_scale, rng):
    """Ordinary RNG consumption, with legacy reset and terminal semantics."""
    if quota <= 0:
        raise ValueError("Do not collect a zero-quota source")
    rows = {key: [] for key in RAW_KEYS}
    episodes = 0
    while len(rows["rewards"]) < quota:
        env.seed(int(rng.integers(0, MAX_SEED)))
        observation = env.reset()
        episodes += 1
        for step in range(1000):
            before = np.asarray(observation, np.float32).copy()
            action = np.asarray(actor.deterministic_action(observation), np.float32)
            if noise_scale > 0:
                action = action + rng.normal(0., noise_scale / np.sqrt(6), (6,)).astype(np.float32)
            action = np.clip(action, -1., 1.).astype(np.float32)
            successor, reward, done, info = env.step(action)
            # Exact reference collection: timeout takes precedence over health failure.
            timeout = step == 999 or bool(info.get("TimeLimit.truncated", False))
            values = (before, action.copy(), np.asarray(successor, np.float32).copy(),
                      reward, bool(done and not timeout), timeout)
            if not all(np.isfinite(value).all() for value in values):
                raise RuntimeError("Nonfinite reference collection transition")
            for key, value in zip(RAW_KEYS, values):
                rows[key].append(value)
            observation = successor
            if done or timeout:
                break
    raw = {key: np.asarray(value, dtype=bool if key in ("terminals", "timeouts") else np.float32)
           for key, value in rows.items()}
    return raw, episodes


def assemble_raw(spec, actor, env_factory, medium_component=None, medium_indices=None):
    """Collect each requested generated component afresh; preserve source order."""
    quotas = requested_quotas(spec)
    parts, source_names, episode_sources = [], [], []
    original_medium_ids = []
    if spec["setting"] == "clean-medium-v2" and quotas["other"]:
        if medium_component is None or medium_indices is None:
            raise ValueError("The selected complete medium episodes are required")
        parts.append(medium_component)
        source_names.append("medium")
        episode_sources.extend(["medium"] * len(medium_indices))
        original_medium_ids.extend(map(int, medium_indices))
    rng = np.random.default_rng(spec["seed"])
    for source, quota, scale in (("clean", quotas["clean"], 0.),
                                  ("noisy", quotas["other"] if spec["setting"].startswith("noise") else 0,
                                   float(spec["setting"][5:]) if spec["setting"].startswith("noise") else 0.)):
        if not quota:
            continue
        env = env_factory()
        try:
            part, episodes = collect_source(env, actor, quota, scale, rng)
        finally:
            env.close()
        parts.append(part)
        source_names.append(source)
        episode_sources.extend([source] * episodes)
        original_medium_ids.extend([None] * episodes)
    if not parts:
        raise ValueError("Composition produced no data")
    raw = {key: np.concatenate([part[key] for part in parts]) for key in RAW_KEYS}
    return raw, source_names, parts, episode_sources, original_medium_ids


def source_counts(dataset, raw, source_names, parts, episode_sources):
    counts = {source: {"raw_transitions": 0, "processed_transitions": 0, "episodes": 0}
              for source in ("clean", "noisy", "medium")}
    for source, part in zip(source_names, parts):
        counts[source]["raw_transitions"] += len(part["rewards"])
    processed_sources = np.asarray(episode_sources)[dataset["episode_ids"]]
    for source in counts:
        counts[source]["episodes"] = episode_sources.count(source)
        counts[source]["processed_transitions"] = int((processed_sources == source).sum())
    if sum(record["raw_transitions"] for record in counts.values()) != len(raw["rewards"]):
        raise RuntimeError("Raw source counts disagree")
    return counts


def verified_expert(path):
    from mixture_data import ReferenceActor
    from verify_setup import sha256

    if sha256(path) != EXPERT_SHA256:
        raise RuntimeError("Expert file differs from the verified reference expert")
    actor = ReferenceActor(path)
    if actor.sha256 != EXPERT_ACTOR_SHA256 or actor.iteration != 1950:
        raise RuntimeError("Expert actor differs from the verified reference expert")
    return actor, {"path": str(path), "url": EXPERT_URL, "sha256": EXPERT_SHA256,
                   "actor_sha256": actor.sha256, "checkpoint_iteration": actor.iteration,
                   "policy": "deterministic mean of the verified reference tanh-Gaussian actor"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", required=True, type=Path)
    commands = parser.add_subparsers(dest="operation", required=True)
    for name in ("preflight", "collect"):
        command = commands.add_parser(name)
        command.add_argument("--request", required=True, type=Path)
        command.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    specs = request["datasets"] if args.operation == "preflight" else [request]
    if not specs:
        parser.error("At least one dataset is required")
    quotas = [requested_quotas(spec) for spec in specs]
    root = args.reference_root.resolve()
    sys.path.insert(0, str(root))
    import run_benchmark as benchmark
    if Path(sys.prefix) != benchmark.ENV or benchmark.ROOT != root:
        parser.error("Invoke through walker-benchmark/benchmark-python")
    proof = benchmark.verify_inputs()
    import gym
    import d4rl  # noqa: F401 -- register legacy environment
    import torch
    import offlinerlkit
    from mixture_data import read_raw, episode_ranges, take_episodes, validate_raw
    from offlinerlkit.utils.load_dataset import qlearning_dataset

    if not Path(offlinerlkit.__file__).resolve().is_relative_to(root / "upstream/OfflineRL-Kit"):
        raise RuntimeError("The worker requires the frozen reference OfflineRL-Kit")
    torch.set_num_threads(1)
    medium = ranges = None
    if any(spec["setting"] == "clean-medium-v2" and quota["other"]
           for spec, quota in zip(specs, quotas)):
        medium = read_raw(benchmark.DATA)
        ranges = episode_ranges(medium, allow_trailing=True)
        for spec, quota in zip(specs, quotas):
            if spec["setting"] == "clean-medium-v2" and quota["other"]:
                medium_selection(ranges, quota["other"], spec["seed"])
    actor = expert = None
    if any(quota["clean"] or (spec["setting"].startswith("noise") and quota["other"])
           for spec, quota in zip(specs, quotas)):
        actor, expert = verified_expert(benchmark.DATA.parent / "walker2d_expert-v2.hdf5")

    def env_factory():
        env = gym.make(TASK)
        try:
            environment_info(env)  # Verify every actual collection instance, not only the preflight env.
        except Exception:
            env.close()
            raise
        return env

    env = env_factory()
    try:
        physics = environment_info(env)
        checks = [{"dataset_spec": spec, "requested_quotas": quota} for spec, quota in zip(specs, quotas)]
        report = {"protocol": PROTOCOL, "datasets": checks, "reference_verification": proof,
                  "environment": physics, "expert": expert,
                  "medium_complete_transitions": int((ranges[:, 1] - ranges[:, 0]).sum()) if ranges is not None else None,
                  "medium_complete_episodes": len(ranges) if ranges is not None else None}
        if args.operation == "preflight":
            with args.output.open("x") as handle:
                json.dump(report, handle, indent=2)
                handle.write("\n")
            return
        if not args.output.is_dir() or any(args.output.iterdir()):
            raise ValueError("Collection output must be an existing empty staging directory")
        spec, quota = specs[0], quotas[0]
        indices = component = None
        if ranges is not None:
            indices = medium_selection(ranges, quota["other"], spec["seed"])
            component = take_episodes(medium, indices)
        raw, names, parts, sources, medium_ids = assemble_raw(spec, actor, env_factory, component, indices)
        validate_raw(raw)
        dataset = convert_dataset(env, raw, qlearning_dataset)
        counts = source_counts(dataset, raw, names, parts, sources)
        raw_count = len(raw["rewards"])
        other_count = counts["noisy"]["raw_transitions"] + counts["medium"]["raw_transitions"]
        other_episodes = counts["noisy"]["episodes"] + counts["medium"]["episodes"]
        metadata = {"dataset_spec": spec, "reference_verification": proof, "environment": physics,
                    "expert": expert, "requested_quotas": quota, "source_counts": counts,
                    "actual_other_episode_fraction": other_episodes / len(sources),
                    "actual_other_transition_fraction": other_count / raw_count,
                    "episode_sources": sources, "original_medium_episode_ids": medium_ids,
                    "medium_source": {"path": str(benchmark.DATA), "sha256": proof["dataset_sha256"]} if indices is not None else None,
                    "raw_transitions": raw_count, "processed_transitions": len(dataset["rewards"]),
                    "processed_terminals": int(dataset["terminals"].sum()), "array_sha256": array_hashes(dataset),
                    "source_order": names, "reset_seeding": "default_rng(seed); integers(0, int32_max) per episode",
                    "noise_seeding": "same generator as clean/noisy reset seeds; normal(0, noise_scale/sqrt(6))",
                    "medium_selection": "separate default_rng(seed) permutation; complete episodes without replacement",
                    "conversion": "upstream qlearning_dataset once after assembly; drop timeouts and global final row"}
        with (args.output / "dataset.npz").open("xb") as handle:
            np.savez(handle, **dataset)
        with (args.output / "metadata.json").open("x") as handle:
            json.dump(metadata, handle, indent=2)
            handle.write("\n")
    finally:
        env.close()


if __name__ == "__main__":
    main()

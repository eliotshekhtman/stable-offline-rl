"""Legacy-runtime worker for walker_sweep.py; invoke via benchmark-python only."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


SETTINGS = ("medium-v2", "clean-medium-v2", "noise0.5", "noise1.0")
TASK = "walker2d-medium-v2"
# MuJoCo 2.1 serialization of the verified reference model, including all physics.
REFERENCE_MODEL_SHA256 = "3c480f4b98ca23f746752540cbdb1dfae3cb78f53e1a9824ee42152bd85c1b46"


def array_hashes(dataset):
    return {key: hashlib.sha256(value.tobytes()).hexdigest()
            for key, value in dataset.items()}


def convert_dataset(env, raw, converter):
    """Keep the reference conversion exact; attach IDs BEFORE dropping timeout rows."""
    dataset = converter(env, dataset=raw)
    keep = ~raw["timeouts"].astype(bool)
    keep[-1] = False  # Upstream deliberately omits the global final row too.
    ends = raw["terminals"].astype(bool) | raw["timeouts"].astype(bool)
    episode_ids = np.r_[0, np.cumsum(ends[:-1], dtype=np.int64)]
    for key, values in dataset.items():
        expected = raw[key][keep].astype(bool if key == "terminals" else np.float32)
        if not np.array_equal(values, expected):
            raise RuntimeError(f"Reference conversion changed: {key}")
    dataset["timeouts"] = raw["timeouts"][keep].astype(bool)
    dataset["episode_ids"] = episode_ids[keep]
    return dataset


def environment_info(env):
    import mujoco_py

    base = env.unwrapped
    model = base.model
    engine = mujoco_py.cymj._mj_version()
    model_hash = hashlib.sha256(model.get_xml().encode()).hexdigest()
    feet = [float(model.geom_friction[model.geom_name2id(name), 0])
            for name in ("foot_geom", "foot_left_geom")]
    if (engine != 210 or model_hash != REFERENCE_MODEL_SHA256 or feet != [0.9, 1.9]
            or env._max_episode_steps != 1000 or base.frame_skip != 4 or base.dt != 0.008
            or env.observation_space.shape != (17,) or env.action_space.shape != (6,)):
        raise RuntimeError("Unexpected reference Walker environment")
    return {"task": TASK, "engine_version": engine, "mujoco_py": mujoco_py.get_version(),
            "horizon": env._max_episode_steps, "frame_skip": base.frame_skip,
            "dt": base.dt, "foot_friction": feet,
            "geom_friction": model.geom_friction.tolist(), "body_mass": model.body_mass.tolist(),
            "model_xml_sha256": model_hash,
            "reference_min_score": env.ref_min_score, "reference_max_score": env.ref_max_score}


def load_actor(path):
    import torch
    from offlinerlkit.modules import ActorProb, TanhDiagGaussian
    from offlinerlkit.nets import MLP

    actor = ActorProb(MLP(17, [256, 256]), TanhDiagGaussian(
        256, 6, unbounded=True, conditioned_sigma=True, max_mu=1.0), device="cpu")
    state = torch.load(path, map_location="cpu", weights_only=True)
    actor.load_state_dict({key[len("actor."):]: value for key, value in state.items()
                           if key.startswith("actor.")}, strict=True)
    if any(not torch.isfinite(value).all() for value in actor.state_dict().values()):
        raise ValueError(f"Nonfinite actor weights: {path}")
    actor.eval()
    return actor


def evaluate(env, request):
    import torch
    from verify_setup import sha256

    if request["episodes"] <= 0:
        raise ValueError("Only positive evaluation requests should reach the worker")
    path = Path(request["policy_path"])
    actor = load_actor(path)
    env.seed(request["seed"])
    returns, lengths = [], []
    with torch.inference_mode():
        for _ in range(request["episodes"]):
            obs, total, length, done = env.reset(), 0.0, 0, False
            while not done:
                action = actor(obs.reshape(1, -1)).mode()[0].numpy()[0]
                obs, reward, done, _ = env.step(action)
                if not np.isfinite(obs).all() or not np.isfinite(reward):
                    raise RuntimeError("Nonfinite reference evaluation transition")
                total += float(reward)
                length += 1
            returns.append(total)
            lengths.append(length)
    scores = 100 * env.get_normalized_score(np.asarray(returns))
    return {**request, "policy_sha256": sha256(path), "returns": returns,
            "lengths": lengths, "normalized_scores": scores.tolist(),
            "return_mean": float(np.mean(returns)), "return_std": float(np.std(returns)),
            "normalized_score_mean": float(scores.mean()),
            "normalized_score_std": float(scores.std()), "std_ddof": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    commands = parser.add_subparsers(dest="operation", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--setting", choices=SETTINGS, required=True)
    prepare.add_argument("--manifest", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--requests", type=Path, required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.reference_root.resolve()
    sys.path.insert(0, str(root))
    import run_benchmark as benchmark
    if Path(sys.prefix) != benchmark.ENV or benchmark.ROOT != root:
        parser.error("Invoke this worker through walker-benchmark/benchmark-python")
    proof = benchmark.verify_inputs()
    import gym
    import d4rl  # noqa: F401 -- registers the legacy environments
    import offlinerlkit
    from offlinerlkit.utils.load_dataset import qlearning_dataset
    from mixture_data import load_mixture, read_raw

    if not Path(offlinerlkit.__file__).resolve().is_relative_to(root / "upstream/OfflineRL-Kit"):
        raise RuntimeError("The worker must use the frozen reference OfflineRL-Kit")
    env = gym.make(TASK)
    try:
        physics = environment_info(env)
        if args.operation == "prepare":
            if args.setting == "medium-v2":
                raw = read_raw(benchmark.DATA)
                source = {"path": str(benchmark.DATA), "sha256": proof["dataset_sha256"]}
            else:
                setting = "medium-v2" if args.setting == "clean-medium-v2" else args.setting
                manifest_bytes = args.manifest.read_bytes()
                raw, manifest = load_mixture(args.manifest, setting)
                if (json.loads(manifest_bytes) != manifest or manifest["environment"] != TASK
                        or manifest["native_mujoco_version"] != physics["engine_version"]
                        or manifest["reference_inputs"] != proof):
                    raise RuntimeError("Mixture provenance does not match the verified reference setup")
                source = {"manifest_path": str(args.manifest.resolve()),
                          "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(), "manifest": manifest}
            dataset = convert_dataset(env, raw, qlearning_dataset)
            np.savez(args.output / "dataset.npz", **dataset)
            metadata = {"setting": args.setting, "reference_verification": proof,
                        "environment": physics, "source": source,
                        "raw_transitions": len(raw["rewards"]),
                        "processed_transitions": len(dataset["rewards"]),
                        "processed_terminals": int(dataset["terminals"].sum()),
                        "array_sha256": array_hashes(dataset),
                        "conversion": "upstream qlearning_dataset; explicit next observations; "
                                      "drop timeouts and global final row"}
            (args.output / "dataset.json").write_text(json.dumps(metadata, indent=2) + "\n")
        else:
            report = {"reference_verification": proof, "environment": physics, "checkpoints": []}
            for request in json.loads(args.requests.read_text()):
                report["checkpoints"].append(evaluate(env, request))
                args.output.write_text(json.dumps(report, indent=2) + "\n")
    finally:
        env.close()


if __name__ == "__main__":
    main()

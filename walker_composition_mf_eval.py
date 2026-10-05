"""Evaluate isolated IQL/TD3BC Walker compositions in the frozen reference runtime.

Reports deliberately use the existing composition evaluation protocol so the
unmodified composition plotter can compare model-free and model-based runs.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

from walker_composition_eval import (
    DEFAULT_REFERENCE_ROOT, FINAL_SEED_OFFSET, METRICS, PROTOCOL, RUN_PROTOCOL,
    atomic_json, cache_lock, checkpoint_requests, episode_statistics, file_hash,
    identity, validate_evaluation, validate_record,
)


ALGORITHMS = ("iql", "td3bc")
IMPLEMENTATION = "walker-composition-mf-v1"


def validate_architecture(algorithm, config):
    if algorithm not in ALGORITHMS:
        raise ValueError("Expected an IQL or TD3BC composition policy")
    if (config.get("implementation") != IMPLEMENTATION
            or config.get("chunk_length") != 1
            or config.get("reward_normalization") is not False):
        raise ValueError("Unsupported model-free composition implementation metadata")
    hidden_dims = config.get(algorithm + "_hidden_dims")
    if (not isinstance(hidden_dims, list) or not hidden_dims
            or any(type(width) is not int or width <= 0 for width in hidden_dims)):
        raise ValueError("Policy hidden dimensions must be positive integers")
    return hidden_dims


def load_action_fn(path, algorithm, training_config):
    """Load deterministic inference only; never reconstruct critics or optimizers."""
    hidden_dims = validate_architecture(algorithm, training_config)
    import torch
    from offlinerlkit.modules import Actor, ActorProb, DiagGaussian
    from offlinerlkit.nets import MLP

    backbone = MLP(17, hidden_dims)
    if algorithm == "iql":
        actor = ActorProb(backbone, DiagGaussian(
            backbone.output_dim, 6, unbounded=False, conditioned_sigma=False,
            max_mu=1.0), device="cpu")
    else:
        actor = Actor(backbone, 6, max_action=1.0, device="cpu")
    state = torch.load(path, map_location="cpu", weights_only=True)
    actor.load_state_dict({key[len("actor."):]: value for key, value in state.items()
                           if key.startswith("actor.")}, strict=True)
    if any(not torch.isfinite(value).all() for value in actor.state_dict().values()):
        raise ValueError("Nonfinite model-free actor weights")
    actor.eval()
    mean, std = None, None
    if algorithm == "td3bc":
        stats = []
        for key in ("observation_mean", "observation_std"):
            value = state.get(key)
            if (not isinstance(value, torch.Tensor) or value.shape != (1, 17)
                    or not value.is_floating_point() or not torch.isfinite(value).all()):
                raise ValueError(f"Missing or invalid TD3BC normalization: {key}")
            stats.append(value.detach().cpu().numpy())
        mean, std = stats
        if not np.all(std > 0):
            raise ValueError("TD3BC observation_std must be positive")

    def action_fn(observation):
        observation = np.asarray(observation)
        if (observation.ndim not in (1, 2) or observation.shape[-1] != 17
                or not np.isfinite(observation).all()):
            raise ValueError("Expected finite Walker observations with 17 dimensions")
        single = observation.ndim == 1
        observation = observation.reshape(-1, 17)
        if mean is not None:
            # Match StandardScaler before Actor converts to float32, including
            # the float64 observations returned by the reference environment.
            observation = (observation - mean) / std
        with torch.inference_mode():
            output = actor(observation)
            action = (output.mode() if algorithm == "iql" else output).cpu().numpy()
        if algorithm == "iql":
            action = np.clip(action, -1.0, 1.0)
        if not np.isfinite(action).all():
            raise ValueError("Nonfinite model-free evaluation action")
        return action[0] if single else action

    return action_fn


def evaluate_run(run_dir, checkpoint_eval_episodes=0, final_eval_episodes=20,
                 reference_root=None, reuse_eval=False, seed=None):
    """Write compatible evaluations separately, without changing training files."""
    if checkpoint_eval_episodes < 0 or final_eval_episodes < 0:
        raise ValueError("Evaluation episode counts cannot be negative")
    if checkpoint_eval_episodes == final_eval_episodes == 0:
        return None
    run_dir = Path(run_dir).resolve()
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("protocol") != RUN_PROTOCOL or manifest.get("version") != 1
            or manifest.get("status") not in ("trained", "complete")):
        raise ValueError("Only completed Walker composition runs may be evaluated")
    validate_architecture(manifest.get("algorithm"), manifest.get("training_config", {}))
    if run_dir.parent.name != "model_free_runs":
        raise ValueError("Expected a composition/model_free_runs/<training-id> directory")
    training_id = str(manifest["training_id"])
    if run_dir.name != training_id or training_id in (".", ".."):
        raise ValueError("Invalid training identity")
    from walker_reference import REFERENCE_MODEL_SHA256
    from walker_composition_data import EXPERT_SHA256, EXPERT_ACTOR_SHA256
    if manifest["environment"].get("model_xml_sha256") != REFERENCE_MODEL_SHA256:
        raise ValueError("Run does not use the verified reference model")
    reference_root = Path(reference_root or manifest.get("reference_root", DEFAULT_REFERENCE_ROOT)).resolve()
    expert_path = Path("/data/shekhe/walker-benchmark/datasets/walker2d_expert-v2.hdf5")
    if file_hash(expert_path) != EXPERT_SHA256:
        raise ValueError("Reference expert file differs from the verified source")
    seed = int(manifest["seed"] if seed is None else seed)
    requests = checkpoint_requests(manifest, checkpoint_eval_episodes, final_eval_episodes, seed)
    expert_episodes = final_eval_episodes or checkpoint_eval_episodes
    expert_seed = seed + (FINAL_SEED_OFFSET if final_eval_episodes else 0)
    expert_request = {"file_sha256": EXPERT_SHA256, "actor_sha256": EXPERT_ACTOR_SHA256,
                      "episodes": expert_episodes,
                      "reset_seeds": list(range(expert_seed, expert_seed + expert_episodes))}
    config = {"protocol": PROTOCOL, "metric_version": 1, "metrics": METRICS,
              "seed_convention": "env.seed(seed + episode); final seed offset 1000000",
              "seed": seed, "checkpoint_eval_episodes": checkpoint_eval_episodes,
              "final_eval_episodes": final_eval_episodes,
              "environment_sha256": identity(manifest["environment"]),
              "checkpoints": [{key: value for key, value in request.items() if key != "policy_path"}
                              for request in requests], "expert": expert_request}
    config_id = identity(config)
    eval_root = run_dir.parent.parent / "evals"
    destination = eval_root / training_id / config_id / "evaluation.json"
    with cache_lock(destination.parent / ".lock"):
        if reuse_eval and destination.exists():
            validate_evaluation(json.loads(destination.read_text()), manifest_path, config, config_id)
            return destination
        with tempfile.TemporaryDirectory(prefix=".worker-", dir=destination.parent) as temporary:
            temporary = Path(temporary)
            request_path, response_path = temporary / "request.json", temporary / "response.json"
            atomic_json(request_path, {"environment": manifest["environment"], "requests": requests,
                                      "expert_request": expert_request, "expert_path": str(expert_path),
                                      "expert_cache_root": str(eval_root / "experts"),
                                      "algorithm": manifest["algorithm"],
                                      "training_config": manifest["training_config"]})
            command = [str(reference_root / "benchmark-python"), str(Path(__file__).resolve()),
                       "_worker", "--reference-root", str(reference_root),
                       "--request", str(request_path), "--output", str(response_path)]
            environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"}
            subprocess.run(command, check=True, env=environment)
            response = json.loads(response_path.read_text())
        for request in requests:
            if file_hash(request["policy_path"]) != request["policy_sha256"]:
                raise ValueError("Checkpoint changed during evaluation")
        report = {"protocol": PROTOCOL, "version": 1, "config_id": config_id,
                  "run_manifest_path": str(manifest_path), "evaluation_config": config,
                  "metrics": METRICS, **response}
        validate_evaluation(report, manifest_path, config, config_id)
        atomic_json(destination, report)
    return destination


def worker(reference_root, request_path, output_path):
    reference_root = Path(reference_root).resolve()
    sys.path.insert(0, str(reference_root))
    import run_benchmark as benchmark
    if Path(sys.prefix) != benchmark.ENV or benchmark.ROOT != reference_root:
        raise RuntimeError("Invoke the evaluation worker through benchmark-python")
    proof = benchmark.verify_inputs()
    import torch
    import gym
    import d4rl  # noqa: F401 -- registers the reference environment
    import offlinerlkit
    from walker_reference import TASK, environment_info
    from walker_composition_data import verified_expert
    if not Path(offlinerlkit.__file__).resolve().is_relative_to(reference_root / "upstream/OfflineRL-Kit"):
        raise RuntimeError("Evaluation must use the frozen reference policy modules")
    torch.set_num_threads(1)
    request = json.loads(Path(request_path).read_text())
    validate_architecture(request["algorithm"], request["training_config"])
    env = gym.make(TASK)
    try:
        physics = environment_info(env)
        if physics != request["environment"]:
            raise ValueError("Evaluation physics differs from training dataset physics")
        expert, _ = verified_expert(Path(request["expert_path"]))
        expert_request = request["expert_request"]
        if expert.sha256 != expert_request["actor_sha256"]:
            raise ValueError("Expert actor identity changed")
        expert_key = {"protocol": PROTOCOL, "metric_version": 1, "environment": physics,
                      "seed_convention": "env.seed(seed + episode)", **expert_request}
        expert_cache = Path(request["expert_cache_root"]) / (identity(expert_key) + ".json")
        with cache_lock(expert_cache.with_suffix(".lock")):
            if expert_cache.exists():
                expert_report = json.loads(expert_cache.read_text())
                if expert_report.get("cache_identity") != expert_key:
                    raise ValueError("Expert cache identity mismatch")
                validate_record(expert_report, expert_request)
            else:
                expert_report = {**expert_request, "cache_identity": expert_key,
                                 **episode_statistics(env, expert.deterministic_action,
                                                      expert_request["reset_seeds"])}
                validate_record(expert_report, expert_request)
                atomic_json(expert_cache, expert_report)
        records = []
        for item in request["requests"]:
            if file_hash(item["policy_path"]) != item["policy_sha256"]:
                raise ValueError("Checkpoint identity changed")
            action_fn = load_action_fn(item["policy_path"], request["algorithm"], request["training_config"])
            stats = episode_statistics(env, action_fn, item["reset_seeds"])
            record = {key: value for key, value in item.items() if key != "policy_path"}
            records.append({**record, **stats})
        atomic_json(output_path, {"environment": physics, "reference_verification": proof,
                                  "records": records, "expert": expert_report})
    finally:
        env.close()


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "_worker":
        parser = argparse.ArgumentParser(description="Internal frozen-runtime model-free evaluation worker")
        parser.add_argument("--reference-root", type=Path, required=True)
        parser.add_argument("--request", type=Path, required=True)
        parser.add_argument("--output", type=Path, required=True)
        args = parser.parse_args(argv[1:])
        worker(args.reference_root, args.request, args.output)
        return
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--run-dir", type=Path)
    selection.add_argument("--root", type=Path, help="Discover completed IQL/TD3BC runs under model_free_runs only")
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--checkpoint-eval-episodes", type=int, default=0)
    parser.add_argument("--final-eval-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--reuse-eval", action="store_true")
    args = parser.parse_args(argv)
    if args.checkpoint_eval_episodes < 0 or args.final_eval_episodes < 0:
        parser.error("Episode counts must be nonnegative")
    if args.run_dir:
        runs = [args.run_dir]
    else:
        runs = []
        for path in sorted(args.root.rglob("run_manifest.json")):
            if path.parent.parent.name != "model_free_runs":
                continue
            manifest = json.loads(path.read_text())
            if (manifest.get("protocol") == RUN_PROTOCOL and manifest.get("version") == 1
                    and manifest.get("status") in ("trained", "complete")
                    and manifest.get("algorithm") in ALGORITHMS):
                runs.append(path.parent)
        if not runs:
            parser.error("No completed model-free Walker composition runs found")
    for run_dir in runs:
        result = evaluate_run(run_dir, args.checkpoint_eval_episodes, args.final_eval_episodes,
                              args.reference_root, args.reuse_eval, args.seed)
        print(result if result is not None else f"Evaluation disabled: {run_dir}")


if __name__ == "__main__":
    main()

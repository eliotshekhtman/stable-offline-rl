"""Evaluate new Walker composition runs in the frozen reference environment only."""

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_REFERENCE_ROOT = ROOT.parent / "walker-benchmark"
PROTOCOL = "walker-composition-eval-v1"
RUN_PROTOCOL = "walker-composition-v1"
FINAL_SEED_OFFSET = 1_000_000
METRICS = {
    "performance": {"name": "Forward displacement", "unit": "m"},
    "returns": {"name": "Episode return", "unit": "reward"},
    "normalized_scores": {"name": "D4RL-normalized return", "unit": "score"},
}


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".evaluation-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def cache_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def episode_statistics(env, action_fn, reset_seeds):
    """Measure displacement directly; rewards also contain survival/action terms."""
    if not reset_seeds:
        raise ValueError("Evaluation needs at least one episode")
    returns, lengths, performance = [], [], []
    for seed in reset_seeds:
        env.seed(int(seed))
        observation = env.reset()
        initial_x = float(env.unwrapped.sim.data.qpos[0])
        total, length, done = 0.0, 0, False
        while not done:
            action = np.asarray(action_fn(observation))
            if not np.isfinite(action).all():
                raise ValueError("Nonfinite evaluation action")
            observation, reward, done, _ = env.step(action)
            if not np.isfinite(observation).all() or not np.isfinite(reward):
                raise ValueError("Nonfinite evaluation transition")
            total += float(reward)
            length += 1
            if length >= env._max_episode_steps and not done:
                raise ValueError("Reference evaluation exceeded the configured horizon")
        displacement = float(env.unwrapped.sim.data.qpos[0]) - initial_x
        if not np.isfinite(displacement):
            raise ValueError("Nonfinite evaluation displacement")
        returns.append(total)
        lengths.append(length)
        performance.append(displacement)
    scores = np.asarray(env.get_normalized_score(np.asarray(returns)), dtype=float) * 100
    result = {"episodes": len(reset_seeds), "reset_seeds": list(reset_seeds),
              "returns": returns, "normalized_scores": scores.tolist(),
              "performance": performance, "lengths": lengths, "std_ddof": 0}
    for key, prefix in (("returns", "return"), ("normalized_scores", "normalized_score"),
                        ("performance", "performance")):
        result[prefix + "_mean"] = float(np.mean(result[key]))
        result[prefix + "_std"] = float(np.std(result[key]))
    return result


def checkpoint_requests(manifest, checkpoint_episodes, final_episodes, seed):
    if checkpoint_episodes < 0 or final_episodes < 0:
        raise ValueError("Evaluation episode counts cannot be negative")
    if seed < 0 or seed + FINAL_SEED_OFFSET + max(checkpoint_episodes, final_episodes) >= 2**32:
        raise ValueError("Evaluation seeds must fit the environment's uint32 range")
    checkpoints = manifest.get("checkpoints", [])
    if not checkpoints or sum(bool(item.get("final")) for item in checkpoints) != 1:
        raise ValueError("A completed run must declare exactly one final checkpoint")
    if (checkpoint_episodes and all(item.get("final") for item in checkpoints)
            and checkpoints[0]["epoch"] > 1):
        raise ValueError("This run saved no intermediate checkpoints; use --checkpoint-eval-episodes 0")
    result = []
    for checkpoint in checkpoints:
        final = bool(checkpoint.get("final", False))
        episodes = final_episodes if final else checkpoint_episodes
        if not episodes:
            continue
        path = Path(checkpoint["policy_path"]).resolve()
        checksum = file_hash(path)
        if "policy_sha256" in checkpoint and checkpoint["policy_sha256"] != checksum:
            raise ValueError(f"Checkpoint checksum differs from the training manifest: {path}")
        offset = FINAL_SEED_OFFSET if final else 0
        result.append({"epoch": checkpoint["epoch"], "step": checkpoint["step"],
                       "final": final, "policy_path": str(path), "policy_sha256": checksum,
                       "episodes": episodes,
                       "reset_seeds": list(range(seed + offset, seed + offset + episodes))})
    return result


def validate_record(record, expected):
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError(f"Evaluation record identity mismatch: {key}")
    n = expected["episodes"]
    for key in ("returns", "normalized_scores", "performance", "lengths"):
        values = np.asarray(record.get(key, []), dtype=float)
        if values.shape != (n,) or not np.isfinite(values).all():
            raise ValueError(f"Invalid cached evaluation array: {key}")
        if key == "lengths" and ((values <= 0).any() or (values > 1000).any()
                                 or (values != values.astype(int)).any()):
            raise ValueError("Invalid cached episode lengths")


def validate_evaluation(report, manifest_path, config, config_id):
    if (report.get("protocol") != PROTOCOL or report.get("version") != 1
            or report.get("run_manifest_path") != str(manifest_path)
            or report.get("evaluation_config") != config or report.get("config_id") != config_id
            or identity(report.get("environment")) != config["environment_sha256"]):
        raise ValueError("Cached evaluation provenance does not match this request")
    if len(report.get("records", [])) != len(config["checkpoints"]):
        raise ValueError("Cached evaluation has missing checkpoint records")
    for record, expected in zip(report["records"], config["checkpoints"]):
        validate_record(record, expected)
    expert_request = config["expert"]
    validate_record(report.get("expert", {}), expert_request)


def evaluate_run(run_dir, checkpoint_eval_episodes=0, final_eval_episodes=20,
                 reference_root=None, reuse_eval=False, seed=None):
    """Return the evaluation path without modifying any training artifact."""
    if checkpoint_eval_episodes < 0 or final_eval_episodes < 0:
        raise ValueError("Evaluation episode counts cannot be negative")
    if checkpoint_eval_episodes == final_eval_episodes == 0:
        return None  # No verification subprocess, expert loading, or environment creation.
    run_dir = Path(run_dir).resolve()
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("protocol") != RUN_PROTOCOL or manifest.get("version") != 1
            or manifest.get("status") not in ("trained", "complete")):
        raise ValueError("Only completed Walker composition runs may be evaluated")
    if manifest.get("algorithm") not in ("mopo", "mobile"):
        raise ValueError("Unsupported reference-policy architecture")
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
    if run_dir.parent.name != "runs":
        raise ValueError("Expected a composition/runs/<training-id> run directory")
    training_id = str(manifest["training_id"])
    if not training_id or Path(training_id).name != training_id or training_id in (".", ".."):
        raise ValueError("Invalid training identity")
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
                                      "expert_cache_root": str(eval_root / "experts")})
            command = [str(reference_root / "benchmark-python"), str(Path(__file__).resolve()),
                       "_worker", "--reference-root", str(reference_root),
                       "--request", str(request_path), "--output", str(response_path)]
            environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"}
            subprocess.run(command, check=True, env=environment)
            response = json.loads(response_path.read_text())
        # Reject a checkpoint replaced while the worker was reading/evaluating it.
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
    from walker_reference import TASK, environment_info, load_actor
    from walker_composition_data import verified_expert
    if not Path(offlinerlkit.__file__).resolve().is_relative_to(reference_root / "upstream/OfflineRL-Kit"):
        raise RuntimeError("Evaluation must use the frozen reference policy modules")
    torch.set_num_threads(1)
    request = json.loads(Path(request_path).read_text())
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
        with torch.inference_mode():
            for item in request["requests"]:
                if file_hash(item["policy_path"]) != item["policy_sha256"]:
                    raise ValueError("Checkpoint identity changed")
                actor = load_actor(Path(item["policy_path"]))
                stats = episode_statistics(env, lambda obs: actor(np.asarray(obs).reshape(1, -1)).mode()[0].numpy()[0],
                                           item["reset_seeds"])
                record = {key: value for key, value in item.items() if key != "policy_path"}
                record.update(stats)
                records.append(record)
        atomic_json(output_path, {"environment": physics, "reference_verification": proof,
                                  "records": records, "expert": expert_report})
    finally:
        env.close()


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "_worker":
        parser = argparse.ArgumentParser(description="Internal frozen-runtime evaluation worker")
        parser.add_argument("--reference-root", type=Path, required=True)
        parser.add_argument("--request", type=Path, required=True)
        parser.add_argument("--output", type=Path, required=True)
        args = parser.parse_args(argv[1:])
        worker(args.reference_root, args.request, args.output)
        return
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--run-dir", type=Path)
    selection.add_argument("--root", type=Path, help="Discover only completed composition runs recursively")
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
            manifest = json.loads(path.read_text())
            if (manifest.get("protocol") == RUN_PROTOCOL and manifest.get("version") == 1
                    and manifest.get("status") in ("trained", "complete")):
                runs.append(path.parent)
        if not runs:
            parser.error("No completed Walker composition runs found")
    for run_dir in runs:
        result = evaluate_run(run_dir, args.checkpoint_eval_episodes, args.final_eval_episodes,
                              args.reference_root, args.reuse_eval, args.seed)
        print(result if result is not None else f"Evaluation disabled: {run_dir}")


if __name__ == "__main__":
    main()

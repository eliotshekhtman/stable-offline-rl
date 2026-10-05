"""Independent Walker-reference composition sweeps; see WALKER_COMPOSITIONS.md."""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np

import walker_sweep
from walker_composition_data import PROTOCOL, EXPERT_ACTOR_SHA256, EXPERT_SHA256, requested_quotas
from walker_reference import array_hashes


ROOT = Path(__file__).resolve().parent
DEFAULT_STORAGE = Path("/data/shekhe/stable-offline-rl/walker-reference/compositions")
COMPOSITIONS = [(1., 0.), (.75, .25), (.5, .5), (.25, .75), (0., 1.)]
TRAIN_ARGUMENTS = ("epoch", "step_per_epoch", "batch_size", "real_ratio", "rollout_length",
                   "rollout_batch_size", "penalty_coef", "actor_lr", "critic_lr")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", nargs="+", choices=("noise0.5", "noise1.0", "clean-medium-v2"),
                        default=["noise0.5", "clean-medium-v2"])
    parser.add_argument("--algos", nargs="+", choices=("mopo", "mobile"), default=["mobile"])
    parser.add_argument("--seed", nargs="+", type=int, default=[10000])
    parser.add_argument("--composition", nargs=2, type=float, action="append", metavar=("CLEAN", "OTHER"))
    parser.add_argument("--num-samples", type=int, default=999995)
    parser.add_argument("--test-fraction", type=float, default=.2)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--reference-root", type=Path, default=walker_sweep.DEFAULT_REFERENCE_ROOT)
    parser.add_argument("--storage-root", type=Path, default=DEFAULT_STORAGE)
    defaults = walker_sweep.parse_args([])
    for name in TRAIN_ARGUMENTS:
        default = getattr(defaults, name)
        parser.add_argument("--" + name.replace("_", "-"), type=type(default), default=default)
    parser.add_argument("--dynamics-max-epochs", type=int, default=None,
                        help="Reference default: MOBILE=30, MOPO=uncapped; 0 means uncapped")
    parser.add_argument("--checkpoint-eval-episodes", type=int, default=0,
                        help="Save 10%% milestones when positive; evaluate them only with --eval")
    parser.add_argument("--final-eval-episodes", type=int, default=20)
    parser.add_argument("--eval", action="store_true", help="Evaluate after training in the reference environment")
    parser.add_argument("--reuse-eval", action="store_true", help="Reuse verified matching evaluation results")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print the grid without making files or starting workers")
    args = parser.parse_args(argv)
    args.composition = args.composition or COMPOSITIONS
    args.smoke = False  # Keep the existing trainer's reference rollout frequency, including in small tests.
    for name in ("epoch", "step_per_epoch", "batch_size", "rollout_length", "rollout_batch_size", "num_samples"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("actor_lr", "critic_lr", "penalty_coef"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0 or (name != "penalty_coef" and value == 0):
            parser.error(f"Invalid --{name.replace('_', '-')}")
    if not 0 <= args.real_ratio <= 1 or not 0 <= args.test_fraction < 1:
        parser.error("Require 0 <= real-ratio <= 1 and 0 <= test-fraction < 1")
    if args.dynamics_max_epochs is not None and args.dynamics_max_epochs < 0:
        parser.error("--dynamics-max-epochs must be nonnegative")
    if min(args.checkpoint_eval_episodes, args.final_eval_episodes) < 0:
        parser.error("Evaluation episode counts must be nonnegative")
    if any(seed < 0 or seed >= 2**31 for seed in args.seed):
        parser.error("Seeds must be between 0 and 2**31-1")
    try:
        for spec in dataset_grid(args):
            requested_quotas(spec)
            if spec["setting"] == "clean-medium-v2" and requested_quotas(spec)["other"] > 999995:
                raise ValueError("Medium-v2 has only 999,995 complete-episode transitions; no replacement is allowed")
    except (ValueError, OverflowError) as error:
        parser.error(str(error))
    args.reference_root = args.reference_root.resolve()
    args.storage_root = args.storage_root.resolve()
    return args


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def dataset_grid(args):
    result = []
    for seed in args.seed:
        for setting in args.settings:
            for clean, other in args.composition:
                spec = {"protocol": PROTOCOL, "setting": "clean" if other == 0 else setting,
                        "seed": seed, "num_samples": args.num_samples,
                        "clean_fraction": float(clean), "other_fraction": float(other),
                        "test_fraction": args.test_fraction}
                if spec not in result:
                    result.append(spec)
    return result


def training_config(args, algo):
    return {**{name: getattr(args, name) for name in TRAIN_ARGUMENTS},
            "dynamics_max_epochs": walker_sweep.dynamics_cap(args, algo),
            "gamma": .99, "tau": .005, "alpha_lr": 1e-4, "target_entropy": -6,
            "rollout_freq": 1000, "model_retain_epochs": 5, "reward_normalization": False,
            "dynamics_patience": 5, "dynamics_holdout": "min(20% of training rows, 1000)",
            "model_termination": "reference termination_fn_walker2d", "chunk_length": 1}


def dataset_identity(spec, proof):
    # Grid-independent identity: clean endpoints match across independently launched families.
    needs_actor = spec["clean_fraction"] > 0 or spec["setting"].startswith("noise")
    return {"dataset_spec": spec, "environment": proof["environment"],
            "reference_verification": proof["reference_verification"],
            "expert_sha256": EXPERT_SHA256 if needs_actor else None,
            "expert_actor_sha256": EXPERT_ACTOR_SHA256 if needs_actor else None}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=".json-", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink()
            raise
    os.replace(temporary, path)


@contextmanager
def locked(path):
    with Path(path).open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def data_worker(args, operation, request, output, workdir):
    request_path, log_path = workdir / "request.json", workdir / "worker.log"
    write_json(request_path, request)
    command = [str(args.reference_root / "benchmark-python"), str(ROOT / "walker_composition_data.py"),
               "--reference-root", str(args.reference_root), operation,
               "--request", str(request_path), "--output", str(output)]
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    with log_path.open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env)
    if result.returncode:
        raise RuntimeError(f"Reference {operation} failed:\n{log_path.read_text()[-8000:]}")


def load_npz(path):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def split_summary(data, metadata):
    ids = np.unique(data["episode_ids"])
    sources = np.asarray(metadata["episode_sources"])
    return {"transitions": len(data["rewards"]), "episodes": len(ids), "episode_ids": ids.tolist(),
            "source_counts": {source: {"episodes": int((sources[ids] == source).sum()),
                                       "transitions": int((sources[data["episode_ids"]] == source).sum())}
                              for source in ("clean", "noisy", "medium")},
            "array_sha256": array_hashes(data)}


def checked_dataset(directory, expected_identity):
    metadata = json.loads((directory / "metadata.json").read_text())
    if metadata.get("cache_identity") != expected_identity or metadata.get("status") != "complete":
        raise RuntimeError(f"Dataset cache is incomplete or has a different identity: {directory}")
    for filename in ("train.npz", "test.npz"):
        if sha256(directory / filename) != metadata["file_sha256"][filename]:
            raise RuntimeError(f"Dataset cache checksum mismatch: {directory / filename}")
    return metadata


def prepare_dataset(args, spec, proof):
    key = dataset_identity(spec, proof)
    dataset_id = identity(key)
    parent = args.storage_root / "datasets"
    parent.mkdir(parents=True, exist_ok=True)
    directory = parent / dataset_id
    with locked(parent / f".{dataset_id}.lock"):
        if directory.exists():
            return directory, checked_dataset(directory, key)
        print(f"Collecting {spec['setting']} seed={spec['seed']} clean/other="
              f"{spec['clean_fraction']:g}/{spec['other_fraction']:g}", flush=True)
        with tempfile.TemporaryDirectory(prefix=f".{dataset_id}-", dir=parent) as temporary:
            workdir = Path(temporary)
            staged = workdir / "dataset"
            staged.mkdir()
            data_worker(args, "collect", spec, staged, workdir)
            metadata = json.loads((staged / "metadata.json").read_text())
            if (metadata["dataset_spec"] != spec or metadata["environment"] != proof["environment"]
                    or metadata["reference_verification"] != proof["reference_verification"]):
                raise RuntimeError("Collection provenance changed after preflight")
            dataset = load_npz(staged / "dataset.npz")
            if array_hashes(dataset) != metadata["array_sha256"]:
                raise RuntimeError("Dataset arrays changed during transfer between interpreters")
            train, test = walker_sweep.split_data(dataset, spec["test_fraction"], spec["seed"])
            metadata["split"] = {"seed": spec["seed"], "test_fraction": spec["test_fraction"],
                                 "train": split_summary(train, metadata), "test": split_summary(test, metadata)}
            for filename, part in (("train.npz", train), ("test.npz", test)):
                np.savez(staged / filename, **part)
            metadata.update(status="complete", dataset_id=dataset_id, cache_identity=key,
                            file_sha256={name: sha256(staged / name) for name in ("train.npz", "test.npz")})
            write_json(staged / "metadata.json", metadata)
            (staged / "dataset.npz").unlink()  # Our temporary unsplit duplicate only.
            staged.rename(directory)
    return directory, metadata


def run_one(args, algo, directory, metadata, provenance):
    config = training_config(args, algo)
    training_id = identity({"protocol": PROTOCOL, "algorithm": algo, "dataset_id": metadata["dataset_id"],
                            "train_array_sha256": metadata["split"]["train"]["array_sha256"],
                            "training_config": config})
    parent = args.storage_root / "runs"
    parent.mkdir(parents=True, exist_ok=True)
    run_dir = parent / training_id
    with locked(parent / f".{training_id}.lock"):
        path = run_dir / "run_manifest.json"
        if run_dir.exists():
            manifest = json.loads(path.read_text())
            if (manifest.get("status") not in ("trained", "complete")
                    or manifest.get("training_id") != training_id or manifest.get("training_config") != config):
                raise RuntimeError(f"Existing run is not a reusable completed training run: {run_dir}")
            missing = set(walker_sweep.checkpoint_epochs(args)) - {item["epoch"] for item in manifest["checkpoints"]}
            if missing:
                raise RuntimeError("This completed run has no saved checkpoints for epochs "
                                   f"{sorted(missing)}; use --checkpoint-eval-episodes 0. "
                                   "Its final policy is retained; earlier checkpoints cannot be reconstructed.")
            for checkpoint in manifest["checkpoints"]:
                if sha256(checkpoint["policy_path"]) != checkpoint["policy_sha256"]:
                    raise RuntimeError(f"Checkpoint checksum mismatch: {checkpoint['policy_path']}")
            print(f"Reusing trained {algo} seed={manifest['seed']}: {run_dir}", flush=True)
        else:
            run_dir.mkdir()
            spec = metadata["dataset_spec"]
            manifest = {"protocol": PROTOCOL, "version": 1, "status": "training", "algorithm": algo,
                        "seed": spec["seed"], "dataset_spec": spec, "dataset_id": metadata["dataset_id"],
                        "dataset_dir": str(directory), "dataset_metadata_path": str(directory / "metadata.json"),
                        "train_dataset_path": str(directory / "train.npz"), "test_dataset_path": str(directory / "test.npz"),
                        "training_config": config, "training_id": training_id, "environment": metadata["environment"],
                        "reference_root": str(args.reference_root), "storage_root": str(args.storage_root),
                        "created_utc": datetime.now(timezone.utc).isoformat(), "learner": provenance,
                        "runtime": {"device": args.device, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")},
                        "checkpoints": []}
            write_json(path, manifest)
            print(f"Training {spec['setting']} {algo} seed={spec['seed']}: {run_dir}", flush=True)
            try:
                walker_sweep.train(args, algo, spec["seed"], load_npz(directory / "train.npz"), run_dir)
                for epoch in walker_sweep.checkpoint_epochs(args) + [args.epoch]:
                    final = epoch == args.epoch
                    policy = run_dir / ("model/policy.pth" if final else f"checkpoint/step_{epoch * args.step_per_epoch}/policy.pth")
                    manifest["checkpoints"].append({"epoch": epoch, "step": epoch * args.step_per_epoch,
                                                    "final": final, "policy_path": str(policy), "policy_sha256": sha256(policy)})
                manifest["status"] = "trained"
            except BaseException as error:
                manifest.update(status="failed", error=repr(error))
                raise
            finally:
                write_json(path, manifest)
        if args.eval:
            from walker_composition_eval import evaluate_run

            evaluate_run(run_dir, checkpoint_eval_episodes=args.checkpoint_eval_episodes,
                         final_eval_episodes=args.final_eval_episodes, reference_root=args.reference_root,
                         reuse_eval=args.reuse_eval)
            manifest["status"] = "complete"
            write_json(path, manifest)
    return run_dir


def main(argv=None):
    args = parse_args(argv)
    specs = dataset_grid(args)
    if args.dry_run:
        print(json.dumps({"protocol": PROTOCOL, "storage_root": str(args.storage_root),
                          "requested_points": len(args.seed) * len(args.settings) * len(args.composition) * len(args.algos),
                          "distinct_datasets": len(specs), "distinct_training_runs": len(specs) * len(set(args.algos)),
                          "training_configs": {algo: training_config(args, algo) for algo in args.algos},
                          "datasets": [{**spec, "requested_quotas": requested_quotas(spec)} for spec in specs]}, indent=2))
        return
    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing to silently switch to CPU")
    provenance = walker_sweep.learner_provenance()
    provenance["source_sha256"][str(Path(__file__).resolve())] = sha256(__file__)
    provenance["source_sha256"][str(ROOT / "walker_composition_data.py")] = sha256(ROOT / "walker_composition_data.py")
    args.storage_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".preflight-", dir=args.storage_root) as temporary:
        workdir = Path(temporary)
        output = workdir / "preflight.json"
        data_worker(args, "preflight", {"datasets": specs}, output, workdir)
        proof = json.loads(output.read_text())
    # Preflight validates the entire grid and finite medium supply before any collection/training.
    for spec in specs:
        directory, metadata = prepare_dataset(args, spec, proof)
        for algo in dict.fromkeys(args.algos):
            run_one(args, algo, directory, metadata, provenance)


if __name__ == "__main__":
    main()

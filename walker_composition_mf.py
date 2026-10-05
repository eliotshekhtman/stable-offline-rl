"""Model-free reference Walker composition sweeps; see WALKER_COMPOSITIONS_MF.md.

This additional entrypoint leaves the running MOPO/MOBILE pipeline untouched.
Dataset collection and splitting are shared; training directories are separate.
"""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace

import numpy as np

import walker_composition as composition
import walker_sweep


ROOT = Path(__file__).resolve().parent
PROTOCOL = composition.PROTOCOL
IMPLEMENTATION = "walker-composition-mf-v1"
ALGORITHMS = ("iql", "td3bc")
IQL_ARGUMENTS = ("iql_temperature", "iql_expectile", "iql_learning_rate",
                 "iql_lr_schedule", "iql_hidden_dims")
TD3BC_ARGUMENTS = ("td3bc_learning_rate", "td3bc_alpha", "td3bc_hidden_dims")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", nargs="+", choices=("noise0.5", "noise1.0", "clean-medium-v2"),
                        default=["noise0.5", "clean-medium-v2"])
    parser.add_argument("--algos", nargs="+", choices=ALGORITHMS, default=list(ALGORITHMS))
    parser.add_argument("--seed", nargs="+", type=int, default=[10000])
    parser.add_argument("--composition", nargs=2, type=float, action="append", metavar=("CLEAN", "OTHER"))
    parser.add_argument("--num-samples", type=int, default=999995)
    parser.add_argument("--test-fraction", type=float, default=.2)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--reference-root", type=Path, default=walker_sweep.DEFAULT_REFERENCE_ROOT)
    parser.add_argument("--storage-root", type=Path, default=composition.DEFAULT_STORAGE)
    parser.add_argument("--epoch", type=int, default=3000)
    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    iql = parser.add_argument_group("IQL options (same defaults as sweep.py)")
    iql.add_argument("--iql-temperature", type=float, default=3.0,
                     help="Multiplier in exp(temperature * advantage), not its reciprocal")
    iql.add_argument("--iql-expectile", type=float, default=.7)
    iql.add_argument("--iql-learning-rate", type=float, default=3e-4)
    iql.add_argument("--iql-lr-schedule", choices=("cosine", "constant"), default="cosine")
    iql.add_argument("--iql-hidden-dims", type=int, nargs="+", default=[256, 256])
    td3bc = parser.add_argument_group("TD3BC options (same defaults as sweep.py)")
    td3bc.add_argument("--td3bc-learning-rate", type=float, default=3e-4)
    td3bc.add_argument("--td3bc-alpha", type=float, default=2.5,
                       help="Weight on Q maximization relative to behavior cloning")
    td3bc.add_argument("--td3bc-hidden-dims", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--checkpoint-eval-episodes", type=int, default=0,
                        help="Save 10%% milestones when positive; evaluate only after training with --eval")
    parser.add_argument("--final-eval-episodes", type=int, default=20)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--reuse-eval", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print the grid without files, collection, or training")
    args = parser.parse_args(argv)
    args.composition = args.composition or composition.COMPOSITIONS
    for name in ("epoch", "step_per_epoch", "batch_size", "num_samples"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("iql_temperature", "iql_learning_rate", "td3bc_learning_rate", "td3bc_alpha"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if not 0 < args.iql_expectile < 1:
        parser.error("--iql-expectile must be strictly between 0 and 1")
    if any(width <= 0 for width in args.iql_hidden_dims + args.td3bc_hidden_dims):
        parser.error("Hidden-layer widths must be positive")
    if not 0 <= args.test_fraction < 1:
        parser.error("Require 0 <= test-fraction < 1")
    if min(args.checkpoint_eval_episodes, args.final_eval_episodes) < 0:
        parser.error("Evaluation episode counts must be nonnegative")
    if any(seed < 0 or seed >= 2**31 for seed in args.seed):
        parser.error("Seeds must be between 0 and 2**31-1")
    try:
        for spec in composition.dataset_grid(args):
            quotas = composition.requested_quotas(spec)
            if spec["setting"] == "clean-medium-v2" and quotas["other"] > 999995:
                raise ValueError("Medium-v2 has only 999,995 complete-episode transitions; no replacement is allowed")
    except (ValueError, OverflowError) as error:
        parser.error(str(error))
    args.reference_root = args.reference_root.resolve()
    args.storage_root = args.storage_root.resolve()
    return args


def training_config(args, algo):
    if algo not in ALGORITHMS:
        raise ValueError(f"Unsupported model-free algorithm: {algo}")
    config = {"implementation": IMPLEMENTATION, "epoch": args.epoch,
              "step_per_epoch": args.step_per_epoch, "batch_size": args.batch_size,
              "gamma": .99, "tau": .005, "chunk_length": 1, "reward_normalization": False}
    names = IQL_ARGUMENTS if algo == "iql" else TD3BC_ARGUMENTS
    config.update({name: getattr(args, name) for name in names})
    if algo == "td3bc":
        config.update(observation_normalization="training observations only",
                      observation_normalization_epsilon=1e-3,
                      policy_noise=.2, noise_clip=.5, update_actor_freq=2)
    else:
        config.update(observation_normalization="none", advantage_weight_max=100.)
    return config


def build_policy(args, algo, data):
    import torch
    from offlinerlkit.buffer import ReplayBuffer
    from policies import build_model_free_policy

    training_config(args, algo)  # Reject unsupported algorithms before allocating replay.
    env = SimpleNamespace(spec=SimpleNamespace(id=walker_sweep.TASK),
                          observation_space=SimpleNamespace(shape=(17,)),
                          action_space=SimpleNamespace(shape=(6,), low=-np.ones(6), high=np.ones(6)))
    buffer = ReplayBuffer(len(data["rewards"]), (17,), np.float32, 6, np.float32, args.device)
    buffer.load_dataset(data)  # Copies arrays; normalization cannot mutate the shared dataset.
    policy, scheduler = build_model_free_policy(algo, env, buffer, args, discount=.99)
    if algo == "td3bc":
        # The existing scaler is not an nn.Module and is absent from state_dict().
        # Save its exact statistics in every checkpoint without changing the policy class.
        for name, values in (("observation_mean", policy.scaler.mu),
                             ("observation_std", policy.scaler.std)):
            if not np.isfinite(values).all() or (name == "observation_std" and (values <= 0).any()):
                raise ValueError("Invalid training-only observation normalization statistics")
            policy.register_buffer(name, torch.as_tensor(values, device=args.device).clone())
    return policy, buffer, scheduler


def train(args, algo, seed, data, run_dir):
    import torch
    from offlinerlkit.policy_trainer import MFPolicyTrainer
    from offlinerlkit.utils.logger import Logger

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    policy, buffer, scheduler = build_policy(args, algo, data)
    logger = Logger(str(run_dir), {"consoleout_backup": "stdout",
                                  "policy_training_progress": "csv"}, console_output=False)
    trainer_closed_logger = False
    try:
        trainer = MFPolicyTrainer(policy, buffer, logger, epoch=args.epoch,
                                  step_per_epoch=args.step_per_epoch, batch_size=args.batch_size,
                                  lr_scheduler=scheduler, checkpoint_epochs=walker_sweep.checkpoint_epochs(args),
                                  show_progress=not args.quiet)
        trainer.train()
        trainer_closed_logger = True
    finally:
        if not trainer_closed_logger:
            logger.close()


def run_one(args, algo, directory, metadata, provenance):
    config = training_config(args, algo)
    training_id = composition.identity({"protocol": PROTOCOL, "algorithm": algo,
                                        "dataset_id": metadata["dataset_id"],
                                        "train_array_sha256": metadata["split"]["train"]["array_sha256"],
                                        "training_config": config})
    parent = args.storage_root / "model_free_runs"
    parent.mkdir(parents=True, exist_ok=True)
    run_dir = parent / training_id
    spec = metadata["dataset_spec"]
    with composition.locked(parent / f".{training_id}.lock"):
        path = run_dir / "run_manifest.json"
        expected = {"protocol": PROTOCOL, "version": 1, "training_id": training_id,
                    "algorithm": algo, "seed": spec["seed"], "dataset_id": metadata["dataset_id"],
                    "dataset_spec": spec, "training_config": config, "environment": metadata["environment"]}
        if run_dir.exists():
            manifest = json.loads(path.read_text())
            if (manifest.get("status") not in ("trained", "complete")
                    or any(manifest.get(key) != value for key, value in expected.items())):
                raise RuntimeError(f"Existing run is not a reusable completed training run: {run_dir}")
            missing = set(walker_sweep.checkpoint_epochs(args)) - {item["epoch"] for item in manifest["checkpoints"]}
            if missing:
                raise RuntimeError("This completed run has no saved checkpoints for epochs "
                                   f"{sorted(missing)}; use --checkpoint-eval-episodes 0. "
                                   "Earlier checkpoints cannot be reconstructed.")
            if sum(item.get("final", False) and item["epoch"] == args.epoch
                   for item in manifest["checkpoints"]) != 1:
                raise RuntimeError("Completed run is missing its final checkpoint")
            for checkpoint in manifest["checkpoints"]:
                if composition.sha256(checkpoint["policy_path"]) != checkpoint["policy_sha256"]:
                    raise RuntimeError(f"Checkpoint checksum mismatch: {checkpoint['policy_path']}")
            print(f"Reusing trained {algo} seed={spec['seed']}: {run_dir}", flush=True)
        else:
            run_dir.mkdir()
            manifest = {**expected, "status": "training", "dataset_dir": str(directory),
                        "dataset_metadata_path": str(directory / "metadata.json"),
                        "train_dataset_path": str(directory / "train.npz"),
                        "test_dataset_path": str(directory / "test.npz"),
                        "reference_root": str(args.reference_root), "storage_root": str(args.storage_root),
                        "created_utc": datetime.now(timezone.utc).isoformat(), "learner": provenance,
                        "runtime": {"device": args.device, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")},
                        "checkpoints": []}
            composition.write_json(path, manifest)
            print(f"Training {spec['setting']} {algo} seed={spec['seed']}: {run_dir}", flush=True)
            try:
                train(args, algo, spec["seed"], composition.load_npz(directory / "train.npz"), run_dir)
                for epoch in walker_sweep.checkpoint_epochs(args) + [args.epoch]:
                    final = epoch == args.epoch
                    policy = run_dir / ("model/policy.pth" if final else f"checkpoint/step_{epoch * args.step_per_epoch}/policy.pth")
                    manifest["checkpoints"].append({"epoch": epoch, "step": epoch * args.step_per_epoch,
                                                    "final": final, "policy_path": str(policy),
                                                    "policy_sha256": composition.sha256(policy)})
                manifest["status"] = "trained"
            except BaseException as error:
                manifest.update(status="failed", error=repr(error))
                raise
            finally:
                composition.write_json(path, manifest)
        if args.eval:
            from walker_composition_mf_eval import evaluate_run

            evaluate_run(run_dir, checkpoint_eval_episodes=args.checkpoint_eval_episodes,
                         final_eval_episodes=args.final_eval_episodes, reference_root=args.reference_root,
                         reuse_eval=args.reuse_eval)
            manifest["status"] = "complete"
            composition.write_json(path, manifest)
    return run_dir


def main(argv=None):
    args = parse_args(argv)
    specs = composition.dataset_grid(args)
    if args.dry_run:
        print(json.dumps({"protocol": PROTOCOL, "storage_root": str(args.storage_root),
                          "requested_points": len(args.seed) * len(args.settings) * len(args.composition) * len(args.algos),
                          "distinct_datasets": len(specs), "distinct_training_runs": len(specs) * len(set(args.algos)),
                          "training_configs": {algo: training_config(args, algo) for algo in args.algos},
                          "datasets": [{**spec, "requested_quotas": composition.requested_quotas(spec)} for spec in specs]}, indent=2))
        return
    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing to silently switch to CPU")
    provenance = walker_sweep.learner_provenance()
    for name in ("walker_composition.py", "walker_composition_data.py", "walker_composition_eval.py",
                 "walker_composition_mf.py", "walker_composition_mf_eval.py"):
        provenance["source_sha256"][str(ROOT / name)] = composition.sha256(ROOT / name)
    args.storage_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".mf-preflight-", dir=args.storage_root) as temporary:
        workdir = Path(temporary)
        output = workdir / "preflight.json"
        composition.data_worker(args, "preflight", {"datasets": specs}, output, workdir)
        proof = json.loads(output.read_text())
    for spec in specs:
        directory, metadata = composition.prepare_dataset(args, spec, proof)
        for algo in dict.fromkeys(args.algos):
            run_one(args, algo, directory, metadata, provenance)


if __name__ == "__main__":
    main()

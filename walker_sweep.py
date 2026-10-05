"""Walker-only D4RL comparison: current learner, isolated legacy data/evaluation.

Run in mujocold. This never calls sweep.py or changes its Minari/test-split behavior.
See WALKER_REFERENCE.md for the deliberately separate experiment protocol.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import tempfile
from types import SimpleNamespace

import numpy as np
import torch

from walker_reference import SETTINGS, TASK, array_hashes


ROOT = Path(__file__).resolve().parent
DEFAULT_REFERENCE_ROOT = ROOT.parent / "walker-benchmark"
DEFAULT_MANIFEST = Path("/data/shekhe/walker-benchmark/mixtures/seed0_n500000/manifest.json")
DEFAULT_STORAGE_ROOT = Path("/data/shekhe/stable-offline-rl/walker-reference")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", nargs="+", choices=SETTINGS, default=["medium-v2"])
    parser.add_argument("--algos", nargs="+", choices=("mopo", "mobile"), default=["mobile"])
    parser.add_argument("--seed", nargs="+", type=int, default=[1])
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--storage-root", type=Path, default=DEFAULT_STORAGE_ROOT)
    parser.add_argument("--epoch", type=int, default=3000)
    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--real-ratio", type=float, default=0.05)
    parser.add_argument("--rollout-length", type=int, default=5)
    parser.add_argument("--rollout-batch-size", type=int, default=50000)
    parser.add_argument("--penalty-coef", type=float, default=0.5)
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--dynamics-max-epochs", type=int, default=None,
                        help="Default: reference MOBILE=30, MOPO=uncapped; 0 means uncapped")
    parser.add_argument("--test-fraction", type=float, default=0.0,
                        help="Optional outer whole-episode holdout; independent of dynamics holdout")
    parser.add_argument("--checkpoint-eval-episodes", type=int, default=0,
                        help="Evaluate saved 10%% milestones AFTER training; 0 disables them")
    parser.add_argument("--final-eval-episodes", type=int, default=10)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true",
                        help="Test only: first 8 episodes, 2 dynamics/policy epochs, 8 updates/epoch")
    args = parser.parse_args(argv)
    for name in ("epoch", "step_per_epoch", "batch_size", "rollout_length", "rollout_batch_size"):
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
    if args.smoke:
        args.epoch, args.step_per_epoch, args.batch_size = 2, 8, 32
        args.rollout_batch_size, args.dynamics_max_epochs = 128, 2
        args.final_eval_episodes = min(args.final_eval_episodes, 2)
        args.checkpoint_eval_episodes = min(args.checkpoint_eval_episodes, 2)
    return args


def reference_worker(args, operation, options, log_path):
    command = [str(args.reference_root / "benchmark-python"), str(ROOT / "walker_reference.py"),
               "--reference-root", str(args.reference_root), operation, *map(str, options)]
    with log_path.open("w") as log:
        try:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as error:
            # Preparation logs can be temporary; expose the cause before they disappear.
            raise RuntimeError(f"Reference worker failed:\n{log_path.read_text()[-8000:]}") from error


def split_data(dataset, fraction, seed):
    if fraction == 0:
        return dataset, {key: value[:0] for key, value in dataset.items()}
    from rollout import split_dataset

    return split_dataset(dataset, test_fraction=fraction, seed=seed)


def build_policy(args, algo):
    from policies import build_dynamics, build_model_based_policy

    # Metadata only: never construct a Gymnasium Walker for reference training/eval.
    env = SimpleNamespace(spec=SimpleNamespace(id=TASK),
                          observation_space=SimpleNamespace(shape=(17,)),
                          action_space=SimpleNamespace(shape=(6,), high=np.ones(6)))
    config = SimpleNamespace(device=args.device, epoch=args.epoch,
                             model_manipulation_settings=False,
                             model_actor_learning_rate=args.actor_lr,
                             model_critic_learning_rate=args.critic_lr,
                             mopo_penalty_coef=args.penalty_coef,
                             mobile_penalty_coef=args.penalty_coef, mobile_return_shift=0.0)
    # Match MOBILE's upstream initialization order without changing the shared builder.
    policy, _, scheduler = build_model_based_policy(
        algo, env, config, discount=0.99, build_dynamics_model=False)
    dynamics = build_dynamics(17, 6, TASK, config, hidden_dims=[200, 200, 200, 200],
                              penalty_coef=args.penalty_coef if algo == "mopo" else 0.0)
    policy.dynamics = dynamics
    return policy, dynamics, scheduler


def dynamics_cap(args, algo):
    if args.dynamics_max_epochs is not None:
        return args.dynamics_max_epochs or None
    return 30 if algo == "mobile" else None


def checkpoint_epochs(args):
    if args.checkpoint_eval_episodes == 0:
        return []
    return sorted({epoch for percent in range(10, 100, 10)
                   if (epoch := math.ceil(args.epoch * percent / 100)) < args.epoch})


def evaluation_requests(args, run_dir, seed):
    requests = [{"epoch": epoch, "episodes": args.checkpoint_eval_episodes, "seed": seed,
                 "policy_path": str(run_dir / "checkpoint" / f"step_{epoch * args.step_per_epoch}" / "policy.pth")}
                for epoch in checkpoint_epochs(args) if epoch < args.epoch]
    if args.final_eval_episodes:
        requests.append({"epoch": args.epoch, "episodes": args.final_eval_episodes, "seed": seed,
                         "policy_path": str(run_dir / "model/policy.pth")})
    return requests


def learner_provenance():
    import offlinerlkit

    library = Path(offlinerlkit.__file__).resolve().parent
    if library.parent != ROOT.parent / "OfflineRL-Kit":
        raise RuntimeError(f"Use mujocold with the current OfflineRL-Kit, not {library}")
    paths = [ROOT / name for name in ("walker_sweep.py", "walker_reference.py", "policies.py", "rollout.py")]
    paths.extend(sorted(library.rglob("*.py")))
    return {"library": str(library), "torch_version": torch.__version__, "numpy_version": np.__version__,
            "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}}


def train(args, algo, seed, data, run_dir):
    from offlinerlkit.buffer import ReplayBuffer
    from offlinerlkit.policy_trainer import MBPolicyTrainer
    from offlinerlkit.utils.logger import Logger

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    policy, dynamics, scheduler = build_policy(args, algo)
    real = ReplayBuffer(len(data["rewards"]), (17,), np.float32, 6, np.float32, args.device)
    real.load_dataset(data)
    fake = ReplayBuffer(args.rollout_batch_size * args.rollout_length * 5,
                        (17,), np.float32, 6, np.float32, args.device)
    logger = Logger(str(run_dir), {"consoleout_backup": "stdout",
                                  "policy_training_progress": "csv",
                                  "dynamics_training_progress": "csv"}, console_output=False)
    try:
        dynamics.train(real.sample_all(), logger, max_epochs=dynamics_cap(args, algo),
                       max_epochs_since_update=5)
        trainer = MBPolicyTrainer(
            policy, real, fake, logger,
            rollout_setting=(8 if args.smoke else 1000, args.rollout_batch_size, args.rollout_length),
            epoch=args.epoch, step_per_epoch=args.step_per_epoch, batch_size=args.batch_size,
            real_ratio=args.real_ratio, lr_scheduler=scheduler,
            checkpoint_epochs=checkpoint_epochs(args), show_progress=not args.quiet)
        trainer.train()
    finally:
        logger.close()


def run_one(args, algo, seed, dataset, metadata, provenance):
    train_data, test_data = split_data(dataset, args.test_fraction, seed)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = args.storage_root / ("smoke" if args.smoke else "runs") / f"{metadata['setting']}_{algo}_seed{seed}_{stamp}"
    run_dir.mkdir(parents=True)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    manifest = {"status": "training", "algorithm": algo, "seed": seed, "arguments": config,
                "dataset": metadata, "learner": provenance, "smoke_only": args.smoke,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
                "split": {"test_fraction": args.test_fraction, "seed": seed,
                          "train_transitions": len(train_data["rewards"]),
                          "test_transitions": len(test_data["rewards"]),
                          "train_episode_ids": np.unique(train_data["episode_ids"]).tolist(),
                          "test_episode_ids": np.unique(test_data["episode_ids"]).tolist(),
                          "train_array_sha256": array_hashes(train_data),
                          "test_array_sha256": array_hashes(test_data)},
                "effective_dynamics_max_epochs": dynamics_cap(args, algo),
                "fixed_parameters": {"gamma": 0.99, "tau": 0.005, "alpha_lr": 1e-4,
                                     "target_entropy": -6, "rollout_freq": 8 if args.smoke else 1000,
                                     "model_retain_epochs": 5, "reward_normalization": False,
                                     "initialization_order": "current actor/critics builder, then dynamics; "
                                                             "matches MOBILE reference order",
                                     "model_termination": "reference termination_fn_walker2d",
                                     "dynamics_patience": 5, "dynamics_holdout": "min(20% of training rows, 1000)"}}
    manifest_path = run_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Training {metadata['setting']} {algo} seed={seed}: {run_dir}", flush=True)
    try:
        train(args, algo, seed, train_data, run_dir)
        manifest["status"] = "trained"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        requests = evaluation_requests(args, run_dir, seed)
        if requests:
            request_path = run_dir / "evaluation_requests.json"
            request_path.write_text(json.dumps(requests, indent=2) + "\n")
            reference_worker(args, "evaluate", ["--requests", request_path, "--output", run_dir / "evaluation.json"],
                             run_dir / "evaluation.log")
            evaluation = json.loads((run_dir / "evaluation.json").read_text())
            if evaluation["environment"] != metadata["environment"]:
                raise RuntimeError("Evaluation physics changed since data preparation")
        manifest["status"] = "complete"
        print(f"Complete: {run_dir}", flush=True)
    except BaseException as error:
        manifest["status"], manifest["error"] = "failed", repr(error)
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    args = parse_args()
    if args.dry_run:
        print(json.dumps(vars(args), indent=2, default=str))
        print("Reference data/physics; current learner; no files created or training started.")
        return
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing to silently switch to CPU")
    provenance = learner_provenance()
    args.storage_root.mkdir(parents=True, exist_ok=True)
    for setting in args.settings:
        # Converted data is temporary, on the requested data volume, not duplicated per run.
        with tempfile.TemporaryDirectory(prefix=".walker-data-", dir=args.storage_root) as temporary:
            temporary = Path(temporary)
            reference_worker(args, "prepare", ["--setting", setting, "--manifest", args.manifest,
                                              "--output", temporary], temporary / "prepare.log")
            metadata = json.loads((temporary / "dataset.json").read_text())
            with np.load(temporary / "dataset.npz", allow_pickle=False) as saved:
                dataset = {key: saved[key] for key in saved.files}
            if array_hashes(dataset) != metadata["array_sha256"]:
                raise RuntimeError("Dataset changed while transferring between runtimes")
        if args.smoke:
            mask = np.isin(dataset["episode_ids"], np.unique(dataset["episode_ids"])[:8])
            dataset = {key: values[mask] for key, values in dataset.items()}
        for seed in args.seed:
            for algo in args.algos:
                run_one(args, algo, seed, dataset, metadata, provenance)


if __name__ == "__main__":
    main()

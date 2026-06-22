# Tasks:
# - Parse the experiment CLI and keep one run focused on one Gymnasium environment.
# - Choose the dataset source: generated rollouts or converted Minari datasets.
# - Cache/load datasets, build OfflineRL-Kit replay buffers, and launch trainers.
# - Own experiment directories, milestone checkpoints, logging, seeding, and run naming.

import argparse
import itertools
import json
import math
import random
import shutil
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

import load_offline
import rollout
from offlinerlkit.buffer import ReplayBuffer
from offlinerlkit.policy_trainer import MBPolicyTrainer, MFPolicyTrainer
from offlinerlkit.utils.logger import Logger
from dql import CLEANDIFFUSER_COMMIT, train_dql
from policies import MODEL_BASED_ALGOS, MODEL_FREE_ALGOS, build_model_based_policy, build_model_free_policy


def main() -> None:
    args = parse_args()
    expert_path = resolve_expert_path(args.expert, args.env)
    run_sweep(env_name=args.env, expert_path=expert_path, args=args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect offline datasets and train OfflineRL-Kit policies.")

    experiment = parser.add_argument_group("experiment")
    experiment.add_argument("--env", required=True, help="Gymnasium environment id to train on, e.g. HalfCheetah-v5")
    experiment.add_argument("--output-dir", default="outputs", help="Root directory for saved datasets, logs, checkpoints, and run manifests")
    experiment.add_argument("--seed", type=int, default=0, help="Random seed used for dataset splitting, generated rollouts, and training")
    experiment.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Torch device used for OfflineRL-Kit policies and dynamics")
    experiment.add_argument("--reuse-datasets", action="store_true", help="Load existing dataset splits from disk instead of recreating them")
    experiment.add_argument("--overwrite", action="store_true", help="Remove an existing run directory before training")

    dataset = parser.add_argument_group("dataset source and split")
    dataset.add_argument("--dataset-source", choices=["generated", "minari"], default="generated", help="Use generated expert/random rollouts or all matching Minari datasets for the environment")
    dataset.add_argument("--test-fraction", type=float, default=0.2, help="Fraction of each dataset held out for post-training evaluation")
    dataset.add_argument("--split-level", choices=["transition", "episode"], default="transition", help="Split train/test by individual transitions or by whole episodes")

    generated = parser.add_argument_group("generated dataset options")
    generated.add_argument("--expert", default="/home/shekhe/stable-offline-rl/experts", help="Expert policy .zip path or directory containing <env>.zip; used only for generated expert data and expert evaluation")
    generated.add_argument("--num-samples", type=int, nargs="+", default=[10000], help="Generated dataset transition counts to sweep over")
    generated.add_argument("--noise-scale", type=float, nargs="+", default=[0.0], help="Gaussian action-noise scales applied to expert actions in generated datasets")
    generated.add_argument("--prop-expert", type=float, nargs="+", default=[1.0], help="Fraction of generated transitions collected from the expert; the rest are random actions")
    generated.add_argument("--max-timesteps", type=int, default=1000, help="Maximum length of each generated rollout trajectory")

    training = parser.add_argument_group("policy training")
    training.add_argument("--algos", nargs="+", default=["cql"], help="Algorithms to train: none, bc, cql, iql, td3bc, edac, dql, mopo, combo, mobile, rambo")
    training.add_argument("--epoch", type=int, default=1000, help="Number of policy-training epochs")
    training.add_argument("--step-per-epoch", type=int, default=1000, help="Gradient-update steps per policy-training epoch")
    training.add_argument("--batch-size", type=int, default=256, help="Policy-training batch size")
    training.add_argument("--eval-episodes", type=int, default=10, help="Episodes used by OfflineRL-Kit trainer evaluation during training")
    training.add_argument("--eval", action="store_true", help="After training, run full evaluation for each trained policy and then generate plots")
    training.add_argument("--jacobian-samples", type=int, default=8, help="Evaluation-only number of dataset and rollout states used for finite-difference Jacobian metrics")
    training.add_argument("--fd-eps", type=float, default=1e-4, help="Evaluation-only central finite-difference perturbation size for Jacobian metrics")

    stability = parser.add_argument_group("stability and conservativity evaluation")
    stability.add_argument("--stability-trajectories", type=int, default=8, help="Evaluation-only number of global trajectories and local perturbed-state pairs")
    stability.add_argument("--stability-horizon", type=int, default=300, help="Evaluation-only maximum rollout steps for global and local stability trajectories")
    stability.add_argument("--global-max-offset", type=int, default=30, help="Evaluation-only maximum timestep offset for global trajectory phase alignment")
    stability.add_argument("--local-perturbation-scale", type=float, default=0.01, help="Evaluation-only local perturbation norm in standardized physical-state coordinates")
    stability.add_argument("--ood-samples", type=int, default=10000, help="Evaluation-only maximum held-out and rollout samples used for OOD metrics")

    model_based = parser.add_argument_group("model-based algorithm options")
    model_based.add_argument("--dynamics-max-epochs", type=int, default=5, help="Maximum epochs for fitting the learned dynamics model before policy training")
    model_based.add_argument("--rollout-freq", type=int, default=1000, help="Policy-training step interval between learned-dynamics rollout generation")
    model_based.add_argument("--rollout-batch-size", type=int, default=10000, help="Number of initial real states used when generating model rollouts")
    model_based.add_argument("--rollout-length", type=int, default=1, help="Number of learned-dynamics steps per synthetic rollout")
    model_based.add_argument("--model-retain-epochs", type=int, default=5, help="How many epochs of synthetic model rollouts to retain in the fake replay buffer")
    model_based.add_argument("--real-ratio", type=float, default=0.05, help="Fraction of each model-based training batch sampled from the real offline dataset rather than the synthetic rollout buffer")
    model_based.add_argument("--dynamics-update-freq", type=int, default=1000, help="RAMBO dynamics-adversary update interval; ignored by other model-based algorithms")
    model_based.add_argument("--adv-batch-size", type=int, default=256, help="RAMBO adversarial dynamics rollout batch size")
    model_based.add_argument("--adv-weight", type=float, default=3e-4, help="RAMBO adversarial dynamics loss weight")
    model_based.add_argument("--bc-epoch", type=int, default=5, help="RAMBO behavior-cloning pretraining epochs")
    model_based.add_argument("--bc-batch-size", type=int, default=256, help="RAMBO behavior-cloning pretraining batch size")
    return parser.parse_args()


def run_sweep(env_name: str, expert_path: Path, args: argparse.Namespace) -> None:
    output_root = Path(args.output_dir) / env_name
    dataset_dir = output_root / "datasets"
    run_dir = output_root / "runs"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset_source == "minari":
        run_minari_sweep(env_name=env_name, dataset_dir=dataset_dir, run_dir=run_dir, args=args)
        maybe_plot(output_root, args)
        return

    for num_samples, noise_scale, prop_expert in itertools.product(
        args.num_samples, args.noise_scale, args.prop_expert
    ):
        dataset_tag = make_dataset_tag(num_samples, noise_scale, prop_expert, args.seed)
        tag_dir = dataset_dir / dataset_tag
        if args.reuse_datasets and (tag_dir / "train.npz").exists():
            print(f"Loading dataset split: {tag_dir}")
            train_dataset = rollout.load_dataset(tag_dir / "train.npz")
            paths = split_paths(tag_dir)
        else:
            dataset, metadata = collect_generated_dataset(
                env_name=env_name,
                expert_path=expert_path,
                num_samples=num_samples,
                noise_scale=noise_scale,
                prop_expert=prop_expert,
                args=args,
            )
            train_dataset, paths = save_dataset_splits(tag_dir, dataset, metadata, args)

        train_algos(env_name, train_dataset, run_dir, dataset_tag, paths, args)

    maybe_plot(output_root, args)


def run_minari_sweep(env_name: str, dataset_dir: Path, run_dir: Path, args: argparse.Namespace) -> None:
    if args.reuse_datasets:
        dataset_ids = [
            json.loads(path.read_text(encoding="utf-8"))["dataset_id"]
            for path in sorted(dataset_dir.glob("minari_*/metadata.json"))
        ]
    else:
        dataset_ids = load_offline.list_minari_dataset_ids(env_name)

    for dataset_id in dataset_ids:
        dataset_tag = load_offline.make_minari_dataset_tag(dataset_id)
        tag_dir = dataset_dir / dataset_tag
        if args.reuse_datasets and (tag_dir / "train.npz").exists():
            print(f"Loading dataset split: {tag_dir}")
            train_dataset = rollout.load_dataset(tag_dir / "train.npz")
            paths = split_paths(tag_dir)
        else:
            dataset, metadata = load_offline.load_minari_dataset(dataset_id, seed=args.seed)
            train_dataset, paths = save_dataset_splits(tag_dir, dataset, metadata, args)

        train_algos(env_name, train_dataset, run_dir, dataset_tag, paths, args)


def collect_generated_dataset(
    env_name: str,
    expert_path: Path,
    num_samples: int,
    noise_scale: float,
    prop_expert: float,
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], dict]:
    if prop_expert > 0.0 and not expert_path.exists():
        raise FileNotFoundError(f"Expert policy not found: {expert_path}")

    print("Collecting generated dataset")
    return rollout.collect_dataset(
        env_name=env_name,
        policy_path=str(expert_path),
        max_timesteps=args.max_timesteps,
        num_samples=num_samples,
        noise_scale=noise_scale,
        prop_expert=prop_expert,
        deterministic=True,
        seed=args.seed,
    )


def save_dataset_splits(
    dataset_dir: Path,
    dataset: dict[str, np.ndarray],
    metadata: dict,
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], dict]:
    full_path = dataset_dir / "full.npz"
    train_path = dataset_dir / "train.npz"
    test_path = dataset_dir / "test.npz"
    metadata_path = dataset_dir / "metadata.json"

    dataset_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        **metadata,
        "split_level": args.split_level,
        "test_fraction": args.test_fraction,
        "full_dataset_path": str(full_path),
        "train_dataset_path": str(train_path),
        "test_dataset_path": str(test_path),
    }
    train_dataset, test_dataset = rollout.split_dataset(
        dataset,
        test_fraction=args.test_fraction,
        split_level=args.split_level,
        seed=args.seed,
    )
    rollout.save_dataset(dataset, full_path)
    rollout.save_dataset(train_dataset, train_path)
    rollout.save_dataset(test_dataset, test_path)
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, sort_keys=True)
    return train_dataset, split_paths(dataset_dir)


def train_algos(
    env_name: str,
    train_dataset: dict[str, np.ndarray],
    run_dir: Path,
    dataset_tag: str,
    paths: dict,
    args: argparse.Namespace,
) -> None:
    for algo in args.algos:
        if algo != "none":
            algo_run_dir = run_dir / f"{algo}_{dataset_tag}"
            train_algo(
                algo=algo,
                env_name=env_name,
                dataset=train_dataset,
                run_dir=algo_run_dir,
                split_paths=paths,
                args=args,
            )
            maybe_evaluate(algo_run_dir, args)


def maybe_evaluate(run_dir: Path, args: argparse.Namespace) -> None:
    if not args.eval:
        return
    from eval import evaluate_run

    evaluate_run(
        run_dir,
        argparse.Namespace(
            device=args.device,
            eval_episodes=args.eval_episodes,
            seed=args.seed,
            jacobian_samples=args.jacobian_samples,
            fd_eps=args.fd_eps,
            stability_trajectories=args.stability_trajectories,
            stability_horizon=args.stability_horizon,
            global_max_offset=args.global_max_offset,
            local_perturbation_scale=args.local_perturbation_scale,
            ood_samples=args.ood_samples,
        ),
    )


def maybe_plot(output_root: Path, args: argparse.Namespace) -> None:
    if not args.eval:
        return
    from plot import plot_root

    plot_root(output_root)


def split_paths(dataset_dir: Path) -> dict:
    return {
        "dataset_dir": str(dataset_dir),
        "full_dataset_path": str(dataset_dir / "full.npz"),
        "train_dataset_path": str(dataset_dir / "train.npz"),
        "test_dataset_path": str(dataset_dir / "test.npz"),
        "dataset_metadata_path": str(dataset_dir / "metadata.json"),
        "dataset_tag": dataset_dir.name,
    }


def train_algo(
    algo: str,
    env_name: str,
    dataset: dict[str, np.ndarray],
    run_dir: Path,
    split_paths: dict,
    args: argparse.Namespace,
) -> None:
    if algo not in MODEL_FREE_ALGOS and algo not in MODEL_BASED_ALGOS:
        raise ValueError(f"Unsupported algorithm: {algo}")

    prepare_run_dir(run_dir, args.overwrite)
    seed_everything(args.seed)

    eval_env = gym.make(env_name)
    eval_env.reset(seed=args.seed)
    eval_env.action_space.seed(args.seed)

    logger = build_logger(run_dir, args, algo, env_name)

    try:
        if algo in MODEL_FREE_ALGOS:
            buffer = build_buffer(dataset, eval_env, args.device)
            policy, lr_scheduler = build_model_free_policy(algo, eval_env, buffer, args)
            if algo != "dql":
                trainer = MFPolicyTrainer(
                    policy=policy,
                    eval_env=eval_env,
                    buffer=buffer,
                    logger=logger,
                    epoch=args.epoch,
                    step_per_epoch=args.step_per_epoch,
                    batch_size=args.batch_size,
                    eval_episodes=args.eval_episodes,
                    lr_scheduler=lr_scheduler,
                    checkpoint_epochs=checkpoint_epochs(args.epoch),
                )
        else:
            real_buffer = build_buffer(dataset, eval_env, args.device)
            obs_mean = obs_std = None
            if algo == "rambo":
                obs_mean, obs_std = real_buffer.normalize_obs()
            fake_buffer = ReplayBuffer(
                buffer_size=args.rollout_batch_size * args.rollout_length * args.model_retain_epochs,
                obs_shape=eval_env.observation_space.shape,
                obs_dtype=np.float32,
                action_dim=int(np.prod(eval_env.action_space.shape)),
                action_dtype=np.float32,
                device=args.device,
            )
            policy, dynamics, lr_scheduler = build_model_based_policy(
                algo, eval_env, args, obs_mean=obs_mean, obs_std=obs_std
            )

            print(f"Training dynamics for {algo}: {run_dir}")
            dynamics.train(real_buffer.sample_all(), logger, max_epochs=args.dynamics_max_epochs, max_epochs_since_update=5)
            if algo == "rambo":
                policy.pretrain(
                    real_buffer.sample_all(),
                    args.bc_epoch,
                    min(args.bc_batch_size, len(dataset["observations"])),
                    1e-4,
                    logger,
                )

            trainer = MBPolicyTrainer(
                policy=policy,
                eval_env=eval_env,
                real_buffer=real_buffer,
                fake_buffer=fake_buffer,
                logger=logger,
                rollout_setting=(args.rollout_freq, args.rollout_batch_size, args.rollout_length),
                epoch=args.epoch,
                step_per_epoch=args.step_per_epoch,
                batch_size=args.batch_size,
                real_ratio=args.real_ratio,
                eval_episodes=args.eval_episodes,
                lr_scheduler=lr_scheduler,
                dynamics_update_freq=args.dynamics_update_freq if algo == "rambo" else 0,
                checkpoint_epochs=checkpoint_epochs(args.epoch),
            )

        initial_checkpoint = Path(logger.checkpoint_dir) / "step_0"
        initial_checkpoint.mkdir(exist_ok=True)
        torch.save(policy.state_dict(), initial_checkpoint / "policy.pth")
        if algo in MODEL_BASED_ALGOS:
            dynamics.save(initial_checkpoint)

        print(f"Training {algo}: {run_dir}")
        if algo == "dql":
            train_dql(
                policy, buffer, logger, args.epoch, args.step_per_epoch,
                args.batch_size, checkpoint_epochs(args.epoch),
            )
        else:
            trainer.train()
        save_run_manifest(run_dir, algo, env_name, split_paths, args)
    finally:
        eval_env.close()


def build_buffer(dataset: dict[str, np.ndarray], env: gym.Env, device: str) -> ReplayBuffer:
    train_dataset = {key: dataset[key] for key in ("observations", "actions", "next_observations", "rewards", "terminals")}
    buffer = ReplayBuffer(
        buffer_size=len(train_dataset["observations"]),
        obs_shape=env.observation_space.shape,
        obs_dtype=np.float32,
        action_dim=int(np.prod(env.action_space.shape)),
        action_dtype=np.float32,
        device=device,
    )
    buffer.load_dataset(train_dataset)
    return buffer


def save_run_manifest(
    run_dir: Path,
    algo: str,
    env_name: str,
    split_paths: dict,
    args: argparse.Namespace,
) -> None:
    manifest = {
        "env_name": env_name,
        "algo": algo,
        "dataset_source": args.dataset_source,
        "model_dir": str(run_dir / "model"),
        "test_fraction": args.test_fraction,
        "split_level": args.split_level,
        "epoch": args.epoch,
        "step_per_epoch": args.step_per_epoch,
        "adv_weight": args.adv_weight,
        "adv_batch_size": args.adv_batch_size,
        "rollout_length": args.rollout_length,
        "expert": str(resolve_expert_path(args.expert, env_name)),
        "checkpoints": checkpoint_manifest(run_dir, algo, args.epoch, args.step_per_epoch),
        **split_paths,
    }
    if algo == "dql":
        manifest["cleandiffuser_commit"] = CLEANDIFFUSER_COMMIT
    with (run_dir / "run_manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, sort_keys=True)


def build_logger(run_dir: Path, args: argparse.Namespace, algo: str, env_name: str) -> Logger:
    output_config = {
        "consoleout_backup": "stdout",
        "policy_training_progress": "csv",
        "tb": "tensorboard",
    }
    logger = Logger(str(run_dir), output_config)
    logger.log_hyperparameters(
        {
            "algo": algo,
            "env": env_name,
            "seed": args.seed,
            "device": args.device,
            "epoch": args.epoch,
            "step_per_epoch": args.step_per_epoch,
            "batch_size": args.batch_size,
            "eval_episodes": args.eval_episodes,
        }
    )
    return logger


def prepare_run_dir(run_dir: Path, overwrite: bool) -> None:
    if run_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Run directory already exists. Use --overwrite to replace it: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def checkpoint_epochs(epochs: int) -> list[int]:
    return [item["epoch"] for item in checkpoint_schedule(epochs) if item["epoch"] > 0]


def checkpoint_schedule(epochs: int) -> list[dict]:
    by_epoch = {0: 0}
    for percent in (1, 5, 10, 25, 50, 75, 100):
        by_epoch[math.ceil(percent * epochs / 100)] = percent
    return [
        {
            "requested_percent": requested_percent,
            "actual_percent": 100.0 * epoch / epochs,
            "epoch": epoch,
        }
        for epoch, requested_percent in sorted(by_epoch.items())
    ]


def checkpoint_manifest(run_dir: Path, algo: str, epochs: int, steps_per_epoch: int) -> list[dict]:
    records = []
    for item in checkpoint_schedule(epochs):
        step = item["epoch"] * steps_per_epoch
        checkpoint_dir = run_dir / "checkpoint" / f"step_{step}"
        record = {
            **item,
            "step": step,
            "policy_path": str(checkpoint_dir / "policy.pth"),
        }
        if algo in MODEL_BASED_ALGOS:
            dynamics_dir = checkpoint_dir if algo == "rambo" else run_dir / "checkpoint" / "step_0"
            record["dynamics_path"] = str(dynamics_dir)
        records.append(record)
    return records


def make_dataset_tag(num_samples: int, noise_scale: float, prop_expert: float, seed: int) -> str:
    return f"samples{num_samples}_expert{prop_expert:g}_noise{noise_scale:g}_seed{seed}"


def resolve_expert_path(expert_arg: str, env_name: str) -> Path:
    expert_path = Path(expert_arg)
    if expert_path.suffix == ".zip":
        return expert_path
    return expert_path / f"{env_name}.zip"


if __name__ == "__main__":
    main()

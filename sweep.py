import argparse
import itertools
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
from policies import MODEL_BASED_ALGOS, MODEL_FREE_ALGOS, build_model_based_policy, build_model_free_policy


def main() -> None:
    args = parse_args()
    expert_path = resolve_expert_path(args.expert, args.env)
    run_sweep(env_name=args.env, expert_path=expert_path, args=args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect offline datasets and train OfflineRL-Kit policies.")
    parser.add_argument("--env", required=True, help="Gymnasium environment id, e.g. HalfCheetah-v5")
    parser.add_argument("--dataset-source", choices=["generated", "minari"], default="generated")
    parser.add_argument("--expert", default="experts", help="Expert policy .zip path or directory containing <env>.zip")
    parser.add_argument("--output-dir", default="outputs", help="Root directory for datasets and runs")
    parser.add_argument("--algos", nargs="+", default=["cql"], help="Algorithms: none, bc, cql, iql, td3bc, edac, mopo, combo, mobile, rambo")
    parser.add_argument("--num-samples", type=int, nargs="+", default=[10000], help="Dataset sample counts")
    parser.add_argument("--noise-scale", type=float, nargs="+", default=[0.0], help="Expert action noise scales")
    parser.add_argument("--prop-expert", type=float, nargs="+", default=[1.0], help="Fraction of samples from expert")
    parser.add_argument("--max-timesteps", type=int, default=1000, help="Maximum steps per collected trajectory")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--dynamics-max-epochs", type=int, default=5)
    parser.add_argument("--rollout-freq", type=int, default=1000)
    parser.add_argument("--rollout-batch-size", type=int, default=10000)
    parser.add_argument("--rollout-length", type=int, default=1)
    parser.add_argument("--model-retain-epochs", type=int, default=5)
    parser.add_argument("--real-ratio", type=float, default=0.05)
    parser.add_argument("--dynamics-update-freq", type=int, default=1000)
    parser.add_argument("--adv-batch-size", type=int, default=256)
    parser.add_argument("--adv-weight", type=float, default=3e-4)
    parser.add_argument("--bc-epoch", type=int, default=5)
    parser.add_argument("--bc-batch-size", type=int, default=256)
    parser.add_argument("--reuse-datasets", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Remove an existing run directory before training")
    return parser.parse_args()


def run_sweep(env_name: str, expert_path: Path, args: argparse.Namespace) -> None:
    output_root = Path(args.output_dir) / env_name
    dataset_dir = output_root / "datasets"
    run_dir = output_root / "runs"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset_source == "minari":
        run_minari_sweep(env_name=env_name, dataset_dir=dataset_dir, run_dir=run_dir, args=args)
        return

    for num_samples, noise_scale, prop_expert in itertools.product(
        args.num_samples, args.noise_scale, args.prop_expert
    ):
        dataset_tag = make_dataset_tag(num_samples, noise_scale, prop_expert, args.seed)
        dataset_path = dataset_dir / f"{dataset_tag}.npz"
        dataset = get_or_collect_dataset(
            env_name=env_name,
            expert_path=expert_path,
            dataset_path=dataset_path,
            num_samples=num_samples,
            noise_scale=noise_scale,
            prop_expert=prop_expert,
            args=args,
        )

        for algo in args.algos:
            if algo == "none":
                continue
            train_algo(
                algo=algo,
                env_name=env_name,
                dataset=dataset,
                run_dir=run_dir / f"{algo}_{dataset_tag}",
                args=args,
            )


def run_minari_sweep(env_name: str, dataset_dir: Path, run_dir: Path, args: argparse.Namespace) -> None:
    for dataset_id in load_offline.list_minari_dataset_ids(env_name):
        dataset_tag = load_offline.make_minari_dataset_tag(dataset_id)
        dataset_path = dataset_dir / f"{dataset_tag}.npz"
        dataset = get_or_load_minari_dataset(dataset_id=dataset_id, dataset_path=dataset_path, args=args)

        for algo in args.algos:
            if algo == "none":
                continue
            train_algo(
                algo=algo,
                env_name=env_name,
                dataset=dataset,
                run_dir=run_dir / f"{algo}_{dataset_tag}",
                args=args,
            )


def get_or_collect_dataset(
    env_name: str,
    expert_path: Path,
    dataset_path: Path,
    num_samples: int,
    noise_scale: float,
    prop_expert: float,
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    if args.reuse_datasets and dataset_path.exists():
        print(f"Loading dataset: {dataset_path}")
        return rollout.load_dataset(dataset_path)

    if prop_expert > 0.0 and not expert_path.exists():
        raise FileNotFoundError(f"Expert policy not found: {expert_path}")

    print(f"Collecting dataset: {dataset_path}")
    dataset, metadata = rollout.collect_dataset(
        env_name=env_name,
        policy_path=str(expert_path),
        max_timesteps=args.max_timesteps,
        num_samples=num_samples,
        noise_scale=noise_scale,
        prop_expert=prop_expert,
        deterministic=True,
        seed=args.seed,
    )
    rollout.save_dataset(dataset, dataset_path, metadata)
    return dataset


def get_or_load_minari_dataset(
    dataset_id: str,
    dataset_path: Path,
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    if args.reuse_datasets and dataset_path.exists():
        print(f"Loading converted Minari dataset: {dataset_path}")
        return rollout.load_dataset(dataset_path)

    print(f"Loading Minari dataset: {dataset_id}")
    dataset, metadata = load_offline.load_minari_dataset(dataset_id, seed=args.seed)
    rollout.save_dataset(dataset, dataset_path, metadata)
    return dataset


def train_algo(
    algo: str,
    env_name: str,
    dataset: dict[str, np.ndarray],
    run_dir: Path,
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
            )

        print(f"Training {algo}: {run_dir}")
        trainer.train()
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


def make_dataset_tag(num_samples: int, noise_scale: float, prop_expert: float, seed: int) -> str:
    return f"samples{num_samples}_expert{prop_expert:g}_noise{noise_scale:g}_seed{seed}"


def resolve_expert_path(expert_arg: str, env_name: str) -> Path:
    expert_path = Path(expert_arg)
    if expert_path.suffix == ".zip":
        return expert_path
    return expert_path / f"{env_name}.zip"


if __name__ == "__main__":
    main()

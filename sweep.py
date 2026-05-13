import argparse
import itertools
import json
import random
import shutil
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

import rollout
from offlinerlkit.buffer import ReplayBuffer
from offlinerlkit.dynamics import EnsembleDynamics
from offlinerlkit.modules import Actor, ActorProb, Critic, DiagGaussian, EnsembleCritic, EnsembleDynamicsModel, TanhDiagGaussian
from offlinerlkit.nets import MLP
from offlinerlkit.policy import BCPolicy, COMBOPolicy, CQLPolicy, EDACPolicy, IQLPolicy, MOBILEPolicy, MOPOPolicy, RAMBOPolicy, TD3BCPolicy
from offlinerlkit.policy_trainer import MBPolicyTrainer, MFPolicyTrainer
from offlinerlkit.utils.logger import Logger
from offlinerlkit.utils.noise import GaussianNoise
from offlinerlkit.utils.scaler import StandardScaler
from offlinerlkit.utils.termination_fns import get_termination_fn, obs_unnormalization


MODEL_FREE_ALGOS = ("bc", "cql", "iql", "td3bc", "edac")
MODEL_BASED_ALGOS = ("mopo", "combo", "mobile", "rambo")


def main() -> None:
    args = parse_args()
    for env_name in args.env:
        expert_path = resolve_expert_path(args.expert, env_name)
        run_sweep(env_name=env_name, expert_path=expert_path, args=args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect offline datasets and train OfflineRL-Kit policies.")
    parser.add_argument("--env", nargs="+", required=True, help="Gymnasium environment id(s), e.g. HalfCheetah-v5")
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
            if algo in MODEL_FREE_ALGOS:
                algo_run_dir = run_dir / f"{algo}_{dataset_tag}"
                train_model_free_algo(
                    algo=algo,
                    env_name=env_name,
                    dataset=dataset,
                    run_dir=algo_run_dir,
                    args=args,
                )
                continue
            if algo in MODEL_BASED_ALGOS:
                algo_run_dir = run_dir / f"{algo}_{dataset_tag}"
                train_model_based_algo(
                    algo=algo,
                    env_name=env_name,
                    dataset=dataset,
                    run_dir=algo_run_dir,
                    args=args,
                )
                continue
            else:
                raise ValueError(f"Unsupported algorithm for this implementation stage: {algo}")


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


def train_model_free_algo(
    algo: str,
    env_name: str,
    dataset: dict[str, np.ndarray],
    run_dir: Path,
    args: argparse.Namespace,
) -> None:
    prepare_run_dir(run_dir, args.overwrite)
    seed_everything(args.seed)

    eval_env = gym.make(env_name)
    eval_env.reset(seed=args.seed)
    eval_env.action_space.seed(args.seed)

    buffer = build_buffer(dataset, eval_env, args.device)
    policy, lr_scheduler = build_model_free_policy(algo, eval_env, buffer, args)
    logger = build_logger(run_dir, args, algo, env_name)

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
    print(f"Training {algo}: {run_dir}")
    trainer.train()
    eval_env.close()


def train_model_based_algo(
    algo: str,
    env_name: str,
    dataset: dict[str, np.ndarray],
    run_dir: Path,
    args: argparse.Namespace,
) -> None:
    prepare_run_dir(run_dir, args.overwrite)
    seed_everything(args.seed)

    eval_env = gym.make(env_name)
    eval_env.reset(seed=args.seed)
    eval_env.action_space.seed(args.seed)

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
    policy, dynamics, lr_scheduler = build_model_based_policy(algo, eval_env, args, obs_mean=obs_mean, obs_std=obs_std)
    logger = build_logger(run_dir, args, algo, env_name)

    print(f"Training dynamics for {algo}: {run_dir}")
    dynamics.train(real_buffer.sample_all(), logger, max_epochs=args.dynamics_max_epochs, max_epochs_since_update=5)

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
    if algo == "rambo":
        policy.pretrain(real_buffer.sample_all(), args.bc_epoch, args.bc_batch_size, 1e-4, logger)
    print(f"Training {algo}: {run_dir}")
    trainer.train()
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


def build_model_free_policy(
    algo: str,
    env: gym.Env,
    buffer: ReplayBuffer,
    args: argparse.Namespace,
):
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    max_action = float(env.action_space.high[0])

    if algo == "bc":
        actor_backbone = MLP(input_dim=obs_dim, hidden_dims=[256, 256])
        actor = Actor(actor_backbone, action_dim, max_action=max_action, device=args.device)
        return BCPolicy(actor, torch.optim.Adam(actor.parameters(), lr=3e-4)), None

    if algo == "cql":
        hidden_dims = [256, 256, 256]
        actor, actor_optim = build_prob_actor(obs_dim, action_dim, max_action, hidden_dims, args.device, 1e-4)
        critic1, critic1_optim = build_critic(obs_dim, action_dim, hidden_dims, args.device, 3e-4)
        critic2, critic2_optim = build_critic(obs_dim, action_dim, hidden_dims, args.device, 3e-4)
        alpha = build_auto_alpha(action_dim, args.device, 1e-4)
        return CQLPolicy(
            actor,
            critic1,
            critic2,
            actor_optim,
            critic1_optim,
            critic2_optim,
            action_space=env.action_space,
            tau=0.005,
            gamma=0.99,
            alpha=alpha,
            cql_weight=5.0,
            temperature=1.0,
            max_q_backup=False,
            deterministic_backup=True,
            with_lagrange=False,
            lagrange_threshold=10.0,
            cql_alpha_lr=3e-4,
            num_repeart_actions=10,
        ), None

    if algo == "iql":
        hidden_dims = [256, 256]
        actor_backbone = MLP(input_dim=obs_dim, hidden_dims=hidden_dims, dropout_rate=None)
        dist = DiagGaussian(
            latent_dim=getattr(actor_backbone, "output_dim"),
            output_dim=action_dim,
            unbounded=False,
            conditioned_sigma=False,
            max_mu=max_action,
        )
        actor = ActorProb(actor_backbone, dist, args.device)
        critic_q1, critic_q1_optim = build_critic(obs_dim, action_dim, hidden_dims, args.device, 3e-4)
        critic_q2, critic_q2_optim = build_critic(obs_dim, action_dim, hidden_dims, args.device, 3e-4)
        critic_v_backbone = MLP(input_dim=obs_dim, hidden_dims=hidden_dims)
        critic_v = Critic(critic_v_backbone, args.device)
        orthogonal_init(actor, critic_q1, critic_q2, critic_v)
        actor_optim = torch.optim.Adam(actor.parameters(), lr=3e-4)
        critic_v_optim = torch.optim.Adam(critic_v.parameters(), lr=3e-4)
        policy = IQLPolicy(
            actor,
            critic_q1,
            critic_q2,
            critic_v,
            actor_optim,
            critic_q1_optim,
            critic_q2_optim,
            critic_v_optim,
            action_space=env.action_space,
            tau=0.005,
            gamma=0.99,
            expectile=0.7,
            temperature=3.0,
        )
        return policy, torch.optim.lr_scheduler.CosineAnnealingLR(actor_optim, args.epoch)

    if algo == "td3bc":
        obs_mean, obs_std = buffer.normalize_obs()
        hidden_dims = [256, 256]
        actor_backbone = MLP(input_dim=obs_dim, hidden_dims=hidden_dims)
        actor = Actor(actor_backbone, action_dim, max_action=max_action, device=args.device)
        critic1, critic1_optim = build_critic(obs_dim, action_dim, hidden_dims, args.device, 3e-4)
        critic2, critic2_optim = build_critic(obs_dim, action_dim, hidden_dims, args.device, 3e-4)
        actor_optim = torch.optim.Adam(actor.parameters(), lr=3e-4)
        return TD3BCPolicy(
            actor,
            critic1,
            critic2,
            actor_optim,
            critic1_optim,
            critic2_optim,
            tau=0.005,
            gamma=0.99,
            max_action=max_action,
            exploration_noise=GaussianNoise(sigma=0.1),
            policy_noise=0.2,
            noise_clip=0.5,
            update_actor_freq=2,
            alpha=2.5,
            scaler=StandardScaler(mu=obs_mean, std=obs_std),
        ), None

    if algo == "edac":
        hidden_dims = [256, 256, 256]
        actor, actor_optim = build_prob_actor(obs_dim, action_dim, max_action, hidden_dims, args.device, 1e-4)
        critics = EnsembleCritic(obs_dim, action_dim, hidden_dims, num_ensemble=10, device=args.device)
        for layer in critics.model[::2]:
            torch.nn.init.constant_(layer.bias, 0.1)
        torch.nn.init.uniform_(critics.model[-1].weight, -3e-3, 3e-3)
        torch.nn.init.uniform_(critics.model[-1].bias, -3e-3, 3e-3)
        critics_optim = torch.optim.Adam(critics.parameters(), lr=3e-4)
        alpha = build_auto_alpha(action_dim, args.device, 1e-4)
        return EDACPolicy(
            actor,
            critics,
            actor_optim,
            critics_optim,
            tau=0.005,
            gamma=0.99,
            alpha=alpha,
            max_q_backup=False,
            deterministic_backup=False,
            eta=1.0,
        ), None

    raise ValueError(f"Unsupported algorithm: {algo}")


def build_model_based_policy(
    algo: str,
    env: gym.Env,
    args: argparse.Namespace,
    obs_mean: np.ndarray | None = None,
    obs_std: np.ndarray | None = None,
):
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    max_action = float(env.action_space.high[0])
    hidden_dims = [256, 256] if algo in ("mopo", "mobile") else [256, 256, 256]
    dynamics = build_dynamics(obs_dim, action_dim, env.spec.id, args, obs_mean=obs_mean, obs_std=obs_std)

    if algo == "mobile":
        actor, actor_optim = build_prob_actor(obs_dim, action_dim, max_action, hidden_dims, args.device, 1e-4)
        critics = torch.nn.ModuleList(
            [Critic(MLP(input_dim=obs_dim + action_dim, hidden_dims=hidden_dims), args.device) for _ in range(2)]
        )
        critics_optim = torch.optim.Adam(critics.parameters(), lr=3e-4)
        alpha = build_auto_alpha(action_dim, args.device, 1e-4)
        policy = MOBILEPolicy(
            dynamics,
            actor,
            critics,
            actor_optim,
            critics_optim,
            tau=0.005,
            gamma=0.99,
            alpha=alpha,
            penalty_coef=1.5,
            num_samples=10,
            deterministic_backup=True,
            max_q_backup=False,
        )
        return policy, dynamics, torch.optim.lr_scheduler.CosineAnnealingLR(actor_optim, args.epoch)

    actor, actor_optim = build_prob_actor(obs_dim, action_dim, max_action, hidden_dims, args.device, 1e-4)
    critic1, critic1_optim = build_critic(obs_dim, action_dim, hidden_dims, args.device, 3e-4)
    critic2, critic2_optim = build_critic(obs_dim, action_dim, hidden_dims, args.device, 3e-4)
    alpha = build_auto_alpha(action_dim, args.device, 1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(actor_optim, args.epoch)

    if algo == "mopo":
        policy = MOPOPolicy(
            dynamics,
            actor,
            critic1,
            critic2,
            actor_optim,
            critic1_optim,
            critic2_optim,
            tau=0.005,
            gamma=0.99,
            alpha=alpha,
        )
        return policy, dynamics, lr_scheduler

    if algo == "rambo":
        if obs_mean is None or obs_std is None:
            raise ValueError("RAMBO requires observation normalization statistics.")
        dynamics_adv_optim = torch.optim.Adam(dynamics.model.parameters(), lr=3e-4)
        policy = RAMBOPolicy(
            dynamics,
            actor,
            critic1,
            critic2,
            actor_optim,
            critic1_optim,
            critic2_optim,
            dynamics_adv_optim,
            tau=0.005,
            gamma=0.99,
            alpha=alpha,
            adv_weight=args.adv_weight,
            adv_rollout_length=args.rollout_length,
            adv_rollout_batch_size=args.adv_batch_size,
            include_ent_in_adv=False,
            scaler=StandardScaler(mu=obs_mean, std=obs_std),
            device=args.device,
        ).to(args.device)
        return policy, dynamics, None

    if algo == "combo":
        policy = COMBOPolicy(
            dynamics,
            actor,
            critic1,
            critic2,
            actor_optim,
            critic1_optim,
            critic2_optim,
            action_space=env.action_space,
            tau=0.005,
            gamma=0.99,
            alpha=alpha,
            cql_weight=5.0,
            temperature=1.0,
            max_q_backup=False,
            deterministic_backup=True,
            with_lagrange=False,
            lagrange_threshold=10.0,
            cql_alpha_lr=3e-4,
            num_repeart_actions=10,
            uniform_rollout=False,
            rho_s="mix",
        )
        return policy, dynamics, lr_scheduler

    raise ValueError(f"Unsupported model-based algorithm: {algo}")


def build_dynamics(
    obs_dim: int,
    action_dim: int,
    task: str,
    args: argparse.Namespace,
    obs_mean: np.ndarray | None = None,
    obs_std: np.ndarray | None = None,
) -> EnsembleDynamics:
    dynamics_model = EnsembleDynamicsModel(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dims=[200, 200, 200, 200],
        num_ensemble=7,
        num_elites=5,
        weight_decays=[2.5e-5, 5e-5, 7.5e-5, 7.5e-5, 1e-4],
        device=args.device,
    )
    dynamics_optim = torch.optim.Adam(dynamics_model.parameters(), lr=1e-3)
    termination_fn = get_termination_fn(task)
    if obs_mean is not None and obs_std is not None:
        termination_fn = obs_unnormalization(termination_fn, obs_mean, obs_std)
    return EnsembleDynamics(
        dynamics_model,
        dynamics_optim,
        StandardScaler(),
        termination_fn,
        penalty_coef=0.5,
    )


def build_prob_actor(obs_dim: int, action_dim: int, max_action: float, hidden_dims: list[int], device: str, lr: float):
    actor_backbone = MLP(input_dim=obs_dim, hidden_dims=hidden_dims)
    dist = TanhDiagGaussian(
        latent_dim=getattr(actor_backbone, "output_dim"),
        output_dim=action_dim,
        unbounded=True,
        conditioned_sigma=True,
        max_mu=max_action,
    )
    actor = ActorProb(actor_backbone, dist, device)
    return actor, torch.optim.Adam(actor.parameters(), lr=lr)


def build_critic(obs_dim: int, action_dim: int, hidden_dims: list[int], device: str, lr: float):
    critic_backbone = MLP(input_dim=obs_dim + action_dim, hidden_dims=hidden_dims)
    critic = Critic(critic_backbone, device)
    return critic, torch.optim.Adam(critic.parameters(), lr=lr)


def build_auto_alpha(action_dim: int, device: str, lr: float):
    target_entropy = -action_dim
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    return target_entropy, log_alpha, torch.optim.Adam([log_alpha], lr=lr)


def orthogonal_init(*modules) -> None:
    for module in modules:
        for layer in module.modules():
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                torch.nn.init.zeros_(layer.bias)


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

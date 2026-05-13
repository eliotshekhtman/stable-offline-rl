# Tasks:
# - Build OfflineRL-Kit policy objects for the algorithms supported by sweep.py.
# - Keep algorithm architecture choices and default coefficients in one place.
# - Build shared actor, critic, entropy-temperature, and dynamics components.
# - Avoid owning datasets, replay-buffer construction, logging, or training loops.

import argparse

import gymnasium as gym
import numpy as np
import torch

from offlinerlkit.buffer import ReplayBuffer
from offlinerlkit.dynamics import EnsembleDynamics
from offlinerlkit.modules import Actor, ActorProb, Critic, DiagGaussian, EnsembleCritic, EnsembleDynamicsModel, TanhDiagGaussian
from offlinerlkit.nets import MLP
from offlinerlkit.policy import BCPolicy, COMBOPolicy, CQLPolicy, EDACPolicy, IQLPolicy, MOBILEPolicy, MOPOPolicy, RAMBOPolicy, TD3BCPolicy
from offlinerlkit.utils.noise import GaussianNoise
from offlinerlkit.utils.scaler import StandardScaler
from offlinerlkit.utils.termination_fns import get_termination_fn, obs_unnormalization


MODEL_FREE_ALGOS = ("bc", "cql", "iql", "td3bc", "edac")
MODEL_BASED_ALGOS = ("mopo", "combo", "mobile", "rambo")

MODEL_BASED_DEFAULTS = {
    "mopo": {"hidden_dims": [256, 256], "dynamics_penalty_coef": 0.5},
    "combo": {"hidden_dims": [256, 256, 256], "dynamics_penalty_coef": 0.0, "cql_weight": 5.0},
    "mobile": {"hidden_dims": [256, 256], "dynamics_penalty_coef": 0.0, "policy_penalty_coef": 1.5},
    "rambo": {"hidden_dims": [256, 256, 256], "dynamics_penalty_coef": 0.0},
}


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
    defaults = MODEL_BASED_DEFAULTS[algo]
    hidden_dims = defaults["hidden_dims"]
    dynamics = build_dynamics(
        obs_dim,
        action_dim,
        env.spec.id,
        args,
        penalty_coef=defaults["dynamics_penalty_coef"],
        obs_mean=obs_mean,
        obs_std=obs_std,
    )

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
            penalty_coef=defaults["policy_penalty_coef"],
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
            cql_weight=defaults["cql_weight"],
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

    raise ValueError(f"Unsupported model-based algorithm: {algo}")


def build_dynamics(
    obs_dim: int,
    action_dim: int,
    task: str,
    args: argparse.Namespace,
    penalty_coef: float,
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
        penalty_coef=penalty_coef,
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

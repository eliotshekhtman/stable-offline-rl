# Tasks:
# - Build OfflineRL-Kit policy objects for the algorithms supported by sweep.py.
# - Keep algorithm architecture choices and default coefficients in one place.
# - Build shared actor, critic, entropy-temperature, and dynamics components.
# - Avoid owning datasets, replay-buffer construction, logging, or training loops.

import argparse

import gymnasium as gym
import numpy as np
import torch

from chunked_dynamics import (
    DIRECT_DYNAMICS_MODE,
    RecursiveChunkDynamics,
    resolve_dynamics_chunk_mode,
)
from offlinerlkit.buffer import ReplayBuffer
from offlinerlkit.dynamics import EnsembleDynamics
from offlinerlkit.modules import Actor, ActorProb, Critic, DiagGaussian, EnsembleCritic, EnsembleDynamicsModel, TanhDiagGaussian
from offlinerlkit.nets import MLP
from offlinerlkit.policy import BCPolicy, COMBOPolicy, CQLPolicy, EDACPolicy, IQLPolicy, MOBILEPolicy, MOPOPolicy, RAMBOPolicy, TD3BCPolicy
from offlinerlkit.utils.noise import GaussianNoise
from offlinerlkit.utils.scaler import StandardScaler
from offlinerlkit.utils.termination_fns import get_termination_fn, obs_unnormalization


TRAINABLE_MODEL_FREE_ALGOS = ("bc", "cql", "iql", "td3bc", "edac")
TRAINABLE_MODEL_BASED_ALGOS = ("mopo", "combo", "mobile")
LOADABLE_MODEL_FREE_ALGOS = (*TRAINABLE_MODEL_FREE_ALGOS, "dql")
LOADABLE_MODEL_BASED_ALGOS = (*TRAINABLE_MODEL_BASED_ALGOS, "rambo")

# Backward-compatible aliases used when interpreting existing manifests. New
# training entry points must use the explicit TRAINABLE_* registries instead.
MODEL_FREE_ALGOS = LOADABLE_MODEL_FREE_ALGOS
MODEL_BASED_ALGOS = LOADABLE_MODEL_BASED_ALGOS
ROBOMIMIC_TASKS = {"can", "lift", "square", "transport", "toolhang"}

MODEL_BASED_DEFAULTS = {
    "mopo": {"hidden_dims": [256, 256]},
    "combo": {"hidden_dims": [256, 256, 256], "dynamics_penalty_coef": 0.0, "cql_weight": 5.0},
    "mobile": {"hidden_dims": [256, 256], "dynamics_penalty_coef": 0.0},
    "rambo": {"hidden_dims": [256, 256, 256], "dynamics_penalty_coef": 0.0},
}


def build_model_free_policy(
    algo: str,
    env: gym.Env,
    buffer: ReplayBuffer,
    args: argparse.Namespace,
    discount: float,
    dql_config: dict | None = None,
):
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    max_action = float(env.action_space.high[0])

    if algo == "dql":
        # Keep old DQL checkpoints loadable without making CleanDiffuser a
        # dependency of ordinary training, evaluation, or plotting imports.
        from dql import build_dql_policy

        return build_dql_policy(
            buffer,
            action_low=env.action_space.low,
            action_high=env.action_space.high,
            total_steps=args.epoch * args.step_per_epoch,
            device=args.device,
            discount=discount,
            config=dql_config,
        ), None

    if algo == "bc":
        actor_backbone = MLP(input_dim=obs_dim, hidden_dims=[256, 256])
        actor = Actor(actor_backbone, action_dim, max_action=max_action, device=args.device)
        return BCPolicy(actor, torch.optim.Adam(actor.parameters(), lr=3e-4)), None

    if algo == "cql":
        # Use robomimic's published low-dimensional CQL policy defaults for manipulation tasks.
        is_robomimic = env.spec.id.lower() in ROBOMIMIC_TASKS
        hidden_dims = [300, 400] if is_robomimic else [256, 256, 256]
        actor_lr = 3e-4 if is_robomimic else 1e-4
        critic_lr = 1e-3 if is_robomimic else 3e-4
        actor, actor_optim = build_prob_actor(obs_dim, action_dim, max_action, hidden_dims, args.device, actor_lr)
        critic1, critic1_optim = build_critic(obs_dim, action_dim, hidden_dims, args.device, critic_lr)
        critic2, critic2_optim = build_critic(obs_dim, action_dim, hidden_dims, args.device, critic_lr)
        if is_robomimic:
            for head in (actor.dist_net.mu, actor.dist_net.sigma):
                torch.nn.init.uniform_(head.weight, -1e-3, 1e-3)
                torch.nn.init.uniform_(head.bias, -1e-3, 1e-3)
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
            gamma=discount,
            alpha=alpha,
            cql_weight=1.0 if is_robomimic else 5.0,
            temperature=1.0,
            max_q_backup=False,
            deterministic_backup=True,
            with_lagrange=is_robomimic,
            lagrange_threshold=5.0 if is_robomimic else 10.0,
            cql_alpha_lr=1e-3 if is_robomimic else 3e-4,
            num_repeart_actions=10,
        ), None

    if algo == "iql":
        hidden_dims = args.iql_hidden_dims
        actor_backbone = MLP(input_dim=obs_dim, hidden_dims=hidden_dims, dropout_rate=None)
        dist = DiagGaussian(
            latent_dim=getattr(actor_backbone, "output_dim"),
            output_dim=action_dim,
            unbounded=False,
            conditioned_sigma=False,
            max_mu=max_action,
        )
        actor = ActorProb(actor_backbone, dist, args.device)
        critic_q1, critic_q1_optim = build_critic(obs_dim, action_dim, hidden_dims, args.device, args.iql_learning_rate)
        critic_q2, critic_q2_optim = build_critic(obs_dim, action_dim, hidden_dims, args.device, args.iql_learning_rate)
        critic_v_backbone = MLP(input_dim=obs_dim, hidden_dims=hidden_dims)
        critic_v = Critic(critic_v_backbone, args.device)
        orthogonal_init(actor, critic_q1, critic_q2, critic_v)
        actor_optim = torch.optim.Adam(actor.parameters(), lr=args.iql_learning_rate)
        critic_v_optim = torch.optim.Adam(critic_v.parameters(), lr=args.iql_learning_rate)
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
            gamma=discount,
            expectile=args.iql_expectile,
            temperature=args.iql_temperature,
        )
        scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(actor_optim, args.epoch)
            if args.iql_lr_schedule == "cosine" else None
        )
        return policy, scheduler

    if algo == "td3bc":
        obs_mean, obs_std = buffer.normalize_obs()
        hidden_dims = args.td3bc_hidden_dims
        actor_backbone = MLP(input_dim=obs_dim, hidden_dims=hidden_dims)
        actor = Actor(actor_backbone, action_dim, max_action=max_action, device=args.device)
        critic1, critic1_optim = build_critic(obs_dim, action_dim, hidden_dims, args.device, args.td3bc_learning_rate)
        critic2, critic2_optim = build_critic(obs_dim, action_dim, hidden_dims, args.device, args.td3bc_learning_rate)
        actor_optim = torch.optim.Adam(actor.parameters(), lr=args.td3bc_learning_rate)
        return TD3BCPolicy(
            actor,
            critic1,
            critic2,
            actor_optim,
            critic1_optim,
            critic2_optim,
            tau=0.005,
            gamma=discount,
            max_action=max_action,
            exploration_noise=GaussianNoise(sigma=0.1),
            policy_noise=0.2,
            noise_clip=0.5,
            update_actor_freq=2,
            alpha=args.td3bc_alpha,
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
            gamma=discount,
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
    discount: float,
    obs_mean: np.ndarray | None = None,
    obs_std: np.ndarray | None = None,
    chunk_length: int = 1,
    base_discount: float = 0.99,
    dynamics_chunk_mode: str = DIRECT_DYNAMICS_MODE,
    primitive_action_dim: int | None = None,
):
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    effective_dynamics_mode = resolve_dynamics_chunk_mode(
        dynamics_chunk_mode, chunk_length
    )
    if effective_dynamics_mode != DIRECT_DYNAMICS_MODE:
        if algo == "rambo":
            raise ValueError("RAMBO does not support recursive chunk dynamics")
        if primitive_action_dim is None:
            if action_dim % chunk_length:
                raise ValueError(
                    f"Chunk action dimension {action_dim} is not divisible by chunk length {chunk_length}"
                )
            primitive_action_dim = action_dim // chunk_length
        if (
            isinstance(primitive_action_dim, bool)
            or not isinstance(primitive_action_dim, (int, np.integer))
            or primitive_action_dim < 1
            or int(primitive_action_dim) * chunk_length != action_dim
        ):
            raise ValueError(
                f"Recursive primitive action dimension {primitive_action_dim!r} "
                f"with chunk length {chunk_length} must reproduce macro action "
                f"dimension {action_dim}"
            )
        expected_discount = base_discount**chunk_length
        if not (
            np.isfinite(discount)
            and np.isfinite(expected_discount)
            and np.isclose(discount, expected_discount, rtol=1e-9, atol=1e-12)
        ):
            raise ValueError(
                f"Recursive macro discount must equal base_discount ** chunk_length "
                f"({expected_discount!r}), got {discount!r}"
            )
        primitive_action_dim = int(primitive_action_dim)
        dynamics_action_dim = primitive_action_dim
    else:
        dynamics_action_dim = action_dim
    max_action = float(env.action_space.high[0])
    defaults = MODEL_BASED_DEFAULTS[algo]
    manipulation_settings = args.model_manipulation_settings and algo in {"mopo", "mobile"}
    hidden_dims = [256, 256, 256] if manipulation_settings else defaults["hidden_dims"]
    dynamics_hidden_dims = [400, 400, 400, 400] if manipulation_settings else [200, 200, 200, 200]
    actor_lr = getattr(args, "model_actor_learning_rate", None)
    if actor_lr is None:
        actor_lr = 3e-5 if manipulation_settings else 1e-4
    critic_lr = args.model_critic_learning_rate
    dynamics_penalty_coef = args.mopo_penalty_coef if algo == "mopo" else defaults["dynamics_penalty_coef"]
    dynamics = build_dynamics(
        obs_dim,
        dynamics_action_dim,
        env.spec.id,
        args,
        hidden_dims=dynamics_hidden_dims,
        penalty_coef=dynamics_penalty_coef,
        obs_mean=obs_mean,
        obs_std=obs_std,
        chunk_length=chunk_length,
        base_discount=base_discount,
        dynamics_chunk_mode=effective_dynamics_mode,
    )

    if algo == "mobile":
        actor, actor_optim = build_prob_actor(obs_dim, action_dim, max_action, hidden_dims, args.device, actor_lr)
        critics = torch.nn.ModuleList(
            [
                Critic(MLP(input_dim=obs_dim + action_dim, hidden_dims=hidden_dims), args.device)
                for _ in range(10 if manipulation_settings else 2)
            ]
        )
        return_shift = args.mobile_return_shift if env.spec.id == "Reacher-v5" else 0.0
        if return_shift:
            with torch.no_grad():
                for critic in critics:
                    critic.last.bias.add_(return_shift)
        critics_optim = torch.optim.Adam(critics.parameters(), lr=critic_lr)
        alpha = build_auto_alpha(action_dim, args.device, 1e-4)
        policy = MOBILEPolicy(
            dynamics,
            actor,
            critics,
            actor_optim,
            critics_optim,
            tau=0.005,
            gamma=discount,
            alpha=alpha,
            penalty_coef=args.mobile_penalty_coef,
            num_samples=10,
            deterministic_backup=True,
            max_q_backup=manipulation_settings,
            clamp_target_q=True,
            return_shift=return_shift,
        )
        return policy, dynamics, torch.optim.lr_scheduler.CosineAnnealingLR(actor_optim, args.epoch)

    actor, actor_optim = build_prob_actor(obs_dim, action_dim, max_action, hidden_dims, args.device, actor_lr)
    critic1, critic1_optim = build_critic(obs_dim, action_dim, hidden_dims, args.device, critic_lr)
    critic2, critic2_optim = build_critic(obs_dim, action_dim, hidden_dims, args.device, critic_lr)
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
            gamma=discount,
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
            gamma=discount,
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
            gamma=discount,
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
    hidden_dims: list[int],
    penalty_coef: float,
    obs_mean: np.ndarray | None = None,
    obs_std: np.ndarray | None = None,
    chunk_length: int = 1,
    base_discount: float = 0.99,
    dynamics_chunk_mode: str = DIRECT_DYNAMICS_MODE,
) -> EnsembleDynamics:
    effective_dynamics_mode = resolve_dynamics_chunk_mode(
        dynamics_chunk_mode, chunk_length
    )
    dynamics_model = EnsembleDynamicsModel(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dims=hidden_dims,
        num_ensemble=7,
        num_elites=5,
        weight_decays=[2.5e-5, 5e-5, 7.5e-5, 7.5e-5, 1e-4],
        device=args.device,
    )
    dynamics_optim = torch.optim.Adam(dynamics_model.parameters(), lr=1e-3)
    task = task.lower()
    if task in ROBOMIMIC_TASKS or task == "reacher-v5":
        termination_fn = termination_fn_never
    elif task == "inverteddoublependulum-v5":
        termination_fn = termination_fn_inverted_double_pendulum
    else:
        termination_fn = get_termination_fn(task)
    if obs_mean is not None and obs_std is not None:
        termination_fn = obs_unnormalization(termination_fn, obs_mean, obs_std)
    if effective_dynamics_mode == DIRECT_DYNAMICS_MODE:
        return EnsembleDynamics(
            dynamics_model,
            dynamics_optim,
            StandardScaler(),
            termination_fn,
            penalty_coef=penalty_coef,
        )
    return RecursiveChunkDynamics(
        dynamics_model,
        dynamics_optim,
        StandardScaler(),
        termination_fn,
        chunk_length=chunk_length,
        primitive_action_dim=action_dim,
        discount=base_discount,
        penalty_coef=penalty_coef,
    )


def termination_fn_never(obs, act, next_obs):
    return np.zeros((len(obs), 1), dtype=bool)


def termination_fn_inverted_double_pendulum(obs, act, next_obs):
    sin_first, sin_second = next_obs[:, 1], next_obs[:, 2]
    cos_first, cos_second = next_obs[:, 3], next_obs[:, 4]
    # Gymnasium terminates at tip height <= 1; recover that height from the observed angles.
    tip_height = 0.6 * (
        cos_first + cos_first * cos_second - sin_first * sin_second
    )
    return (tip_height <= 1.0)[:, None]


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

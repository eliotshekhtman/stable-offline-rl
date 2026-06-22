# Tasks:
# - Reload trained OfflineRL-Kit runs from their run_manifest.json files.
# - Evaluate learned policies and configured expert policies in the true environment.
# - For model-based runs, evaluate learned next-state prediction MSE on held-out data.
# - For model-based runs, evaluate learned next-state prediction MSE along policy rollouts.
# - Measure empirical global/local trajectory convergence and dataset conservativity.

import argparse
import json
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
import torch

import metrics
import rollout
from policies import MODEL_BASED_ALGOS, build_model_based_policy, build_model_free_policy
from sweep import build_buffer


def main() -> None:
    args = parse_args()
    evaluate_run(args.run_dir, args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained stable-offline-rl runs.")

    run = parser.add_argument_group("run")
    run.add_argument("--run-dir", type=Path, required=True, help="Directory containing run_manifest.json and the trained model files")
    run.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Torch device used to reload the trained policy and dynamics")
    run.add_argument("--seed", type=int, default=0, help="Random seed for evaluation rollouts and sampled Jacobian states")

    rollout_eval = parser.add_argument_group("rollout evaluation")
    rollout_eval.add_argument("--eval-episodes", type=int, default=10, help="Number of true-environment episodes used to estimate policy and expert returns")

    jacobian_eval = parser.add_argument_group("model-based jacobian evaluation")
    jacobian_eval.add_argument("--jacobian-samples", type=int, default=8, help="Number of held-out dataset states and policy-rollout states used for finite-difference Jacobian evaluation")
    jacobian_eval.add_argument("--fd-eps", type=float, default=1e-4, help="Central finite-difference perturbation size for closed-loop Jacobian estimates")

    stability = parser.add_argument_group("stability and conservativity evaluation")
    stability.add_argument("--stability-trajectories", type=int, default=8, help="Number of global trajectories and local perturbed-state pairs used for stability metrics")
    stability.add_argument("--stability-horizon", type=int, default=300, help="Maximum rollout steps used for each global and local stability trajectory")
    stability.add_argument("--global-max-offset", type=int, default=30, help="Largest positive or negative timestep offset considered when phase-aligning global trajectories")
    stability.add_argument("--local-perturbation-scale", type=float, default=0.01, help="Initial local-state perturbation norm in training-standardized physical-state coordinates")
    stability.add_argument("--ood-samples", type=int, default=10000, help="Maximum held-out and policy-rollout samples used for state and state-action OOD metrics")
    return parser.parse_args()


def evaluate_run(run_dir: Path, args: argparse.Namespace) -> None:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    with (run_dir / "run_manifest.json").open("r", encoding="utf-8") as file:
        manifest = json.load(file)
    train_dataset = rollout.load_dataset(manifest["train_dataset_path"])
    test_dataset = rollout.load_dataset(manifest["test_dataset_path"])
    policy, dynamics, obs_mean, obs_std = load_policy_and_dynamics(
        manifest,
        args.device,
        Path(manifest["model_dir"]) / "policy.pth",
        Path(manifest["model_dir"]) if manifest["algo"] in MODEL_BASED_ALGOS else None,
        train_dataset,
    )
    rollout_info, global_stability, local_stability, conservativity = evaluate_policy_behavior(
        policy, manifest["env_name"], train_dataset, test_dataset, args,
        dynamics=dynamics, obs_mean=obs_mean, obs_std=obs_std,
    )
    expert_info = evaluate_expert(
        env_name=manifest["env_name"],
        expert_path=Path(manifest["expert"]),
        episodes=args.eval_episodes,
        seed=args.seed,
    )

    eval_dir = run_dir / "eval"
    eval_dir.mkdir(exist_ok=True)
    results = {
        "env_name": manifest["env_name"],
        "algo": manifest["algo"],
        "dataset_tag": manifest["dataset_tag"],
        "policy_return_mean": float(np.mean(rollout_info["returns"])),
        "policy_return_std": float(np.std(rollout_info["returns"])),
        "expert_return_mean": float(np.mean(expert_info["returns"])),
        "expert_return_std": float(np.std(expert_info["returns"])),
    }
    np.savez_compressed(
        eval_dir / "returns.npz",
        policy_episode_returns=rollout_info["returns"],
        expert_episode_returns=expert_info["returns"],
    )

    np.savez_compressed(eval_dir / "global_stability.npz", **global_stability)
    np.savez_compressed(eval_dir / "local_stability.npz", **local_stability)
    np.savez_compressed(eval_dir / "conservativity.npz", **conservativity)
    results.update(
        global_stability_c=float(global_stability["c"]),
        global_stability_rho=float(global_stability["rho"]),
        local_stability_c=float(local_stability["c"]),
        local_stability_rho=float(local_stability["rho"]),
        state_ood_ratio=float(conservativity["state_ood_ratio"]),
        state_action_ood_ratio=float(conservativity["state_action_ood_ratio"]),
    )

    if dynamics is not None:
        dataset_errors = evaluate_dynamics_on_dataset(
            dynamics=dynamics,
            dataset=test_dataset,
            obs_mean=obs_mean,
            obs_std=obs_std,
        )
        np.savez_compressed(eval_dir / "dynamics_dataset.npz", next_obs_sq_error=dataset_errors)
        np.savez_compressed(
            eval_dir / "dynamics_rollout.npz",
            next_obs_sq_error=rollout_info["next_obs_sq_error"],
            episode_ids=rollout_info["dynamics_episode_ids"],
            timesteps=rollout_info["dynamics_timesteps"],
        )
        results["dataset_next_obs_mse"] = float(np.mean(dataset_errors))
        results["rollout_next_obs_mse"] = float(np.mean(rollout_info["next_obs_sq_error"]))
        dataset_jacobians = evaluate_jacobians_on_observations(
            policy=policy,
            dynamics=dynamics,
            env_name=manifest["env_name"],
            observations=test_dataset["observations"],
            sample_count=args.jacobian_samples,
            seed=args.seed,
            fd_eps=args.fd_eps,
            obs_mean=obs_mean,
            obs_std=obs_std,
        )
        rollout_jacobians = evaluate_jacobians_on_observations(
            policy=policy,
            dynamics=dynamics,
            env_name=manifest["env_name"],
            observations=rollout_info["dynamics_observations"],
            sample_count=args.jacobian_samples,
            seed=args.seed,
            fd_eps=args.fd_eps,
            obs_mean=obs_mean,
            obs_std=obs_std,
        )
        np.savez_compressed(eval_dir / "jacobian_dataset.npz", **dataset_jacobians)
        np.savez_compressed(eval_dir / "jacobian_rollout.npz", **rollout_jacobians)
        results["dataset_closed_loop_jacobian_mse"] = float(np.mean(dataset_jacobians["closed_loop_jacobian_sq_error"]))
        results["rollout_closed_loop_jacobian_mse"] = float(np.mean(rollout_jacobians["closed_loop_jacobian_sq_error"]))

    with (eval_dir / "results.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, sort_keys=True)
    evaluate_history(manifest, train_dataset, test_dataset, expert_info, eval_dir, args)
    print(f"Saved evaluation: {eval_dir}")


def load_policy_and_dynamics(
    manifest: dict,
    device: str,
    policy_path: Path,
    dynamics_path: Path | None,
    train_dataset: dict[str, np.ndarray],
):
    env = gym.make(manifest["env_name"])
    buffer = build_buffer(train_dataset, env, device)
    build_args = argparse.Namespace(
        device=device,
        epoch=manifest["epoch"],
        step_per_epoch=manifest.get("step_per_epoch", 1),
        adv_weight=manifest["adv_weight"],
        rollout_length=manifest["rollout_length"],
        adv_batch_size=manifest["adv_batch_size"],
    )

    obs_mean = obs_std = None
    if manifest["algo"] in MODEL_BASED_ALGOS:
        if manifest["algo"] == "rambo":
            obs_mean, obs_std = buffer.normalize_obs()
        policy, dynamics, _ = build_model_based_policy(
            manifest["algo"], env, build_args, obs_mean=obs_mean, obs_std=obs_std
        )
        dynamics.load(str(dynamics_path))
    else:
        policy, _ = build_model_free_policy(manifest["algo"], env, buffer, build_args)
        dynamics = None

    policy.load_state_dict(torch.load(policy_path, map_location=device, weights_only=True))
    policy.eval()
    env.close()
    return policy, dynamics, obs_mean, obs_std


def evaluate_policy_behavior(
    policy,
    env_name: str,
    train_dataset: dict[str, np.ndarray],
    test_dataset: dict[str, np.ndarray],
    args: argparse.Namespace,
    dynamics=None,
    obs_mean: np.ndarray | None = None,
    obs_std: np.ndarray | None = None,
):
    rollout_info = evaluate_policy_rollouts(
        policy, env_name, args.eval_episodes, args.seed,
        dynamics=dynamics, obs_mean=obs_mean, obs_std=obs_std,
    )
    global_stability = evaluate_global_stability(
        policy, env_name, train_dataset, args.stability_trajectories,
        args.stability_horizon, args.global_max_offset, args.seed,
    )
    local_stability = evaluate_local_stability(
        policy, env_name, train_dataset, test_dataset,
        args.stability_trajectories, args.stability_horizon,
        args.local_perturbation_scale, args.seed,
    )
    conservativity = evaluate_conservativity(
        train_dataset, test_dataset, rollout_info["observations"],
        rollout_info["actions"], args.ood_samples, args.seed,
    )
    return rollout_info, global_stability, local_stability, conservativity


def evaluate_history(
    manifest: dict,
    train_dataset: dict[str, np.ndarray],
    test_dataset: dict[str, np.ndarray],
    expert_info: dict[str, np.ndarray],
    eval_dir: Path,
    args: argparse.Namespace,
) -> None:
    records = []
    first_checkpoint = manifest["checkpoints"][0]
    first_dynamics_path = Path(first_checkpoint["dynamics_path"]) if "dynamics_path" in first_checkpoint else None
    policy, dynamics, _, _ = load_policy_and_dynamics(
        manifest, args.device, Path(first_checkpoint["policy_path"]),
        first_dynamics_path, train_dataset,
    )
    for checkpoint_index, checkpoint in enumerate(manifest["checkpoints"]):
        seed_policy_randomness(args.seed)
        if checkpoint_index > 0:
            policy.load_state_dict(
                torch.load(Path(checkpoint["policy_path"]), map_location=args.device, weights_only=True)
            )
            if dynamics is not None:
                dynamics.load(checkpoint["dynamics_path"])
            policy.eval()
        rollout_info, global_stability, local_stability, conservativity = evaluate_policy_behavior(
            policy, manifest["env_name"], train_dataset, test_dataset, args
        )
        records.append(
            {
                "requested_percent": checkpoint["requested_percent"],
                "actual_percent": checkpoint["actual_percent"],
                "step": checkpoint["step"],
                "policy_return_mean": float(rollout_info["returns"].mean()),
                "policy_return_std": float(rollout_info["returns"].std()),
                "global_stability_c": float(global_stability["c"]),
                "global_stability_rho": float(global_stability["rho"]),
                "global_survival_fraction": float(global_stability["support"][-1] / global_stability["support"][0]),
                "local_stability_c": float(local_stability["c"]),
                "local_stability_rho": float(local_stability["rho"]),
                "local_survival_fraction": float(local_stability["support"][-1] / local_stability["support"][0]),
                "state_ood_ratio": float(conservativity["state_ood_ratio"]),
                "state_action_ood_ratio": float(conservativity["state_action_ood_ratio"]),
            }
        )

    history = {
        "env_name": manifest["env_name"],
        "algo": manifest["algo"],
        "dataset_tag": manifest["dataset_tag"],
        "expert_return_mean": float(expert_info["returns"].mean()),
        "records": records,
    }
    with (eval_dir / "history.json").open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=2, sort_keys=True)
    np.savez_compressed(
        eval_dir / "history.npz",
        **{key: np.asarray([record[key] for record in records]) for key in records[0]},
    )


def evaluate_policy_rollouts(
    policy,
    env_name: str,
    episodes: int,
    seed: int,
    dynamics=None,
    obs_mean: np.ndarray | None = None,
    obs_std: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    env = gym.make(env_name)
    returns = []
    observations, actions = [], []
    next_obs_errors, error_observations, error_episode_ids, error_timesteps = [], [], [], []

    try:
        for episode in range(episodes):
            obs, _ = env.reset(seed=seed + episode)
            episode_return = 0.0
            episode_length = 0
            terminated = truncated = False

            while not (terminated or truncated):
                action = policy.select_action(obs.reshape(1, -1), deterministic=True).reshape(-1)
                observations.append(np.asarray(obs, dtype=np.float32).copy())
                actions.append(np.asarray(action, dtype=np.float32).copy())
                next_obs, reward, terminated, truncated, _ = env.step(action)
                if dynamics is not None:
                    error_observations.append(np.asarray(obs, dtype=np.float32).copy())
                    pred_next_obs = predict_next_obs(
                        dynamics,
                        obs.reshape(1, -1),
                        action.reshape(1, -1),
                        obs_mean=obs_mean,
                        obs_std=obs_std,
                    )[0]
                    next_obs_errors.append(float(np.sum((pred_next_obs - next_obs) ** 2)))
                    error_episode_ids.append(episode)
                    error_timesteps.append(episode_length)

                episode_return += float(reward)
                episode_length += 1
                obs = next_obs

            returns.append(episode_return)
    finally:
        env.close()

    return {
        "returns": np.asarray(returns, dtype=np.float32),
        "observations": np.asarray(observations, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "next_obs_sq_error": np.asarray(next_obs_errors, dtype=np.float32),
        "dynamics_observations": np.asarray(error_observations, dtype=np.float32),
        "dynamics_episode_ids": np.asarray(error_episode_ids, dtype=np.int64),
        "dynamics_timesteps": np.asarray(error_timesteps, dtype=np.int64),
    }


def evaluate_expert(env_name: str, expert_path: Path, episodes: int, seed: int) -> dict[str, np.ndarray]:
    policy = rollout.load_expert_policy(env_name, str(expert_path))
    env = gym.make(env_name)
    returns = []
    try:
        for episode in range(episodes):
            obs, _ = env.reset(seed=seed + episode)
            episode_return = 0.0
            terminated = truncated = False

            while not (terminated or truncated):
                action, _ = policy.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_return += float(reward)

            returns.append(episode_return)
    finally:
        env.close()

    return {"returns": np.asarray(returns, dtype=np.float32)}


def evaluate_global_stability(
    policy,
    env_name: str,
    train_dataset: dict[str, np.ndarray],
    trajectory_count: int,
    horizon: int,
    max_offset: int,
    seed: int,
) -> dict[str, np.ndarray]:
    env = gym.make(env_name)
    columns = reconstructible_observation_columns(env)
    state_std = train_dataset["observations"][:, columns].std(axis=0)
    state_std[state_std == 0.0] = 1.0
    trajectories = []
    try:
        for index in range(trajectory_count):
            obs, _ = env.reset(seed=seed + index)
            seed_policy_randomness(seed + 100000)
            trajectories.append(rollout_state_trajectory(env, policy, obs, columns, state_std, horizon))
    finally:
        env.close()

    offsets, curves = [], []
    for first in range(len(trajectories)):
        for second in range(first + 1, len(trajectories)):
            offset, distances = metrics.align_trajectory_pair(
                trajectories[first], trajectories[second], max_offset
            )
            offsets.append(offset)
            curves.append(distances)

    c, rho, envelope, support = metrics.fit_empirical_bound(curves)
    return {
        "c": np.asarray(c, dtype=np.float32),
        "rho": np.asarray(rho, dtype=np.float32),
        "distance_curves": pad_curves(curves),
        "envelope": envelope,
        "support": support,
        "offsets": np.asarray(offsets, dtype=np.int64),
        "trajectory_lengths": np.asarray([len(item) for item in trajectories], dtype=np.int64),
        "state_columns": columns,
    }


def evaluate_local_stability(
    policy,
    env_name: str,
    train_dataset: dict[str, np.ndarray],
    test_dataset: dict[str, np.ndarray],
    pair_count: int,
    horizon: int,
    perturbation_scale: float,
    seed: int,
) -> dict[str, np.ndarray]:
    env = gym.make(env_name)
    columns = reconstructible_observation_columns(env)
    state_std = train_dataset["observations"][:, columns].std(axis=0)
    state_std[state_std == 0.0] = 1.0
    rng = np.random.default_rng(seed)
    pair_count = min(pair_count, len(test_dataset["observations"]))
    indices = rng.choice(len(test_dataset["observations"]), size=pair_count, replace=False)
    curves = []
    base_lengths, perturbed_lengths = [], []

    try:
        env.reset(seed=seed)
        for pair_index, dataset_index in enumerate(indices):
            base_obs = test_dataset["observations"][dataset_index].copy()
            direction = rng.normal(size=len(columns))
            direction /= np.linalg.norm(direction)
            perturbed_obs = base_obs.copy()
            perturbed_obs[columns] += perturbation_scale * state_std * direction

            seed_policy_randomness(seed + 100000 + pair_index)
            base = rollout_state_trajectory(env, policy, base_obs, columns, state_std, horizon)
            seed_policy_randomness(seed + 100000 + pair_index)
            perturbed = rollout_state_trajectory(env, policy, perturbed_obs, columns, state_std, horizon)
            overlap = min(len(base), len(perturbed))
            curves.append(np.linalg.norm(base[:overlap] - perturbed[:overlap], axis=1).astype(np.float32))
            base_lengths.append(len(base))
            perturbed_lengths.append(len(perturbed))
    finally:
        env.close()

    c, rho, envelope, support = metrics.fit_empirical_bound(curves)
    return {
        "c": np.asarray(c, dtype=np.float32),
        "rho": np.asarray(rho, dtype=np.float32),
        "distance_curves": pad_curves(curves),
        "envelope": envelope,
        "support": support,
        "sample_indices": indices.astype(np.int64),
        "base_lengths": np.asarray(base_lengths, dtype=np.int64),
        "perturbed_lengths": np.asarray(perturbed_lengths, dtype=np.int64),
        "state_columns": columns,
    }


def rollout_state_trajectory(
    env: gym.Env,
    policy,
    initial_obs: np.ndarray,
    columns: np.ndarray,
    state_std: np.ndarray,
    horizon: int,
) -> np.ndarray:
    env.reset()
    set_env_from_obs(env, initial_obs)
    obs = env.unwrapped._get_obs().astype(np.float32)
    states = [obs[columns] / state_std]
    for _ in range(horizon):
        action = policy.select_action(obs.reshape(1, -1), deterministic=True).reshape(-1)
        obs, _, terminated, truncated, _ = env.step(action)
        states.append(np.asarray(obs, dtype=np.float32)[columns] / state_std)
        if terminated or truncated:
            break
    return np.asarray(states, dtype=np.float32)


def evaluate_conservativity(
    train_dataset: dict[str, np.ndarray],
    test_dataset: dict[str, np.ndarray],
    rollout_observations: np.ndarray,
    rollout_actions: np.ndarray,
    sample_count: int,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_indices = rng.choice(len(train_dataset["observations"]), size=min(50000, len(train_dataset["observations"])), replace=False)
    test_indices = rng.choice(len(test_dataset["observations"]), size=min(sample_count, len(test_dataset["observations"])), replace=False)
    rollout_indices = rng.choice(len(rollout_observations), size=min(sample_count, len(rollout_observations)), replace=False)

    train_states = train_dataset["observations"][train_indices]
    state_mean = train_dataset["observations"].mean(axis=0)
    state_std = train_dataset["observations"].std(axis=0)
    state_std[state_std == 0.0] = 1.0
    state_reference = (train_states - state_mean) / state_std
    test_state_distances = metrics.knn_distances(
        state_reference, (test_dataset["observations"][test_indices] - state_mean) / state_std
    )
    rollout_state_distances = metrics.knn_distances(
        state_reference, (rollout_observations[rollout_indices] - state_mean) / state_std
    )

    train_state_actions = np.concatenate([train_states, train_dataset["actions"][train_indices]], axis=1)
    state_action_mean = np.concatenate(
        [state_mean, train_dataset["actions"].mean(axis=0)]
    )
    state_action_std = np.concatenate(
        [state_std, train_dataset["actions"].std(axis=0)]
    )
    state_action_std[state_action_std == 0.0] = 1.0
    state_action_reference = (train_state_actions - state_action_mean) / state_action_std
    test_state_actions = np.concatenate(
        [test_dataset["observations"][test_indices], test_dataset["actions"][test_indices]], axis=1
    )
    rollout_state_actions = np.concatenate(
        [rollout_observations[rollout_indices], rollout_actions[rollout_indices]], axis=1
    )
    test_state_action_distances = metrics.knn_distances(
        state_action_reference, (test_state_actions - state_action_mean) / state_action_std
    )
    rollout_state_action_distances = metrics.knn_distances(
        state_action_reference, (rollout_state_actions - state_action_mean) / state_action_std
    )

    return {
        "state_ood_ratio": np.asarray(rollout_state_distances.mean() / max(test_state_distances.mean(), metrics.EPS), dtype=np.float32),
        "state_action_ood_ratio": np.asarray(rollout_state_action_distances.mean() / max(test_state_action_distances.mean(), metrics.EPS), dtype=np.float32),
        "test_state_distances": test_state_distances,
        "rollout_state_distances": rollout_state_distances,
        "test_state_action_distances": test_state_action_distances,
        "rollout_state_action_distances": rollout_state_action_distances,
        "train_indices": train_indices.astype(np.int64),
        "test_indices": test_indices.astype(np.int64),
        "rollout_indices": rollout_indices.astype(np.int64),
    }


def seed_policy_randomness(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pad_curves(curves: list[np.ndarray]) -> np.ndarray:
    padded = np.full((len(curves), max(map(len, curves))), np.nan, dtype=np.float32)
    for index, curve in enumerate(curves):
        padded[index, : len(curve)] = curve
    return padded


def evaluate_dynamics_on_dataset(
    dynamics,
    dataset: dict[str, np.ndarray],
    obs_mean: np.ndarray | None = None,
    obs_std: np.ndarray | None = None,
) -> np.ndarray:
    errors = []
    for start in range(0, len(dataset["observations"]), 8192):
        end = start + 8192
        obs = dataset["observations"][start:end]
        actions = dataset["actions"][start:end]
        pred_next_obs = predict_next_obs(dynamics, obs, actions, obs_mean=obs_mean, obs_std=obs_std)
        errors.append(np.sum((pred_next_obs - dataset["next_observations"][start:end]) ** 2, axis=1))
    return np.concatenate(errors, axis=0).astype(np.float32)


def predict_next_obs(
    dynamics,
    obs: np.ndarray,
    actions: np.ndarray,
    obs_mean: np.ndarray | None = None,
    obs_std: np.ndarray | None = None,
) -> np.ndarray:
    model_obs = obs if obs_mean is None else (obs - obs_mean) / obs_std
    model_input = dynamics.scaler.transform(np.concatenate([model_obs, actions], axis=-1))
    with torch.no_grad():
        mean, _ = dynamics.model(model_input)
    elite_indices = dynamics.model.elites.detach().cpu().numpy()
    elite_mean = mean[elite_indices].mean(dim=0).cpu().numpy()
    pred_model_next_obs = model_obs + elite_mean[:, : obs.shape[1]]
    if obs_mean is not None:
        return pred_model_next_obs * obs_std + obs_mean
    return pred_model_next_obs


def evaluate_jacobians_on_observations(
    policy,
    dynamics,
    env_name: str,
    observations: np.ndarray,
    sample_count: int,
    seed: int,
    fd_eps: float,
    obs_mean: np.ndarray | None = None,
    obs_std: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    env = gym.make(env_name)
    env.reset(seed=seed)
    columns = reconstructible_observation_columns(env)
    sample_count = min(sample_count, len(observations))
    indices = np.random.default_rng(seed).choice(len(observations), size=sample_count, replace=False)
    errors = []

    try:
        for index in indices:
            true_jacobian = closed_loop_jacobian(
                lambda obs: true_next_obs(env, policy, obs),
                env,
                observations[index],
                columns,
                fd_eps,
            )
            learned_jacobian = closed_loop_jacobian(
                lambda obs: learned_next_obs(policy, dynamics, obs, obs_mean, obs_std),
                env,
                observations[index],
                columns,
                fd_eps,
            )
            errors.append(float(np.sum((learned_jacobian - true_jacobian) ** 2)))
    finally:
        env.close()

    return {
        "closed_loop_jacobian_sq_error": np.asarray(errors, dtype=np.float32),
        "sample_indices": indices.astype(np.int64),
        "columns": columns.astype(np.int64),
    }


def closed_loop_jacobian(next_obs_fn, env: gym.Env, obs: np.ndarray, columns: np.ndarray, fd_eps: float) -> np.ndarray:
    jacobian = np.empty((len(obs), len(columns)), dtype=np.float32)
    for column_index, obs_index in enumerate(columns):
        obs_plus = np.asarray(obs, dtype=np.float32).copy()
        obs_minus = np.asarray(obs, dtype=np.float32).copy()
        obs_plus[obs_index] += fd_eps
        obs_minus[obs_index] -= fd_eps
        set_env_from_obs(env, obs_plus)
        physical_obs_plus = env.unwrapped._get_obs().astype(np.float32)
        set_env_from_obs(env, obs_minus)
        physical_obs_minus = env.unwrapped._get_obs().astype(np.float32)
        jacobian[:, column_index] = (next_obs_fn(physical_obs_plus) - next_obs_fn(physical_obs_minus)) / (2.0 * fd_eps)
    return jacobian


def true_next_obs(env: gym.Env, policy, obs: np.ndarray) -> np.ndarray:
    set_env_from_obs(env, obs)
    action = policy.select_action(obs.reshape(1, -1), deterministic=True).reshape(-1)
    next_obs, *_ = env.unwrapped.step(action)
    return np.asarray(next_obs, dtype=np.float32)


def learned_next_obs(
    policy,
    dynamics,
    obs: np.ndarray,
    obs_mean: np.ndarray | None,
    obs_std: np.ndarray | None,
) -> np.ndarray:
    action = policy.select_action(obs.reshape(1, -1), deterministic=True).reshape(1, -1)
    return predict_next_obs(dynamics, obs.reshape(1, -1), action, obs_mean=obs_mean, obs_std=obs_std)[0]


def reconstructible_observation_columns(env: gym.Env) -> np.ndarray:
    structure = env.unwrapped.observation_structure
    return np.arange(structure["qpos"] + structure["qvel"], dtype=np.int64)


def set_env_from_obs(env: gym.Env, obs: np.ndarray) -> None:
    unwrapped = env.unwrapped
    structure = unwrapped.observation_structure
    skipped_qpos = structure["skipped_qpos"]
    qpos = np.zeros(unwrapped.model.nq, dtype=np.float64)
    qvel = np.zeros(unwrapped.model.nv, dtype=np.float64)
    offset = 0
    qpos[skipped_qpos:] = obs[offset : offset + structure["qpos"]]
    offset += structure["qpos"]
    qvel[:] = obs[offset : offset + structure["qvel"]]
    mujoco.mj_normalizeQuat(unwrapped.model, qpos)
    unwrapped.set_state(qpos, qvel)

if __name__ == "__main__":
    main()

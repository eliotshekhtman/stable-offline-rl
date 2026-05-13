# Tasks:
# - Reload trained OfflineRL-Kit runs from their run_manifest.json files.
# - Evaluate learned policies and configured expert policies in the true environment.
# - For model-based runs, evaluate learned next-state prediction MSE on held-out data.
# - For model-based runs, evaluate learned next-state prediction MSE along policy rollouts.

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

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
    return parser.parse_args()


def evaluate_run(run_dir: Path, args: argparse.Namespace) -> None:
    with (run_dir / "run_manifest.json").open("r", encoding="utf-8") as file:
        manifest = json.load(file)
    policy, dynamics, obs_mean, obs_std = load_policy_and_dynamics(manifest, args.device)
    rollout_info = evaluate_policy_rollouts(
        policy=policy,
        env_name=manifest["env_name"],
        episodes=args.eval_episodes,
        seed=args.seed,
        dynamics=dynamics,
        obs_mean=obs_mean,
        obs_std=obs_std,
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

    if dynamics is not None:
        test_dataset = rollout.load_dataset(manifest["test_dataset_path"])
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
    print(f"Saved evaluation: {eval_dir}")

def load_policy_and_dynamics(manifest: dict, device: str):
    env = gym.make(manifest["env_name"])
    train_dataset = rollout.load_dataset(manifest["train_dataset_path"])
    buffer = build_buffer(train_dataset, env, device)
    build_args = argparse.Namespace(
        device=device,
        epoch=manifest["epoch"],
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
        dynamics.load(manifest["model_dir"])
    else:
        policy, _ = build_model_free_policy(manifest["algo"], env, buffer, build_args)
        dynamics = None

    policy.load_state_dict(torch.load(Path(manifest["model_dir"]) / "policy.pth", map_location=device, weights_only=True))
    policy.eval()
    env.close()
    return policy, dynamics, obs_mean, obs_std


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
    next_obs_errors, error_observations, error_episode_ids, error_timesteps = [], [], [], []

    try:
        for episode in range(episodes):
            obs, _ = env.reset(seed=seed + episode)
            episode_return = 0.0
            episode_length = 0
            terminated = truncated = False

            while not (terminated or truncated):
                action = policy.select_action(obs.reshape(1, -1), deterministic=True).reshape(-1)
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
    unwrapped.set_state(qpos, qvel)

if __name__ == "__main__":
    main()

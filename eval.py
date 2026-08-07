# Tasks:
# - Reload trained OfflineRL-Kit runs from their run_manifest.json files.
# - Evaluate learned policies and configured expert policies in the true environment.
# - Execute learned action chunks open-loop while retaining primitive-step metrics.
# - For model-based runs, evaluate learned macro next-state prediction MSE on held-out data.
# - For model-based runs, evaluate learned macro next-state prediction MSE along policy rollouts.
# - Measure empirical global/local trajectory convergence and dataset conservativity.

import argparse
import json
import shutil
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
import torch

import chunking
import metrics
import load_offline
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
    run.add_argument("--expert", default=None, help="Expert policy .zip path or directory containing <env>.zip; defaults to the expert recorded during training")

    rollout_eval = parser.add_argument_group("rollout evaluation")
    rollout_eval.add_argument("--eval-episodes", type=int, default=10, help="Number of true-environment episodes used to estimate policy and expert returns")

    jacobian_eval = parser.add_argument_group("model-based jacobian evaluation")
    jacobian_eval.add_argument("--jacobian-samples", type=int, default=8, help="Number of held-out dataset states and policy-rollout states used for finite-difference Jacobian evaluation")
    jacobian_eval.add_argument("--fd-eps", type=float, default=1e-4, help="Central finite-difference perturbation size for closed-loop Jacobian estimates")

    stability = parser.add_argument_group("stability and conservativity evaluation")
    stability.add_argument("--stability-trajectories", type=int, default=8, help="Number of global trajectories and local perturbed-state pairs used for stability metrics")
    stability.add_argument("--stability-horizon", type=int, default=300, help="Maximum primitive environment steps used for each global and local stability trajectory")
    stability.add_argument("--global-max-offset", type=int, default=30, help="Largest primitive-step offset considered when phase-aligning global trajectories")
    stability.add_argument("--local-perturbation-scale", type=float, default=0.01, help="Initial local-state perturbation norm in training-standardized physical-state coordinates")
    stability.add_argument("--ood-samples", type=int, default=10000, help="Maximum held-out and policy decision-boundary samples used for state and state-action OOD metrics")
    return parser.parse_args()


def evaluate_run(run_dir: Path, args: argparse.Namespace) -> Path:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    manifest_path = (run_dir / "run_manifest.json").resolve()
    with manifest_path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)
    if args.expert is not None:
        expert_path = Path(args.expert).expanduser()
        if expert_path.suffix != ".zip":
            expert_path = expert_path / f"{manifest['env_name']}.zip"
        manifest["expert"] = str(expert_path.resolve())
    train_dataset = chunking.make_action_chunk_dataset(
        rollout.load_dataset(manifest["train_dataset_path"]),
        manifest["chunk_length"],
        manifest["base_discount"],
    )
    test_dataset = chunking.make_action_chunk_dataset(
        rollout.load_dataset(manifest["test_dataset_path"]),
        manifest["chunk_length"],
        manifest["base_discount"],
    )
    policy, dynamics, obs_mean, obs_std = load_policy_and_dynamics(
        manifest,
        args.device,
        Path(manifest["model_dir"]) / "policy.pth",
        Path(manifest["model_dir"]) if manifest["algo"] in MODEL_BASED_ALGOS else None,
        train_dataset,
    )
    rollout_info, global_stability, local_stability, conservativity = evaluate_policy_behavior(
        policy, manifest, train_dataset, test_dataset, args,
        dynamics=dynamics, obs_mean=obs_mean, obs_std=obs_std,
    )
    expert_info = evaluate_expert(manifest, args.eval_episodes, args.seed)

    eval_dir = Path(manifest["eval_dir"])
    if eval_dir.exists():
        shutil.rmtree(eval_dir)
    eval_dir.mkdir(parents=True)
    results = {
        "run_manifest_path": str(manifest_path),
        "evaluation_config": {
            "device": args.device,
            "eval_episodes": args.eval_episodes,
            "expert": manifest["expert"],
            "seed": args.seed,
            "jacobian_samples": args.jacobian_samples,
            "fd_eps": args.fd_eps,
            "stability_trajectories": args.stability_trajectories,
            "stability_horizon_primitive_steps": args.stability_horizon,
            "global_max_offset_primitive_steps": args.global_max_offset,
            "local_perturbation_scale": args.local_perturbation_scale,
            "ood_samples": args.ood_samples,
        },
        "env_name": manifest["env_name"],
        "algo": manifest["algo"],
        "dataset_tag": manifest["dataset_tag"],
        "chunk_length": manifest["chunk_length"],
        "base_discount": manifest["base_discount"],
        "macro_discount": manifest["macro_discount"],
        "rollout_timestep_unit": "primitive_step",
        "stability_timestep_unit": "primitive_step",
        "policy_return_mean": float(np.mean(rollout_info["returns"])),
        "policy_return_std": float(np.std(rollout_info["returns"])),
        "expert_return_mean": expert_info["return_mean"],
        "expert_return_std": expert_info["return_std"],
    }
    np.savez_compressed(
        eval_dir / "returns.npz",
        policy_episode_returns=rollout_info["returns"],
        expert_episode_returns=expert_info["returns"],
    )

    np.savez_compressed(eval_dir / "global_stability.npz", **global_stability)
    np.savez_compressed(eval_dir / "conservativity.npz", **conservativity)
    results.update(
        global_stability_c=float(global_stability["c"]),
        global_stability_rho=float(global_stability["rho"]),
        state_ood_ratio=float(conservativity["state_ood_ratio"]),
        state_action_ood_ratio=float(conservativity["state_action_ood_ratio"]),
    )
    if local_stability is not None:
        np.savez_compressed(eval_dir / "local_stability.npz", **local_stability)
        results.update(
            local_stability_c=float(local_stability["c"]),
            local_stability_rho=float(local_stability["rho"]),
        )

    if dynamics is not None:
        dataset_errors = evaluate_dynamics_on_dataset(
            dynamics=dynamics,
            dataset=test_dataset,
            obs_mean=obs_mean,
            obs_std=obs_std,
        )
        np.savez_compressed(eval_dir / "dynamics_dataset.npz", next_obs_mse=dataset_errors)
        np.savez_compressed(
            eval_dir / "dynamics_rollout.npz",
            next_obs_mse=rollout_info["next_obs_mse"],
            episode_ids=rollout_info["dynamics_episode_ids"],
            primitive_timesteps=rollout_info["dynamics_primitive_timesteps"],
        )
        results["dynamics_horizon_primitive_steps"] = manifest["chunk_length"]
        results["dataset_next_obs_mse"] = float(np.mean(dataset_errors))
        results["rollout_next_obs_mse"] = float(np.mean(rollout_info["next_obs_mse"]))
        if manifest["dataset_source"] != "robomimic":
            dataset_jacobians = evaluate_jacobians_on_observations(
                policy=policy,
                dynamics=dynamics,
                env_name=manifest["env_name"],
                observations=test_dataset["observations"],
                sample_count=args.jacobian_samples,
                seed=args.seed,
                fd_eps=args.fd_eps,
                chunk_length=manifest["chunk_length"],
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
                chunk_length=manifest["chunk_length"],
                obs_mean=obs_mean,
                obs_std=obs_std,
            )
            np.savez_compressed(eval_dir / "jacobian_dataset.npz", **dataset_jacobians)
            np.savez_compressed(eval_dir / "jacobian_rollout.npz", **rollout_jacobians)
            results["dataset_closed_loop_jacobian_mse"] = float(np.mean(dataset_jacobians["closed_loop_jacobian_mse"]))
            results["rollout_closed_loop_jacobian_mse"] = float(np.mean(rollout_jacobians["closed_loop_jacobian_mse"]))

    with (eval_dir / "results.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, sort_keys=True)
    evaluate_history(manifest, train_dataset, test_dataset, expert_info, eval_dir, args)
    print(f"Saved evaluation: {eval_dir}")
    return eval_dir


def load_policy_and_dynamics(
    manifest: dict,
    device: str,
    policy_path: Path,
    dynamics_path: Path | None,
    train_dataset: dict[str, np.ndarray],
):
    env = chunking.ActionChunkWrapper(make_eval_env(manifest), manifest["chunk_length"])
    buffer = build_buffer(train_dataset, env, device)
    build_args = argparse.Namespace(
        device=device,
        epoch=manifest["epoch"],
        step_per_epoch=manifest["step_per_epoch"],
        adv_weight=manifest["adv_weight"],
        rollout_length=manifest["rollout_length"],
        adv_batch_size=manifest["adv_batch_size"],
    )

    obs_mean = obs_std = None
    if manifest["algo"] in MODEL_BASED_ALGOS:
        if manifest["algo"] == "rambo":
            obs_mean, obs_std = buffer.normalize_obs()
        policy, dynamics, _ = build_model_based_policy(
            manifest["algo"], env, build_args, discount=manifest["macro_discount"],
            obs_mean=obs_mean, obs_std=obs_std,
        )
        dynamics.load(str(dynamics_path))
    else:
        dql_config = manifest["dql_config"] if manifest["algo"] == "dql" else None
        policy, _ = build_model_free_policy(
            manifest["algo"], env, buffer, build_args,
            discount=manifest["macro_discount"], dql_config=dql_config,
        )
        dynamics = None

    policy.load_state_dict(torch.load(policy_path, map_location=device, weights_only=True))
    policy.eval()
    env.close()
    return policy, dynamics, obs_mean, obs_std


def evaluate_policy_behavior(
    policy,
    manifest: dict,
    train_dataset: dict[str, np.ndarray],
    test_dataset: dict[str, np.ndarray],
    args: argparse.Namespace,
    dynamics=None,
    obs_mean: np.ndarray | None = None,
    obs_std: np.ndarray | None = None,
):
    rollout_info = evaluate_policy_rollouts(
        policy, manifest, args.eval_episodes, args.seed,
        dynamics=dynamics, obs_mean=obs_mean, obs_std=obs_std,
    )
    global_stability = evaluate_global_stability(
        policy, manifest, train_dataset, args.stability_trajectories,
        args.stability_horizon, args.global_max_offset, args.seed,
    )
    local_stability = None
    if manifest["dataset_source"] != "robomimic":
        local_stability = evaluate_local_stability(
            policy, manifest["env_name"], train_dataset, test_dataset,
            args.stability_trajectories, args.stability_horizon,
            args.local_perturbation_scale, args.seed, manifest["chunk_length"],
        )
    conservativity = evaluate_conservativity(
        train_dataset, test_dataset, rollout_info["decision_observations"],
        rollout_info["action_chunks"], args.ood_samples, args.seed,
    )
    return rollout_info, global_stability, local_stability, conservativity


def make_eval_env(manifest: dict):
    if manifest["dataset_source"] == "robomimic":
        metadata = load_offline.load_metadata(manifest["dataset_metadata_path"])
        return load_offline.make_robomimic_env(metadata)
    return gym.make(manifest["env_name"])


def robomimic_expert_metadata(manifest: dict) -> dict:
    metadata_path = Path(manifest["dataset_metadata_path"])
    metadata = load_offline.load_metadata(metadata_path)
    if metadata["dataset_type"] == "ph":
        return metadata
    ph_dataset_dir = metadata_path.parents[2] / f"robomimic_{metadata['task'].lower()}_ph"
    ph_metadata_paths = sorted(ph_dataset_dir.glob("*/metadata.json"), reverse=True)
    if ph_metadata_paths:
        return load_offline.load_metadata(ph_metadata_paths[0])
    ph_spec = load_offline.list_robomimic_dataset_specs(metadata["task"], "ph")[0]
    _, ph_metadata = load_offline.load_robomimic_dataset(ph_spec)
    return ph_metadata


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
            policy, manifest, train_dataset, test_dataset, args
        )
        record = {
            "requested_percent": checkpoint["requested_percent"],
            "actual_percent": checkpoint["actual_percent"],
            "step": checkpoint["step"],
            "policy_return_mean": float(rollout_info["returns"].mean()),
            "policy_return_std": float(rollout_info["returns"].std()),
            "global_stability_c": float(global_stability["c"]),
            "global_stability_rho": float(global_stability["rho"]),
            "global_survival_fraction": float(global_stability["support"][-1] / global_stability["support"][0]),
            "state_ood_ratio": float(conservativity["state_ood_ratio"]),
            "state_action_ood_ratio": float(conservativity["state_action_ood_ratio"]),
        }
        if local_stability is not None:
            record.update(
                local_stability_c=float(local_stability["c"]),
                local_stability_rho=float(local_stability["rho"]),
                local_survival_fraction=float(local_stability["support"][-1] / local_stability["support"][0]),
            )
        records.append(record)

    history = {
        "env_name": manifest["env_name"],
        "algo": manifest["algo"],
        "dataset_tag": manifest["dataset_tag"],
        "chunk_length": manifest["chunk_length"],
        "base_discount": manifest["base_discount"],
        "macro_discount": manifest["macro_discount"],
        "stability_timestep_unit": "primitive_step",
        "expert_return_mean": expert_info["return_mean"],
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
    manifest: dict,
    episodes: int,
    seed: int,
    dynamics=None,
    obs_mean: np.ndarray | None = None,
    obs_std: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    env = make_eval_env(manifest)
    chunk_length = manifest["chunk_length"]
    returns = []
    decision_observations, action_chunks = [], []
    next_obs_errors, error_observations = [], []
    error_episode_ids, error_primitive_timesteps = [], []

    try:
        for episode in range(episodes):
            obs, _ = env.reset(seed=seed + episode)
            episode_return = 0.0
            episode_primitive_steps = 0
            terminated = truncated = False

            while not (terminated or truncated):
                action_chunk = policy.select_action(obs.reshape(1, -1), deterministic=True).reshape(-1)
                decision_observations.append(np.asarray(obs, dtype=np.float32).copy())
                action_chunks.append(np.asarray(action_chunk, dtype=np.float32).copy())
                next_obs, reward, terminated, truncated, chunk_info = chunking.execute_action_chunk(
                    env, action_chunk, chunk_length
                )
                primitive_steps = chunk_info["primitive_steps"]
                if dynamics is not None and primitive_steps == chunk_length:
                    error_observations.append(np.asarray(obs, dtype=np.float32).copy())
                    pred_next_obs = predict_next_obs(
                        dynamics,
                        obs.reshape(1, -1),
                        action_chunk.reshape(1, -1),
                        obs_mean=obs_mean,
                        obs_std=obs_std,
                    )[0]
                    next_obs_errors.append(float(np.mean((pred_next_obs - next_obs) ** 2)))
                    error_episode_ids.append(episode)
                    error_primitive_timesteps.append(episode_primitive_steps)

                episode_return += float(reward)
                episode_primitive_steps += primitive_steps
                obs = next_obs

            returns.append(episode_return)
    finally:
        env.close()

    return {
        "returns": np.asarray(returns, dtype=np.float32),
        "decision_observations": np.asarray(decision_observations, dtype=np.float32),
        "action_chunks": np.asarray(action_chunks, dtype=np.float32),
        "next_obs_mse": np.asarray(next_obs_errors, dtype=np.float32),
        "dynamics_observations": np.asarray(error_observations, dtype=np.float32),
        "dynamics_episode_ids": np.asarray(error_episode_ids, dtype=np.int64),
        "dynamics_primitive_timesteps": np.asarray(error_primitive_timesteps, dtype=np.int64),
    }


def evaluate_expert(manifest: dict, episodes: int, seed: int) -> dict[str, np.ndarray]:
    if manifest["dataset_source"] == "robomimic":
        metadata = robomimic_expert_metadata(manifest)
        return {
            "returns": np.asarray([metadata["episode_return_mean"]], dtype=np.float32),
            "return_mean": float(metadata["episode_return_mean"]),
            "return_std": float(metadata["episode_return_std"]),
        }

    env_name = manifest["env_name"]
    expert_path = Path(manifest["expert"])
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

    returns = np.asarray(returns, dtype=np.float32)
    return {
        "returns": returns,
        "return_mean": float(returns.mean()),
        "return_std": float(returns.std()),
    }


def evaluate_global_stability(
    policy,
    manifest: dict,
    train_dataset: dict[str, np.ndarray],
    trajectory_count: int,
    horizon: int,
    max_offset: int,
    seed: int,
) -> dict[str, np.ndarray]:
    env = make_eval_env(manifest)
    chunk_length = manifest["chunk_length"]
    columns = observation_columns(env, manifest)
    state_std = train_dataset["observations"][:, columns].std(axis=0)
    state_std[state_std == 0.0] = 1.0
    trajectories = []
    try:
        for index in range(trajectory_count):
            obs, _ = env.reset(seed=seed + index)
            seed_policy_randomness(seed + 100000)
            if manifest["dataset_source"] == "robomimic":
                trajectories.append(
                    rollout_current_env_trajectory(
                        env, policy, obs, columns, state_std, horizon, chunk_length
                    )
                )
            else:
                trajectories.append(
                    rollout_state_trajectory(
                        env, policy, obs, columns, state_std, horizon, chunk_length
                    )
                )
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
    chunk_length: int,
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
            base = rollout_state_trajectory(
                env, policy, base_obs, columns, state_std, horizon, chunk_length
            )
            seed_policy_randomness(seed + 100000 + pair_index)
            perturbed = rollout_state_trajectory(
                env, policy, perturbed_obs, columns, state_std, horizon, chunk_length
            )
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
    chunk_length: int,
) -> np.ndarray:
    env.reset()
    set_env_from_obs(env, initial_obs)
    obs = env.unwrapped._get_obs().astype(np.float32)
    return rollout_current_env_trajectory(
        env, policy, obs, columns, state_std, horizon, chunk_length
    )


def rollout_current_env_trajectory(
    env: gym.Env,
    policy,
    obs: np.ndarray,
    columns: np.ndarray,
    state_std: np.ndarray,
    horizon: int,
    chunk_length: int,
) -> np.ndarray:
    states = [np.asarray(obs, dtype=np.float32)[columns] / state_std]
    primitive_steps = 0
    terminated = truncated = False
    while primitive_steps < horizon and not (terminated or truncated):
        action_chunk = policy.select_action(obs.reshape(1, -1), deterministic=True).reshape(-1)
        obs, _, terminated, truncated, chunk_info = chunking.execute_action_chunk(
            env,
            action_chunk,
            chunk_length,
            max_primitive_steps=horizon - primitive_steps,
        )
        primitive_next_observations = chunk_info["primitive_next_observations"]
        states.extend(primitive_next_observations[:, columns] / state_std)
        primitive_steps += chunk_info["primitive_steps"]
    return np.asarray(states, dtype=np.float32)


def evaluate_conservativity(
    train_dataset: dict[str, np.ndarray],
    test_dataset: dict[str, np.ndarray],
    rollout_decision_observations: np.ndarray,
    rollout_action_chunks: np.ndarray,
    sample_count: int,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_indices = rng.choice(len(train_dataset["observations"]), size=min(50000, len(train_dataset["observations"])), replace=False)
    test_indices = rng.choice(len(test_dataset["observations"]), size=min(sample_count, len(test_dataset["observations"])), replace=False)
    rollout_indices = rng.choice(
        len(rollout_decision_observations),
        size=min(sample_count, len(rollout_decision_observations)),
        replace=False,
    )

    train_states = train_dataset["observations"][train_indices]
    state_mean = train_dataset["observations"].mean(axis=0)
    state_std = train_dataset["observations"].std(axis=0)
    state_std[state_std == 0.0] = 1.0
    state_reference = (train_states - state_mean) / state_std
    test_state_distances = metrics.knn_distances(
        state_reference, (test_dataset["observations"][test_indices] - state_mean) / state_std
    )
    rollout_state_distances = metrics.knn_distances(
        state_reference, (rollout_decision_observations[rollout_indices] - state_mean) / state_std
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
        [rollout_decision_observations[rollout_indices], rollout_action_chunks[rollout_indices]], axis=1
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
        action_chunks = dataset["actions"][start:end]
        pred_next_obs = predict_next_obs(
            dynamics, obs, action_chunks, obs_mean=obs_mean, obs_std=obs_std
        )
        errors.append(np.mean((pred_next_obs - dataset["next_observations"][start:end]) ** 2, axis=1))
    return np.concatenate(errors, axis=0).astype(np.float32)


def predict_next_obs(
    dynamics,
    obs: np.ndarray,
    action_chunks: np.ndarray,
    obs_mean: np.ndarray | None = None,
    obs_std: np.ndarray | None = None,
) -> np.ndarray:
    model_obs = obs if obs_mean is None else (obs - obs_mean) / obs_std
    model_input = dynamics.scaler.transform(np.concatenate([model_obs, action_chunks], axis=-1))
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
    chunk_length: int,
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
        for sample_index, index in enumerate(indices):
            finite_difference_seed = seed + sample_index * len(columns)
            true_jacobian = closed_loop_jacobian(
                lambda obs: true_next_obs(env, policy, obs, chunk_length),
                env,
                observations[index],
                columns,
                fd_eps,
                finite_difference_seed,
            )
            learned_jacobian = closed_loop_jacobian(
                lambda obs: learned_next_obs(policy, dynamics, obs, obs_mean, obs_std),
                env,
                observations[index],
                columns,
                fd_eps,
                finite_difference_seed,
            )
            errors.append(float(np.mean((learned_jacobian - true_jacobian) ** 2)))
    finally:
        env.close()

    return {
        "closed_loop_jacobian_mse": np.asarray(errors, dtype=np.float32),
        "sample_indices": indices.astype(np.int64),
        "columns": columns.astype(np.int64),
    }


def closed_loop_jacobian(
    next_obs_fn,
    env: gym.Env,
    obs: np.ndarray,
    columns: np.ndarray,
    fd_eps: float,
    seed: int,
) -> np.ndarray:
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
        seed_policy_randomness(seed + column_index)
        next_obs_plus = next_obs_fn(physical_obs_plus)
        seed_policy_randomness(seed + column_index)
        next_obs_minus = next_obs_fn(physical_obs_minus)
        jacobian[:, column_index] = (next_obs_plus - next_obs_minus) / (2.0 * fd_eps)
    return jacobian


def true_next_obs(env: gym.Env, policy, obs: np.ndarray, chunk_length: int) -> np.ndarray:
    set_env_from_obs(env, obs)
    action_chunk = policy.select_action(obs.reshape(1, -1), deterministic=True).reshape(-1)
    next_obs, *_ = chunking.execute_action_chunk(env.unwrapped, action_chunk, chunk_length)
    return np.asarray(next_obs, dtype=np.float32)


def learned_next_obs(
    policy,
    dynamics,
    obs: np.ndarray,
    obs_mean: np.ndarray | None,
    obs_std: np.ndarray | None,
) -> np.ndarray:
    action_chunk = policy.select_action(obs.reshape(1, -1), deterministic=True).reshape(1, -1)
    return predict_next_obs(
        dynamics, obs.reshape(1, -1), action_chunk,
        obs_mean=obs_mean, obs_std=obs_std,
    )[0]


def reconstructible_observation_columns(env: gym.Env) -> np.ndarray:
    structure = env.unwrapped.observation_structure
    return np.arange(structure["qpos"] + structure["qvel"], dtype=np.int64)


def observation_columns(env: gym.Env, manifest: dict) -> np.ndarray:
    if manifest["dataset_source"] == "robomimic":
        return np.arange(env.observation_space.shape[0], dtype=np.int64)
    return reconstructible_observation_columns(env)


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

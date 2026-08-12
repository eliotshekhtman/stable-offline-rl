# Tasks:
# - Reload trained OfflineRL-Kit runs from their run_manifest.json files.
# - Evaluate learned policies and configured expert policies in the true environment.
# - Execute learned action chunks open-loop while retaining primitive-step metrics.
# - Measure final-policy contraction after perturbing only controlled-agent coordinates.
# - Measure policy conservativity over training checkpoints.
# - Retain dormant learned-dynamics and Jacobian evaluation implementations for future use.

import argparse
import json
import shutil
from pathlib import Path

import gymnasium as gym
import h5py
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
    run.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Torch device used to reload the trained policy")
    run.add_argument("--seed", type=int, default=0, help="Random seed for evaluation rollouts, perturbations, and OOD sampling")
    run.add_argument("--expert", default=None, help="Expert policy .zip path or directory containing <env>.zip for Gymnasium tasks; Robomimic uses its PH dataset as the expert baseline")

    rollout_eval = parser.add_argument_group("rollout evaluation")
    rollout_eval.add_argument("--eval-episodes", type=int, default=10, help="Number of true-environment episodes used to estimate policy performance and Gymnasium expert performance")

    behavior = parser.add_argument_group("contraction and conservativity evaluation")
    behavior.add_argument("--contraction-trajectories", type=int, default=8, help="Number of matched unperturbed and agent-state-perturbed trajectory pairs")
    behavior.add_argument("--contraction-horizon", type=int, default=300, help="Maximum primitive steps in each contraction trajectory")
    behavior.add_argument("--perturbation-scale", type=float, default=0.01, help="Euclidean norm of the initial perturbation applied to controlled-agent qpos/qvel coordinates")
    behavior.add_argument("--ood-samples", type=int, default=10000, help="Maximum held-out and policy decision-boundary samples used for OOD metrics")
    args = parser.parse_args()
    if args.eval_episodes <= 0 or args.contraction_trajectories <= 0 or args.contraction_horizon <= 0:
        parser.error("evaluation episode, trajectory, and horizon counts must be positive")
    if args.perturbation_scale < 0.0:
        parser.error("--perturbation-scale must be nonnegative")
    return args


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
    policy, _, _, _ = load_policy_and_dynamics(
        manifest,
        args.device,
        Path(manifest["model_dir"]) / "policy.pth",
        None,
        train_dataset,
    )
    rollout_info = evaluate_policy_rollouts(
        policy, manifest, args.eval_episodes, args.seed
    )
    contraction = evaluate_contraction(
        policy, manifest, args.contraction_trajectories,
        args.contraction_horizon, args.perturbation_scale, args.seed,
    )
    conservativity = evaluate_conservativity(
        train_dataset, test_dataset, rollout_info["decision_observations"],
        rollout_info["action_chunks"], args.ood_samples, args.seed,
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
            "contraction_trajectories": args.contraction_trajectories,
            "contraction_horizon_primitive_steps": args.contraction_horizon,
            "perturbation_scale": args.perturbation_scale,
            "ood_samples": args.ood_samples,
        },
        "env_name": manifest["env_name"],
        "algo": manifest["algo"],
        "dataset_tag": manifest["dataset_tag"],
        "chunk_length": manifest["chunk_length"],
        "base_discount": manifest["base_discount"],
        "macro_discount": manifest["macro_discount"],
        "rollout_timestep_unit": "primitive_step",
        "contraction_timestep_unit": "primitive_step",
        "policy_return_mean": float(np.mean(rollout_info["returns"])),
        "policy_return_std": float(np.std(rollout_info["returns"])),
        "performance_metric": rollout_info["performance_metric"],
        "performance_label": rollout_info["performance_label"],
        "performance_higher_is_better": rollout_info["performance_higher_is_better"],
        "policy_performance_mean": float(np.mean(rollout_info["performance"])),
        "policy_performance_std": float(np.std(rollout_info["performance"])),
        "expert_return_mean": expert_info["return_mean"],
        "expert_return_std": expert_info["return_std"],
        "expert_performance_mean": expert_info["performance_mean"],
        "expert_performance_std": expert_info["performance_std"],
    }
    np.savez_compressed(
        eval_dir / "returns.npz",
        policy_episode_returns=rollout_info["returns"],
        expert_episode_returns=expert_info["returns"],
        policy_episode_performance=rollout_info["performance"],
        expert_episode_performance=expert_info["performance"],
    )

    np.savez_compressed(eval_dir / "contraction.npz", **contraction)
    np.savez_compressed(eval_dir / "conservativity.npz", **conservativity)
    results.update(
        state_ood_ratio=float(conservativity["state_ood_ratio"]),
        state_action_ood_ratio=float(conservativity["state_action_ood_ratio"]),
    )

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
        if dynamics_path is not None:
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


def make_eval_env(manifest: dict):
    if manifest["dataset_source"] == "robomimic":
        metadata = load_offline.load_metadata(manifest["dataset_metadata_path"])
        metadata["env_args"]["env_kwargs"]["reward_shaping"] = False
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
    expert_info: dict,
    eval_dir: Path,
    args: argparse.Namespace,
) -> None:
    records = []
    first_checkpoint = manifest["checkpoints"][0]
    policy, _, _, _ = load_policy_and_dynamics(
        manifest, args.device, Path(first_checkpoint["policy_path"]),
        None, train_dataset,
    )
    for checkpoint_index, checkpoint in enumerate(manifest["checkpoints"]):
        seed_policy_randomness(args.seed)
        if checkpoint_index > 0:
            policy.load_state_dict(
                torch.load(Path(checkpoint["policy_path"]), map_location=args.device, weights_only=True)
            )
            policy.eval()
        rollout_info = evaluate_policy_rollouts(
            policy, manifest, args.eval_episodes, args.seed
        )
        conservativity = evaluate_conservativity(
            train_dataset, test_dataset, rollout_info["decision_observations"],
            rollout_info["action_chunks"], args.ood_samples, args.seed,
        )
        record = {
            "requested_percent": checkpoint["requested_percent"],
            "actual_percent": checkpoint["actual_percent"],
            "step": checkpoint["step"],
            "policy_return_mean": float(rollout_info["returns"].mean()),
            "policy_return_std": float(rollout_info["returns"].std()),
            "policy_performance_mean": float(rollout_info["performance"].mean()),
            "policy_performance_std": float(rollout_info["performance"].std()),
            "state_ood_ratio": float(conservativity["state_ood_ratio"]),
            "state_action_ood_ratio": float(conservativity["state_action_ood_ratio"]),
        }
        records.append(record)

    performance_metric, performance_label, _ = performance_definition(manifest["env_name"])
    history = {
        "env_name": manifest["env_name"],
        "algo": manifest["algo"],
        "dataset_tag": manifest["dataset_tag"],
        "chunk_length": manifest["chunk_length"],
        "base_discount": manifest["base_discount"],
        "macro_discount": manifest["macro_discount"],
        "performance_metric": performance_metric,
        "performance_label": performance_label,
        "expert_return_mean": expert_info["return_mean"],
        "expert_performance_mean": expert_info["performance_mean"],
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
) -> dict:
    env = make_eval_env(manifest)
    env_name = manifest["env_name"]
    chunk_length = manifest["chunk_length"]
    returns, performance = [], []
    decision_observations, action_chunks = [], []
    next_obs_errors, error_observations = [], []
    error_episode_ids, error_primitive_timesteps = [], []

    try:
        for episode in range(episodes):
            obs, reset_info = env.reset(seed=seed + episode)
            obs = current_observation(env, manifest)
            episode_return = 0.0
            episode_primitive_steps = 0
            episode_success = False
            final_info = reset_info
            terminated = truncated = False

            while not (terminated or truncated):
                action_chunk = policy.select_action(obs.reshape(1, -1), deterministic=True).reshape(-1)
                decision_observations.append(np.asarray(obs, dtype=np.float32).copy())
                action_chunks.append(np.asarray(action_chunk, dtype=np.float32).copy())
                next_obs, reward, terminated, truncated, chunk_info = chunking.execute_action_chunk(
                    env, action_chunk, chunk_length
                )
                primitive_steps = chunk_info["primitive_steps"]
                episode_success |= bool(np.any(chunk_info["primitive_rewards"] > 0.0))
                final_info = chunk_info
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
            performance.append(
                episode_performance(
                    env_name, env, episode_return, episode_primitive_steps,
                    episode_success, reset_info, final_info,
                )
            )
    finally:
        env.close()

    performance_metric, performance_label, higher_is_better = performance_definition(env_name)
    return {
        "returns": np.asarray(returns, dtype=np.float32),
        "performance": np.asarray(performance, dtype=np.float32),
        "performance_metric": performance_metric,
        "performance_label": performance_label,
        "performance_higher_is_better": higher_is_better,
        "decision_observations": np.asarray(decision_observations, dtype=np.float32),
        "action_chunks": np.asarray(action_chunks, dtype=np.float32),
        "next_obs_mse": np.asarray(next_obs_errors, dtype=np.float32),
        "dynamics_observations": np.asarray(error_observations, dtype=np.float32),
        "dynamics_episode_ids": np.asarray(error_episode_ids, dtype=np.int64),
        "dynamics_primitive_timesteps": np.asarray(error_primitive_timesteps, dtype=np.int64),
    }


def evaluate_expert(manifest: dict, episodes: int, seed: int) -> dict:
    if manifest["dataset_source"] == "robomimic":
        metadata = robomimic_expert_metadata(manifest)
        with h5py.File(metadata["hdf5_path"], "r") as file:
            episode_rewards = [
                np.asarray(file["data"][key]["rewards"], dtype=np.float32)
                for key in sorted(file["data"].keys())
            ]
        returns = np.asarray([rewards.sum() for rewards in episode_rewards], dtype=np.float32)
        performance = np.asarray(
            [np.any(rewards > 0.0) for rewards in episode_rewards], dtype=np.float32
        )
        return {
            "returns": returns,
            "return_mean": float(returns.mean()),
            "return_std": float(returns.std()),
            "performance": performance,
            "performance_mean": float(performance.mean()),
            "performance_std": float(performance.std()),
        }

    env_name = manifest["env_name"]
    expert_path = Path(manifest["expert"])
    policy = rollout.load_expert_policy(env_name, str(expert_path))
    env = gym.make(env_name)
    returns, performance = [], []
    try:
        for episode in range(episodes):
            obs, reset_info = env.reset(seed=seed + episode)
            episode_return = 0.0
            episode_steps = 0
            episode_success = False
            final_info = reset_info
            terminated = truncated = False

            while not (terminated or truncated):
                action, _ = policy.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, final_info = env.step(action)
                episode_return += float(reward)
                episode_steps += 1
                episode_success |= reward > 0.0

            returns.append(episode_return)
            performance.append(
                episode_performance(
                    env_name, env, episode_return, episode_steps,
                    episode_success, reset_info, final_info,
                )
            )
    finally:
        env.close()

    returns = np.asarray(returns, dtype=np.float32)
    performance = np.asarray(performance, dtype=np.float32)
    return {
        "returns": returns,
        "return_mean": float(returns.mean()),
        "return_std": float(returns.std()),
        "performance": performance,
        "performance_mean": float(performance.mean()),
        "performance_std": float(performance.std()),
    }


def performance_definition(env_name: str) -> tuple[str, str, bool]:
    if env_name in {"Can", "Lift"}:
        return "success_rate", "task success rate", True
    if env_name == "Reacher-v5":
        return "final_target_distance", "final fingertip-target distance (m)", False
    if env_name == "InvertedDoublePendulum-v5":
        return "balance_duration", "balance duration (primitive steps)", True
    if env_name == "HalfCheetah-v5":
        return "forward_displacement", "forward displacement (m)", True
    return "episode_return", "episode return", True


def episode_performance(
    env_name: str,
    env: gym.Env,
    episode_return: float,
    primitive_steps: int,
    succeeded: bool,
    reset_info: dict,
    final_info: dict,
) -> float:
    if env_name in {"Can", "Lift"}:
        return float(succeeded)
    if env_name == "Reacher-v5":
        fingertip = env.unwrapped.get_body_com("fingertip")
        target = env.unwrapped.get_body_com("target")
        return float(np.linalg.norm(fingertip - target))
    if env_name == "InvertedDoublePendulum-v5":
        return float(primitive_steps)
    if env_name == "HalfCheetah-v5":
        return float(final_info["x_position"] - reset_info["x_position"])
    return episode_return


def evaluate_contraction(
    policy,
    manifest: dict,
    trajectory_count: int,
    horizon: int,
    perturbation_scale: float,
    seed: int,
) -> dict[str, np.ndarray]:
    env = make_eval_env(manifest)
    rng = np.random.default_rng(seed)
    curves = []
    qpos_indices = qvel_indices = None

    try:
        for pair_index in range(trajectory_count):
            reset_seed = seed + pair_index
            obs, _ = env.reset(seed=reset_seed)
            obs = current_observation(env, manifest)
            qpos, qvel = simulator_state(env, manifest)
            if qpos_indices is None:
                qpos_indices, qvel_indices = controlled_agent_indices(env, manifest)

            direction = rng.normal(size=len(qpos_indices) + len(qvel_indices))
            direction /= np.linalg.norm(direction)
            perturbed_qpos = qpos.copy()
            perturbed_qvel = qvel.copy()
            split = len(qpos_indices)
            perturbed_qpos[qpos_indices] += perturbation_scale * direction[:split]
            perturbed_qvel[qvel_indices] += perturbation_scale * direction[split:]

            rollout_seed = seed + 100000 + pair_index
            seed_policy_randomness(rollout_seed)
            base = rollout_agent_trajectory(
                env, policy, obs, manifest, qpos_indices, qvel_indices, horizon
            )
            perturbed_obs = reset_to_simulator_state(
                env, manifest, reset_seed, perturbed_qpos, perturbed_qvel
            )
            seed_policy_randomness(rollout_seed)
            perturbed = rollout_agent_trajectory(
                env, policy, perturbed_obs, manifest, qpos_indices, qvel_indices, horizon
            )

            overlap = min(len(base), len(perturbed))
            curves.append(
                np.linalg.norm(base[:overlap] - perturbed[:overlap], axis=1).astype(np.float32)
            )
    finally:
        env.close()

    return {
        "distance_curves": pad_curves(curves),
        "qpos_indices": qpos_indices,
        "qvel_indices": qvel_indices,
        "perturbation_scale": np.asarray(perturbation_scale, dtype=np.float32),
    }


def rollout_agent_trajectory(
    env: gym.Env,
    policy,
    obs: np.ndarray,
    manifest: dict,
    qpos_indices: np.ndarray,
    qvel_indices: np.ndarray,
    horizon: int,
) -> np.ndarray:
    states = [agent_state(env, manifest, qpos_indices, qvel_indices)]
    primitive_steps = 0
    terminated = truncated = False
    chunk_length = manifest["chunk_length"]

    while primitive_steps < horizon and not (terminated or truncated):
        action_chunk = policy.select_action(obs.reshape(1, -1), deterministic=True)
        actions = np.asarray(action_chunk).reshape((chunk_length, *env.action_space.shape))
        for action in actions:
            obs, _, terminated, truncated, _ = env.step(action)
            states.append(agent_state(env, manifest, qpos_indices, qvel_indices))
            primitive_steps += 1
            if primitive_steps == horizon or terminated or truncated:
                break
    return np.asarray(states, dtype=np.float32)


def controlled_agent_indices(env: gym.Env, manifest: dict) -> tuple[np.ndarray, np.ndarray]:
    if manifest["dataset_source"] == "robomimic":
        robot = env.unwrapped.robots[0]
        qpos = list(robot._ref_joint_pos_indexes)
        qvel = list(robot._ref_joint_vel_indexes)
        for arm in robot.arms:
            qpos.extend(robot._ref_gripper_joint_pos_indexes[arm])
            qvel.extend(robot._ref_gripper_joint_vel_indexes[arm])
        return np.asarray(qpos, dtype=np.int64), np.asarray(qvel, dtype=np.int64)

    if manifest["env_name"] == "Reacher-v5":
        return np.arange(2, dtype=np.int64), np.arange(2, dtype=np.int64)
    return (
        np.arange(env.unwrapped.model.nq, dtype=np.int64),
        np.arange(env.unwrapped.model.nv, dtype=np.int64),
    )


def simulator_state(env: gym.Env, manifest: dict) -> tuple[np.ndarray, np.ndarray]:
    data = env.unwrapped.sim.data if manifest["dataset_source"] == "robomimic" else env.unwrapped.data
    return np.asarray(data.qpos).copy(), np.asarray(data.qvel).copy()


def agent_state(
    env: gym.Env,
    manifest: dict,
    qpos_indices: np.ndarray,
    qvel_indices: np.ndarray,
) -> np.ndarray:
    qpos, qvel = simulator_state(env, manifest)
    return np.concatenate((qpos[qpos_indices], qvel[qvel_indices])).astype(np.float32)


def reset_to_simulator_state(
    env: gym.Env,
    manifest: dict,
    seed: int,
    qpos: np.ndarray,
    qvel: np.ndarray,
) -> np.ndarray:
    env.reset(seed=seed)
    if manifest["dataset_source"] == "robomimic":
        robosuite_env = env.unwrapped
        robosuite_env.sim.data.qpos[:] = qpos
        robosuite_env.sim.data.qvel[:] = qvel
        robosuite_env.sim.forward()
        return current_observation(env, manifest)

    env.unwrapped.set_state(qpos, qvel)
    return current_observation(env, manifest)


def current_observation(env: gym.Env, manifest: dict) -> np.ndarray:
    if manifest["dataset_source"] == "robomimic":
        observations = env.unwrapped._get_observations(force_update=True)
        return env._flatten_obs(observations).astype(np.float32)
    return env.unwrapped._get_obs().astype(np.float32)


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

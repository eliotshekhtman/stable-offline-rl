# Tasks:
# - Reload trained OfflineRL-Kit runs from their run_manifest.json files.
# - Evaluate learned policies and configured expert policies in the true environment.
# - Execute learned action chunks open-loop while retaining primitive-step metrics.
# - Measure final-policy contraction after perturbing only controlled-agent coordinates.
# - Measure policy conservativity over training checkpoints.
# - Evaluate learned dynamics on held-out transitions for validation reports.

import argparse
import fcntl
import hashlib
import inspect
import json
import shutil
import tempfile
from pathlib import Path

import gymnasium as gym
import h5py
import numpy as np
import torch
from scipy.spatial import cKDTree

import chunking
import metrics
import load_offline
import rollout
import task_support
from chunked_dynamics import DIRECT_DYNAMICS_MODE, resolve_dynamics_chunk_mode
from policies import MODEL_BASED_ALGOS, MODEL_FREE_ALGOS, build_model_based_policy, build_model_free_policy


EVALUATION_SCHEMA_VERSION = 2
ROLLOUT_CACHE_VERSION = 1
EXPERT_CACHE_VERSION = 1
FINAL_EVAL_SEED_OFFSET = 1_000_000


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
    run.add_argument("--reuse-eval", action="store_true", help="Reuse matching cached checkpoint rollouts and completed evaluation results")

    rollout_eval = parser.add_argument_group("rollout evaluation")
    rollout_eval.add_argument("--checkpoint-eval-episodes", type=int, default=20, help="Number of true-environment episodes used to monitor each non-final training checkpoint; use 0 to evaluate only the final policy")
    rollout_eval.add_argument("--final-eval-episodes", type=int, default=100, help="Number of true-environment episodes used for the final policy and Gymnasium expert report")

    behavior = parser.add_argument_group("contraction and conservativity evaluation")
    behavior.add_argument("--contraction-trajectories", type=int, default=8, help="Number of matched unperturbed and agent-state-perturbed trajectory pairs")
    behavior.add_argument("--contraction-horizon", type=int, default=300, help="Maximum primitive steps in each contraction trajectory")
    behavior.add_argument("--perturbation-scale", type=float, default=0.01, help="Euclidean norm of the initial perturbation applied to controlled-agent qpos/qvel coordinates")
    behavior.add_argument("--ood-samples", type=int, default=10000, help="Maximum held-out and policy decision-boundary samples used for OOD metrics")
    args = parser.parse_args()
    if args.checkpoint_eval_episodes < 0 or args.final_eval_episodes <= 0:
        parser.error(
            "checkpoint evaluation episodes must be nonnegative and final "
            "evaluation episodes must be positive"
        )
    if args.contraction_trajectories <= 0 or args.contraction_horizon <= 0:
        parser.error("evaluation episode, trajectory, and horizon counts must be positive")
    if args.contraction_trajectories > args.final_eval_episodes:
        parser.error("--contraction-trajectories cannot exceed --final-eval-episodes")
    if args.perturbation_scale < 0.0:
        parser.error("--perturbation-scale must be nonnegative")
    return args


def evaluate_run(run_dir: Path, args: argparse.Namespace) -> Path:
    manifest_path = (Path(run_dir) / "run_manifest.json").resolve()
    with manifest_path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)
    task_support.require_supported_task(
        manifest["env_name"], manifest["dataset_source"]
    )
    eval_dir = Path(manifest["eval_dir"])
    eval_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = eval_dir.parent / f".{eval_dir.name}.evaluation.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        return _evaluate_run_locked(run_dir, args)


def _evaluate_run_locked(run_dir: Path, args: argparse.Namespace) -> Path:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    manifest_path = (run_dir / "run_manifest.json").resolve()
    with manifest_path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)
    task_support.require_supported_task(
        manifest["env_name"], manifest["dataset_source"]
    )
    if args.expert is not None:
        expert_path = Path(args.expert).expanduser()
        if expert_path.suffix != ".zip":
            expert_path = expert_path / f"{manifest['env_name']}.zip"
        manifest["expert"] = str(expert_path.resolve())

    eval_dir = Path(manifest["eval_dir"])
    config = evaluation_config(manifest, args)
    if args.reuse_eval and evaluation_is_complete(eval_dir, config):
        return eval_dir
    if eval_dir.exists() and not args.reuse_eval:
        shutil.rmtree(eval_dir)
    elif eval_dir.exists():
        (eval_dir / "results.json").unlink(missing_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

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
    checkpoints = manifest["checkpoints"]
    last_checkpoint = checkpoints[-1]
    conservativity_reference = prepare_conservativity(
        train_dataset, test_dataset, args.ood_samples, args.seed
    )
    rollout_infos = {}
    records = []
    last_conservativity = None
    final_rollout_seed = args.seed + FINAL_EVAL_SEED_OFFSET
    env = make_eval_env(manifest)
    try:
        chunk_env = chunking.ActionChunkWrapper(env, manifest["chunk_length"])
        policy, _ = load_policy_and_dynamics(
            manifest, args.device, Path(last_checkpoint["policy_path"]),
            None, train_dataset, chunk_env,
        )
        body_ids, body_names = agent_position_bodies(env, manifest)
        last_loaded_path = Path(last_checkpoint["policy_path"])
        for checkpoint in checkpoints:
            is_final = checkpoint["step"] == last_checkpoint["step"]
            if not is_final and args.checkpoint_eval_episodes == 0:
                continue
            episodes = args.final_eval_episodes if is_final else args.checkpoint_eval_episodes
            rollout_seed = final_rollout_seed if is_final else args.seed
            checkpoint_path = Path(checkpoint["policy_path"])
            rollout_info = load_cached_rollout(
                eval_dir, manifest, checkpoint, args, episodes, rollout_seed,
                include_contraction_state=is_final,
            )
            if rollout_info is None:
                if checkpoint_path != last_loaded_path:
                    load_policy_checkpoint(policy, checkpoint_path, args.device)
                    last_loaded_path = checkpoint_path
                rollout_info = evaluate_policy_rollouts(
                    policy, env, manifest, episodes, rollout_seed, body_ids,
                    include_contraction_state=is_final,
                )
                save_cached_rollout(
                    eval_dir, manifest, checkpoint, episodes, rollout_seed,
                    rollout_info, include_contraction_state=is_final,
                )
            rollout_infos[checkpoint["step"]] = rollout_info
            conservativity = evaluate_conservativity(
                conservativity_reference,
                rollout_info["decision_observations"],
                rollout_info["action_chunks"],
                args.ood_samples,
                args.seed,
            )
            records.append(
                checkpoint_record(checkpoint, rollout_info, conservativity, episodes, rollout_seed)
            )
            if is_final:
                last_conservativity = conservativity

        contraction = load_cached_contraction(
            eval_dir, manifest, last_checkpoint, args, final_rollout_seed
        )
        if contraction is None:
            if Path(last_checkpoint["policy_path"]) != last_loaded_path:
                load_policy_checkpoint(policy, Path(last_checkpoint["policy_path"]), args.device)
            contraction = evaluate_contraction(
                policy, env, manifest, rollout_infos[last_checkpoint["step"]],
                args.contraction_trajectories, args.contraction_horizon,
                args.perturbation_scale, final_rollout_seed, body_ids, body_names,
            )
            save_cached_contraction(
                eval_dir, manifest, last_checkpoint, args, final_rollout_seed, contraction
            )
    finally:
        env.close()

    last_info = rollout_infos[last_checkpoint["step"]]
    if last_conservativity is None:
        raise RuntimeError("Final-checkpoint conservativity was not evaluated.")
    expert_info = load_or_evaluate_expert(eval_dir, manifest, args, final_rollout_seed)

    results = {
        "run_manifest_path": str(manifest_path),
        "evaluation_config": config,
        "env_name": manifest["env_name"],
        "algo": manifest["algo"],
        "dataset_tag": manifest["dataset_tag"],
        "chunk_length": manifest["chunk_length"],
        "base_discount": manifest["base_discount"],
        "macro_discount": manifest["macro_discount"],
        "rollout_timestep_unit": "primitive_step",
        "contraction_timestep_unit": "primitive_step",
        "contraction_metric": "L2 distance over controlled-agent Cartesian body positions",
        "contraction_distance_unit": "m",
        "performance_metric": last_info["performance_metric"],
        "performance_label": last_info["performance_label"],
        "performance_higher_is_better": last_info["performance_higher_is_better"],
        "last_checkpoint_step": last_checkpoint["step"],
        "last_checkpoint_percent": last_checkpoint["actual_percent"],
        "last_policy_return_mean": float(last_info["returns"].mean()),
        "last_policy_return_std": float(last_info["returns"].std()),
        "last_policy_performance_mean": float(last_info["performance"].mean()),
        "last_policy_performance_std": float(last_info["performance"].std()),
        "expert_return_mean": expert_info["return_mean"],
        "expert_return_std": expert_info["return_std"],
        "expert_performance_mean": expert_info["performance_mean"],
        "expert_performance_std": expert_info["performance_std"],
    }
    atomic_save_npz(
        eval_dir / "returns_last.npz",
        policy_episode_returns=last_info["returns"],
        expert_episode_returns=expert_info["returns"],
        policy_episode_performance=last_info["performance"],
        expert_episode_performance=expert_info["performance"],
    )
    atomic_save_npz(eval_dir / "conservativity.npz", **last_conservativity)
    results.update(
        last_state_ood_ratio=float(last_conservativity["state_ood_ratio"]),
        last_state_action_ood_ratio=float(last_conservativity["state_action_ood_ratio"]),
    )
    save_history(manifest, expert_info, records, eval_dir)
    atomic_write_json(eval_dir / "results.json", results)
    return eval_dir


def evaluation_config(manifest: dict, args: argparse.Namespace) -> dict:
    config = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "device": args.device,
        "checkpoint_eval_episodes": args.checkpoint_eval_episodes,
        "final_eval_episodes": args.final_eval_episodes,
        "expert": manifest["expert"],
        "seed": args.seed,
        "contraction_trajectories": args.contraction_trajectories,
        "contraction_horizon_primitive_steps": args.contraction_horizon,
        "perturbation_scale": args.perturbation_scale,
        "ood_samples": args.ood_samples,
    }
    task_semantics = manifest["training_schema"]["dataset"].get("task_semantics")
    if task_semantics is not None:
        config["task_semantics"] = task_semantics
    return config


def evaluation_is_complete(eval_dir: Path, config: dict) -> bool:
    required = (
        "results.json", "history.json", "history.npz", "returns_last.npz",
        "contraction_last.npz", "conservativity.npz",
    )
    results_path = eval_dir / "results.json"
    if not results_path.exists() or not all((eval_dir / name).exists() for name in required):
        return False
    try:
        with results_path.open("r", encoding="utf-8") as file:
            results = json.load(file)
        return (
            isinstance(results, dict)
            and results.get("evaluation_config") == config
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def load_policy_and_dynamics(
    manifest: dict,
    device: str,
    policy_path: Path,
    dynamics_path: Path | None,
    train_dataset: dict[str, np.ndarray],
    env: gym.Env,
):
    build_args = argparse.Namespace(
        device=device,
        epoch=manifest["epoch"],
        step_per_epoch=manifest["step_per_epoch"],
    )
    if manifest["algo"] in MODEL_BASED_ALGOS:
        model_based = manifest["training_schema"]["model_based"]
        build_args.model_manipulation_settings = "manipulation_settings" in model_based
        build_args.model_actor_learning_rate = model_based.get("actor_learning_rate")
        build_args.model_critic_learning_rate = model_based.get("critic_learning_rate", 3e-4)
        build_args.mopo_penalty_coef = manifest["training_schema"].get("mopo", {}).get("penalty_coef", 0.5)
        build_args.mobile_penalty_coef = manifest["training_schema"].get("mobile", {}).get("penalty_coef", 1.5)
    if manifest["algo"] == "mobile":
        mobile = manifest["training_schema"].get("mobile", {})
        build_args.mobile_return_shift = mobile.get("return_shift", 0.0)
    elif manifest["algo"] == "iql":
        iql = manifest["training_schema"]["iql"]
        build_args.iql_temperature = iql["temperature"]
        build_args.iql_expectile = iql["expectile"]
        build_args.iql_learning_rate = iql["learning_rate"]
        build_args.iql_lr_schedule = iql["lr_schedule"]
        build_args.iql_hidden_dims = iql["hidden_dims"]
    elif manifest["algo"] == "td3bc":
        td3bc = manifest["training_schema"]["td3bc"]
        build_args.td3bc_learning_rate = td3bc["learning_rate"]
        build_args.td3bc_alpha = td3bc["alpha"]
        build_args.td3bc_hidden_dims = td3bc["hidden_dims"]

    if manifest["algo"] in MODEL_BASED_ALGOS:
        chunk_length = manifest["chunk_length"]
        action_dim = int(np.prod(env.action_space.shape))
        if action_dim % chunk_length:
            raise ValueError(
                f"Chunk action dimension {action_dim} is not divisible by "
                f"chunk length {chunk_length}"
            )
        builder_kwargs = dict(
            discount=manifest["macro_discount"],
            chunk_length=chunk_length,
            base_discount=manifest["base_discount"],
            dynamics_chunk_mode=manifest_dynamics_chunk_mode(manifest),
            primitive_action_dim=action_dim // chunk_length,
        )
        builder_parameters = inspect.signature(
            build_model_based_policy
        ).parameters
        if (
            "build_dynamics_model" in builder_parameters
            or any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in builder_parameters.values()
            )
        ):
            builder_kwargs["build_dynamics_model"] = dynamics_path is not None
        policy, dynamics, _ = build_model_based_policy(
            manifest["algo"], env, build_args, **builder_kwargs
        )
        if dynamics_path is not None:
            dynamics.load(str(dynamics_path))
    elif manifest["algo"] in MODEL_FREE_ALGOS:
        buffer = None
        if manifest["algo"] == "td3bc":
            observations = np.asarray(train_dataset["observations"])
            obs_mean = observations.mean(axis=0, keepdims=True)
            obs_std = observations.std(axis=0, keepdims=True) + 1e-3
            buffer = argparse.Namespace(
                normalize_obs=lambda: (obs_mean, obs_std)
            )
        policy, _ = build_model_free_policy(
            manifest["algo"], env, buffer, build_args,
            discount=manifest["macro_discount"],
        )
        dynamics = None
    else:
        raise ValueError(f"Unsupported algorithm: {manifest['algo']}")

    policy.load_state_dict(torch.load(policy_path, map_location=device, weights_only=True))
    policy.eval()
    return policy, dynamics


def manifest_dynamics_chunk_mode(manifest: dict) -> str:
    chunk_dynamics = manifest["training_schema"]["model_based"].get(
        "chunk_dynamics"
    )
    if chunk_dynamics is None:
        requested_mode = DIRECT_DYNAMICS_MODE
    else:
        if chunk_dynamics.get("version") != 1:
            raise ValueError(
                "Unsupported chunk dynamics schema version: "
                f"{chunk_dynamics.get('version')!r}"
            )
        requested_mode = chunk_dynamics["mode"]
    return resolve_dynamics_chunk_mode(requested_mode, manifest["chunk_length"])


def load_policy_checkpoint(policy, path: Path, device: str) -> None:
    policy.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    policy.eval()


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
    ph_spec = load_offline.list_robomimic_dataset_specs(metadata["task"], "ph")[0]
    ph_dataset_dir = metadata_path.parents[2] / load_offline.make_robomimic_dataset_tag(
        ph_spec
    )
    ph_metadata_paths = sorted(ph_dataset_dir.glob("*/metadata.json"), reverse=True)
    if ph_metadata_paths:
        return load_offline.load_metadata(ph_metadata_paths[0])
    _, ph_metadata = load_offline.load_robomimic_dataset(ph_spec)
    return ph_metadata


def checkpoint_record(
    checkpoint: dict,
    rollout_info: dict,
    conservativity: dict,
    episodes: int,
    rollout_seed: int,
) -> dict:
    return {
        "requested_percent": checkpoint["requested_percent"],
        "actual_percent": checkpoint["actual_percent"],
        "step": checkpoint["step"],
        "evaluation_episodes": episodes,
        "evaluation_seed": rollout_seed,
        "policy_return_mean": float(rollout_info["returns"].mean()),
        "policy_return_std": float(rollout_info["returns"].std()),
        "policy_performance_mean": float(rollout_info["performance"].mean()),
        "policy_performance_std": float(rollout_info["performance"].std()),
        "state_ood_ratio": float(conservativity["state_ood_ratio"]),
        "state_action_ood_ratio": float(conservativity["state_action_ood_ratio"]),
    }


def save_history(manifest: dict, expert_info: dict, records: list[dict], eval_dir: Path) -> None:
    performance_metric, performance_label, higher_is_better = performance_definition(manifest["env_name"])
    history = {
        "env_name": manifest["env_name"],
        "algo": manifest["algo"],
        "dataset_tag": manifest["dataset_tag"],
        "chunk_length": manifest["chunk_length"],
        "base_discount": manifest["base_discount"],
        "macro_discount": manifest["macro_discount"],
        "performance_metric": performance_metric,
        "performance_label": performance_label,
        "performance_higher_is_better": higher_is_better,
        "expert_return_mean": expert_info["return_mean"],
        "expert_performance_mean": expert_info["performance_mean"],
        "records": records,
    }
    atomic_save_npz(
        eval_dir / "history.npz",
        **{key: np.asarray([record[key] for record in records]) for key in records[0]},
    )
    atomic_write_json(eval_dir / "history.json", history)


def evaluate_policy_rollouts(
    policy,
    env: gym.Env,
    manifest: dict,
    episodes: int,
    seed: int,
    body_ids: np.ndarray,
    include_contraction_state: bool = True,
) -> dict:
    env_name = manifest["env_name"]
    returns, performance = [], []
    decision_observations, action_chunks = [], []
    position_trajectories, initial_qpos, initial_qvel = [], [], []

    for episode in range(episodes):
        episode_info = rollout_policy_episode(
            env, policy, manifest, seed + episode, body_ids,
            collect_positions=include_contraction_state,
        )
        returns.append(episode_info["return"])
        performance.append(episode_info["performance"])
        decision_observations.extend(episode_info["decision_observations"])
        action_chunks.extend(episode_info["action_chunks"])
        if include_contraction_state:
            position_trajectories.append(episode_info["positions"])
            initial_qpos.append(episode_info["initial_qpos"])
            initial_qvel.append(episode_info["initial_qvel"])

    performance_metric, performance_label, higher_is_better = performance_definition(env_name)
    result = {
        "returns": np.asarray(returns, dtype=np.float32),
        "performance": np.asarray(performance, dtype=np.float32),
        "performance_metric": performance_metric,
        "performance_label": performance_label,
        "performance_higher_is_better": higher_is_better,
        "decision_observations": np.asarray(decision_observations, dtype=np.float32),
        "action_chunks": np.asarray(action_chunks, dtype=np.float32),
    }
    if include_contraction_state:
        result.update(
            position_trajectories=pad_trajectories(position_trajectories),
            position_lengths=np.asarray(
                [len(item) for item in position_trajectories], dtype=np.int64
            ),
            initial_qpos=np.asarray(initial_qpos, dtype=np.float64),
            initial_qvel=np.asarray(initial_qvel, dtype=np.float64),
        )
    return result


def rollout_policy_episode(
    env: gym.Env,
    policy,
    manifest: dict,
    reset_seed: int,
    body_ids: np.ndarray,
    horizon: int | None = None,
    initial_qpos: np.ndarray | None = None,
    initial_qvel: np.ndarray | None = None,
    stop_on_success: bool = True,
    collect_positions: bool = True,
) -> dict:
    if initial_qpos is None:
        _, reset_info = env.reset(seed=reset_seed)
        obs = current_observation(env, manifest)
    else:
        _, reset_info = env.reset(seed=reset_seed)
        obs = set_simulator_state(env, manifest, initial_qpos, initial_qvel)
    if collect_positions:
        start_qpos, start_qvel = simulator_state(env, manifest)
        positions = [agent_positions(env, manifest, body_ids)]
    decision_observations, action_chunks = [], []
    episode_return = 0.0
    primitive_steps = 0
    succeeded = False
    final_info = reset_info
    done = False

    seed_policy_randomness(reset_seed + 100000)
    while not done and (horizon is None or primitive_steps < horizon):
        action_chunk = policy.select_action(obs.reshape(1, -1), deterministic=True).reshape(-1)
        decision_observations.append(np.asarray(obs, dtype=np.float32).copy())
        action_chunks.append(np.asarray(action_chunk, dtype=np.float32).copy())
        actions = action_chunk.reshape((manifest["chunk_length"], *env.action_space.shape))
        for action in actions:
            obs, reward, terminated, truncated, final_info = env.step(action)
            episode_return += float(reward)
            primitive_steps += 1
            if collect_positions:
                positions.append(agent_positions(env, manifest, body_ids))
            task_succeeded = (
                manifest["dataset_source"] == "robomimic"
                and bool(env.unwrapped._check_success())
            )
            succeeded |= task_succeeded
            done = terminated or truncated or (stop_on_success and task_succeeded)
            if done or (horizon is not None and primitive_steps == horizon):
                break

    result = {
        "return": episode_return,
        "performance": episode_performance(
            manifest["env_name"], env, episode_return, primitive_steps,
            succeeded, reset_info, final_info,
        ),
        "decision_observations": decision_observations,
        "action_chunks": action_chunks,
    }
    if collect_positions:
        result.update(
            positions=np.asarray(positions, dtype=np.float32),
            initial_qpos=start_qpos,
            initial_qvel=start_qvel,
        )
    return result


def rollout_cache_config(
    manifest: dict,
    checkpoint: dict,
    episodes: int,
    rollout_seed: int,
    include_contraction_state: bool = True,
) -> dict:
    return {
        "version": ROLLOUT_CACHE_VERSION,
        "env_name": manifest["env_name"],
        "dataset_source": manifest["dataset_source"],
        "chunk_length": manifest["chunk_length"],
        "checkpoint_step": checkpoint["step"],
        "policy_path": checkpoint["policy_path"],
        "episodes": episodes,
        "seed": rollout_seed,
        "contraction_state": include_contraction_state,
    }


def load_cached_rollout(
    eval_dir: Path,
    manifest: dict,
    checkpoint: dict,
    args: argparse.Namespace,
    episodes: int,
    rollout_seed: int,
    include_contraction_state: bool = True,
) -> dict | None:
    if not args.reuse_eval:
        return None
    path = eval_dir / "rollouts" / f"step_{checkpoint['step']}.npz"
    config_path = path.with_suffix(".json")
    if not path.exists() or not config_path.exists():
        return None
    expected_config = rollout_cache_config(
        manifest, checkpoint, episodes, rollout_seed, include_contraction_state
    )
    legacy_config = dict(expected_config)
    legacy_config.pop("contraction_state")
    try:
        with config_path.open("r", encoding="utf-8") as file:
            if json.load(file) not in (expected_config, legacy_config):
                return None
        with np.load(path) as data:
            rollout_info = {key: data[key] for key in data.files}
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None
    required = {"returns", "performance", "decision_observations", "action_chunks"}
    if include_contraction_state:
        required.update({
            "position_trajectories", "position_lengths", "initial_qpos", "initial_qvel",
        })
    if not required.issubset(rollout_info):
        return None
    metric, label, higher_is_better = performance_definition(manifest["env_name"])
    rollout_info.update(
        performance_metric=metric,
        performance_label=label,
        performance_higher_is_better=higher_is_better,
    )
    return rollout_info


def save_cached_rollout(
    eval_dir: Path,
    manifest: dict,
    checkpoint: dict,
    episodes: int,
    rollout_seed: int,
    rollout_info: dict,
    include_contraction_state: bool = True,
) -> None:
    path = eval_dir / "rollouts" / f"step_{checkpoint['step']}.npz"
    path.parent.mkdir(exist_ok=True)
    arrays = {
        key: value for key, value in rollout_info.items()
        if isinstance(value, np.ndarray) and key != "decision_episode_ids"
    }
    if not include_contraction_state:
        for key in (
            "position_trajectories", "position_lengths", "initial_qpos", "initial_qvel",
        ):
            arrays.pop(key, None)
    config_path = path.with_suffix(".json")
    config_path.unlink(missing_ok=True)
    atomic_save_npz(path, **arrays)
    atomic_write_json(
        config_path,
        rollout_cache_config(
            manifest, checkpoint, episodes, rollout_seed,
            include_contraction_state,
        ),
    )


def contraction_cache_config(
    manifest: dict,
    checkpoint: dict,
    args: argparse.Namespace,
    rollout_seed: int,
) -> dict:
    config = {
        "version": EVALUATION_SCHEMA_VERSION,
        "checkpoint_step": checkpoint["step"],
        "trajectories": args.contraction_trajectories,
        "horizon": args.contraction_horizon,
        "perturbation_scale": args.perturbation_scale,
        "seed": rollout_seed,
    }
    task_semantics = manifest["training_schema"]["dataset"].get("task_semantics")
    if task_semantics is not None:
        config["task_semantics"] = task_semantics
    return config


def load_cached_contraction(
    eval_dir: Path,
    manifest: dict,
    checkpoint: dict,
    args: argparse.Namespace,
    rollout_seed: int,
) -> dict | None:
    if not args.reuse_eval:
        return None
    path = eval_dir / "contraction_last.npz"
    config_path = eval_dir / "contraction_last.json"
    if not path.exists() or not config_path.exists():
        return None
    try:
        with config_path.open("r", encoding="utf-8") as file:
            if json.load(file) != contraction_cache_config(
                manifest, checkpoint, args, rollout_seed
            ):
                return None
        with np.load(path) as data:
            return {key: data[key] for key in data.files}
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


def save_cached_contraction(
    eval_dir: Path,
    manifest: dict,
    checkpoint: dict,
    args: argparse.Namespace,
    rollout_seed: int,
    contraction: dict,
) -> None:
    config_path = eval_dir / "contraction_last.json"
    config_path.unlink(missing_ok=True)
    atomic_save_npz(eval_dir / "contraction_last.npz", **contraction)
    atomic_write_json(
        config_path,
        contraction_cache_config(manifest, checkpoint, args, rollout_seed),
    )


def load_or_evaluate_expert(
    eval_dir: Path,
    manifest: dict,
    args: argparse.Namespace,
    rollout_seed: int,
) -> dict:
    shared_config = expert_cache_config(manifest, args, rollout_seed)
    shared_path, shared_config_path, lock_path = shared_expert_cache_paths(
        eval_dir, shared_config
    )
    cached = load_expert_cache(shared_path, shared_config_path, shared_config)
    if cached is not None:
        return cached

    legacy_path = eval_dir / "expert.npz"
    legacy_config_path = eval_dir / "expert.json"
    legacy_config = {
        "version": EVALUATION_SCHEMA_VERSION,
        "env_name": manifest["env_name"],
        "expert": manifest["expert"],
        "episodes": args.final_eval_episodes,
        "seed": rollout_seed,
    }
    legacy = None
    if args.reuse_eval:
        legacy = load_expert_cache(
            legacy_path, legacy_config_path, legacy_config
        )

    shared_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        cached = load_expert_cache(
            shared_path, shared_config_path, shared_config
        )
        if cached is not None:
            return cached
        expert_info = legacy or evaluate_expert(
            manifest, args.final_eval_episodes, rollout_seed
        )
        save_expert_cache(
            shared_path, shared_config_path, shared_config, expert_info
        )
        return expert_info


def expert_cache_config(
    manifest: dict,
    args: argparse.Namespace,
    rollout_seed: int,
) -> dict:
    config = {
        "version": EXPERT_CACHE_VERSION,
        "env_name": manifest["env_name"],
    }
    if manifest["dataset_source"] == "robomimic":
        ph_spec = load_offline.list_robomimic_dataset_specs(
            manifest["env_name"], "ph"
        )[0]
        config.update(
            baseline="robomimic_ph_demonstrations",
            dataset_source="robomimic",
            repo_id=load_offline.ROBOMIMIC_HF_REPO_ID,
            repo_path=ph_spec["repo_path"],
        )
    else:
        config.update(
            expert=manifest["expert"],
            episodes=args.final_eval_episodes,
            seed=rollout_seed,
        )
    return config


def shared_expert_cache_paths(
    eval_dir: Path,
    config: dict,
) -> tuple[Path, Path, Path]:
    encoded = json.dumps(
        config, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    cache_key = hashlib.sha256(encoded).hexdigest()
    cache_dir = eval_dir.parents[1] / "_expert_cache"
    path = cache_dir / f"{cache_key}.npz"
    return path, path.with_suffix(".json"), path.with_suffix(".lock")


def load_expert_cache(
    path: Path,
    config_path: Path,
    expected_config: dict,
) -> dict | None:
    if not path.exists() or not config_path.exists():
        return None
    try:
        with config_path.open("r", encoding="utf-8") as file:
            if json.load(file) != expected_config:
                return None
        with np.load(path) as data:
            arrays = {key: data[key] for key in data.files}
        returns = arrays["returns"]
        performance = arrays["performance"]
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None
    return {
        "returns": returns,
        "return_mean": float(returns.mean()),
        "return_std": float(returns.std()),
        "performance": performance,
        "performance_mean": float(performance.mean()),
        "performance_std": float(performance.std()),
    }


def save_expert_cache(
    path: Path,
    config_path: Path,
    config: dict,
    expert_info: dict,
) -> None:
    atomic_save_npz(
        path,
        returns=expert_info["returns"],
        performance=expert_info["performance"],
    )
    atomic_write_json(config_path, config)


def atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    with tempfile.NamedTemporaryFile(
        dir=path.parent, suffix=".npz", delete=False
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
    try:
        np.savez_compressed(temporary_path, **arrays)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def atomic_write_json(path: Path, data: dict) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as temporary_file:
        temporary_config_path = Path(temporary_file.name)
        json.dump(data, temporary_file, indent=2, sort_keys=True)
    try:
        temporary_config_path.replace(path)
    finally:
        temporary_config_path.unlink(missing_ok=True)


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
    if env_name == "HalfCheetah-v5":
        return "forward_displacement", "forward displacement (m)", True
    raise ValueError(f"Unsupported task {env_name!r}.")


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
    if env_name == "HalfCheetah-v5":
        return float(final_info["x_position"] - reset_info["x_position"])
    raise ValueError(f"Unsupported task {env_name!r}.")


def evaluate_contraction(
    policy,
    env: gym.Env,
    manifest: dict,
    base_rollouts: dict,
    trajectory_count: int,
    horizon: int,
    perturbation_scale: float,
    seed: int,
    body_ids: np.ndarray,
    body_names: list[str],
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    curves = []
    qpos_indices, qvel_indices = controlled_agent_indices(env, manifest)
    continuing = (
        manifest["training_schema"]["dataset"].get("task_semantics") == "continuing"
    )

    for pair_index in range(trajectory_count):
        qpos = base_rollouts["initial_qpos"][pair_index]
        qvel = base_rollouts["initial_qvel"][pair_index]
        direction = rng.normal(size=len(qpos_indices) + len(qvel_indices))
        direction /= np.linalg.norm(direction)
        perturbed_qpos = qpos.copy()
        perturbed_qvel = qvel.copy()
        split = len(qpos_indices)
        perturbed_qpos[qpos_indices] += perturbation_scale * direction[:split]
        perturbed_qvel[qvel_indices] += perturbation_scale * direction[split:]

        if continuing:
            base = rollout_policy_episode(
                env, policy, manifest, seed + pair_index, body_ids,
                horizon=horizon,
                initial_qpos=qpos,
                initial_qvel=qvel,
                stop_on_success=False,
            )["positions"]
        else:
            base_length = min(int(base_rollouts["position_lengths"][pair_index]), horizon + 1)
            base = base_rollouts["position_trajectories"][pair_index, :base_length]
        perturbed = rollout_policy_episode(
            env, policy, manifest, seed + pair_index, body_ids,
            horizon=horizon,
            initial_qpos=perturbed_qpos,
            initial_qvel=perturbed_qvel,
            stop_on_success=not continuing,
        )["positions"]
        overlap = min(len(base), len(perturbed))
        curves.append(
            np.linalg.norm(base[:overlap] - perturbed[:overlap], axis=1).astype(np.float32)
        )

    return {
        "distance_curves": pad_curves(curves),
        "qpos_indices": qpos_indices,
        "qvel_indices": qvel_indices,
        "position_body_ids": body_ids,
        "position_body_names": np.asarray(body_names),
        "perturbation_scale": np.asarray(perturbation_scale, dtype=np.float32),
    }


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


def agent_position_bodies(env: gym.Env, manifest: dict) -> tuple[np.ndarray, list[str]]:
    if manifest["dataset_source"] == "robomimic":
        robot = env.unwrapped.robots[0]
        body_names = list(robot.robot_model.bodies)
        for arm in robot.arms:
            body_names.extend(robot.gripper[arm].bodies)
        body_names = list(dict.fromkeys(body_names))
        body_ids = [env.unwrapped.sim.model.body_name2id(name) for name in body_names]
        return np.asarray(body_ids, dtype=np.int64), body_names

    model = env.unwrapped.model
    excluded = {"world", "target"} if manifest["env_name"] == "Reacher-v5" else {"world"}
    body_names = [model.body(index).name for index in range(model.nbody)]
    body_names = [name for name in body_names if name not in excluded]
    body_ids = [model.body(name).id for name in body_names]
    return np.asarray(body_ids, dtype=np.int64), body_names


def simulator_state(env: gym.Env, manifest: dict) -> tuple[np.ndarray, np.ndarray]:
    data = env.unwrapped.sim.data if manifest["dataset_source"] == "robomimic" else env.unwrapped.data
    return np.asarray(data.qpos).copy(), np.asarray(data.qvel).copy()


def agent_positions(
    env: gym.Env,
    manifest: dict,
    body_ids: np.ndarray,
) -> np.ndarray:
    if manifest["dataset_source"] == "robomimic":
        positions = env.unwrapped.sim.data.body_xpos[body_ids]
    else:
        positions = env.unwrapped.data.xpos[body_ids]
    return np.asarray(positions, dtype=np.float32).reshape(-1).copy()


def set_simulator_state(
    env: gym.Env,
    manifest: dict,
    qpos: np.ndarray,
    qvel: np.ndarray,
) -> np.ndarray:
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


def prepare_conservativity(
    train_dataset: dict[str, np.ndarray],
    test_dataset: dict[str, np.ndarray],
    sample_count: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    train_indices = rng.choice(len(train_dataset["observations"]), size=min(50000, len(train_dataset["observations"])), replace=False)
    test_indices = rng.choice(len(test_dataset["observations"]), size=min(sample_count, len(test_dataset["observations"])), replace=False)

    train_states = train_dataset["observations"][train_indices]
    state_mean = train_dataset["observations"].mean(axis=0)
    state_std = train_dataset["observations"].std(axis=0)
    state_std[state_std == 0.0] = 1.0
    state_reference = (train_states - state_mean) / state_std
    state_tree = cKDTree(state_reference)
    test_state_distances = metrics.knn_distances(
        state_tree, (test_dataset["observations"][test_indices] - state_mean) / state_std
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
    state_action_tree = cKDTree(state_action_reference)
    test_state_actions = np.concatenate(
        [test_dataset["observations"][test_indices], test_dataset["actions"][test_indices]], axis=1
    )
    test_state_action_distances = metrics.knn_distances(
        state_action_tree, (test_state_actions - state_action_mean) / state_action_std
    )

    return {
        "state_tree": state_tree,
        "state_action_tree": state_action_tree,
        "state_mean": state_mean,
        "state_std": state_std,
        "state_action_mean": state_action_mean,
        "state_action_std": state_action_std,
        "test_state_distances": test_state_distances,
        "test_state_action_distances": test_state_action_distances,
        "train_indices": train_indices.astype(np.int64),
        "test_indices": test_indices.astype(np.int64),
    }


def evaluate_conservativity(
    reference: dict,
    rollout_decision_observations: np.ndarray,
    rollout_action_chunks: np.ndarray,
    sample_count: int,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    rollout_indices = rng.choice(
        len(rollout_decision_observations),
        size=min(sample_count, len(rollout_decision_observations)),
        replace=False,
    )
    rollout_state_distances = metrics.knn_distances(
        reference["state_tree"],
        (rollout_decision_observations[rollout_indices] - reference["state_mean"])
        / reference["state_std"],
    )
    rollout_state_actions = np.concatenate(
        [rollout_decision_observations[rollout_indices], rollout_action_chunks[rollout_indices]], axis=1
    )
    rollout_state_action_distances = metrics.knn_distances(
        reference["state_action_tree"],
        (rollout_state_actions - reference["state_action_mean"]) / reference["state_action_std"],
    )

    return {
        "state_ood_ratio": np.asarray(rollout_state_distances.mean() / max(reference["test_state_distances"].mean(), metrics.EPS), dtype=np.float32),
        "state_action_ood_ratio": np.asarray(rollout_state_action_distances.mean() / max(reference["test_state_action_distances"].mean(), metrics.EPS), dtype=np.float32),
        "test_state_distances": reference["test_state_distances"],
        "rollout_state_distances": rollout_state_distances,
        "test_state_action_distances": reference["test_state_action_distances"],
        "rollout_state_action_distances": rollout_state_action_distances,
        "train_indices": reference["train_indices"],
        "test_indices": reference["test_indices"],
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


def pad_trajectories(trajectories: list[np.ndarray]) -> np.ndarray:
    shape = (len(trajectories), max(map(len, trajectories)), trajectories[0].shape[1])
    padded = np.full(shape, np.nan, dtype=np.float32)
    for index, trajectory in enumerate(trajectories):
        padded[index, : len(trajectory)] = trajectory
    return padded


def evaluate_dynamics_on_dataset(
    dynamics,
    dataset: dict[str, np.ndarray],
) -> np.ndarray:
    errors = []
    for start in range(0, len(dataset["observations"]), 8192):
        end = start + 8192
        obs = dataset["observations"][start:end]
        action_chunks = dataset["actions"][start:end]
        pred_next_obs = predict_next_obs(dynamics, obs, action_chunks)
        errors.append(np.mean((pred_next_obs - dataset["next_observations"][start:end]) ** 2, axis=1))
    return np.concatenate(errors, axis=0).astype(np.float32)


def predict_next_obs(
    dynamics,
    obs: np.ndarray,
    action_chunks: np.ndarray,
) -> np.ndarray:
    if hasattr(dynamics, "mean_next_obss"):
        return dynamics.mean_next_obss(obs, action_chunks)
    else:
        model_input = dynamics.scaler.transform(
            np.concatenate([obs, action_chunks], axis=-1)
        )
        with torch.no_grad():
            mean, _ = dynamics.model(model_input)
        elite_indices = dynamics.model.elites.detach().cpu().numpy()
        elite_mean = mean[elite_indices].mean(dim=0).cpu().numpy()
        return obs + elite_mean[:, : obs.shape[1]]


if __name__ == "__main__":
    main()

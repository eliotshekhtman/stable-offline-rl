# Tasks:
# - Evaluate saved policy checkpoints on each run's held-out episode split.
# - Record offline diagnostics without changing training, checkpoint selection, or run identity.
# - Reuse validation results independently of trained-run and environment-evaluation caches.

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import chunking
import rollout


VALIDATION_SCHEMA_VERSION = 1
VALIDATION_SEED_OFFSET = 2_000_000


def validate_run(run_dir: Path, device: str) -> Path:
    """Backfill deterministic held-out metrics for every saved checkpoint."""
    import eval as evaluation

    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    output_path = run_dir / "record" / "validation.csv"
    checkpoints = manifest["checkpoints"]
    if validation_is_complete(output_path, checkpoints):
        return output_path

    train_dataset = chunking.make_action_chunk_dataset(
        rollout.load_dataset(manifest["train_dataset_path"]),
        manifest["chunk_length"],
        manifest["base_discount"],
    )
    heldout_dataset = chunking.make_action_chunk_dataset(
        rollout.load_dataset(manifest["test_dataset_path"]),
        manifest["chunk_length"],
        manifest["base_discount"],
    )
    env = evaluation.make_eval_env(manifest)
    chunk_env = chunking.ActionChunkWrapper(env, manifest["chunk_length"])
    progress = load_training_progress(run_dir)
    first_checkpoint = checkpoints[0]

    try:
        policy, dynamics, obs_mean, obs_std = evaluation.load_policy_and_dynamics(
            manifest,
            device,
            Path(first_checkpoint["policy_path"]),
            Path(first_checkpoint["dynamics_path"]) if "dynamics_path" in first_checkpoint else None,
            train_dataset,
            chunk_env,
        )
        policy_dataset = prepare_policy_dataset(
            manifest, policy, train_dataset, heldout_dataset, obs_mean, obs_std
        )
        loaded_dynamics_path = first_checkpoint.get("dynamics_path")
        dynamics_errors = {}
        records = []

        for checkpoint in checkpoints:
            evaluation.load_policy_checkpoint(policy, Path(checkpoint["policy_path"]), device)
            restore_logged_alpha(policy, progress.get(checkpoint["step"]))

            dynamics_path = checkpoint.get("dynamics_path")
            if dynamics is not None and dynamics_path != loaded_dynamics_path:
                dynamics.load(dynamics_path)
                loaded_dynamics_path = dynamics_path
            if dynamics is not None:
                dynamics.model.eval()

            rng_devices = cuda_rng_devices(device)
            with torch.random.fork_rng(devices=rng_devices):
                seed = manifest["seed"] + VALIDATION_SEED_OFFSET
                torch.manual_seed(seed)
                if rng_devices:
                    torch.cuda.manual_seed_all(seed)
                metrics = validate_policy(
                    manifest["algo"], policy, policy_dataset,
                    batch_size=manifest["batch_size"],
                )

            if dynamics is not None:
                if dynamics_path not in dynamics_errors:
                    errors = evaluation.evaluate_dynamics_on_dataset(
                        dynamics, heldout_dataset, obs_mean=obs_mean, obs_std=obs_std
                    )
                    dynamics_errors[dynamics_path] = float(errors.mean())
                metrics["heldout/dynamics_next_obs_mse"] = dynamics_errors[dynamics_path]

            records.append(
                {
                    "validation_schema_version": VALIDATION_SCHEMA_VERSION,
                    "step": checkpoint["step"],
                    "requested_percent": checkpoint["requested_percent"],
                    "actual_percent": checkpoint["actual_percent"],
                    "heldout/num_transitions": len(heldout_dataset["observations"]),
                    **metrics,
                }
            )
    finally:
        chunk_env.close()

    write_validation_csv(output_path, records)
    return output_path


def validation_is_complete(path: Path, checkpoints: list[dict]) -> bool:
    if not path.exists():
        return False
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    return (
        len(rows) == len(checkpoints)
        and all(int(row["validation_schema_version"]) == VALIDATION_SCHEMA_VERSION for row in rows)
        and [int(row["step"]) for row in rows] == [item["step"] for item in checkpoints]
    )


def prepare_policy_dataset(
    manifest: dict,
    policy,
    train_dataset: dict[str, np.ndarray],
    heldout_dataset: dict[str, np.ndarray],
    obs_mean: np.ndarray | None,
    obs_std: np.ndarray | None,
) -> dict[str, np.ndarray]:
    dataset = {
        key: np.asarray(heldout_dataset[key], dtype=np.float32).copy()
        for key in ("observations", "actions", "next_observations", "rewards", "terminals")
    }
    dataset["rewards"] = dataset["rewards"].reshape(-1, 1)
    dataset["terminals"] = dataset["terminals"].reshape(-1, 1)
    if manifest["algo"] == "td3bc":
        dataset["observations"] = policy.scaler.transform(dataset["observations"])
        dataset["next_observations"] = policy.scaler.transform(dataset["next_observations"])
    elif manifest["algo"] == "rambo":
        dataset["observations"] = (dataset["observations"] - obs_mean) / obs_std
        dataset["next_observations"] = (dataset["next_observations"] - obs_mean) / obs_std

    model_based = manifest["training_schema"].get("model_based", {})
    manipulation = model_based.get("manipulation_settings")
    if manipulation and manipulation["reward_normalization"] == "zscore":
        train_rewards = train_dataset["rewards"]
        dataset["rewards"] = (
            (dataset["rewards"] - train_rewards.mean()) / (train_rewards.std() + 1e-3)
        ).astype(np.float32)
    return dataset


def validate_policy(
    algo: str,
    policy,
    dataset: dict[str, np.ndarray],
    batch_size: int,
) -> dict[str, float]:
    sums = defaultdict(float)
    maxima = defaultdict(lambda: -np.inf)
    count = len(dataset["observations"])

    for start in range(0, count, batch_size):
        batch_count = min(batch_size, count - start)
        batch = {
            key: torch.as_tensor(value[start:start + batch_count], device=policy_device(policy))
            for key, value in dataset.items()
        }
        with torch.no_grad():
            means, batch_maxima = validation_batch(algo, policy, batch)
        for key, value in means.items():
            sums[key] += float(value) * batch_count
        for key, value in batch_maxima.items():
            maxima[key] = max(maxima[key], float(value))

    metrics = {key: value / count for key, value in sums.items()}
    metrics.update(maxima)
    if algo == "td3bc":
        metrics["heldout/td3bc_lambda"] = float(policy._alpha) / metrics["heldout/policy_q_abs_mean"]
    return metrics


def validation_batch(algo: str, policy, batch: dict[str, torch.Tensor]):
    if algo == "bc":
        action_mse = (policy.actor(batch["observations"]) - batch["actions"]).square().mean()
        return {"heldout/action_mse": action_mse}, {}
    if algo == "td3bc":
        return td3bc_metrics(policy, batch)
    if algo == "iql":
        return iql_metrics(policy, batch)
    if algo == "dql":
        return dql_metrics(policy, batch)
    if algo == "edac":
        return ensemble_sac_metrics(policy, batch)
    if algo == "mobile":
        return mobile_metrics(policy, batch)
    return twin_sac_metrics(policy, batch)


def td3bc_metrics(policy, batch):
    obs, actions = batch["observations"], batch["actions"]
    q1, q2 = policy.critic1(obs, actions), policy.critic2(obs, actions)
    noise = (torch.randn_like(actions) * policy._policy_noise).clamp(
        -policy._noise_clip, policy._noise_clip
    )
    next_actions = (policy.actor_old(batch["next_observations"]) + noise).clamp(
        -policy._max_action, policy._max_action
    )
    next_q = torch.minimum(
        policy.critic1_old(batch["next_observations"], next_actions),
        policy.critic2_old(batch["next_observations"], next_actions),
    )
    target = batch["rewards"] + policy._gamma * (1.0 - batch["terminals"]) * next_q
    policy_actions = policy.actor(obs)
    policy_q = policy.critic1(obs, policy_actions)
    data_q = torch.minimum(q1, q2)
    return {
        "heldout/q1_td_mse": (q1 - target).square().mean(),
        "heldout/q2_td_mse": (q2 - target).square().mean(),
        "heldout/action_mse": (policy_actions - actions).square().mean(),
        "heldout/q_data_mean": data_q.mean(),
        "heldout/policy_q_mean": policy_q.mean(),
        "heldout/policy_q_abs_mean": policy_q.abs().mean(),
        "heldout/policy_data_q_gap": policy_q.mean() - data_q.mean(),
    }, {"heldout/q_abs_max": torch.maximum(q1.abs().max(), q2.abs().max())}


def iql_metrics(policy, batch):
    obs, actions = batch["observations"], batch["actions"]
    q1, q2 = policy.critic_q1(obs, actions), policy.critic_q2(obs, actions)
    old_q = torch.minimum(
        policy.critic_q1_old(obs, actions), policy.critic_q2_old(obs, actions)
    )
    value = policy.critic_v(obs)
    target = batch["rewards"] + policy._gamma * (1.0 - batch["terminals"]) * policy.critic_v(
        batch["next_observations"]
    )
    weights = torch.exp((old_q - value) * policy._temperature).clamp(max=100.0)
    dist = policy.actor(obs)
    actor_loss = -(weights * dist.log_prob(actions)).mean()
    policy_actions = dist.mode()
    data_q = torch.minimum(q1, q2)
    return {
        "heldout/q1_td_mse": (q1 - target).square().mean(),
        "heldout/q2_td_mse": (q2 - target).square().mean(),
        "heldout/value_expectile_loss": policy._expectile_regression(old_q - value).mean(),
        "heldout/actor_loss": actor_loss,
        "heldout/action_mse": (policy_actions - actions).square().mean(),
        "heldout/advantage_weight_mean": weights.mean(),
        "heldout/q_data_mean": data_q.mean(),
    }, {
        "heldout/advantage_weight_max": weights.max(),
        "heldout/q_abs_max": torch.maximum(q1.abs().max(), q2.abs().max()),
    }


def dql_metrics(policy, batch):
    obs = policy._normalize(batch["observations"])
    next_obs = policy._normalize(batch["next_observations"])
    actions = policy._normalize_action(batch["actions"])
    q1, q2 = policy.critic(obs, actions)
    next_actions = policy._sample(
        next_obs, len(next_obs), use_ema=True, temperature=1.0
    )
    target = batch["rewards"] * policy.reward_scale
    target = target + (1.0 - batch["terminals"]) * policy.discount * torch.minimum(
        *policy.critic_target(next_obs, next_actions)
    )
    return {
        "heldout/diffusion_bc_loss": policy.actor.loss(actions, obs),
        "heldout/q1_td_mse": (q1 - target).square().mean(),
        "heldout/q2_td_mse": (q2 - target).square().mean(),
        "heldout/q_data_mean": torch.minimum(q1, q2).mean(),
        "heldout/target_q_mean": target.mean(),
    }, {"heldout/q_abs_max": torch.maximum(q1.abs().max(), q2.abs().max())}


def twin_sac_metrics(policy, batch):
    obs, actions, next_obs = batch["observations"], batch["actions"], batch["next_observations"]
    q1, q2 = policy.critic1(obs, actions), policy.critic2(obs, actions)
    next_actions, next_log_probs = policy.actforward(next_obs)
    next_q = torch.minimum(
        policy.critic1_old(next_obs, next_actions), policy.critic2_old(next_obs, next_actions)
    )
    if not getattr(policy, "_deterministic_backup", False):
        next_q = next_q - policy._alpha * next_log_probs
    target = batch["rewards"] + policy._gamma * (1.0 - batch["terminals"]) * next_q
    policy_actions, _ = policy.actforward(obs, deterministic=True)
    policy_q = torch.minimum(
        policy.critic1(obs, policy_actions), policy.critic2(obs, policy_actions)
    )
    data_q = torch.minimum(q1, q2)
    return {
        "heldout/q1_td_mse": (q1 - target).square().mean(),
        "heldout/q2_td_mse": (q2 - target).square().mean(),
        "heldout/action_mse": (policy_actions - actions).square().mean(),
        "heldout/q_data_mean": data_q.mean(),
        "heldout/policy_q_mean": policy_q.mean(),
        "heldout/policy_data_q_gap": policy_q.mean() - data_q.mean(),
    }, {"heldout/q_abs_max": torch.maximum(q1.abs().max(), q2.abs().max())}


def ensemble_sac_metrics(policy, batch):
    obs, actions, next_obs = batch["observations"], batch["actions"], batch["next_observations"]
    qs = policy.critics(obs, actions)
    next_actions, next_log_probs = policy.actforward(next_obs)
    next_q = policy.critics_old(next_obs, next_actions).min(0)[0]
    if not policy._deterministic_backup:
        next_q = next_q - policy._alpha * next_log_probs
    target = batch["rewards"] + policy._gamma * (1.0 - batch["terminals"]) * next_q
    policy_actions, _ = policy.actforward(obs, deterministic=True)
    policy_q = policy.critics(obs, policy_actions).min(0)[0]
    data_q = qs.min(0)[0]
    return {
        "heldout/critic_td_mse": (qs - target.unsqueeze(0)).square().mean(),
        "heldout/action_mse": (policy_actions - actions).square().mean(),
        "heldout/q_data_mean": data_q.mean(),
        "heldout/policy_q_mean": policy_q.mean(),
        "heldout/policy_data_q_gap": policy_q.mean() - data_q.mean(),
    }, {"heldout/q_abs_max": qs.abs().max()}


def mobile_metrics(policy, batch):
    obs, actions, next_obs = batch["observations"], batch["actions"], batch["next_observations"]
    qs = torch.stack([critic(obs, actions) for critic in policy.critics])
    if policy._max_q_backup:
        repeated_obs = next_obs.unsqueeze(1).repeat(1, 10, 1).flatten(0, 1)
        repeated_actions, _ = policy.actforward(repeated_obs)
        repeated_qs = torch.cat(
            [critic(repeated_obs, repeated_actions) for critic in policy.critics_old], dim=1
        )
        next_q = (
            repeated_qs.view(len(next_obs), 10, len(policy.critics_old))
            .max(1).values.min(1).values[:, None]
        )
    else:
        next_actions, next_log_probs = policy.actforward(next_obs)
        next_q = torch.cat(
            [critic(next_obs, next_actions) for critic in policy.critics_old], dim=1
        ).min(1).values[:, None]
        if not policy._deteterministic_backup:
            next_q = next_q - policy._alpha * next_log_probs
    shifted_rewards = batch["rewards"] + (1.0 - policy._gamma) * policy._return_shift
    target = shifted_rewards + policy._gamma * (1.0 - batch["terminals"]) * next_q
    if policy._clamp_target_q:
        target = target.clamp(min=0.0)
    policy_actions, _ = policy.actforward(obs, deterministic=True)
    policy_q = torch.cat(
        [critic(obs, policy_actions) for critic in policy.critics], dim=1
    ).min(1).values[:, None]
    data_q = qs.min(0)[0]
    return {
        "heldout/critic_td_mse": (qs - target).square().mean(),
        "heldout/action_mse": (policy_actions - actions).square().mean(),
        "heldout/q_data_mean": data_q.mean(),
        "heldout/policy_q_mean": policy_q.mean(),
        "heldout/policy_data_q_gap": policy_q.mean() - data_q.mean(),
    }, {"heldout/q_abs_max": qs.abs().max()}


def load_training_progress(run_dir: Path) -> dict[int, dict[str, str]]:
    path = run_dir / "record" / "policy_training_progress.csv"
    with path.open(newline="", encoding="utf-8") as file:
        return {int(float(row["timestep"])): row for row in csv.DictReader(file)}


def restore_logged_alpha(policy, row: dict[str, str] | None) -> None:
    if row is None or not row.get("alpha") or not hasattr(policy, "_alpha"):
        return
    policy._alpha = torch.as_tensor(float(row["alpha"]), device=policy_device(policy))


def policy_device(policy) -> torch.device:
    return next(policy.parameters()).device


def cuda_rng_devices(device: str) -> list[int]:
    torch_device = torch.device(device)
    if torch_device.type != "cuda":
        return []
    return [torch_device.index if torch_device.index is not None else torch.cuda.current_device()]


def write_validation_csv(path: Path, records: list[dict]) -> None:
    metadata = [
        "validation_schema_version", "step", "requested_percent", "actual_percent",
        "heldout/num_transitions",
    ]
    metric_names = sorted(set().union(*(record.keys() for record in records)) - set(metadata))
    temporary_path = path.with_suffix(".tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=metadata + metric_names)
        writer.writeheader()
        writer.writerows(records)
    temporary_path.replace(path)

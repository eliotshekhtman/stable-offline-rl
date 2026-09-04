# Tasks:
# - Parse the experiment CLI and keep one run focused on one Gymnasium environment.
# - Choose generated, Minari, clean/Minari mixture, or robomimic data.
# - Cache/load datasets, build OfflineRL-Kit replay buffers, and launch trainers.
# - Sweep algorithms and action chunk lengths over each requested random seed.
# - Own experiment directories, milestone checkpoints, logging, seeding, and run naming.

import argparse
import filecmp
import fcntl
import itertools
import json
import math
import os
import random
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

import chunking
import load_offline
import rollout
import task_support
from chunked_dynamics import DYNAMICS_CHUNK_MODES, resolve_dynamics_chunk_mode
from offlinerlkit.buffer import ReplayBuffer
from offlinerlkit.policy_trainer import MBPolicyTrainer, MFPolicyTrainer
from offlinerlkit.utils.logger import Logger
from policies import (
    CQL_ACTION_VOLUME_LAGRANGE_TARGET_MODE,
    CQL_DEFAULT_ENTROPY_ALPHA_MAX,
    CQL_DEFAULT_ENTROPY_LEARNING_RATE,
    CQL_DEFAULT_LAGRANGE_TARGET_MODE,
    CQL_LAGRANGE_TARGET_MODES,
    MODEL_BASED_ALGOS,
    MODEL_FREE_ALGOS,
    build_model_based_policy,
    build_model_free_policy,
)


DEFAULT_STORAGE_ROOT = Path("/data/shekhe/stable-offline-rl")
DATASET_SCHEMA_VERSION = 4
TRAINING_SCHEMA_VERSION = 3
BASE_DISCOUNT = 0.99


def positive_float_or_none(value: str) -> float | None:
    if value.lower() == "none":
        return None
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"expected a positive finite float or 'none', got {value!r}"
        ) from error
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError(
            f"expected a positive finite float or 'none', got {value!r}"
        )
    return parsed


def prepare_storage_root(storage_root: Path) -> None:
    for name in ("datasets", "trained", "evals"):
        directory = storage_root / name
        try:
            directory.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                prefix=".storage-write-test-",
                dir=directory,
            ):
                pass
        except OSError as error:
            raise OSError(
                f"Cannot use storage directory {directory}: {error}"
            ) from error


def main() -> None:
    args = parse_args()
    prepare_storage_root(args.storage_root)
    expert_path = resolve_expert_path(args.expert, args.env)
    eval_dirs = []
    for seed in args.seeds:
        seed_args = argparse.Namespace(**vars(args))
        seed_args.seed = seed
        eval_dirs.extend(
            run_sweep(
                env_name=args.env,
                expert_path=expert_path,
                storage_root=args.storage_root,
                args=seed_args,
            )
        )
    maybe_plot(args.storage_root / "evals" / args.env, eval_dirs, args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect offline datasets and train OfflineRL-Kit policies.")

    experiment = parser.add_argument_group("experiment")
    experiment.add_argument("--env", required=True, help="Gymnasium environment id to train on, e.g. HalfCheetah-v5")
    experiment.add_argument("--seed", dest="seeds", type=int, nargs="+", default=[0], help="Random seeds used for dataset splitting, generated rollouts, training, and evaluation")
    experiment.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Torch device used for OfflineRL-Kit policies and dynamics")
    experiment.add_argument("--quiet", action="store_true", help="Suppress routine output, including the epoch-level training progress bar; warnings and errors remain visible")
    experiment.add_argument(
        "--storage-root",
        type=Path,
        default=DEFAULT_STORAGE_ROOT,
        help=(
            "Parent directory containing datasets, trained runs, and evaluations "
            f"(default: {DEFAULT_STORAGE_ROOT})"
        ),
    )

    dataset = parser.add_argument_group("dataset source and split")
    dataset.add_argument("--dataset-source", choices=["generated", "minari", "clean-minari", "robomimic"], default="generated", help="Use generated rollouts, full Minari datasets, clean-expert/Minari mixtures, or low-dimensional robomimic datasets")
    dataset.add_argument("--dataset", default=None, help="Limit a premade-data sweep to one Robomimic type or Minari leaf/full id; required for clean-minari and omitted to use all datasets for other premade sources")
    dataset.add_argument("--test-fraction", type=float, default=0.2, help="Fraction of each dataset held out for post-training evaluation")

    generated = parser.add_argument_group("generated and clean-Minari dataset options")
    generated.add_argument("--expert", default="/home/shekhe/stable-offline-rl/experts", help="Expert policy .zip path or directory containing <env>.zip; used for generated or clean-Minari expert trajectories and expert evaluation")
    generated.add_argument("--num-samples", type=int, nargs="+", default=[1000000], help="Minimum transition counts for generated and clean-minari datasets; collection always retains complete trajectories")
    generated.add_argument(
        "--noise-scale", type=float, nargs="+", default=[0.0],
        help=(
            "Gaussian action-noise scales to sweep for generated noisy-expert "
            "trajectories; each action coordinate uses standard deviation "
            "scale / sqrt(action dimension)"
        ),
    )
    generated.add_argument(
        "--composition", type=float, nargs=2, action="append", metavar=("CLEAN_EXPERT", "NOISY_EXPERT"),
        help="Generated-data clean and noisy expert trajectory proportions; repeat for multiple compositions, with random trajectories filling the remainder (default: 1 0)",
    )
    generated.add_argument("--max-timesteps", type=int, default=1000, help="Maximum length of each generated clean, noisy, or random rollout trajectory")
    generated.add_argument("--minari-fraction", dest="minari_fractions", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0], help="Clean-minari trajectory fractions to sweep; the remainder comes from clean expert trajectories")

    training = parser.add_argument_group("policy training")
    training.add_argument(
        "--algos", nargs="+", choices=(
            "none", *MODEL_FREE_ALGOS, *MODEL_BASED_ALGOS,
        ),
        default=["cql"], help="Algorithms to train, or none to collect the dataset without training",
    )
    training.add_argument(
        "--chunk-lengths", type=int, nargs="+", default=[1],
        help="Action chunk lengths to sweep over; each policy emits this many primitive actions",
    )
    training.add_argument("--epoch", type=int, default=1000, help="Number of policy-training epochs")
    training.add_argument("--step-per-epoch", type=int, default=1000, help="Gradient-update steps per policy-training epoch")
    training.add_argument("--batch-size", type=int, default=512, help="Policy-training batch size")

    cql = parser.add_argument_group("CQL options")
    cql.add_argument(
        "--cql-entropy-learning-rate",
        type=float,
        default=CQL_DEFAULT_ENTROPY_LEARNING_RATE,
        help="Learning rate for CQL's automatically tuned entropy coefficient",
    )
    cql.add_argument(
        "--cql-entropy-alpha-max",
        type=positive_float_or_none,
        default=CQL_DEFAULT_ENTROPY_ALPHA_MAX,
        help=(
            "Upper bound for CQL's automatically tuned entropy coefficient; "
            "use 'none' for no upper bound"
        ),
    )
    cql.add_argument(
        "--cql-lagrange-target-mode",
        choices=CQL_LAGRANGE_TARGET_MODES,
        default=CQL_DEFAULT_LAGRANGE_TARGET_MODE,
        help=(
            "Lagrange target for CQL on Robomimic tasks: action-volume adds "
            "the chunk's extra log Box volume to the legacy fixed target; "
            "fixed preserves the legacy target of 5"
        ),
    )

    iql = parser.add_argument_group("IQL options")
    iql.add_argument("--iql-temperature", type=float, default=3.0, help="IQL advantage-weighting temperature; larger values favor higher-advantage dataset actions more strongly")
    iql.add_argument("--iql-expectile", type=float, default=0.7, help="IQL value-function expectile, strictly between 0 and 1")
    iql.add_argument("--iql-learning-rate", type=float, default=3e-4, help="Learning rate shared by the IQL actor, two Q-functions, and value function")
    iql.add_argument("--iql-lr-schedule", choices=["cosine", "constant"], default="cosine", help="IQL actor learning-rate schedule; cosine anneals over policy-training epochs")
    iql.add_argument("--iql-hidden-dims", type=int, nargs="+", default=[256, 256], help="Hidden-layer widths shared by the IQL actor, two Q-functions, and value function")

    td3bc = parser.add_argument_group("TD3+BC options")
    td3bc.add_argument("--td3bc-learning-rate", type=float, default=3e-4, help="Learning rate shared by the TD3+BC actor and two critics")
    td3bc.add_argument("--td3bc-alpha", type=float, default=2.5, help="TD3+BC weight on Q-value maximization relative to behavior cloning")
    td3bc.add_argument("--td3bc-hidden-dims", type=int, nargs="+", default=[256, 256], help="Hidden-layer widths shared by the TD3+BC actor and two critics")

    evaluation = parser.add_argument_group("post-training evaluation")
    evaluation.add_argument("--eval", action="store_true", help="After training, run full evaluation for each trained policy and then generate plots")
    evaluation.add_argument("--reuse-eval", action="store_true", help="With --eval, reuse matching cached checkpoint rollouts and completed evaluation results")
    evaluation.add_argument("--checkpoint-eval-episodes", type=int, default=20, help="Number of true-environment episodes used to monitor each non-final training checkpoint; use 0 to evaluate only the final policy")
    evaluation.add_argument("--final-eval-episodes", type=int, default=100, help="Number of true-environment episodes used for the final policy and generated-data expert report")
    evaluation.add_argument("--contraction-trajectories", type=int, default=16, help="Number of matched unperturbed and agent-state-perturbed trajectories for the final policy")
    evaluation.add_argument("--contraction-horizon", type=int, default=300, help="Maximum primitive steps in each final-policy contraction trajectory")
    evaluation.add_argument("--perturbation-scale", type=float, default=0.1, help="Euclidean norm of the initial perturbation applied to controlled-agent qpos/qvel coordinates")
    evaluation.add_argument("--ood-samples", type=int, default=10000, help="Maximum held-out and policy decision-boundary samples used for OOD metrics")

    model_based = parser.add_argument_group("model-based algorithm options")
    model_based.add_argument("--model-actor-learning-rate", type=float, default=None, help="Override the actor learning rate for model-based algorithms; defaults to 1e-4, or 3e-5 with --model-manipulation-settings")
    model_based.add_argument("--model-critic-learning-rate", type=float, default=3e-4, help="Learning rate for model-based critics; the default preserves the existing 3e-4 setting")
    model_based.add_argument("--dynamics-max-epochs", type=int, default=30, help="Maximum epochs for fitting the learned dynamics model before policy training")
    model_based.add_argument(
        "--dynamics-chunk-mode",
        choices=DYNAMICS_CHUNK_MODES,
        default="direct",
        help="Model action chunks directly, or fit primitive one-step dynamics and recursively execute each chunk; length one always uses the legacy direct path",
    )
    model_based.add_argument("--rollout-freq", type=int, default=1000, help="Policy-training step interval between learned-dynamics rollout generation")
    model_based.add_argument("--rollout-batch-size", type=int, default=10000, help="Number of initial real states used when generating model rollouts")
    model_based.add_argument("--rollout-length", type=int, default=5, help="Number of learned macro-transitions per synthetic rollout")
    model_based.add_argument("--model-retain-epochs", type=int, default=5, help="How many epochs of synthetic model rollouts to retain in the fake replay buffer")
    model_based.add_argument("--real-ratio", type=float, default=0.50, help="Fraction of each model-based training batch sampled from the real offline dataset rather than the synthetic rollout buffer")
    model_based.add_argument("--model-manipulation-settings", action="store_true", help="Use MOBILE's published Adroit manipulation architecture, actor learning rate, reward normalization, and MOBILE critic settings for MOPO/MOBILE")
    model_based.add_argument("--mopo-penalty-coef", type=float, default=0.5, help="MOPO coefficient multiplying learned-dynamics uncertainty in synthetic rewards")
    model_based.add_argument("--mobile-penalty-coef", type=float, default=1.5, help="MOBILE coefficient multiplying model-Bellman inconsistency on synthetic targets")
    model_based.add_argument("--mobile-return-shift", type=float, default=30.0, help="Reacher-only MOBILE return/Q shift D used to place the clamped target floor at -D")
    args = parser.parse_args()
    args.storage_root = args.storage_root.expanduser().resolve()
    try:
        task_support.require_supported_task(args.env, args.dataset_source)
    except ValueError as error:
        parser.error(str(error))
    if any(chunk_length <= 0 for chunk_length in args.chunk_lengths):
        parser.error("--chunk-lengths values must be positive")
    if (
        not math.isfinite(args.cql_entropy_learning_rate)
        or args.cql_entropy_learning_rate <= 0.0
    ):
        parser.error("--cql-entropy-learning-rate must be a positive finite float")
    if args.iql_temperature <= 0.0:
        parser.error("--iql-temperature must be positive")
    if not 0.0 < args.iql_expectile < 1.0:
        parser.error("--iql-expectile must be strictly between 0 and 1")
    if args.iql_learning_rate <= 0.0 or any(width <= 0 for width in args.iql_hidden_dims):
        parser.error("IQL learning rate and hidden dimensions must be positive")
    if args.td3bc_learning_rate <= 0.0 or args.td3bc_alpha <= 0.0:
        parser.error("TD3+BC learning rate and alpha must be positive")
    if any(width <= 0 for width in args.td3bc_hidden_dims):
        parser.error("TD3+BC hidden dimensions must be positive")
    if not math.isfinite(args.mobile_return_shift) or args.mobile_return_shift < 0.0:
        parser.error("--mobile-return-shift must be finite and nonnegative")
    if not math.isfinite(args.mopo_penalty_coef) or args.mopo_penalty_coef < 0.0:
        parser.error("--mopo-penalty-coef must be finite and nonnegative")
    if not math.isfinite(args.mobile_penalty_coef) or args.mobile_penalty_coef < 0.0:
        parser.error("--mobile-penalty-coef must be finite and nonnegative")
    if args.model_actor_learning_rate is not None and (
        not math.isfinite(args.model_actor_learning_rate)
        or args.model_actor_learning_rate <= 0.0
    ):
        parser.error("--model-actor-learning-rate must be finite and positive")
    if not math.isfinite(args.model_critic_learning_rate) or args.model_critic_learning_rate <= 0.0:
        parser.error("--model-critic-learning-rate must be finite and positive")
    args.seeds = list(dict.fromkeys(args.seeds))
    args.chunk_lengths = list(dict.fromkeys(args.chunk_lengths))
    args.algos = list(dict.fromkeys(args.algos))
    args.minari_fractions = list(dict.fromkeys(args.minari_fractions))
    if any(
        not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0
        for fraction in args.minari_fractions
    ):
        parser.error("--minari-fraction values must be finite and between 0 and 1")
    args.composition = [(1.0, 0.0)] if args.composition is None else [tuple(values) for values in args.composition]
    args.composition = list(dict.fromkeys(args.composition))
    for prop_clean, prop_noisy in args.composition:
        if prop_clean < 0.0 or prop_noisy < 0.0 or prop_clean + prop_noisy > 1.0:
            parser.error("--composition values must be nonnegative and sum to at most 1")
    if "none" in args.algos and args.algos != ["none"]:
        parser.error("--algos none cannot be combined with training algorithms")
    if args.dataset is not None and args.dataset_source == "generated":
        parser.error("--dataset applies only to premade dataset sources")
    if args.dataset_source == "clean-minari" and args.dataset is None:
        parser.error("--dataset is required with --dataset-source clean-minari")
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
    if args.reuse_eval and not args.eval:
        parser.error("--reuse-eval requires --eval")
    return args


def run_sweep(
    env_name: str,
    expert_path: Path,
    storage_root: Path,
    args: argparse.Namespace,
) -> list[Path]:
    task_support.require_supported_task(env_name, args.dataset_source)
    dataset_root = storage_root / "datasets" / env_name
    trained_root = storage_root / "trained" / env_name
    eval_root = storage_root / "evals" / env_name
    dataset_root.mkdir(parents=True, exist_ok=True)
    trained_root.mkdir(parents=True, exist_ok=True)
    eval_dirs = []

    if args.dataset_source == "minari":
        eval_dirs = run_minari_sweep(env_name, dataset_root, trained_root, eval_root, args)
    elif args.dataset_source == "clean-minari":
        eval_dirs = run_clean_minari_sweep(
            env_name, expert_path, dataset_root, trained_root, eval_root, args
        )
    elif args.dataset_source == "robomimic":
        eval_dirs = run_robomimic_sweep(env_name, dataset_root, trained_root, eval_root, args)
    else:
        for num_samples, noise_scale, composition in itertools.product(
            args.num_samples, args.noise_scale, args.composition
        ):
            prop_clean, prop_noisy = composition
            eval_dirs.extend(
                run_generated_configuration(
                    env_name, expert_path, dataset_root, trained_root, eval_root,
                    num_samples, noise_scale, prop_clean, prop_noisy, args,
                )
            )

    return eval_dirs


def run_generated_configuration(
    env_name: str,
    expert_path: Path,
    dataset_root: Path,
    trained_root: Path,
    eval_root: Path,
    num_samples: int,
    noise_scale: float,
    prop_clean: float,
    prop_noisy: float,
    args: argparse.Namespace,
) -> list[Path]:
    prop_random = 1.0 - prop_clean - prop_noisy
    dataset_tag = make_dataset_tag(
        num_samples, noise_scale, prop_clean, prop_noisy, args.seed
    )
    dataset_schema = {
        "version": DATASET_SCHEMA_VERSION,
        "source": "generated",
        "env_name": env_name,
        "expert_path": str(expert_path),
        "max_timesteps": args.max_timesteps,
        "num_samples": num_samples,
        "noise_scale": noise_scale,
        "prop_clean_expert": prop_clean,
        "prop_noisy_expert": prop_noisy,
        "prop_random": prop_random,
        "prop_expert": prop_clean + prop_noisy,
        "deterministic": True,
        "seed": args.seed,
        "test_fraction": args.test_fraction,
    }
    return train_algos(
        env_name, trained_root, eval_root, dataset_root / dataset_tag,
        dataset_tag, dataset_schema, args,
        lambda: collect_generated_dataset(
            env_name, expert_path, num_samples, noise_scale,
            prop_clean, prop_noisy, args,
        ),
    )


def run_minari_sweep(
    env_name: str,
    dataset_root: Path,
    trained_root: Path,
    eval_root: Path,
    args: argparse.Namespace,
) -> list[Path]:
    eval_dirs = []
    dataset_ids = load_offline.list_minari_dataset_ids(env_name, args.dataset)
    for dataset_id in dataset_ids:
        dataset_tag = load_offline.make_minari_dataset_tag(dataset_id)
        dataset_schema = {
            "version": DATASET_SCHEMA_VERSION,
            "source": "minari",
            "env_name": env_name,
            "dataset_id": dataset_id,
            "seed": args.seed,
            "test_fraction": args.test_fraction,
        }
        eval_dirs.extend(
            train_algos(
                env_name, trained_root, eval_root, dataset_root / dataset_tag,
                dataset_tag, dataset_schema, args,
                lambda dataset_id=dataset_id: load_offline.load_minari_dataset(dataset_id, seed=args.seed),
            )
        )
    return eval_dirs


def run_clean_minari_sweep(
    env_name: str,
    expert_path: Path,
    dataset_root: Path,
    trained_root: Path,
    eval_root: Path,
    args: argparse.Namespace,
) -> list[Path]:
    dataset_id = load_offline.list_minari_dataset_ids(env_name, args.dataset)[0]
    eval_dirs = []

    for num_samples, minari_fraction in itertools.product(
        args.num_samples, args.minari_fractions
    ):
        if minari_fraction == 0.0:
            eval_dirs.extend(
                run_generated_clean_endpoint(
                    env_name, expert_path, dataset_root, trained_root, eval_root,
                    num_samples, args,
                )
            )
            continue

        dataset_tag = make_clean_minari_dataset_tag(
            dataset_id, num_samples, minari_fraction, args.seed
        )
        dataset_schema = {
            "version": DATASET_SCHEMA_VERSION,
            "source": "clean-minari",
            "env_name": env_name,
            "expert_path": str(expert_path),
            "dataset_id": dataset_id,
            "max_timesteps": args.max_timesteps,
            "num_samples": num_samples,
            "minari_fraction": minari_fraction,
            "deterministic": True,
            "seed": args.seed,
            "test_fraction": args.test_fraction,
        }
        eval_dirs.extend(
            train_algos(
                env_name, trained_root, eval_root, dataset_root / dataset_tag,
                dataset_tag, dataset_schema, args,
                lambda num_samples=num_samples, minari_fraction=minari_fraction: collect_clean_minari_dataset(
                    env_name, expert_path, dataset_id, num_samples,
                    minari_fraction, args,
                ),
            )
        )

    return eval_dirs


def run_generated_clean_endpoint(
    env_name: str,
    expert_path: Path,
    dataset_root: Path,
    trained_root: Path,
    eval_root: Path,
    num_samples: int,
    args: argparse.Namespace,
) -> list[Path]:
    generated_args = argparse.Namespace(**vars(args))
    generated_args.dataset_source = "generated"
    configurations = (
        [(None, None)] if args.algos == ["none"]
        else itertools.product(args.algos, args.chunk_lengths)
    )
    eval_dirs = []
    for algo, chunk_length in configurations:
        endpoint_args = argparse.Namespace(**vars(generated_args))
        if algo is not None:
            endpoint_args.algos = [algo]
            endpoint_args.chunk_lengths = [chunk_length]

        existing = find_generated_clean_dataset(
            dataset_root, trained_root, env_name, expert_path,
            num_samples, endpoint_args,
        )
        if existing is None:
            eval_dirs.extend(
                run_generated_configuration(
                    env_name, expert_path, dataset_root, trained_root, eval_root,
                    num_samples, 0.0, 1.0, 0.0, endpoint_args,
                )
            )
            continue

        dataset_parent, dataset_tag, dataset_schema = existing
        eval_dirs.extend(
            train_algos(
                env_name, trained_root, eval_root, dataset_parent,
                dataset_tag, dataset_schema, endpoint_args,
                lambda dataset_schema=dataset_schema: collect_generated_dataset(
                    env_name, expert_path, num_samples,
                    dataset_schema["noise_scale"], 1.0, 0.0, endpoint_args,
                ),
            )
        )
    return eval_dirs


def run_robomimic_sweep(
    env_name: str,
    dataset_root: Path,
    trained_root: Path,
    eval_root: Path,
    args: argparse.Namespace,
) -> list[Path]:
    eval_dirs = []
    for spec in load_offline.list_robomimic_dataset_specs(env_name, args.dataset):
        dataset_tag = load_offline.make_robomimic_dataset_tag(spec)
        dataset_schema = {
            "version": DATASET_SCHEMA_VERSION,
            **spec,
            "seed": args.seed,
            "test_fraction": args.test_fraction,
        }
        eval_dirs.extend(
            train_algos(
                env_name, trained_root, eval_root, dataset_root / dataset_tag,
                dataset_tag, dataset_schema, args,
                lambda spec=spec: load_offline.load_robomimic_dataset(spec, seed=args.seed),
            )
        )
    return eval_dirs


def collect_generated_dataset(
    env_name: str,
    expert_path: Path,
    num_samples: int,
    noise_scale: float,
    prop_clean_expert: float,
    prop_noisy_expert: float,
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], dict]:
    if prop_clean_expert + prop_noisy_expert > 0.0 and not expert_path.exists():
        raise FileNotFoundError(f"Expert policy not found: {expert_path}")

    return rollout.collect_dataset(
        env_name=env_name,
        policy_path=str(expert_path),
        max_timesteps=args.max_timesteps,
        num_samples=num_samples,
        noise_scale=noise_scale,
        prop_clean_expert=prop_clean_expert,
        prop_noisy_expert=prop_noisy_expert,
        deterministic=True,
        seed=args.seed,
    )


def collect_clean_minari_dataset(
    env_name: str,
    expert_path: Path,
    dataset_id: str,
    num_samples: int,
    minari_fraction: float,
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], dict]:
    env = gym.make(env_name)
    try:
        trajectory_horizon = min(args.max_timesteps, env.spec.max_episode_steps)
    finally:
        env.close()

    target_num_trajectories = max(2, math.ceil(num_samples / trajectory_horizon))
    initial_num_minari = max(1, int(round(target_num_trajectories * minari_fraction)))
    target_counts = np.asarray(
        [target_num_trajectories - initial_num_minari, initial_num_minari],
        dtype=np.int64,
    )
    proportions = np.asarray([1.0 - minari_fraction, minari_fraction])
    trajectory_counts = np.zeros(2, dtype=np.int64)
    transition_counts = np.zeros(2, dtype=np.int64)
    components = []
    next_episode_id = 0
    clean_rng = np.random.default_rng(args.seed)
    minari_metadata = None

    while transition_counts.sum() < num_samples:
        additions = target_counts - trajectory_counts
        num_clean, num_minari = (int(value) for value in additions)
        if num_clean:
            if not expert_path.exists():
                raise FileNotFoundError(f"Expert policy not found: {expert_path}")
            clean_dataset = rollout.collect_expert(
                env_name=env_name,
                policy_path=str(expert_path),
                num_trajectories=num_clean,
                max_timesteps=args.max_timesteps,
                deterministic=True,
                rng=clean_rng,
                episode_id_start=next_episode_id,
            )
            components.append(clean_dataset)
            trajectory_counts[0] += num_clean
            transition_counts[0] += len(clean_dataset["rewards"])
            next_episode_id += num_clean

        if num_minari:
            minari_dataset, minari_metadata = load_offline.load_minari_episode_subset(
                dataset_id=dataset_id,
                num_episodes=num_minari,
                seed=args.seed,
                episode_id_start=next_episode_id,
                episode_offset=int(trajectory_counts[1]),
            )
            components.append(minari_dataset)
            trajectory_counts[1] += num_minari
            transition_counts[1] += len(minari_dataset["rewards"])
            next_episode_id += num_minari

        if transition_counts.sum() < num_samples:
            mean_length = transition_counts.sum() / trajectory_counts.sum()
            shortfall = num_samples - transition_counts.sum()
            target_num_trajectories = int(trajectory_counts.sum()) + max(
                1, math.ceil(shortfall / mean_length)
            )
            target_counts = grow_source_counts(
                trajectory_counts, target_num_trajectories, proportions
            )

    dataset = load_offline.concat_datasets(components)
    num_clean = int(trajectory_counts[0])
    num_minari = int(trajectory_counts[1])
    num_trajectories = num_clean + num_minari
    num_clean_transitions = int(transition_counts[0])
    num_minari_transitions = int(transition_counts[1])
    num_transitions = num_clean_transitions + num_minari_transitions

    return dataset, {
        "source": "clean-minari",
        "env_name": env_name,
        "policy_path": str(expert_path),
        "dataset_id": dataset_id,
        "minari_env_id": minari_metadata["env_id"],
        "max_timesteps": args.max_timesteps,
        "requested_num_samples": num_samples,
        "requested_minari_trajectory_fraction": minari_fraction,
        "requested_clean_expert_trajectory_fraction": 1.0 - minari_fraction,
        "num_trajectories": num_trajectories,
        "num_clean_expert_trajectories": num_clean,
        "num_minari_trajectories": num_minari,
        "actual_minari_trajectory_fraction": num_minari / num_trajectories,
        "num_transitions": num_transitions,
        "num_clean_expert_transitions": num_clean_transitions,
        "num_minari_transitions": num_minari_transitions,
        "actual_minari_transition_fraction": num_minari_transitions / num_transitions,
        "available_minari_episodes": minari_metadata["available_num_episodes"],
        "available_minari_transitions": minari_metadata["available_num_transitions"],
        "deterministic": True,
        "seed": args.seed,
    }


def grow_source_counts(
    current_counts: np.ndarray,
    target_total: int,
    proportions: np.ndarray,
) -> np.ndarray:
    """Grow source counts while staying closest to requested proportions."""
    counts = np.asarray(current_counts, dtype=np.int64).copy()
    while counts.sum() < target_total:
        next_total = int(counts.sum()) + 1
        deficits = next_total * proportions - counts
        counts[int(np.argmax(deficits))] += 1
    return counts


def find_generated_clean_dataset(
    dataset_root: Path,
    trained_root: Path,
    env_name: str,
    expert_path: Path,
    num_samples: int,
    args: argparse.Namespace,
) -> tuple[Path, str, dict] | None:
    expected = {
        "version": DATASET_SCHEMA_VERSION,
        "source": "generated",
        "env_name": env_name,
        "expert_path": str(expert_path),
        "max_timesteps": args.max_timesteps,
        "num_samples": num_samples,
        "prop_clean_expert": 1.0,
        "prop_noisy_expert": 0.0,
        "prop_random": 0.0,
        "prop_expert": 1.0,
        "deterministic": True,
        "seed": args.seed,
        "test_fraction": args.test_fraction,
    }
    candidates = []
    for metadata_path in sorted(dataset_root.glob("*/*/metadata.json"), reverse=True):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        schema = metadata.get("dataset_schema", {})
        if {key: value for key, value in schema.items() if key != "noise_scale"} != expected:
            continue
        dataset_dir = metadata_path.parent
        if dataset_cache_is_complete(dataset_dir):
            candidates.append((dataset_dir.parent, dataset_dir.parent.name, schema))
    if not candidates or args.algos == ["none"]:
        return None if not candidates else candidates[0]

    def reusable_run_count(candidate: tuple[Path, str, dict]) -> int:
        _, dataset_tag, schema = candidate
        return sum(
            find_trained_run(
                trained_root / f"{algo}_chunk{chunk_length}_{dataset_tag}",
                make_training_schema(algo, env_name, schema, chunk_length, args),
            ) is not None
            for algo, chunk_length in itertools.product(args.algos, args.chunk_lengths)
        )

    return max(candidates, key=reusable_run_count)


def get_or_create_dataset(
    dataset_parent: Path,
    dataset_schema: dict,
    create_dataset,
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], dict]:
    dataset_parent.mkdir(parents=True, exist_ok=True)
    with (dataset_parent / ".creation.lock").open("a+b") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        cached = find_cached_dataset(dataset_parent, dataset_schema)
        if cached is not None:
            return cached

        dataset, metadata = create_dataset()
        dataset_dir = dataset_parent / timestamp_name()
        return save_dataset_splits(dataset_dir, dataset, metadata, dataset_schema, args)


def find_cached_dataset(
    dataset_parent: Path,
    dataset_schema: dict,
) -> tuple[dict[str, np.ndarray], dict] | None:
    for metadata_path in sorted(dataset_parent.glob("*/metadata.json"), reverse=True):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        dataset_dir = metadata_path.parent
        if (
            metadata.get("dataset_schema") == dataset_schema
            and dataset_cache_is_complete(dataset_dir)
        ):
            try:
                dataset = rollout.load_dataset(dataset_dir / "train.npz")
                rollout.load_dataset(dataset_dir / "test.npz")
            except (OSError, ValueError, KeyError):
                continue
            return dataset, split_paths(dataset_dir)
    return None


def dataset_cache_is_complete(dataset_dir: Path) -> bool:
    """Return whether the canonical split cache is present; full.npz is legacy-only."""
    return all(
        (dataset_dir / filename).is_file()
        for filename in ("metadata.json", "train.npz", "test.npz")
    )


def save_dataset_splits(
    dataset_dir: Path,
    dataset: dict[str, np.ndarray],
    metadata: dict,
    dataset_schema: dict,
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], dict]:
    train_path = dataset_dir / "train.npz"
    test_path = dataset_dir / "test.npz"
    metadata_path = dataset_dir / "metadata.json"

    dataset_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        **metadata,
        "dataset_schema": dataset_schema,
        "test_fraction": args.test_fraction,
        "train_dataset_path": str(train_path.resolve()),
        "test_dataset_path": str(test_path.resolve()),
    }
    train_dataset, test_dataset = rollout.split_dataset(
        dataset,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    rollout.save_dataset(train_dataset, train_path)
    rollout.save_dataset(test_dataset, test_path)
    write_json_atomic(metadata_path, metadata)
    return train_dataset, split_paths(dataset_dir)


def train_algos(
    env_name: str,
    trained_root: Path,
    eval_root: Path,
    dataset_parent: Path,
    dataset_tag: str,
    dataset_schema: dict,
    args: argparse.Namespace,
    create_dataset,
) -> list[Path]:
    eval_dirs = []
    dataset_and_paths = None
    test_dataset = None
    if args.algos == ["none"]:
        get_or_create_dataset(dataset_parent, dataset_schema, create_dataset, args)
        return eval_dirs

    for chunk_length in args.chunk_lengths:
        chunk_dataset = None
        for algo in args.algos:
            run_name = f"{algo}_chunk{chunk_length}_{dataset_tag}"
            schema = make_training_schema(algo, env_name, dataset_schema, chunk_length, args)
            run_dir = find_trained_run(trained_root / run_name, schema)
            if run_dir is None:
                if dataset_and_paths is None:
                    dataset_and_paths = get_or_create_dataset(
                        dataset_parent, dataset_schema, create_dataset, args
                    )
                primitive_dataset, paths = dataset_and_paths
                if chunk_dataset is None:
                    chunk_dataset = chunking.make_action_chunk_dataset(
                        primitive_dataset, chunk_length, BASE_DISCOUNT
                    )
                    if test_dataset is None:
                        test_dataset = rollout.load_dataset(paths["test_dataset_path"])
                    # Fail before training if this length cannot be evaluated
                    # on any complete episode in the held-out split.
                    chunking.make_action_chunk_dataset(
                        test_dataset, chunk_length, BASE_DISCOUNT
                    )
                variant = timestamp_name()
                run_dir = trained_root / run_name / variant
                train_algo(
                    algo=algo,
                    env_name=env_name,
                    primitive_dataset=primitive_dataset,
                    chunk_dataset=chunk_dataset,
                    chunk_length=chunk_length,
                    run_dir=run_dir,
                    eval_dir=eval_root / run_name / variant,
                    split_paths=paths,
                    training_schema=schema,
                    args=args,
                )
            maybe_validate(run_dir, args)
            eval_dir = maybe_evaluate(run_dir, args)
            if eval_dir is not None:
                eval_dirs.append(eval_dir)
    return eval_dirs


def make_training_schema(
    algo: str,
    env_name: str,
    dataset_schema: dict,
    chunk_length: int,
    args: argparse.Namespace,
) -> dict:
    macro_discount = BASE_DISCOUNT**chunk_length
    schema = {
        "version": TRAINING_SCHEMA_VERSION,
        "env_name": env_name,
        "algo": algo,
        "dataset": dataset_schema,
        "chunk_length": chunk_length,
        "base_discount": BASE_DISCOUNT,
        "macro_discount": macro_discount,
        "chunk_reward": "discounted_sum",
        "seed": args.seed,
        "epoch": args.epoch,
        "step_per_epoch": args.step_per_epoch,
        "batch_size": args.batch_size,
    }
    if algo == "iql":
        schema["iql"] = {
            "temperature": args.iql_temperature,
            "expectile": args.iql_expectile,
            "learning_rate": args.iql_learning_rate,
            "lr_schedule": args.iql_lr_schedule,
            "hidden_dims": args.iql_hidden_dims,
        }
    if algo == "td3bc":
        schema["td3bc"] = {
            "learning_rate": args.td3bc_learning_rate,
            "alpha": args.td3bc_alpha,
            "hidden_dims": args.td3bc_hidden_dims,
        }
    if algo == "cql":
        entropy_learning_rate = getattr(
            args,
            "cql_entropy_learning_rate",
            CQL_DEFAULT_ENTROPY_LEARNING_RATE,
        )
        entropy_alpha_max = getattr(
            args,
            "cql_entropy_alpha_max",
            CQL_DEFAULT_ENTROPY_ALPHA_MAX,
        )
        lagrange_target_mode = getattr(
            args,
            "cql_lagrange_target_mode",
            CQL_DEFAULT_LAGRANGE_TARGET_MODE,
        )
        cql_schema = {}
        if (
            entropy_learning_rate != CQL_DEFAULT_ENTROPY_LEARNING_RATE
            or entropy_alpha_max != CQL_DEFAULT_ENTROPY_ALPHA_MAX
        ):
            cql_schema.update(
                entropy_learning_rate=entropy_learning_rate,
                entropy_alpha_max=entropy_alpha_max,
            )
        if (
            env_name in task_support.ROBOMIMIC_TASKS
            and lagrange_target_mode == CQL_ACTION_VOLUME_LAGRANGE_TARGET_MODE
        ):
            cql_schema["lagrange_target_mode"] = lagrange_target_mode
        if cql_schema:
            schema["cql"] = cql_schema
    if algo == "mopo" and args.mopo_penalty_coef != 0.5:
        schema["mopo"] = {"penalty_coef": args.mopo_penalty_coef}
    if algo == "mobile":
        mobile_schema = {}
        if env_name == "Reacher-v5":
            mobile_schema.update(
                return_shift=args.mobile_return_shift,
                clamp_target_q=True,
            )
        if args.mobile_penalty_coef != 1.5:
            mobile_schema["penalty_coef"] = args.mobile_penalty_coef
        if args.model_manipulation_settings:
            mobile_schema.update(num_critics=10, max_q_backup=True)
        if mobile_schema:
            schema["mobile"] = mobile_schema
    if algo in {"cql", "combo"}:
        schema["implementation_version"] = 2
    if algo in MODEL_BASED_ALGOS:
        schema["model_based"] = {
            "dynamics_max_epochs": args.dynamics_max_epochs,
            "rollout_freq": args.rollout_freq,
            "rollout_batch_size": args.rollout_batch_size,
            "rollout_length": args.rollout_length,
            "model_retain_epochs": args.model_retain_epochs,
            "real_ratio": args.real_ratio,
        }
        dynamics_chunk_mode = resolve_dynamics_chunk_mode(
            getattr(args, "dynamics_chunk_mode", "direct"), chunk_length
        )
        if dynamics_chunk_mode == "recursive":
            schema["model_based"]["chunk_dynamics"] = {
                "version": 1,
                "mode": "recursive",
            }
        if args.model_actor_learning_rate is not None:
            schema["model_based"]["actor_learning_rate"] = args.model_actor_learning_rate
        if args.model_critic_learning_rate != 3e-4:
            schema["model_based"]["critic_learning_rate"] = args.model_critic_learning_rate
        if env_name == "Lift":
            schema["model_based"]["synthetic_termination"] = "never"
        if args.model_manipulation_settings and algo in {"mopo", "mobile"}:
            schema["model_based"]["manipulation_settings"] = {
                "actor_hidden_dims": [256, 256, 256],
                "dynamics_hidden_dims": [400, 400, 400, 400],
                "actor_learning_rate": args.model_actor_learning_rate or 3e-5,
                "reward_normalization": "zscore",
            }
    return schema


def find_trained_run(run_parent: Path, training_schema: dict) -> Path | None:
    for manifest_path in sorted(run_parent.glob("*/run_manifest.json"), reverse=True):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if manifest.get("training_schema") == training_schema and run_is_complete(manifest):
            return manifest_path.parent
    return None


def run_is_complete(manifest: dict) -> bool:
    try:
        paths = [
            Path(manifest["model_dir"]) / "policy.pth",
            Path(manifest["train_dataset_path"]),
            Path(manifest["test_dataset_path"]),
            Path(manifest["dataset_metadata_path"]),
        ]
        for checkpoint in manifest["checkpoints"]:
            paths.append(Path(checkpoint["policy_path"]))
            if "dynamics_path" in checkpoint:
                dynamics_dir = Path(checkpoint["dynamics_path"])
                paths.extend((
                    dynamics_dir / "dynamics.pth",
                    dynamics_dir / "mu.npy",
                    dynamics_dir / "std.npy",
                ))
        if manifest["algo"] in MODEL_BASED_ALGOS:
            model_dir = Path(manifest["model_dir"])
            paths.extend((
                model_dir / "dynamics.pth",
                model_dir / "mu.npy",
                model_dir / "std.npy",
            ))
    except (KeyError, TypeError):
        return False
    return all(path.is_file() for path in paths)


def maybe_evaluate(run_dir: Path, args: argparse.Namespace) -> Path | None:
    if not args.eval:
        return None
    from eval import evaluate_run

    return evaluate_run(
        run_dir,
        argparse.Namespace(
            device=args.device,
            checkpoint_eval_episodes=args.checkpoint_eval_episodes,
            final_eval_episodes=args.final_eval_episodes,
            expert=args.expert,
            seed=args.seed,
            contraction_trajectories=args.contraction_trajectories,
            contraction_horizon=args.contraction_horizon,
            perturbation_scale=args.perturbation_scale,
            ood_samples=args.ood_samples,
            reuse_eval=args.reuse_eval,
        ),
    )


def maybe_validate(run_dir: Path, args: argparse.Namespace) -> None:
    if not args.eval:
        return
    from validation import validate_run

    validate_run(run_dir, args.device)


def maybe_plot(eval_root: Path, eval_dirs: list[Path], args: argparse.Namespace) -> None:
    if not args.eval or not eval_dirs:
        return
    from plot import plot_root

    plot_root(eval_root, eval_dirs=eval_dirs)


def split_paths(dataset_dir: Path) -> dict:
    with (dataset_dir / "metadata.json").open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    paths = {
        "dataset_dir": str(dataset_dir.resolve()),
        "train_dataset_path": str((dataset_dir / "train.npz").resolve()),
        "test_dataset_path": str((dataset_dir / "test.npz").resolve()),
        "dataset_metadata_path": str((dataset_dir / "metadata.json").resolve()),
        "dataset_tag": dataset_dir.parent.name,
        "test_fraction": metadata["test_fraction"],
    }
    legacy_full_path = metadata.get("full_dataset_path")
    if legacy_full_path is not None:
        paths["full_dataset_path"] = legacy_full_path
    elif (dataset_dir / "full.npz").exists():
        paths["full_dataset_path"] = str((dataset_dir / "full.npz").resolve())
    return paths


def train_algo(
    algo: str,
    env_name: str,
    primitive_dataset: dict[str, np.ndarray],
    chunk_dataset: dict[str, np.ndarray],
    chunk_length: int,
    run_dir: Path,
    eval_dir: Path,
    split_paths: dict,
    training_schema: dict,
    args: argparse.Namespace,
) -> None:
    if algo not in (*MODEL_FREE_ALGOS, *MODEL_BASED_ALGOS):
        raise ValueError(f"Unsupported algorithm: {algo}")

    run_dir.mkdir(parents=True)
    seed_everything(args.seed)

    macro_discount = training_schema["macro_discount"]
    eval_env = None
    logger = None
    trainer_closed_logger = False
    try:
        eval_env = make_env(env_name, split_paths, args)
        eval_env = chunking.ActionChunkWrapper(eval_env, chunk_length)
        eval_env.reset(seed=args.seed)
        eval_env.action_space.seed(args.seed)
        logger = build_logger(
            run_dir, args, algo, env_name, chunk_length, macro_discount
        )

        if algo in MODEL_FREE_ALGOS:
            buffer = build_buffer(chunk_dataset, eval_env, args.device)
            policy, lr_scheduler = build_model_free_policy(
                algo, eval_env, buffer, args,
                discount=macro_discount,
                **({"chunk_length": chunk_length} if algo == "cql" else {}),
            )
            trainer = MFPolicyTrainer(
                policy=policy,
                buffer=buffer,
                logger=logger,
                epoch=args.epoch,
                step_per_epoch=args.step_per_epoch,
                batch_size=args.batch_size,
                lr_scheduler=lr_scheduler,
                checkpoint_epochs=checkpoint_epochs(args.epoch),
                show_progress=not args.quiet,
            )
        else:
            dynamics_chunk_mode = resolve_dynamics_chunk_mode(
                getattr(args, "dynamics_chunk_mode", "direct"), chunk_length
            )
            policy_dataset, dynamics_dataset = prepare_model_based_datasets(
                algo,
                primitive_dataset,
                chunk_dataset,
                chunk_length,
                BASE_DISCOUNT,
                dynamics_chunk_mode,
                args.model_manipulation_settings,
            )
            real_buffer = build_buffer(policy_dataset, eval_env, args.device)
            fake_buffer = ReplayBuffer(
                buffer_size=args.rollout_batch_size * args.rollout_length * args.model_retain_epochs,
                obs_shape=eval_env.observation_space.shape,
                obs_dtype=np.float32,
                action_dim=int(np.prod(eval_env.action_space.shape)),
                action_dtype=np.float32,
                device=args.device,
            )
            policy, dynamics, lr_scheduler = build_model_based_policy(
                algo, eval_env, args,
                discount=macro_discount,
                chunk_length=chunk_length,
                base_discount=BASE_DISCOUNT,
                dynamics_chunk_mode=dynamics_chunk_mode,
                primitive_action_dim=int(np.prod(eval_env.env.action_space.shape)),
            )

            if dynamics_dataset is None:
                dynamics_dataset = real_buffer.sample_all()
            dynamics.train(
                dynamics_dataset,
                logger,
                max_epochs=args.dynamics_max_epochs,
                max_epochs_since_update=5,
            )

            trainer = MBPolicyTrainer(
                policy=policy,
                real_buffer=real_buffer,
                fake_buffer=fake_buffer,
                logger=logger,
                rollout_setting=(args.rollout_freq, args.rollout_batch_size, args.rollout_length),
                epoch=args.epoch,
                step_per_epoch=args.step_per_epoch,
                batch_size=args.batch_size,
                real_ratio=args.real_ratio,
                lr_scheduler=lr_scheduler,
                checkpoint_epochs=checkpoint_epochs(args.epoch),
                show_progress=not args.quiet,
            )

        initial_checkpoint = Path(logger.checkpoint_dir) / "step_0"
        initial_checkpoint.mkdir(exist_ok=True)
        torch.save(policy.state_dict(), initial_checkpoint / "policy.pth")
        if algo in MODEL_BASED_ALGOS:
            dynamics.save(initial_checkpoint)

        trainer.train()
        trainer_closed_logger = True
        compact_run_artifacts(
            run_dir, algo, args.epoch, args.step_per_epoch
        )
        save_run_manifest(
            run_dir, eval_dir, algo, env_name, split_paths,
            training_schema, chunk_length, macro_discount, args,
        )
    finally:
        try:
            if logger is not None and not trainer_closed_logger:
                logger.close()
        finally:
            if eval_env is not None:
                eval_env.close()


def compact_run_artifacts(
    run_dir: Path,
    algo: str,
    epochs: int,
    steps_per_epoch: int,
) -> None:
    """Deduplicate successful-run artifacts without changing referenced paths."""
    final_policy = (
        run_dir / "checkpoint" / f"step_{epochs * steps_per_epoch}" / "policy.pth"
    )
    rolling_policy = run_dir / "checkpoint" / "policy.pth"
    if final_policy.is_file():
        try:
            rolling_policy.unlink(missing_ok=True)
        except OSError:
            pass
        replace_identical_file_with_hardlink(
            final_policy, run_dir / "model" / "policy.pth"
        )

    if algo in MODEL_BASED_ALGOS:
        initial_dynamics = run_dir / "checkpoint" / "step_0"
        model_dir = run_dir / "model"
        for filename in ("dynamics.pth", "mu.npy", "std.npy"):
            replace_identical_file_with_hardlink(
                initial_dynamics / filename, model_dir / filename
            )


def replace_identical_file_with_hardlink(source: Path, duplicate: Path) -> bool:
    """Replace an exact duplicate with a hardlink, falling back without error."""
    if not source.is_file() or not duplicate.is_file():
        return False
    try:
        if source.samefile(duplicate):
            return True
    except OSError:
        return False
    try:
        identical = filecmp.cmp(source, duplicate, shallow=False)
    except OSError:
        return False
    if not identical:
        return False

    temporary = duplicate.with_name(
        f".{duplicate.name}.{uuid.uuid4().hex}.link"
    )
    try:
        os.link(source, temporary)
        os.replace(temporary, duplicate)
    except OSError:
        temporary.unlink(missing_ok=True)
        return False
    return True


def prepare_model_based_datasets(
    algo: str,
    primitive_dataset: dict[str, np.ndarray],
    chunk_dataset: dict[str, np.ndarray],
    chunk_length: int,
    base_discount: float,
    dynamics_chunk_mode: str,
    model_manipulation_settings: bool,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray] | None]:
    """Prepare macro policy data and, when recursive, primitive dynamics data."""
    dynamics_chunk_mode = resolve_dynamics_chunk_mode(
        dynamics_chunk_mode, chunk_length
    )

    manipulation_settings = model_manipulation_settings and algo in {"mopo", "mobile"}
    policy_dataset = chunk_dataset
    reward_mean = reward_scale = None
    if manipulation_settings:
        macro_rewards = np.asarray(chunk_dataset["rewards"], dtype=np.float32)
        reward_mean = float(macro_rewards.mean())
        reward_scale = float(macro_rewards.std()) + 1e-3
        policy_dataset = dict(chunk_dataset)
        policy_dataset["rewards"] = (
            (macro_rewards - reward_mean) / reward_scale
        ).astype(np.float32)

    # The direct path must continue training on real_buffer.sample_all() so H=1
    # uses the exact legacy arrays, shapes, and RNG sequence.
    if dynamics_chunk_mode == "direct":
        return policy_dataset, None

    dynamics_dataset = {
        key: np.asarray(primitive_dataset[key])
        for key in ("observations", "actions", "next_observations")
    }
    primitive_rewards = np.asarray(primitive_dataset["rewards"], dtype=np.float32)
    if manipulation_settings:
        discount_mass = float(np.sum(base_discount ** np.arange(chunk_length)))
        primitive_rewards = (
            (primitive_rewards - reward_mean / discount_mass) / reward_scale
        ).astype(np.float32)
    dynamics_dataset["rewards"] = primitive_rewards.reshape(-1, 1)
    dynamics_dataset["terminals"] = np.asarray(
        primitive_dataset["terminals"], dtype=np.float32
    ).reshape(-1, 1)
    return policy_dataset, dynamics_dataset


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


def make_env(env_name: str, split_paths: dict, args: argparse.Namespace):
    if args.dataset_source == "robomimic":
        metadata = load_offline.load_metadata(split_paths["dataset_metadata_path"])
        return load_offline.make_robomimic_env(metadata)
    return gym.make(env_name)


def save_run_manifest(
    run_dir: Path,
    eval_dir: Path,
    algo: str,
    env_name: str,
    split_paths: dict,
    training_schema: dict,
    chunk_length: int,
    macro_discount: float,
    args: argparse.Namespace,
) -> None:
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "env_name": env_name,
        "algo": algo,
        "chunk_length": chunk_length,
        "base_discount": BASE_DISCOUNT,
        "macro_discount": macro_discount,
        "chunk_reward": "discounted_sum",
        "dataset_source": args.dataset_source,
        "run_dir": str(run_dir.resolve()),
        "model_dir": str((run_dir / "model").resolve()),
        "eval_dir": str(eval_dir.resolve()),
        "training_schema": training_schema,
        "seed": args.seed,
        "device": args.device,
        "test_fraction": args.test_fraction,
        "epoch": args.epoch,
        "step_per_epoch": args.step_per_epoch,
        "batch_size": args.batch_size,
        "rollout_length": args.rollout_length,
        "expert": str(resolve_expert_path(args.expert, env_name)),
        "checkpoints": checkpoint_manifest(run_dir, algo, args.epoch, args.step_per_epoch),
        **split_paths,
    }
    write_json_atomic(run_dir / "run_manifest.json", manifest)


def write_json_atomic(path: Path, payload: dict) -> None:
    """Write JSON through a same-directory temporary file and atomic rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(payload, temporary, indent=2, sort_keys=True)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def build_logger(
    run_dir: Path,
    args: argparse.Namespace,
    algo: str,
    env_name: str,
    chunk_length: int,
    macro_discount: float,
) -> Logger:
    output_config = {
        "consoleout_backup": "stdout",
        "policy_training_progress": "csv",
        "tb": "tensorboard",
    }
    logger = Logger(str(run_dir), output_config, console_output=False)
    hyperparameters = {
        "algo": algo,
        "env": env_name,
        "chunk_length": chunk_length,
        "base_discount": BASE_DISCOUNT,
        "macro_discount": macro_discount,
        "seed": args.seed,
        "device": args.device,
        "epoch": args.epoch,
        "step_per_epoch": args.step_per_epoch,
        "batch_size": args.batch_size,
    }
    try:
        logger.log_hyperparameters(hyperparameters)
    except BaseException:
        logger.close()
        raise
    return logger


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def checkpoint_epochs(epochs: int) -> list[int]:
    return [item["epoch"] for item in checkpoint_schedule(epochs) if item["epoch"] > 0]


def checkpoint_schedule(epochs: int) -> list[dict]:
    by_epoch = {0: 0}
    for percent in range(10, 101, 10):
        by_epoch[math.ceil(percent * epochs / 100)] = percent
    return [
        {
            "requested_percent": requested_percent,
            "actual_percent": 100.0 * epoch / epochs,
            "epoch": epoch,
        }
        for epoch, requested_percent in sorted(by_epoch.items())
    ]


def checkpoint_manifest(run_dir: Path, algo: str, epochs: int, steps_per_epoch: int) -> list[dict]:
    records = []
    for item in checkpoint_schedule(epochs):
        step = item["epoch"] * steps_per_epoch
        checkpoint_dir = run_dir / "checkpoint" / f"step_{step}"
        record = {
            **item,
            "step": step,
            "policy_path": str((checkpoint_dir / "policy.pth").resolve()),
        }
        if algo in MODEL_BASED_ALGOS:
            dynamics_dir = run_dir / "checkpoint" / "step_0"
            record["dynamics_path"] = str(dynamics_dir.resolve())
        records.append(record)
    return records


def make_clean_minari_dataset_tag(
    dataset_id: str,
    num_samples: int,
    minari_fraction: float,
    seed: int,
) -> str:
    source = dataset_id.replace("/", "_")
    return (
        f"clean_minari_{source}_samples{num_samples}_"
        f"minari{minari_fraction:g}_seed{seed}"
    )


def make_dataset_tag(
    num_samples: int,
    noise_scale: float,
    prop_clean_expert: float,
    prop_noisy_expert: float,
    seed: int,
) -> str:
    return (
        f"samples{num_samples}_clean{prop_clean_expert:g}_noisy{prop_noisy_expert:g}_"
        f"noise{noise_scale:g}_seed{seed}"
    )


def resolve_expert_path(expert_arg: str, env_name: str) -> Path:
    expert_path = Path(expert_arg).expanduser()
    if expert_path.suffix == ".zip":
        return expert_path.resolve()
    return (expert_path / f"{env_name}.zip").resolve()


def timestamp_name() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S-%f")


if __name__ == "__main__":
    main()

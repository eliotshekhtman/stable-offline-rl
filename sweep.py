# Tasks:
# - Parse the experiment CLI and keep one run focused on one Gymnasium environment.
# - Choose the dataset source: generated rollouts, converted Minari datasets, or robomimic datasets.
# - Cache/load datasets, build OfflineRL-Kit replay buffers, and launch trainers.
# - Sweep algorithms and action chunk lengths over each requested random seed.
# - Own experiment directories, milestone checkpoints, logging, seeding, and run naming.

import argparse
import itertools
import json
import math
import random
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

import chunking
import load_offline
import rollout
from offlinerlkit.buffer import ReplayBuffer
from offlinerlkit.policy_trainer import MBPolicyTrainer, MFPolicyTrainer
from offlinerlkit.utils.logger import Logger
from dql import CLEANDIFFUSER_COMMIT, resolve_dql_config, train_dql
from policies import MODEL_BASED_ALGOS, MODEL_FREE_ALGOS, build_model_based_policy, build_model_free_policy


PROJECT_DIR = Path(__file__).resolve().parent
DATASET_ROOT = PROJECT_DIR / "datasets"
TRAINED_ROOT = PROJECT_DIR / "trained"
EVAL_ROOT = PROJECT_DIR / "evals"
DATASET_SCHEMA_VERSION = 4
TRAINING_SCHEMA_VERSION = 3
BASE_DISCOUNT = 0.99


def main() -> None:
    args = parse_args()
    expert_path = resolve_expert_path(args.expert, args.env)
    eval_dirs = []
    for seed in args.seeds:
        seed_args = argparse.Namespace(**vars(args))
        seed_args.seed = seed
        eval_dirs.extend(
            run_sweep(env_name=args.env, expert_path=expert_path, args=seed_args)
        )
    maybe_plot(EVAL_ROOT / args.env, eval_dirs, args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect offline datasets and train OfflineRL-Kit policies.")

    experiment = parser.add_argument_group("experiment")
    experiment.add_argument("--env", required=True, help="Gymnasium environment id to train on, e.g. HalfCheetah-v5")
    experiment.add_argument("--seed", dest="seeds", type=int, nargs="+", default=[0], help="Random seeds used for dataset splitting, generated rollouts, training, and evaluation")
    experiment.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Torch device used for OfflineRL-Kit policies and dynamics")

    dataset = parser.add_argument_group("dataset source and split")
    dataset.add_argument("--dataset-source", choices=["generated", "minari", "robomimic"], default="generated", help="Use generated expert/random rollouts, all matching Minari datasets, or low-dimensional robomimic datasets for the environment")
    dataset.add_argument("--dataset", default=None, help="Limit a premade-data sweep to one Robomimic type (e.g. mh) or Minari leaf/full id (e.g. medium-v0); omit to use all matching datasets")
    dataset.add_argument("--test-fraction", type=float, default=0.2, help="Fraction of each dataset held out for post-training evaluation")

    generated = parser.add_argument_group("generated dataset options")
    generated.add_argument("--expert", default="/home/shekhe/stable-offline-rl/experts", help="Expert policy .zip path or directory containing <env>.zip; used only for generated expert data and expert evaluation")
    generated.add_argument("--num-samples", type=int, nargs="+", default=[1000000], help="Minimum generated transition counts to sweep over; collection always retains complete trajectories")
    generated.add_argument("--noise-scale", type=float, nargs="+", default=[0.0], help="Gaussian action-noise scales applied to noisy expert actions in generated datasets")
    generated.add_argument(
        "--composition", type=float, nargs=2, action="append", metavar=("CLEAN_EXPERT", "NOISY_EXPERT"),
        help="Generated-data clean and noisy expert trajectory proportions; repeat for multiple compositions, with random trajectories filling the remainder (default: 1 0)",
    )
    generated.add_argument("--max-timesteps", type=int, default=1000, help="Maximum length of each generated rollout trajectory")

    training = parser.add_argument_group("policy training")
    training.add_argument(
        "--algos", nargs="+", choices=("none", *MODEL_FREE_ALGOS, *MODEL_BASED_ALGOS),
        default=["cql"], help="Algorithms to train, or none to collect the dataset without training",
    )
    training.add_argument(
        "--chunk-lengths", type=int, nargs="+", default=[1],
        help="Action chunk lengths to sweep over; each policy emits this many primitive actions",
    )
    training.add_argument("--epoch", type=int, default=1000, help="Number of policy-training epochs")
    training.add_argument("--step-per-epoch", type=int, default=1000, help="Gradient-update steps per policy-training epoch")
    training.add_argument("--batch-size", type=int, default=512, help="Policy-training batch size")

    dql = parser.add_argument_group("DQL options")
    dql.add_argument("--dql-eta", type=float, default=None, help="Override the DQL Q-guidance loss weight; defaults to CleanDiffuser's locomotion value of 1")
    dql.add_argument("--dql-weight-temperature", type=float, default=None, help="Override the DQL candidate-action softmax weight; defaults to a CleanDiffuser task value when available")
    dql.add_argument("--dql-reward-normalization", choices=["auto", "none", "episode-range"], default="auto", help="DQL reward scaling: auto uses CleanDiffuser episode-return-range scaling for Minari data and no scaling otherwise")

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
    evaluation.add_argument("--eval-episodes", type=int, default=100, help="Number of true-environment episodes used to estimate final and checkpoint policy performance; also used for generated-data expert evaluation")
    evaluation.add_argument("--contraction-trajectories", type=int, default=16, help="Number of matched unperturbed and agent-state-perturbed trajectories for each best and last policy")
    evaluation.add_argument("--contraction-horizon", type=int, default=300, help="Maximum primitive steps in each best- and last-policy contraction trajectory")
    evaluation.add_argument("--perturbation-scale", type=float, default=0.1, help="Euclidean norm of the initial perturbation applied to controlled-agent qpos/qvel coordinates")
    evaluation.add_argument("--ood-samples", type=int, default=10000, help="Maximum held-out and policy decision-boundary samples used for OOD metrics")

    model_based = parser.add_argument_group("model-based algorithm options")
    model_based.add_argument("--dynamics-max-epochs", type=int, default=30, help="Maximum epochs for fitting the learned dynamics model before policy training")
    model_based.add_argument("--rollout-freq", type=int, default=1000, help="Policy-training step interval between learned-dynamics rollout generation")
    model_based.add_argument("--rollout-batch-size", type=int, default=10000, help="Number of initial real states used when generating model rollouts")
    model_based.add_argument("--rollout-length", type=int, default=5, help="Number of learned macro-transitions per synthetic rollout")
    model_based.add_argument("--model-retain-epochs", type=int, default=5, help="How many epochs of synthetic model rollouts to retain in the fake replay buffer")
    model_based.add_argument("--real-ratio", type=float, default=0.50, help="Fraction of each model-based training batch sampled from the real offline dataset rather than the synthetic rollout buffer")
    model_based.add_argument("--dynamics-update-freq", type=int, default=1000, help="RAMBO dynamics-adversary update interval; ignored by other model-based algorithms")
    model_based.add_argument("--adv-batch-size", type=int, default=256, help="RAMBO adversarial dynamics rollout batch size")
    model_based.add_argument("--adv-weight", type=float, default=3e-4, help="RAMBO adversarial dynamics loss weight")
    model_based.add_argument("--bc-epoch", type=int, default=5, help="RAMBO behavior-cloning pretraining epochs")
    model_based.add_argument("--bc-batch-size", type=int, default=256, help="RAMBO behavior-cloning pretraining batch size")
    args = parser.parse_args()
    if any(chunk_length <= 0 for chunk_length in args.chunk_lengths):
        parser.error("--chunk-lengths values must be positive")
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
    args.seeds = list(dict.fromkeys(args.seeds))
    args.chunk_lengths = list(dict.fromkeys(args.chunk_lengths))
    args.algos = list(dict.fromkeys(args.algos))
    args.composition = [(1.0, 0.0)] if args.composition is None else [tuple(values) for values in args.composition]
    args.composition = list(dict.fromkeys(args.composition))
    for prop_clean, prop_noisy in args.composition:
        if prop_clean < 0.0 or prop_noisy < 0.0 or prop_clean + prop_noisy > 1.0:
            parser.error("--composition values must be nonnegative and sum to at most 1")
    if "none" in args.algos and args.algos != ["none"]:
        parser.error("--algos none cannot be combined with training algorithms")
    if args.dataset is not None and args.dataset_source == "generated":
        parser.error("--dataset applies only to minari and robomimic dataset sources")
    if args.eval_episodes <= 0 or args.contraction_trajectories <= 0 or args.contraction_horizon <= 0:
        parser.error("evaluation episode, trajectory, and horizon counts must be positive")
    if args.contraction_trajectories > args.eval_episodes:
        parser.error("--contraction-trajectories cannot exceed --eval-episodes")
    if args.perturbation_scale < 0.0:
        parser.error("--perturbation-scale must be nonnegative")
    if args.reuse_eval and not args.eval:
        parser.error("--reuse-eval requires --eval")
    return args


def run_sweep(env_name: str, expert_path: Path, args: argparse.Namespace) -> list[Path]:
    dataset_root = DATASET_ROOT / env_name
    trained_root = TRAINED_ROOT / env_name
    eval_root = EVAL_ROOT / env_name
    dataset_root.mkdir(parents=True, exist_ok=True)
    trained_root.mkdir(parents=True, exist_ok=True)
    eval_dirs = []

    if args.dataset_source == "minari":
        eval_dirs = run_minari_sweep(env_name, dataset_root, trained_root, eval_root, args)
    elif args.dataset_source == "robomimic":
        eval_dirs = run_robomimic_sweep(env_name, dataset_root, trained_root, eval_root, args)
    else:
        for num_samples, noise_scale, composition in itertools.product(
            args.num_samples, args.noise_scale, args.composition
        ):
            prop_clean, prop_noisy = composition
            prop_random = 1.0 - prop_clean - prop_noisy
            prop_expert = prop_clean + prop_noisy
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
                "prop_expert": prop_expert,
                "deterministic": True,
                "seed": args.seed,
                "test_fraction": args.test_fraction,
            }
            eval_dirs.extend(
                train_algos(
                    env_name, trained_root, eval_root, dataset_root / dataset_tag,
                    dataset_tag, dataset_schema, args,
                    lambda: collect_generated_dataset(
                        env_name, expert_path, num_samples, noise_scale,
                        prop_clean, prop_noisy, args,
                    ),
                )
            )

    return eval_dirs


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

    print("Collecting generated dataset")
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


def get_or_create_dataset(
    dataset_parent: Path,
    dataset_schema: dict,
    create_dataset,
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], dict]:
    for metadata_path in sorted(dataset_parent.glob("*/metadata.json"), reverse=True):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        dataset_dir = metadata_path.parent
        if metadata.get("dataset_schema") == dataset_schema and all(
            (dataset_dir / filename).exists() for filename in ("full.npz", "train.npz", "test.npz")
        ):
            print(f"Loading dataset split: {dataset_dir}")
            return rollout.load_dataset(dataset_dir / "train.npz"), split_paths(dataset_dir)

    dataset, metadata = create_dataset()
    dataset_dir = dataset_parent / timestamp_name()
    return save_dataset_splits(dataset_dir, dataset, metadata, dataset_schema, args)


def save_dataset_splits(
    dataset_dir: Path,
    dataset: dict[str, np.ndarray],
    metadata: dict,
    dataset_schema: dict,
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], dict]:
    full_path = dataset_dir / "full.npz"
    train_path = dataset_dir / "train.npz"
    test_path = dataset_dir / "test.npz"
    metadata_path = dataset_dir / "metadata.json"

    dataset_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        **metadata,
        "dataset_schema": dataset_schema,
        "test_fraction": args.test_fraction,
        "full_dataset_path": str(full_path.resolve()),
        "train_dataset_path": str(train_path.resolve()),
        "test_dataset_path": str(test_path.resolve()),
    }
    train_dataset, test_dataset = rollout.split_dataset(
        dataset,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    rollout.save_dataset(dataset, full_path)
    rollout.save_dataset(train_dataset, train_path)
    rollout.save_dataset(test_dataset, test_path)
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, sort_keys=True)
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
            else:
                print(f"Reusing trained run: {run_dir}")

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
    if algo == "dql":
        schema["dql"] = {
            "eta": args.dql_eta,
            "weight_temperature": args.dql_weight_temperature,
            "reward_normalization": args.dql_reward_normalization,
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
    if algo == "rambo":
        schema["rambo"] = {
            "dynamics_update_freq": args.dynamics_update_freq,
            "adv_batch_size": args.adv_batch_size,
            "adv_weight": args.adv_weight,
            "bc_epoch": args.bc_epoch,
            "bc_batch_size": args.bc_batch_size,
        }
    return schema


def find_trained_run(run_parent: Path, training_schema: dict) -> Path | None:
    for manifest_path in sorted(run_parent.glob("*/run_manifest.json"), reverse=True):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("training_schema") == training_schema and run_is_complete(manifest):
            return manifest_path.parent
    return None


def run_is_complete(manifest: dict) -> bool:
    paths = [
        Path(manifest["model_dir"]) / "policy.pth",
        Path(manifest["full_dataset_path"]),
        Path(manifest["train_dataset_path"]),
        Path(manifest["test_dataset_path"]),
        Path(manifest["dataset_metadata_path"]),
    ]
    for checkpoint in manifest["checkpoints"]:
        paths.append(Path(checkpoint["policy_path"]))
        if "dynamics_path" in checkpoint:
            dynamics_dir = Path(checkpoint["dynamics_path"])
            paths.extend((dynamics_dir / "dynamics.pth", dynamics_dir / "mu.npy", dynamics_dir / "std.npy"))
    if manifest["algo"] in MODEL_BASED_ALGOS:
        model_dir = Path(manifest["model_dir"])
        paths.extend((model_dir / "dynamics.pth", model_dir / "mu.npy", model_dir / "std.npy"))
    return all(path.exists() for path in paths)


def maybe_evaluate(run_dir: Path, args: argparse.Namespace) -> Path | None:
    if not args.eval:
        return None
    from eval import evaluate_run

    return evaluate_run(
        run_dir,
        argparse.Namespace(
            device=args.device,
            eval_episodes=args.eval_episodes,
            expert=args.expert,
            seed=args.seed,
            contraction_trajectories=args.contraction_trajectories,
            contraction_horizon=args.contraction_horizon,
            perturbation_scale=args.perturbation_scale,
            ood_samples=args.ood_samples,
            reuse_eval=args.reuse_eval,
        ),
    )


def maybe_plot(eval_root: Path, eval_dirs: list[Path], args: argparse.Namespace) -> None:
    if not args.eval or not eval_dirs:
        return
    from plot import plot_root

    plot_root(eval_root, eval_dirs=eval_dirs)


def split_paths(dataset_dir: Path) -> dict:
    with (dataset_dir / "metadata.json").open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    return {
        "dataset_dir": str(dataset_dir.resolve()),
        "full_dataset_path": str((dataset_dir / "full.npz").resolve()),
        "train_dataset_path": str((dataset_dir / "train.npz").resolve()),
        "test_dataset_path": str((dataset_dir / "test.npz").resolve()),
        "dataset_metadata_path": str((dataset_dir / "metadata.json").resolve()),
        "dataset_tag": dataset_dir.parent.name,
        "test_fraction": metadata["test_fraction"],
    }


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
    if algo not in MODEL_FREE_ALGOS and algo not in MODEL_BASED_ALGOS:
        raise ValueError(f"Unsupported algorithm: {algo}")

    run_dir.mkdir(parents=True)
    seed_everything(args.seed)

    dql_config = None
    if algo == "dql":
        dql_config = resolve_dql_config(
            env_name=env_name,
            dataset_tag=split_paths["dataset_tag"],
            dataset=primitive_dataset,
            dataset_source=args.dataset_source,
            eta_override=args.dql_eta,
            weight_temperature_override=args.dql_weight_temperature,
            reward_normalization=args.dql_reward_normalization,
        )
    macro_discount = training_schema["macro_discount"]
    eval_env = chunking.ActionChunkWrapper(make_env(env_name, split_paths, args), chunk_length)
    eval_env.reset(seed=args.seed)
    eval_env.action_space.seed(args.seed)
    logger = build_logger(
        run_dir, args, algo, env_name, chunk_length, macro_discount, dql_config
    )

    try:
        if algo in MODEL_FREE_ALGOS:
            buffer = build_buffer(chunk_dataset, eval_env, args.device)
            policy, lr_scheduler = build_model_free_policy(
                algo, eval_env, buffer, args,
                discount=macro_discount,
                dql_config=dql_config,
            )
            if algo != "dql":
                trainer = MFPolicyTrainer(
                    policy=policy,
                    buffer=buffer,
                    logger=logger,
                    epoch=args.epoch,
                    step_per_epoch=args.step_per_epoch,
                    batch_size=args.batch_size,
                    lr_scheduler=lr_scheduler,
                    checkpoint_epochs=checkpoint_epochs(args.epoch),
                )
        else:
            real_buffer = build_buffer(chunk_dataset, eval_env, args.device)
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
            policy, dynamics, lr_scheduler = build_model_based_policy(
                algo, eval_env, args,
                discount=macro_discount,
                obs_mean=obs_mean,
                obs_std=obs_std,
            )

            print(f"Training dynamics for {algo}: {run_dir}")
            dynamics.train(real_buffer.sample_all(), logger, max_epochs=args.dynamics_max_epochs, max_epochs_since_update=5)
            if algo == "rambo":
                policy.pretrain(
                    real_buffer.sample_all(),
                    args.bc_epoch,
                    min(args.bc_batch_size, len(chunk_dataset["observations"])),
                    1e-4,
                    logger,
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
                dynamics_update_freq=args.dynamics_update_freq if algo == "rambo" else 0,
                checkpoint_epochs=checkpoint_epochs(args.epoch),
            )

        initial_checkpoint = Path(logger.checkpoint_dir) / "step_0"
        initial_checkpoint.mkdir(exist_ok=True)
        torch.save(policy.state_dict(), initial_checkpoint / "policy.pth")
        if algo in MODEL_BASED_ALGOS:
            dynamics.save(initial_checkpoint)

        print(f"Training {algo}: {run_dir}")
        if algo == "dql":
            train_dql(
                policy, buffer, logger, args.epoch, args.step_per_epoch,
                args.batch_size, checkpoint_epochs(args.epoch),
            )
        else:
            trainer.train()
        save_run_manifest(
            run_dir, eval_dir, algo, env_name, split_paths,
            training_schema, chunk_length, macro_discount, args, dql_config,
        )
    finally:
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
    dql_config: dict | None,
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
        "adv_weight": args.adv_weight,
        "adv_batch_size": args.adv_batch_size,
        "rollout_length": args.rollout_length,
        "expert": str(resolve_expert_path(args.expert, env_name)),
        "checkpoints": checkpoint_manifest(run_dir, algo, args.epoch, args.step_per_epoch),
        **split_paths,
    }
    if algo == "dql":
        manifest["cleandiffuser_commit"] = CLEANDIFFUSER_COMMIT
        manifest["dql_config"] = dql_config
    with (run_dir / "run_manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, sort_keys=True)


def build_logger(
    run_dir: Path,
    args: argparse.Namespace,
    algo: str,
    env_name: str,
    chunk_length: int,
    macro_discount: float,
    dql_config: dict | None,
) -> Logger:
    output_config = {
        "consoleout_backup": "stdout",
        "policy_training_progress": "csv",
        "tb": "tensorboard",
    }
    logger = Logger(str(run_dir), output_config)
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
    if dql_config is not None:
        hyperparameters["dql"] = dql_config
    logger.log_hyperparameters(hyperparameters)
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
            dynamics_dir = checkpoint_dir if algo == "rambo" else run_dir / "checkpoint" / "step_0"
            record["dynamics_path"] = str(dynamics_dir.resolve())
        records.append(record)
    return records


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

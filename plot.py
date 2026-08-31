# Tasks:
# - Load completed sweep runs from run manifests and evaluation result files.
# - Average matching results across the random seeds selected by the sweep.
# - Plot final-policy performance and contraction against action chunk length.
# - Plot task performance and contraction against generated noisy-trajectory fractions.
# - Plot task performance and contraction against clean-expert/Minari fractions.
# - Plot state and state-action conservativity over policy-training checkpoints.
# - Plot task performance over policy-training checkpoints.
# - Retain dormant learned-dynamics mismatch plotting for future use.

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


EPS = 1e-12
EVALUATION_SCHEMA_VERSION = 2
BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_PERCENTILES = (10.0, 90.0)
BOOTSTRAP_SEED = 0


def main() -> None:
    args = parse_args()
    plot_root(args.root, args.out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot stable-offline-rl sweep and evaluation results.")
    parser.add_argument("--root", type=Path, required=True, help="Environment directory under stable-offline-rl/evals")
    parser.add_argument("--out", type=Path, default=None, help="Directory for saved plots; defaults to <root>/plots")
    args = parser.parse_args()
    if args.out is None:
        args.out = args.root / "plots"
    return args


def plot_root(root: Path, out: Path | None = None, eval_dirs: list[Path] | None = None) -> None:
    out = root / "plots" if out is None else out
    if eval_dirs is None:
        eval_dirs = latest_eval_dirs(root)
    rows = load_rows(eval_dirs)
    histories = load_histories(eval_dirs, rows)
    rows = average_seed_rows(rows)
    histories = average_seed_histories(histories)
    out.mkdir(parents=True, exist_ok=True)
    plot_generated_ablation(rows, out)
    plot_clean_minari_ablation(rows, out)

    dataset_tags = sorted({row["plot_dataset_tag"] for row in rows} | {history["plot_dataset_tag"] for history in histories})
    for dataset_tag in dataset_tags:
        dataset_out = out / dataset_tag
        dataset_out.mkdir(exist_ok=True)
        dataset_rows = [row for row in rows if row["plot_dataset_tag"] == dataset_tag]
        dataset_histories = [history for history in histories if history["plot_dataset_tag"] == dataset_tag]
        selection_out = dataset_out / "final"
        selection_out.mkdir(exist_ok=True)
        plot_performance_vs_chunk_length(dataset_rows, selection_out)
        plot_contraction_vs_chunk_length(dataset_rows, selection_out)
        plot_training_histories(dataset_histories, dataset_out)


def latest_eval_dirs(root: Path) -> list[Path]:
    latest = {}
    for results_path in sorted(root.glob("*/*/results.json")):
        results = load_json(results_path)
        manifest_path = Path(results["run_manifest_path"])
        if not manifest_path.exists():
            continue
        manifest = load_json(manifest_path)
        latest[json.dumps(manifest["training_schema"], sort_keys=True)] = results_path.parent
    return sorted(latest.values())


def load_rows(eval_dirs: list[Path]) -> list[dict]:
    rows = []
    for eval_dir in sorted(eval_dirs):
        results_path = eval_dir / "results.json"
        if not results_path.exists():
            continue

        results = load_json(results_path)
        if results["evaluation_config"].get("schema_version") != EVALUATION_SCHEMA_VERSION:
            continue
        manifest_path = Path(results["run_manifest_path"])
        manifest = load_json(manifest_path)
        metadata = load_json(Path(manifest["dataset_metadata_path"]))
        row = {
            **manifest,
            **results,
            **dataset_fields(metadata),
            "run_dir": str(manifest_path.parent),
            "eval_dir": str(eval_dir.resolve()),
        }
        training_schema = {
            key: value for key, value in manifest["training_schema"].items()
            if key != "seed"
        }
        training_schema["dataset"] = {
            key: value for key, value in training_schema["dataset"].items()
            if key != "seed"
        }
        row["seed_group"] = json.dumps(training_schema, sort_keys=True)
        row["plot_dataset_tag"] = row["dataset_tag"]
        if row["dataset_source"] == "generated":
            dataset_schema = manifest["training_schema"]["dataset"]
            row["seedless_dataset_tag"] = (
                f"samples{dataset_schema['num_samples']}_"
                f"clean{dataset_schema['prop_clean_expert']:g}_"
                f"noisy{dataset_schema['prop_noisy_expert']:g}_"
                f"noise{dataset_schema['noise_scale']:g}"
            )
        elif row["dataset_source"] == "clean-minari":
            dataset_schema = manifest["training_schema"]["dataset"]
            source = dataset_schema["dataset_id"].replace("/", "_")
            row["seedless_dataset_tag"] = (
                f"clean_minari_{source}_samples{dataset_schema['num_samples']}_"
                f"minari{dataset_schema['minari_fraction']:g}"
            )
        else:
            row["seedless_dataset_tag"] = row["dataset_tag"]
        row["label"] = policy_label(row)
        rows.append(row)
    return rows


def average_seed_rows(rows: list[dict]) -> list[dict]:
    groups = {}
    for row in rows:
        groups.setdefault(row["seed_group"], []).append(row)

    averaged = []
    for seed_rows in groups.values():
        row = dict(seed_rows[0])
        row["seed_rows"] = seed_rows
        for key in (
            "last_policy_performance_mean",
            "expert_performance_mean",
        ):
            row[key] = float(np.mean([seed_row[key] for seed_row in seed_rows]))
        if row["dataset_source"] == "generated":
            row["noisy_trajectory_fraction"] = float(np.mean([
                seed_row["noisy_trajectory_fraction"] for seed_row in seed_rows
            ]))
        elif row["dataset_source"] == "clean-minari":
            row["minari_trajectory_fraction"] = float(np.mean([
                seed_row["minari_trajectory_fraction"] for seed_row in seed_rows
            ]))
        if len(seed_rows) > 1 and row["dataset_source"] in {
            "generated", "clean-minari"
        }:
            row["plot_dataset_tag"] = row["seedless_dataset_tag"]
        averaged.append(row)
    return averaged


def dataset_fields(metadata: dict) -> dict:
    if metadata.get("source") == "minari":
        return {
            "dataset_source": "minari",
            "minari_dataset": metadata["dataset_id"].split("/")[-1].removesuffix("-v0"),
        }
    if metadata.get("source") == "clean-minari":
        dataset_schema = metadata["dataset_schema"]
        return {
            "dataset_source": "clean-minari",
            "minari_dataset_id": metadata["dataset_id"],
            "minari_dataset": metadata["dataset_id"].split("/")[-1].removesuffix("-v0"),
            "num_samples": dataset_schema["num_samples"],
            "requested_minari_trajectory_fraction": dataset_schema["minari_fraction"],
            "minari_trajectory_fraction": metadata["actual_minari_trajectory_fraction"],
        }
    if metadata.get("source") == "robomimic":
        return {
            "dataset_source": "robomimic",
            "robomimic_task": metadata["task"],
            "robomimic_dataset": metadata["dataset_type"],
        }

    dataset_schema = metadata["dataset_schema"]
    return {
        "dataset_source": "generated",
        "num_samples": dataset_schema["num_samples"],
        "noise_scale": dataset_schema["noise_scale"],
        "requested_prop_clean_expert": dataset_schema["prop_clean_expert"],
        "requested_prop_noisy_expert": dataset_schema["prop_noisy_expert"],
        "requested_prop_random": dataset_schema["prop_random"],
        "noisy_trajectory_fraction": metadata["actual_prop_noisy_expert_trajectories"],
    }


def load_histories(eval_dirs: list[Path], rows: list[dict]) -> list[dict]:
    rows_by_eval_dir = {row["eval_dir"]: row for row in rows}
    histories = []
    for eval_dir in sorted(eval_dirs):
        history_path = eval_dir / "history.json"
        row = rows_by_eval_dir.get(str(eval_dir.resolve()))
        if not history_path.exists() or row is None:
            continue
        history = load_json(history_path)
        history["eval_dir"] = str(eval_dir.resolve())
        history["seed_group"] = row["seed_group"]
        history["plot_dataset_tag"] = row["plot_dataset_tag"]
        history["seedless_dataset_tag"] = row["seedless_dataset_tag"]
        history["label"] = policy_label(history)
        histories.append(history)
    return histories


def average_seed_histories(histories: list[dict]) -> list[dict]:
    groups = {}
    for history in histories:
        groups.setdefault(history["seed_group"], []).append(history)

    averaged = []
    for seed_histories in groups.values():
        history = dict(seed_histories[0])
        history["seed_histories"] = seed_histories
        history["expert_performance_mean"] = float(np.mean([
            seed_history["expert_performance_mean"]
            for seed_history in seed_histories
        ]))
        if len(seed_histories) > 1 and history["dataset_tag"] != history["seedless_dataset_tag"]:
            history["plot_dataset_tag"] = history["seedless_dataset_tag"]

        records_by_percent = {}
        for seed_history in seed_histories:
            for record in seed_history["records"]:
                records_by_percent.setdefault(record["actual_percent"], []).append(record)
        history["records"] = []
        for actual_percent, seed_records in sorted(records_by_percent.items()):
            record = dict(seed_records[0])
            for key in (
                "policy_performance_mean",
                "state_ood_ratio",
                "state_action_ood_ratio",
            ):
                values = [seed_record[key] for seed_record in seed_records if key in seed_record]
                if values:
                    record[key] = float(np.mean(values))
            history["records"].append(record)
        averaged.append(history)
    return averaged


def policy_label(record: dict) -> str:
    return f"{record['algo']} (l={record['chunk_length']})"


def bootstrap_mean(samples_by_seed: list[np.ndarray]) -> tuple[float, float, float]:
    samples_by_seed = [np.asarray(samples, dtype=np.float64).reshape(-1) for samples in samples_by_seed]
    center = float(np.mean([samples.mean() for samples in samples_by_seed]))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    seed_count = len(samples_by_seed)
    seed_choices = rng.integers(0, seed_count, size=(BOOTSTRAP_REPLICATES, seed_count))
    bootstrap_means = np.empty((BOOTSTRAP_REPLICATES, seed_count), dtype=np.float64)

    for slot in range(seed_count):
        for seed_index, samples in enumerate(samples_by_seed):
            rows = np.flatnonzero(seed_choices[:, slot] == seed_index)
            indices = rng.integers(0, len(samples), size=(len(rows), len(samples)))
            bootstrap_means[rows, slot] = samples[indices].mean(axis=1)

    low, high = np.percentile(bootstrap_means.mean(axis=1), BOOTSTRAP_PERCENTILES)
    return center, float(low), float(high)


def finite_mean(values: np.ndarray, axis: int) -> np.ndarray:
    counts = np.sum(np.isfinite(values), axis=axis)
    return np.divide(
        np.nansum(values, axis=axis),
        counts,
        out=np.full(counts.shape, np.nan, dtype=np.float64),
        where=counts > 0,
    )


def bootstrap_curve(
    curves_by_seed: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    width = max(curves.shape[1] for curves in curves_by_seed)
    padded = []
    for curves in curves_by_seed:
        seed_curves = np.full((len(curves), width), np.nan, dtype=np.float64)
        seed_curves[:, : curves.shape[1]] = curves
        padded.append(seed_curves)

    center = finite_mean(
        np.stack([finite_mean(curves, axis=0) for curves in padded]), axis=0
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    seed_count = len(padded)
    seed_choices = rng.integers(0, seed_count, size=(BOOTSTRAP_REPLICATES, seed_count))
    sums = np.zeros((BOOTSTRAP_REPLICATES, width), dtype=np.float64)
    counts = np.zeros((BOOTSTRAP_REPLICATES, width), dtype=np.int16)

    for slot in range(seed_count):
        for seed_index, curves in enumerate(padded):
            rows = np.flatnonzero(seed_choices[:, slot] == seed_index)
            for start in range(0, len(rows), 500):
                batch_rows = rows[start : start + 500]
                indices = rng.integers(
                    0, len(curves), size=(len(batch_rows), len(curves))
                )
                means = finite_mean(curves[indices], axis=1)
                valid = np.isfinite(means)
                sums[batch_rows] += np.where(valid, means, 0.0)
                counts[batch_rows] += valid

    bootstrap_means = np.divide(
        sums,
        counts,
        out=np.full(sums.shape, np.nan, dtype=np.float64),
        where=counts > 0,
    )
    low = np.full(width, np.nan, dtype=np.float64)
    high = np.full(width, np.nan, dtype=np.float64)
    for timestep in range(width):
        values = bootstrap_means[:, timestep]
        values = values[np.isfinite(values)]
        if len(values):
            low[timestep], high[timestep] = np.percentile(values, BOOTSTRAP_PERCENTILES)
    return center, low, high


def final_performance_samples(row: dict) -> list[np.ndarray]:
    samples = []
    for seed_row in row["seed_rows"]:
        with np.load(Path(seed_row["eval_dir"]) / "returns_last.npz") as data:
            samples.append(data["policy_episode_performance"])
    return samples


def history_performance_samples(history: dict, step: int) -> list[np.ndarray]:
    samples = []
    for seed_history in history["seed_histories"]:
        record = next(record for record in seed_history["records"] if record["step"] == step)
        with np.load(Path(seed_history["eval_dir"]) / "rollouts" / f"step_{record['step']}.npz") as data:
            samples.append(data["performance"])
    return samples


def history_scalar_samples(history: dict, step: int, key: str) -> list[np.ndarray]:
    samples = []
    for seed_history in history["seed_histories"]:
        record = next(record for record in seed_history["records"] if record["step"] == step)
        samples.append(np.asarray([record[key]], dtype=np.float64))
    return samples


def plot_performance_vs_chunk_length(rows: list[dict], out: Path) -> None:
    if len({row["chunk_length"] for row in rows}) < 2:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for algo in sorted({row["algo"] for row in rows}):
        algo_rows = sorted((row for row in rows if row["algo"] == algo), key=lambda row: row["chunk_length"])
        x = [row["chunk_length"] for row in algo_rows]
        intervals = [bootstrap_mean(final_performance_samples(row)) for row in algo_rows]
        center, low, high = map(np.asarray, zip(*intervals))
        line, = ax.plot(x, center, marker="o", label=algo)
        ax.fill_between(x, low, high, color=line.get_color(), alpha=0.2)
    ax.axhline(
        np.mean([row["expert_performance_mean"] for row in rows]),
        color="black", linestyle=":", label="expert",
    )
    ax.set_xlabel("action chunk length")
    ax.set_ylabel(rows[0]["performance_label"])
    ax.set_title(f"Final-policy {rows[0]['performance_label']} vs action chunk length")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "performance_vs_chunk_length.png", dpi=200)
    plt.close(fig)


def plot_contraction_vs_chunk_length(rows: list[dict], out: Path) -> None:
    if len({row["chunk_length"] for row in rows}) < 2:
        return
    contraction_curve_plot(
        rows, "chunk_length", "chunk length",
        "Final-policy contraction by action chunk length",
        out / "contraction_vs_chunk_length.png",
    )


def plot_generated_ablation(rows: list[dict], out: Path) -> None:
    generated = [row for row in rows if row["dataset_source"] == "generated"]
    if not generated or len({row["noisy_trajectory_fraction"] for row in generated}) < 2:
        return
    fixed = ("num_samples", "noise_scale", "chunk_length")
    if any(len({row[key] for row in generated}) > 1 for key in fixed):
        return

    if all(row["requested_prop_clean_expert"] == 0.0 for row in generated):
        family = "random_noisy"
        title = "Random/noisy trajectory ablation"
    elif len({row["requested_prop_random"] for row in generated}) == 1:
        family = "expert_noisy"
        title = "Clean-expert/noisy-expert trajectory ablation"
    else:
        return

    experiment_out = out / family / "final"
    experiment_out.mkdir(parents=True, exist_ok=True)
    performance_ablation_plot(
        generated, title, experiment_out / "performance_vs_noisy_fraction.png",
        "noisy_trajectory_fraction", "fraction of trajectories collected from the noisy expert",
    )
    contraction_curve_plot(
        generated, "noisy_trajectory_fraction", "noisy trajectory fraction",
        f"Final-policy {title.lower()}",
        experiment_out / "contraction_vs_noisy_fraction.png",
    )


def plot_clean_minari_ablation(rows: list[dict], out: Path) -> None:
    mixed = [row for row in rows if row["dataset_source"] == "clean-minari"]
    if not mixed:
        return
    clean = [
        row for row in rows
        if row["dataset_source"] == "generated"
        and row["requested_prop_clean_expert"] == 1.0
        and row["requested_prop_noisy_expert"] == 0.0
        and row["requested_prop_random"] == 0.0
    ]

    groups = {}
    for row in mixed:
        key = (row["minari_dataset_id"], row["num_samples"], row["chunk_length"])
        groups.setdefault(key, []).append(row)

    for (dataset_id, num_samples, chunk_length), mixture_rows in groups.items():
        plot_rows = list(mixture_rows)
        baselines = {}
        for row in clean:
            if row["num_samples"] == num_samples and row["chunk_length"] == chunk_length:
                if row["algo"] not in baselines:
                    baselines[row["algo"]] = {
                        **row,
                        "seed_rows": list(row["seed_rows"]),
                        "minari_trajectory_fraction": 0.0,
                    }
                else:
                    baselines[row["algo"]]["seed_rows"].extend(row["seed_rows"])
        plot_rows.extend(baselines.values())
        if len({row["minari_trajectory_fraction"] for row in plot_rows}) < 2:
            continue

        dataset_name = dataset_id.split("/")[-1].removesuffix("-v0")
        experiment_out = (
            out / "clean_minari" / dataset_name
            / f"samples{num_samples}_chunk{chunk_length}" / "final"
        )
        experiment_out.mkdir(parents=True, exist_ok=True)
        title = f"Clean-expert/{dataset_name} Minari trajectory ablation"
        performance_ablation_plot(
            plot_rows, title,
            experiment_out / "performance_vs_minari_fraction.png",
            "minari_trajectory_fraction",
            "fraction of trajectories drawn from the Minari dataset",
        )
        contraction_curve_plot(
            plot_rows, "minari_trajectory_fraction", "Minari trajectory fraction",
            f"Final-policy {title.lower()}",
            experiment_out / "contraction_vs_minari_fraction.png",
        )


def performance_ablation_plot(
    rows: list[dict],
    title: str,
    path: Path,
    value_key: str,
    value_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for algo in sorted({row["algo"] for row in rows}):
        algo_rows = sorted(
            (row for row in rows if row["algo"] == algo),
            key=lambda row: row[value_key],
        )
        x = [row[value_key] for row in algo_rows]
        intervals = [bootstrap_mean(final_performance_samples(row)) for row in algo_rows]
        center, low, high = map(np.asarray, zip(*intervals))
        line, = ax.plot(x, center, marker="o", label=algo)
        ax.fill_between(x, low, high, color=line.get_color(), alpha=0.2)
    ax.axhline(
        np.mean([row["expert_performance_mean"] for row in rows]),
        color="black", linestyle=":", label="expert",
    )
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel(value_label)
    ax.set_ylabel(rows[0]["performance_label"])
    ax.set_title(f"Final-policy {rows[0]['performance_label']}\n{title}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def contraction_curve_plot(
    rows: list[dict],
    value_key: str,
    value_label: str,
    title: str,
    path: Path,
) -> None:
    filename = "contraction_last.npz"
    available = [
        row for row in rows
        if any((Path(seed_row["eval_dir"]) / filename).exists() for seed_row in row["seed_rows"])
    ]
    if not available:
        return
    algorithms = sorted({row["algo"] for row in available})
    fig, axes = plt.subplots(1, len(algorithms), figsize=(5 * len(algorithms), 4), squeeze=False)
    for axis, algo in zip(axes[0], algorithms):
        for row in sorted(
            (row for row in available if row["algo"] == algo), key=lambda row: row[value_key]
        ):
            seed_curves = []
            for seed_row in row["seed_rows"]:
                contraction_path = Path(seed_row["eval_dir"]) / filename
                if contraction_path.exists():
                    with np.load(contraction_path) as data:
                        seed_curves.append(data["distance_curves"])
            center, low, high = bootstrap_curve(seed_curves)
            timesteps = np.arange(len(center))
            line, = axis.plot(center, label=f"{value_label}={row[value_key]:g}")
            axis.fill_between(
                timesteps, low, high, color=line.get_color(), alpha=0.2
            )
        axis.set_title(algo)
        axis.set_xlabel("primitive timestep")
        axis.legend(fontsize=8)
    axes[0, 0].set_ylabel("agent Cartesian-position distance (m)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_mismatch_ratios(rows: list[dict], out: Path) -> None:
    next_obs_rows = [
        row for row in rows
        if "dataset_next_obs_mse" in row and "rollout_next_obs_mse" in row
    ]
    for row in next_obs_rows:
        row["next_obs_ratio"] = row["dataset_next_obs_mse"] / (row["rollout_next_obs_mse"] + EPS)

    if next_obs_rows:
        scatter(
            next_obs_rows,
            x_key="next_obs_ratio",
            y_key="policy_return_mean",
            color_key="expert_return_mean",
            xlabel="dataset macro next-state MSE / rollout macro next-state MSE",
            ylabel="policy return",
            color_label="expert return",
            path=out / "reward_vs_next_obs_ratio.png",
        )

    jacobian_rows = [
        row for row in rows
        if "dataset_next_obs_mse" in row and "dataset_closed_loop_jacobian_mse" in row
    ]
    if not jacobian_rows:
        return

    for row in jacobian_rows:
        row["jacobian_ratio"] = row["dataset_closed_loop_jacobian_mse"] / (row["rollout_closed_loop_jacobian_mse"] + EPS)

    scatter(
        jacobian_rows,
        x_key="next_obs_ratio",
        y_key="jacobian_ratio",
        color_key="policy_return_mean",
        xlabel="dataset macro next-state MSE / rollout macro next-state MSE",
        ylabel="dataset Jacobian MSE / rollout Jacobian MSE",
        color_label="policy return",
        path=out / "mismatch_ratio_reward_scatter.png",
    )
    scatter(
        jacobian_rows,
        x_key="jacobian_ratio",
        y_key="policy_return_mean",
        color_key="next_obs_ratio",
        xlabel="dataset Jacobian MSE / rollout Jacobian MSE",
        ylabel="policy return",
        color_label="next-state mismatch ratio",
        path=out / "reward_vs_jacobian_ratio.png",
    )


def plot_training_histories(histories: list[dict], out: Path) -> None:
    if not histories:
        return

    performance_history_plot(histories, out / "performance_vs_training_percent.png")
    history_line_plot(
        histories, ("state_ood_ratio", "state_action_ood_ratio"),
        ("state OOD", "state-action OOD"), out / "ood_vs_training_percent.png",
    )


def performance_history_plot(histories: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for history in histories:
        records = history["records"]
        x = [record["actual_percent"] for record in records]
        intervals = [
            bootstrap_mean(history_performance_samples(history, record["step"]))
            for record in records
        ]
        center, low, high = map(np.asarray, zip(*intervals))
        line, = ax.plot(x, center, marker="o", label=history["label"])
        ax.fill_between(x, low, high, color=line.get_color(), alpha=0.2)
    ax.axhline(
        np.mean([history["expert_performance_mean"] for history in histories]),
        color="black", linestyle=":", label="expert",
    )
    ax.set_xlabel("training completed (%)")
    ax.set_ylabel(histories[0]["performance_label"])
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def history_line_plot(histories: list[dict], keys: tuple[str, ...], names: tuple[str, ...], path: Path) -> None:
    fig, axes = plt.subplots(len(keys), 1, figsize=(8, 3 * len(keys)), squeeze=False, sharex=True)
    for axis, key, name in zip(axes[:, 0], keys, names):
        key_histories = [history for history in histories if history["records"] and key in history["records"][0]]
        if not key_histories:
            axis.set_visible(False)
            continue
        for history in key_histories:
            records = history["records"]
            x = [record["actual_percent"] for record in records]
            intervals = [
                bootstrap_mean(history_scalar_samples(history, record["step"], key))
                for record in records
            ]
            center, low, high = map(np.asarray, zip(*intervals))
            line, = axis.plot(x, center, marker="o", label=history["label"])
            axis.fill_between(x, low, high, color=line.get_color(), alpha=0.2)
        if key.endswith("_ood_ratio"):
            axis.axhline(1.0, color="gray", linestyle=":")
        axis.set_ylabel(name)
    axes[0, 0].legend(fontsize=8)
    axes[-1, 0].set_xlabel("training completed (%)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def scatter(rows: list[dict], x_key: str, y_key: str, color_key: str, xlabel: str, ylabel: str, color_label: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    points = ax.scatter(
        [row[x_key] for row in rows],
        [row[y_key] for row in rows],
        c=[row[color_key] for row in rows],
        cmap="viridis",
    )
    for row in rows:
        ax.annotate(row["label"], (row[x_key], row[y_key]), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.colorbar(points, ax=ax, label=color_label)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


if __name__ == "__main__":
    main()

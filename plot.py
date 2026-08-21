# Tasks:
# - Load completed sweep runs from run manifests and evaluation result files.
# - Average matching results across the random seeds selected by the sweep.
# - Plot best- and last-checkpoint performance and contraction against action chunk length.
# - Plot task performance and contraction against generated noisy-trajectory fractions.
# - Plot state and state-action conservativity over policy-training checkpoints.
# - Plot task performance over policy-training checkpoints.
# - Retain dormant learned-dynamics mismatch plotting for future use.

import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


EPS = 1e-12


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
        latest = {}
        for results_path in sorted(root.glob("*/*/results.json")):
            latest[results_path.parent.parent.name] = results_path.parent
        eval_dirs = sorted(latest.values())
    rows = load_rows(eval_dirs)
    histories = load_histories(eval_dirs, rows)
    rows = average_seed_rows(rows)
    histories = average_seed_histories(histories)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for selection in ("best", "last"):
        plot_generated_ablation(rows, out, selection)

    dataset_tags = sorted({row["plot_dataset_tag"] for row in rows} | {history["plot_dataset_tag"] for history in histories})
    for dataset_tag in dataset_tags:
        dataset_out = out / dataset_tag
        dataset_out.mkdir(exist_ok=True)
        dataset_rows = [row for row in rows if row["plot_dataset_tag"] == dataset_tag]
        dataset_histories = [history for history in histories if history["plot_dataset_tag"] == dataset_tag]
        for selection in ("best", "last"):
            selection_out = dataset_out / selection
            selection_out.mkdir(exist_ok=True)
            plot_performance_vs_chunk_length(dataset_rows, selection_out, selection)
            plot_contraction_vs_chunk_length(dataset_rows, selection_out, selection)
        plot_training_histories(dataset_histories, dataset_out)


def load_rows(eval_dirs: list[Path]) -> list[dict]:
    rows = []
    for eval_dir in sorted(eval_dirs):
        results_path = eval_dir / "results.json"
        if not results_path.exists():
            print(f"Skipping incomplete evaluation: {eval_dir}")
            continue

        results = load_json(results_path)
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
            "best_policy_performance_mean",
            "last_policy_performance_mean",
            "expert_performance_mean",
        ):
            row[key] = float(np.mean([seed_row[key] for seed_row in seed_rows]))
        if row["dataset_source"] == "generated":
            row["noisy_trajectory_fraction"] = float(np.mean([
                seed_row["noisy_trajectory_fraction"] for seed_row in seed_rows
            ]))
            if len(seed_rows) > 1:
                row["plot_dataset_tag"] = row["seedless_dataset_tag"]
        averaged.append(row)
    return averaged


def dataset_fields(metadata: dict) -> dict:
    if metadata.get("source") == "minari":
        return {
            "dataset_source": "minari",
            "minari_dataset": metadata["dataset_id"].split("/")[-1].removesuffix("-v0"),
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


def plot_performance_vs_chunk_length(rows: list[dict], out: Path, selection: str) -> None:
    if len({row["chunk_length"] for row in rows}) < 2:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for algo in sorted({row["algo"] for row in rows}):
        algo_rows = sorted((row for row in rows if row["algo"] == algo), key=lambda row: row["chunk_length"])
        ax.plot(
            [row["chunk_length"] for row in algo_rows],
            [row[f"{selection}_policy_performance_mean"] for row in algo_rows],
            marker="o",
            label=algo,
        )
    ax.axhline(
        np.mean([row["expert_performance_mean"] for row in rows]),
        color="black", linestyle=":", label="expert",
    )
    ax.set_xlabel("action chunk length")
    ax.set_ylabel(rows[0]["performance_label"])
    ax.set_title(f"{selection.capitalize()}-checkpoint {rows[0]['performance_label']} vs action chunk length")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "performance_vs_chunk_length.png", dpi=200)
    plt.close(fig)


def plot_contraction_vs_chunk_length(rows: list[dict], out: Path, selection: str) -> None:
    if len({row["chunk_length"] for row in rows}) < 2:
        return
    contraction_curve_plot(
        rows, "chunk_length", "chunk length",
        f"{selection.capitalize()}-checkpoint contraction by action chunk length",
        out / "contraction_vs_chunk_length.png",
        selection,
    )


def plot_generated_ablation(rows: list[dict], out: Path, selection: str) -> None:
    generated = [row for row in rows if row["dataset_source"] == "generated"]
    if not generated or len({row["noisy_trajectory_fraction"] for row in generated}) < 2:
        return
    fixed = ("num_samples", "noise_scale", "chunk_length")
    if any(len({row[key] for row in generated}) > 1 for key in fixed):
        print("Skipping generated composition plots: sample count, noise scale, and chunk length must be fixed.")
        return

    if all(row["requested_prop_clean_expert"] == 0.0 for row in generated):
        family = "random_noisy"
        title = "Random/noisy trajectory ablation"
    elif len({row["requested_prop_random"] for row in generated}) == 1:
        family = "expert_noisy"
        title = "Clean-expert/noisy-expert trajectory ablation"
    else:
        print("Skipping generated composition plots: rows do not form one requested composition ablation.")
        return

    experiment_out = out / family / selection
    experiment_out.mkdir(parents=True, exist_ok=True)
    performance_ablation_plot(
        generated, title, experiment_out / "performance_vs_noisy_fraction.png", selection
    )
    contraction_curve_plot(
        generated, "noisy_trajectory_fraction", "noisy trajectory fraction",
        f"{selection.capitalize()}-checkpoint {title.lower()}",
        experiment_out / "contraction_vs_noisy_fraction.png", selection,
    )


def performance_ablation_plot(rows: list[dict], title: str, path: Path, selection: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for algo in sorted({row["algo"] for row in rows}):
        algo_rows = sorted(
            (row for row in rows if row["algo"] == algo),
            key=lambda row: row["noisy_trajectory_fraction"],
        )
        ax.plot(
            [row["noisy_trajectory_fraction"] for row in algo_rows],
            [row[f"{selection}_policy_performance_mean"] for row in algo_rows],
            marker="o", label=algo,
        )
    ax.axhline(
        np.mean([row["expert_performance_mean"] for row in rows]),
        color="black", linestyle=":", label="expert",
    )
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("fraction of trajectories collected from the noisy expert")
    ax.set_ylabel(rows[0]["performance_label"])
    ax.set_title(f"{selection.capitalize()}-checkpoint {rows[0]['performance_label']}\n{title}")
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
    selection: str,
) -> None:
    filename = f"contraction_{selection}.npz"
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
                        seed_curves.append(np.nanmean(data["distance_curves"], axis=0))
            mean_curve = np.nanmean(seed_curves, axis=0)
            axis.plot(mean_curve, label=f"{value_label}={row[value_key]:g}")
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
        ax.plot(
            [record["actual_percent"] for record in records],
            [record["policy_performance_mean"] for record in records],
            marker="o",
            label=history["label"],
        )
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
            axis.plot(
                [record["actual_percent"] for record in records],
                [record[key] for record in records],
                marker="o",
                label=history["label"],
            )
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

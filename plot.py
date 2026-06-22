# Tasks:
# - Load completed sweep runs from run manifests and evaluation result files.
# - Plot generated-dataset reward ablations when exactly one generated axis varies.
# - Plot Minari reward bars without implying a numeric ablation.
# - Plot run-level dynamics mismatch ratios against learned policy reward.
# - Plot stability and conservativity over policy-training checkpoints.

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DATASET_ORDER = ("simple", "medium", "expert")
EPS = 1e-12


def main() -> None:
    args = parse_args()
    plot_root(args.root, args.out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot stable-offline-rl sweep and evaluation results.")
    parser.add_argument("--root", type=Path, required=True, help="Environment output directory containing runs/ and datasets/")
    parser.add_argument("--out", type=Path, default=None, help="Directory for saved plots; defaults to <root>/plots")
    args = parser.parse_args()
    if args.out is None:
        args.out = args.root / "plots"
    return args


def plot_root(root: Path, out: Path | None = None) -> None:
    out = root / "plots" if out is None else out
    rows = load_rows(root)
    histories = load_histories(root)
    out.mkdir(parents=True, exist_ok=True)
    plot_generated_reward_ablation(rows, out)
    plot_minari_reward_bars(rows, out)

    dataset_tags = sorted({row["dataset_tag"] for row in rows} | {history["dataset_tag"] for history in histories})
    for dataset_tag in dataset_tags:
        dataset_out = out / dataset_tag
        dataset_out.mkdir(exist_ok=True)
        dataset_rows = [row for row in rows if row["dataset_tag"] == dataset_tag]
        dataset_histories = [history for history in histories if history["dataset_tag"] == dataset_tag]
        plot_stability_distances(dataset_rows, dataset_out)
        plot_mismatch_ratios(dataset_rows, dataset_out)
        plot_history_relationships(dataset_histories, dataset_out)
        plot_training_histories(dataset_histories, dataset_out)


def load_rows(root: Path) -> list[dict]:
    rows = []
    for manifest_path in sorted((root / "runs").glob("*/run_manifest.json")):
        run_dir = manifest_path.parent
        results_path = run_dir / "eval" / "results.json"
        if not results_path.exists():
            print(f"Skipping unevaluated run: {run_dir}")
            continue

        manifest = load_json(manifest_path)
        results = load_json(results_path)
        metadata = load_json(Path(manifest["dataset_metadata_path"]))
        row = {**manifest, **results, **dataset_fields(metadata), "run_dir": str(run_dir.resolve())}
        rows.append(row)
    return rows


def dataset_fields(metadata: dict) -> dict:
    if metadata.get("source") == "minari":
        return {
            "dataset_source": "minari",
            "minari_dataset": metadata["dataset_id"].split("/")[-1].removesuffix("-v0"),
        }

    num_expert = metadata["num_expert"]
    num_suboptimal = metadata["num_suboptimal"]
    num_samples = num_expert + num_suboptimal
    return {
        "dataset_source": "generated",
        "num_samples": num_samples,
        "noise_scale": metadata["noise_scale"],
        "prop_expert": num_expert / num_samples,
    }


def load_histories(root: Path) -> list[dict]:
    histories = []
    for history_path in sorted((root / "runs").glob("*/eval/history.json")):
        history = load_json(history_path)
        history["label"] = history["algo"]
        histories.append(history)
    return histories


def plot_generated_reward_ablation(rows: list[dict], out: Path) -> None:
    generated = [row for row in rows if row["dataset_source"] == "generated"]
    if not generated:
        return

    axes = ("num_samples", "noise_scale", "prop_expert")
    varying = [axis for axis in axes if len({row[axis] for row in generated}) > 1]
    if len(varying) != 1:
        print(f"Skipping generated reward ablation: expected one varying axis, found {varying}.")
        return

    axis = varying[0]
    fig, ax = plt.subplots(figsize=(8, 5))
    for algo in sorted({row["algo"] for row in generated}):
        algo_rows = sorted((row for row in generated if row["algo"] == algo), key=lambda row: row[axis])
        ax.plot([row[axis] for row in algo_rows], [row["policy_return_mean"] for row in algo_rows], marker="o", label=algo)

    ax.axhline(np.mean([row["expert_return_mean"] for row in generated]), color="black", linestyle=":", label="expert")
    ax.set_xlabel(axis.replace("_", " "))
    ax.set_ylabel("policy return")
    ax.set_title(f"Generated dataset ablation: {axis.replace('_', ' ')}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / f"generated_reward_vs_{axis}.png", dpi=200)
    plt.close(fig)


def plot_minari_reward_bars(rows: list[dict], out: Path) -> None:
    minari = [row for row in rows if row["dataset_source"] == "minari"]
    if not minari:
        return

    datasets = [name for name in DATASET_ORDER if any(row["minari_dataset"] == name for row in minari)]
    algos = sorted({row["algo"] for row in minari})
    width = 0.8 / max(len(algos), 1)
    x = np.arange(len(datasets))

    fig, ax = plt.subplots(figsize=(8, 5))
    for algo_index, algo in enumerate(algos):
        values = []
        for dataset in datasets:
            matching = [row["policy_return_mean"] for row in minari if row["algo"] == algo and row["minari_dataset"] == dataset]
            values.append(np.mean(matching) if matching else np.nan)
        ax.bar(x + (algo_index - (len(algos) - 1) / 2) * width, values, width=width, label=algo)

    ax.axhline(np.mean([row["expert_return_mean"] for row in minari]), color="black", linestyle=":", label="expert")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_xlabel("Minari dataset")
    ax.set_ylabel("policy return")
    ax.set_title("Policy reward by Minari dataset")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "minari_reward_by_dataset.png", dpi=200)
    plt.close(fig)


def plot_stability_distances(rows: list[dict], out: Path) -> None:
    stability_distance_plot(rows, out, "global")
    stability_distance_plot(rows, out, "local")


def stability_distance_plot(rows: list[dict], out: Path, stability_type: str) -> None:
    filename = f"{stability_type}_stability.npz"
    available = [row for row in rows if (Path(row["run_dir"]) / "eval" / filename).exists()]
    if not available:
        return

    available.sort(key=lambda row: row["algo"])
    fig, axes = plt.subplots(
        2, len(available), figsize=(4 * len(available), 7),
        squeeze=False, sharex="col",
    )
    for column, row in enumerate(available):
        with np.load(Path(row["run_dir"]) / "eval" / filename) as data:
            curves = data["distance_curves"]
            envelope = data["envelope"]
            support = data["support"]
            c = float(data["c"])
            rho = float(data["rho"])

        normalized = curves / np.maximum(curves[:, :1], EPS)
        normalized[normalized <= 0.0] = np.nan
        timesteps = np.arange(len(envelope))
        bound = c * rho**timesteps
        bound[bound <= 0.0] = np.nan

        distance_axis = axes[0, column]
        pair_lines = distance_axis.plot(normalized.T, color="tab:blue", alpha=0.2, linewidth=0.7)
        pair_lines[0].set_label("trajectory pairs")
        distance_axis.plot(envelope, color="tab:red", linewidth=2, label="maximum envelope")
        distance_axis.plot(bound, color="black", linestyle="--", linewidth=1.5, label=r"tight $C\rho^t$ upper bound")
        distance_axis.axhline(1.0, color="gray", linestyle=":", label="initial distance")
        distance_axis.set_yscale("log")
        distance_axis.set_title(f"{row['algo'].upper()}\nC={c:.3g}, rho={rho:.6g}")

        support_axis = axes[1, column]
        support_axis.plot(support / support[0], color="tab:purple", linewidth=2)
        support_axis.set_ylim(-0.02, 1.02)
        support_axis.set_xlabel("timestep")

    axes[0, 0].set_ylabel("normalized state distance")
    axes[1, 0].set_ylabel("pair support fraction")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"{stability_type.capitalize()} final-policy stability distances")
    fig.tight_layout()
    fig.savefig(out / f"{stability_type}_stability_distances.png", dpi=200)
    plt.close(fig)


def plot_mismatch_ratios(rows: list[dict], out: Path) -> None:
    model_rows = [
        row for row in rows
        if "dataset_next_obs_mse" in row and "dataset_closed_loop_jacobian_mse" in row
    ]
    if not model_rows:
        return

    for row in model_rows:
        row["next_obs_ratio"] = row["dataset_next_obs_mse"] / (row["rollout_next_obs_mse"] + EPS)
        row["jacobian_ratio"] = row["dataset_closed_loop_jacobian_mse"] / (row["rollout_closed_loop_jacobian_mse"] + EPS)
        row["normalized_return"] = row["policy_return_mean"] / (row["expert_return_mean"] + EPS)

    scatter(
        model_rows,
        x_key="next_obs_ratio",
        y_key="jacobian_ratio",
        color_key="policy_return_mean",
        xlabel="dataset next-state MSE / rollout next-state MSE",
        ylabel="dataset Jacobian MSE / rollout Jacobian MSE",
        color_label="policy return",
        path=out / "mismatch_ratio_reward_scatter.png",
    )
    scatter(
        model_rows,
        x_key="next_obs_ratio",
        y_key="policy_return_mean",
        color_key="jacobian_ratio",
        xlabel="dataset next-state MSE / rollout next-state MSE",
        ylabel="policy return",
        color_label="Jacobian mismatch ratio",
        path=out / "reward_vs_next_obs_ratio.png",
    )
    scatter(
        model_rows,
        x_key="jacobian_ratio",
        y_key="policy_return_mean",
        color_key="next_obs_ratio",
        xlabel="dataset Jacobian MSE / rollout Jacobian MSE",
        ylabel="policy return",
        color_label="next-state mismatch ratio",
        path=out / "reward_vs_jacobian_ratio.png",
    )


def plot_history_relationships(histories: list[dict], out: Path) -> None:
    if not histories:
        return

    history_relationship_plot(
        histories, "global_stability_rho", "global_stability_c",
        "global empirical rho", out / "global_stability_vs_reward.png", reference_x=1.0,
    )
    history_relationship_plot(
        histories, "local_stability_rho", "local_stability_c",
        "local empirical rho", out / "local_stability_vs_reward.png", reference_x=1.0,
    )
    history_relationship_plot(
        histories, "state_ood_ratio", None,
        "state OOD ratio", out / "state_ood_vs_reward.png", reference_x=1.0,
    )
    history_relationship_plot(
        histories, "state_action_ood_ratio", None,
        "state-action OOD ratio", out / "state_action_ood_vs_reward.png", reference_x=1.0,
    )


def history_relationship_plot(
    histories: list[dict],
    x_key: str,
    c_key: str | None,
    xlabel: str,
    path: Path,
    reference_x: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    color_scale = plt.Normalize(0, 100)
    color_map = plt.get_cmap("viridis")
    for history in histories:
        records = history["records"]
        x = np.asarray([record[x_key] for record in records])
        reward = np.asarray([record["policy_return_mean"] for record in records])
        percent = np.asarray([record["actual_percent"] for record in records])
        sizes = 35 if c_key is None else 25 + 20 * np.log1p([record[c_key] for record in records])
        ax.plot(x, reward, linewidth=1, alpha=0.7, label=history["label"])
        ax.scatter(x, reward, c=percent, s=sizes, cmap=color_map, norm=color_scale)

    if reference_x is not None:
        ax.axvline(reference_x, color="gray", linestyle=":", label=f"{xlabel} = 1")
    ax.axhline(np.mean([history["expert_return_mean"] for history in histories]), color="black", linestyle=":", label="expert return")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("policy return")
    if c_key is not None:
        ax.set_title("Marker size scales with log(1 + C)")
    ax.legend(fontsize=8)
    fig.colorbar(plt.cm.ScalarMappable(norm=color_scale, cmap=color_map), ax=ax, label="training completed (%)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_training_histories(histories: list[dict], out: Path) -> None:
    if not histories:
        return

    history_line_plot(
        histories, ("policy_return_mean",), ("policy return",),
        out / "reward_vs_training_percent.png",
    )
    history_line_plot(
        histories,
        ("global_stability_rho", "local_stability_rho", "global_stability_c", "local_stability_c"),
        ("global rho", "local rho", "global C", "local C"),
        out / "stability_vs_training_percent.png",
    )
    history_line_plot(
        histories, ("state_ood_ratio", "state_action_ood_ratio"),
        ("state OOD", "state-action OOD"), out / "ood_vs_training_percent.png",
    )


def history_line_plot(histories: list[dict], keys: tuple[str, ...], names: tuple[str, ...], path: Path) -> None:
    fig, axes = plt.subplots(len(keys), 1, figsize=(8, 3 * len(keys)), squeeze=False, sharex=True)
    for axis, key, name in zip(axes[:, 0], keys, names):
        for history in histories:
            records = history["records"]
            axis.plot(
                [record["actual_percent"] for record in records],
                [record[key] for record in records],
                marker="o",
                label=history["label"],
            )
        if key.endswith("_rho") or key.endswith("_ood_ratio"):
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
        ax.annotate(row["algo"], (row[x_key], row[y_key]), fontsize=8, xytext=(4, 4), textcoords="offset points")
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

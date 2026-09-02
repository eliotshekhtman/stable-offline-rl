# Tasks:
# - Load completed sweep runs from run manifests and evaluation result files.
# - Average matching results across the random seeds selected by the sweep.
# - Plot final-policy performance and contraction against action chunk length.
# - Plot task performance and contraction against generated noisy-trajectory fractions.
# - Plot task performance and contraction against clean-expert/Minari fractions.
# - Plot state and state-action conservativity over policy-training checkpoints.
# - Plot task performance over policy-training checkpoints.

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import task_support


EVALUATION_SCHEMA_VERSION = 2
PLOT_COHORT_VERSION = 1
BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_PERCENTILES = (10.0, 90.0)
BOOTSTRAP_SEED = 0


def main() -> None:
    args = parse_args()
    plot_root(args.root, args.out, cohort=args.cohort)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot stable-offline-rl sweep and evaluation results.")
    parser.add_argument("--root", type=Path, required=True, help="Evaluation environment directory, normally <storage-root>/evals/<environment>")
    parser.add_argument("--out", type=Path, default=None, help="Directory for saved plots; defaults to <root>/plots")
    parser.add_argument(
        "--cohort",
        type=Path,
        default=None,
        help=(
            "JSON cohort selecting one or more explicitly matched algorithm series; "
            "required when historical runs contain multiple configurations of the "
            "same algorithm at one plotted x-value"
        ),
    )
    args = parser.parse_args()
    if args.out is None:
        args.out = args.root / "plots"
    return args


def plot_root(
    root: Path,
    out: Path | None = None,
    eval_dirs: list[Path] | None = None,
    cohort: Path | dict | None = None,
) -> None:
    out = root / "plots" if out is None else out
    if eval_dirs is None:
        eval_dirs = latest_eval_dirs(root)
    rows = load_rows(eval_dirs)
    histories = load_histories(eval_dirs, rows)
    rows = average_seed_rows(rows)
    histories = average_seed_histories(histories)
    if cohort is not None:
        cohort_config = (
            load_plot_cohort(cohort) if isinstance(cohort, Path)
            else validate_plot_cohort(cohort)
        )
        rows = select_plot_cohort(rows, cohort_config)
        histories = select_cohort_histories(histories, rows)
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
        task_support.require_supported_task(
            row["env_name"], row["dataset_source"]
        )
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
        history["training_schema"] = row["training_schema"]
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


def load_plot_cohort(path: Path) -> dict:
    return validate_plot_cohort(load_json(path))


def validate_plot_cohort(cohort: dict) -> dict:
    if not isinstance(cohort, dict):
        raise ValueError("Plot cohort must be a JSON object")
    if cohort.get("version") != PLOT_COHORT_VERSION:
        raise ValueError(
            f"Plot cohort version must be {PLOT_COHORT_VERSION}"
        )
    if set(cohort) != {"version", "series"}:
        raise ValueError("Plot cohort supports only 'version' and 'series'")

    series = cohort["series"]
    if not isinstance(series, list) or not series:
        raise ValueError("Plot cohort 'series' must be a non-empty list")
    validated_series = []
    for index, item in enumerate(series):
        if not isinstance(item, dict):
            raise ValueError(f"Plot cohort series {index} must be an object")
        unknown = set(item) - {"algo", "match", "label"}
        if unknown:
            raise ValueError(
                f"Plot cohort series {index} has unsupported fields: "
                f"{sorted(unknown)}"
            )
        algo = item.get("algo")
        if not isinstance(algo, str) or not algo:
            raise ValueError(
                f"Plot cohort series {index} requires a non-empty 'algo'"
            )
        match = item.get("match", {})
        if not isinstance(match, dict):
            raise ValueError(
                f"Plot cohort series {index} 'match' must be an object"
            )
        for path in match:
            if (
                not isinstance(path, str)
                or not path
                or any(not component for component in path.split("."))
            ):
                raise ValueError(
                    f"Plot cohort series {index} has an invalid match path"
                )
            if "seed" in path.split("."):
                raise ValueError(
                    "Plot cohorts select seed-averaged configurations; "
                    f"series {index} cannot match '{path}'"
                )
        label = item.get("label")
        if label is not None and (not isinstance(label, str) or not label):
            raise ValueError(
                f"Plot cohort series {index} 'label' must be non-empty"
            )
        validated = {"algo": algo, "match": dict(match)}
        if label is not None:
            validated["label"] = label
        validated_series.append(validated)
    return {"version": PLOT_COHORT_VERSION, "series": validated_series}


def training_schema_value(record: dict, path: str):
    value = record["training_schema"]
    for component in path.split("."):
        if not isinstance(value, dict) or component not in value:
            return None, False
        value = value[component]
    return value, True


def cohort_series_matches(record: dict, series: dict) -> bool:
    if record["algo"] != series["algo"]:
        return False
    for path, expected in series["match"].items():
        actual, present = training_schema_value(record, path)
        if not present and expected is None:
            continue
        if not present or actual != expected:
            return False
    return True


def select_plot_cohort(rows: list[dict], cohort: dict) -> list[dict]:
    selected = []
    owners = {}
    for series_index, series in enumerate(cohort["series"]):
        matches = [
            (row_index, row) for row_index, row in enumerate(rows)
            if cohort_series_matches(row, series)
        ]
        if not matches:
            raise ValueError(
                f"Plot cohort series {series_index} ({series['algo']}) "
                "matched no seed-averaged configurations"
            )
        for row_index, row in matches:
            if row_index in owners:
                raise ValueError(
                    f"Plot cohort series {owners[row_index]} and {series_index} "
                    "select the same seed-averaged configuration"
                )
            owners[row_index] = series_index
            selected.append({
                **row,
                "_plot_cohort_series": series_index,
                "_plot_cohort_spec": series,
            })
    return selected


def select_cohort_histories(
    histories: list[dict], selected_rows: list[dict]
) -> list[dict]:
    selections = {
        row["seed_group"]: (
            row["_plot_cohort_series"], row["_plot_cohort_spec"]
        )
        for row in selected_rows
    }
    selected = []
    for history in histories:
        selection = selections.get(history["seed_group"])
        if selection is None:
            continue
        series_index, series = selection
        selected_history = {
            **history,
            "_plot_cohort_series": series_index,
            "_plot_cohort_spec": series,
        }
        selected_history["label"] = cohort_policy_label(
            selected_history, series
        )
        selected.append(selected_history)
    return selected


def dynamics_chunk_mode(record: dict) -> str | None:
    training_schema = record.get("training_schema", {})
    model_based = training_schema.get("model_based")
    if model_based is None:
        return None
    if record["chunk_length"] == 1:
        return "direct"
    return model_based.get("chunk_dynamics", {}).get("mode", "direct")


def algorithm_group_key(record: dict) -> tuple[str, str]:
    return record["algo"], dynamics_chunk_mode(record) or ""


def algorithm_label(record: dict) -> str:
    mode = dynamics_chunk_mode(record)
    if mode is not None and mode != "direct":
        return f"{record['algo']} ({mode} dynamics)"
    return record["algo"]


def parameter_value_label(path: str, value) -> str:
    name = {
        "model_based.real_ratio": "real ratio",
    }.get(path, path.replace("_", " "))
    if path == "model_based.real_ratio" and isinstance(value, (int, float)):
        rendered = f"{value:.2f}"
    elif isinstance(value, float):
        rendered = f"{value:g}"
    elif isinstance(value, (dict, list)):
        rendered = json.dumps(value, sort_keys=True, separators=(",", ":"))
    else:
        rendered = str(value)
    return f"{name}={rendered}"


def cohort_parameter_labels(series: dict) -> list[str]:
    return [
        parameter_value_label(path, value)
        for path, value in sorted(series["match"].items())
    ]


def cohort_policy_label(record: dict, series: dict) -> str:
    qualifiers = [f"l={record['chunk_length']}"]
    mode = dynamics_chunk_mode(record)
    if mode is not None and mode != "direct":
        qualifiers.append(f"{mode} dynamics")
    qualifiers.extend(cohort_parameter_labels(series))
    return f"{series.get('label', record['algo'])} ({', '.join(qualifiers)})"


def plot_series_key(record: dict) -> tuple[int, str, str]:
    series_index = record.get("_plot_cohort_series")
    if series_index is not None:
        return series_index, record["algo"], ""
    return -1, *algorithm_group_key(record)


AXIS_SCHEMA_PATHS = {
    "chunk_length": {
        ("chunk_length",),
        ("macro_discount",),
    },
    "noisy_trajectory_fraction": {
        ("dataset", "prop_clean_expert"),
        ("dataset", "prop_noisy_expert"),
        ("dataset", "prop_random"),
        ("dataset", "prop_expert"),
    },
    "minari_trajectory_fraction": {
        ("dataset", "source"),
        ("dataset", "dataset_id"),
        ("dataset", "minari_fraction"),
        ("dataset", "noise_scale"),
        ("dataset", "prop_clean_expert"),
        ("dataset", "prop_noisy_expert"),
        ("dataset", "prop_random"),
        ("dataset", "prop_expert"),
    },
}


def line_training_schema(record: dict, value_key: str) -> dict:
    excluded = AXIS_SCHEMA_PATHS.get(value_key, set())

    def keep_fields(value, path=()):
        if isinstance(value, dict):
            return {
                key: keep_fields(item, path + (key,))
                for key, item in value.items()
                if key != "seed" and path + (key,) not in excluded
            }
        if isinstance(value, list):
            return [keep_fields(item, path) for item in value]
        return value

    return keep_fields(record["training_schema"])


def schema_leaf_values(schema: dict, prefix=()) -> dict:
    values = {}
    for key, value in schema.items():
        path = prefix + (key,)
        if isinstance(value, dict):
            values.update(schema_leaf_values(value, path))
        else:
            values[".".join(path)] = json.dumps(value, sort_keys=True)
    return values


def inconsistent_schema_paths(records: list[dict], value_key: str) -> list[str]:
    flattened = [
        schema_leaf_values(line_training_schema(record, value_key))
        for record in records
    ]
    paths = set().union(*(values.keys() for values in flattened))
    return sorted(
        path for path in paths
        if len({values.get(path, "<missing>") for values in flattened}) > 1
    )


def seed_row_identity(row: dict):
    schema = row.get("training_schema", {})
    if "seed" in schema:
        return "seed", schema["seed"]
    if "seed" in row:
        return "seed", row["seed"]
    return "eval_dir", row.get("eval_dir", row.get("run_dir", id(row)))


def seed_row_recency(row: dict) -> tuple[str, str]:
    return row.get("created_at", ""), row.get("eval_dir", "")


def coalesce_clean_minari_baselines(rows: list[dict]) -> list[dict]:
    groups = {}
    for row in rows:
        schema = json.dumps(
            line_training_schema(row, "minari_trajectory_fraction"),
            sort_keys=True,
        )
        groups.setdefault((plot_series_key(row), schema), []).append(row)

    coalesced = []
    for group in groups.values():
        representative = max(group, key=seed_row_recency)
        by_seed = {}
        for row in group:
            for seed_row in row["seed_rows"]:
                identity = seed_row_identity(seed_row)
                previous = by_seed.get(identity)
                if (
                    previous is None
                    or seed_row_recency(seed_row) > seed_row_recency(previous)
                ):
                    by_seed[identity] = seed_row
        seed_rows = [by_seed[key] for key in sorted(by_seed, key=repr)]
        baseline = {
            **representative,
            "seed_rows": seed_rows,
            "minari_trajectory_fraction": 0.0,
        }
        for key in (
            "last_policy_performance_mean",
            "expert_performance_mean",
        ):
            if seed_rows and all(key in seed_row for seed_row in seed_rows):
                baseline[key] = float(np.mean([
                    seed_row[key] for seed_row in seed_rows
                ]))
        coalesced.append(baseline)
    return coalesced


def plot_series_label(record: dict) -> str:
    series = record.get("_plot_cohort_spec")
    if series is None:
        return algorithm_label(record)

    qualifiers = []
    mode = dynamics_chunk_mode(record)
    if mode is not None and mode != "direct":
        qualifiers.append(f"{mode} dynamics")
    qualifiers.extend(cohort_parameter_labels(series))
    label = series.get("label", record["algo"])
    if qualifiers:
        return f"{label} ({', '.join(qualifiers)})"
    return label


def algorithm_groups(
    records: list[dict], value_key: str | None = None
) -> list[tuple[str, list[dict]]]:
    groups = {}
    for record in records:
        groups.setdefault(plot_series_key(record), []).append(record)

    result = []
    for _, group in sorted(groups.items()):
        label = plot_series_label(group[0])
        if value_key is not None:
            by_value = {}
            for record in group:
                by_value.setdefault(record[value_key], []).append(record)
            duplicates = {
                value: matching for value, matching in by_value.items()
                if len(matching) > 1
            }
            if duplicates:
                values = ", ".join(f"{value:g}" for value in duplicates)
                raise ValueError(
                    f"Plot series '{label}' has multiple seed-averaged "
                    f"configurations at {value_key}={values}. Select each "
                    "intended variant as a separate --cohort series."
                )
            inconsistent = inconsistent_schema_paths(group, value_key)
            if inconsistent:
                paths = ", ".join(inconsistent)
                raise ValueError(
                    f"Plot series '{label}' changes non-axis training "
                    f"parameters across {value_key}: {paths}. Add these "
                    "paths to separate cohort series match objects."
                )
        result.append((label, group))
    return result


def policy_label(record: dict) -> str:
    mode = dynamics_chunk_mode(record)
    if mode is not None and mode != "direct":
        return f"{record['algo']} (l={record['chunk_length']}, {mode} dynamics)"
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
    for label, group_rows in algorithm_groups(rows, "chunk_length"):
        algo_rows = sorted(group_rows, key=lambda row: row["chunk_length"])
        x = [row["chunk_length"] for row in algo_rows]
        intervals = [bootstrap_mean(final_performance_samples(row)) for row in algo_rows]
        center, low, high = map(np.asarray, zip(*intervals))
        line, = ax.plot(x, center, marker="o", label=label)
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
        matching_clean = [
            row for row in clean
            if row["num_samples"] == num_samples
            and row["chunk_length"] == chunk_length
        ]
        plot_rows.extend(coalesce_clean_minari_baselines(matching_clean))
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
    for label, group_rows in algorithm_groups(rows, value_key):
        algo_rows = sorted(group_rows, key=lambda row: row[value_key])
        x = [row[value_key] for row in algo_rows]
        intervals = [bootstrap_mean(final_performance_samples(row)) for row in algo_rows]
        center, low, high = map(np.asarray, zip(*intervals))
        line, = ax.plot(x, center, marker="o", label=label)
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
    algorithms = algorithm_groups(available, value_key)
    fig, axes = plt.subplots(1, len(algorithms), figsize=(5 * len(algorithms), 4), squeeze=False)
    for axis, (label, group_rows) in zip(axes[0], algorithms):
        for row in sorted(group_rows, key=lambda row: row[value_key]):
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
        axis.set_title(label)
        axis.set_xlabel("primitive timestep")
        axis.legend(fontsize=8)
    axes[0, 0].set_ylabel("agent Cartesian-position distance (m)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


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


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


if __name__ == "__main__":
    main()

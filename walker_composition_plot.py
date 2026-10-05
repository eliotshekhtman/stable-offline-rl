"""Plot only the isolated reference-Walker composition protocol, without simulation.

Cohorts use the ordinary version-2 seeds/match/series structure, plus ``setting``
and the explicit ``other_fraction`` ablation. Filters are dotted manifest paths;
optional ``eval_match`` filters are dotted evaluation_config paths. Selection uses
requested transition fractions; x coordinates use actual trajectory fractions,
matching the ordinary composition plots. No existing plot schema is changed.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot import (
    BOOTSTRAP_PERCENTILES,
    BOOTSTRAP_REPLICATES,
    algorithm_name,
    bootstrap_mean,
    parameter_value_label,
    validate_cohort_match,
)
from plot_labels import mobile_names
from plot_rendering import X_LABEL_FONTSIZE, latex_plot, save_plot, set_composition_xticks, set_data_xticks, tex_text, validate_plot_options
from walker_composition_eval import identity, validate_evaluation


METRICS = {
    "performance": ("performance", "Forward displacement (m)"),
    "raw_return": ("returns", "Episode return"),
    "normalized_return": ("normalized_scores", "D4RL-normalized return"),
}


def read_json(path):
    return json.loads(Path(path).read_text())


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def matches(value, filters):
    for dotted, expected in filters.items():
        actual = value
        for key in dotted.split("."):
            if not isinstance(actual, dict) or key not in actual:
                return False
            actual = actual[key]
        if actual != expected:
            return False
    return True


def validate_cohort(cohort):
    required = {"version", "setting", "seeds", "ablation", "match", "series"}
    if not isinstance(cohort, dict) or required - cohort.keys() or cohort.keys() - required - {"eval_match", "figure_height", "show_legend"}:
        raise ValueError("Cohort requires version, setting, seeds, ablation, match, series; optional eval_match, figure_height, show_legend")
    validate_plot_options(cohort)
    if cohort["version"] != 2 or cohort["setting"] not in {"noise0.5", "noise1.0", "clean-medium-v2"}:
        raise ValueError("Expected version 2 and an explicit reference composition setting")
    seeds = cohort["seeds"]
    if not isinstance(seeds, list) or not seeds or any(type(seed) is not int for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be unique integers")
    ablation = cohort["ablation"]
    if not isinstance(ablation, dict) or set(ablation) != {"name", "values"} or ablation["name"] != "other_fraction":
        raise ValueError("Composition ablation must declare other_fraction and values")
    values = ablation["values"]
    if not isinstance(values, list) or not values or any(type(x) not in (int, float) or not np.isfinite(x) or not 0 <= x <= 1 for x in values) or len(set(values)) != len(values):
        raise ValueError("Composition values must be unique finite fractions in [0, 1]")
    validate_cohort_match(cohort["match"], "match")
    validate_cohort_match(cohort.get("eval_match", {}), "eval_match")
    series = cohort["series"]
    if not isinstance(series, list) or not series:
        raise ValueError("At least one algorithm series is required")
    for item in series:
        if not isinstance(item, dict) or set(item) - {"algo", "match", "label"} or not isinstance(item.get("algo"), str) or not item["algo"]:
            raise ValueError("Each series requires algo, with optional match and label")
        validate_cohort_match(item.get("match", {}), "series.match")
        if "label" in item and (not isinstance(item["label"], str) or not item["label"].strip()):
            raise ValueError("Series labels must be non-empty")
    return cohort


def resolve_path(value, relative_to):
    path = Path(value)
    return path if path.is_absolute() else relative_to / path


def evaluation_signature(config, include_checkpoints=True):
    """Retain evaluation semantics, excluding seed-specific policy/episode IDs."""
    common = {key: value for key, value in config.items() if key not in {"seed", "checkpoints"}}
    if "expert" in common:
        common["expert"] = {key: value for key, value in common["expert"].items() if key != "reset_seeds"}
    if include_checkpoints:
        common["checkpoints"] = [
            {key: value for key, value in checkpoint.items() if key not in {"policy_sha256", "reset_seeds", "policy_path"}}
            for checkpoint in config.get("checkpoints", [])
        ]
    return canonical(common)


def series_signature(manifest):
    dataset = {
        key: value for key, value in manifest["dataset_spec"].items()
        if key not in {"seed", "setting", "clean_fraction", "other_fraction"}
    }
    return canonical({"dataset": dataset, "training": manifest["training_config"]})


def metric_samples(record, field, context):
    samples = np.asarray(record.get(field, []), dtype=np.float64)
    if samples.ndim != 1 or not len(samples) or not np.isfinite(samples).all():
        raise ValueError(f"Missing or nonfinite {field} samples: {context}; evaluate with the new metric protocol")
    if record.get("episodes", len(samples)) != len(samples):
        raise ValueError(f"Episode count disagrees with {field} samples: {context}")
    return samples


def load_candidates(root, cohort):
    candidates = []
    for path in sorted(Path(root).rglob("evaluation.json")):
        evaluation = read_json(path)
        if evaluation.get("protocol") != "walker-composition-eval-v1":
            continue
        if evaluation.get("version") != 1:
            raise ValueError(f"Unsupported composition evaluation version: {path}")
        manifest_path = resolve_path(evaluation["run_manifest_path"], path.parent)
        manifest = read_json(manifest_path)
        if manifest.get("protocol") != "walker-composition-v1" or manifest.get("version") != 1:
            raise ValueError(f"Unsupported composition run protocol: {manifest_path}")
        if manifest.get("status") not in {"trained", "complete"}:
            continue
        if manifest["seed"] not in cohort["seeds"] or not matches(manifest, cohort["match"]):
            continue
        spec = manifest["dataset_spec"]
        fraction = spec["other_fraction"]
        if fraction not in cohort["ablation"]["values"]:
            continue
        if spec["setting"] != cohort["setting"] and not (fraction == 0 and spec["setting"] == "clean"):
            continue
        if not matches(evaluation["evaluation_config"], cohort.get("eval_match", {})):
            continue
        if spec["seed"] != manifest["seed"]:
            raise ValueError(f"Dataset seed and training seed differ: {manifest_path}")
        metadata_path = resolve_path(manifest["dataset_metadata_path"], manifest_path.parent)
        metadata = read_json(metadata_path)
        if any(metadata.get(key) != manifest.get(key) for key in ("dataset_id", "dataset_spec", "environment")):
            raise ValueError(f"Dataset metadata does not match the training manifest: {metadata_path}")
        actual = metadata["actual_other_episode_fraction"]
        if type(actual) not in (float, int) or not np.isfinite(actual) or not 0 <= actual <= 1:
            raise ValueError(f"Invalid actual trajectory fraction: {metadata_path}")
        environment = evaluation.get("environment", {})
        if not environment.get("model_xml_sha256"):
            raise ValueError(f"Missing reference physics identity: {path}")
        if manifest.get("environment") != environment:
            raise ValueError(f"Evaluation and training physics differ: {path}")
        config = evaluation["evaluation_config"]
        validate_evaluation(evaluation, manifest_path, config, identity(config))
        for checkpoint in config["checkpoints"]:
            episode_key = "final_eval_episodes" if checkpoint["final"] else "checkpoint_eval_episodes"
            if checkpoint["episodes"] != config[episode_key]:
                raise ValueError(f"Checkpoint episodes disagree with evaluation configuration: {path}")
        candidates.append({"path": str(path), "manifest": manifest, "evaluation": evaluation, "actual_fraction": actual})
    return candidates


def select_grid(candidates, cohort):
    """Resolve exactly one run and evaluation for every series/fraction/seed."""
    grids = []
    global_physics = set()
    global_evaluation = set()
    evaluation_seeds = set()
    evaluation_offsets = set()
    used_paths = set()
    for series in cohort["series"]:
        grid = {}
        configurations = set()
        series_evaluations = set()
        for fraction in cohort["ablation"]["values"]:
            for seed in cohort["seeds"]:
                selected = [
                    candidate for candidate in candidates
                    if candidate["manifest"]["algorithm"] == series["algo"]
                    and candidate["manifest"]["seed"] == seed
                    and candidate["manifest"]["dataset_spec"]["other_fraction"] == fraction
                    and matches(candidate["manifest"], series.get("match", {}))
                ]
                context = f"{series['algo']}, fraction={fraction}, seed={seed}"
                if not selected:
                    raise ValueError(f"Missing evaluated run: {context}")
                ids = {item["manifest"]["training_id"] for item in selected}
                if len(ids) != 1:
                    raise ValueError(f"Ambiguous training replicas/configurations: {context}; narrow cohort match")
                signatures = {
                    canonical({key: item["evaluation"].get(key) for key in ("evaluation_config", "records", "expert", "environment")})
                    for item in selected
                }
                if len(signatures) != 1:
                    raise ValueError(f"Ambiguous evaluations: {context}; narrow eval_match")
                item = sorted(selected, key=lambda item: item["path"])[-1]
                if item["path"] in used_paths:
                    raise ValueError(f"Overlapping cohort series select the same run: {context}")
                used_paths.add(item["path"])
                grid[fraction, seed] = item
                configurations.add(series_signature(item["manifest"]))
                global_physics.add(canonical(item["evaluation"]["environment"]))
                config = item["evaluation"]["evaluation_config"]
                series_evaluations.add(evaluation_signature(config))
                global_evaluation.add(evaluation_signature(config, include_checkpoints=False))
                evaluation_seed = item["evaluation"]["evaluation_config"]["seed"]
                evaluation_seeds.add(evaluation_seed)
                evaluation_offsets.add(evaluation_seed - seed)
        if len(configurations) != 1:
            raise ValueError(f"Mixed dataset protocols or training configurations in series {series['algo']}; narrow cohort match")
        if len(global_physics) != 1:
            raise ValueError("Mixed reference physics in plot")
        if len(series_evaluations) != 1:
            raise ValueError(f"Mixed evaluation configurations/checkpoint grids in series {series['algo']}; narrow eval_match")
        grids.append(grid)
    if len(global_evaluation) != 1:
        raise ValueError("Mixed evaluation configurations/checkpoint grids in plot; narrow eval_match")
    if len(evaluation_seeds) != 1 and len(evaluation_offsets) != 1:
        raise ValueError("Mixed evaluation seed conventions; expected a common seed or common offset from training seed")
    return grids


def series_labels(cohort, grids=None):
    if grids is None:
        ratios = [
            {**cohort["match"], **series.get("match", {})}.get("training_config.real_ratio")
            for series in cohort["series"]
        ]
    else:
        ratios = [next(iter(grid.values()))["manifest"]["training_config"].get("real_ratio") for grid in grids]
    names = mobile_names([series["algo"] for series in cohort["series"]], ratios)
    labels = []
    for series, name in zip(cohort["series"], names):
        label = algorithm_name({"algo": series["algo"]}, series) if "label" in series or name is None else name
        peers = [item for item in cohort["series"] if item["algo"] == series["algo"]]
        if "label" not in series and len(peers) > 1:
            varying = {
                key for item in peers for key in item.get("match", {})
                if len({canonical(peer.get("match", {}).get(key)) for peer in peers}) > 1
            }
            if name is not None:
                varying.discard("training_config.real_ratio")
            details = [parameter_value_label(key.removeprefix("training_config."), value) for key, value in sorted(series.get("match", {}).items()) if key in varying]
            if details:
                label += " (" + ", ".join(details) + ")"
        labels.append(label)
    if len(set(labels)) != len(labels):
        raise ValueError("Series labels are ambiguous; provide explicit labels")
    return labels


def summarize(root, cohort, metric="performance"):
    cohort = validate_cohort(cohort)
    field, ylabel = METRICS[metric]
    grids = select_grid(load_candidates(root, cohort), cohort)
    result = {
        "protocol": "walker-composition-plot-v1", "cohort": cohort, "metric": metric,
        "performance_label": ylabel, "x_semantics": "Mean actual second-source trajectory fraction across seeds",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES, "bootstrap_percentiles": list(BOOTSTRAP_PERCENTILES),
        "series": [],
    }
    for label, grid in zip(series_labels(cohort, grids), grids):
        points = []
        for fraction in sorted(cohort["ablation"]["values"]):
            items = [grid[fraction, seed] for seed in cohort["seeds"]]
            records_by_seed = []
            for item in items:
                records = item["evaluation"]["records"]
                final = [record for record in records if record.get("final")]
                if len(final) != 1 or final[0]["epoch"] != item["manifest"]["training_config"]["epoch"]:
                    raise ValueError(f"Expected one final evaluation at the configured final epoch: {item['path']}")
                record_map = {(record["epoch"], record["step"], bool(record.get("final"))): record for record in records}
                if len(record_map) != len(records):
                    raise ValueError(f"Duplicate evaluated checkpoints: {item['path']}")
                records_by_seed.append(record_map)
            if any(set(records) != set(records_by_seed[0]) for records in records_by_seed[1:]):
                raise ValueError(f"Missing checkpoint evaluations between seeds at fraction {fraction}")
            history = []
            for key in sorted(records_by_seed[0]):
                mean, low, high = bootstrap_mean([metric_samples(records[key], field, f"{label}/{fraction}/{seed}/{key}") for seed, records in zip(cohort["seeds"], records_by_seed)])
                history.append({"epoch": key[0], "step": key[1], "final": key[2], "mean": mean, "low": low, "high": high})
            final_summary = next(record for record in history if record["final"])
            expert = [metric_samples(item["evaluation"]["expert"], field, item["path"] + "/expert") for item in items]
            points.append({
                "requested_fraction": fraction, "actual_fraction": float(np.mean([item["actual_fraction"] for item in items])),
                "mean": final_summary["mean"], "low": final_summary["low"], "high": final_summary["high"],
                "expert_mean": float(np.mean([samples.mean() for samples in expert])), "history": history,
                "seeds": list(cohort["seeds"]), "evaluations": [item["path"] for item in items],
            })
        result["series"].append({"label": label, "points": points})
    return result


@latex_plot
def render(summary, out):
    options = validate_plot_options(summary["cohort"])
    height = options.get("figure_height", 5)
    show_legend = options.get("show_legend", True)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    setting = summary["cohort"]["setting"]
    xlabel = "Fraction of trajectories from D4RL medium-v2" if setting == "clean-medium-v2" else "Fraction of data from the noisy expert"
    fig, ax = plt.subplots(figsize=(8, height))
    for series in summary["series"]:
        points = series["points"]
        x = [point["actual_fraction"] for point in points]
        line, = ax.plot(x, [point["mean"] for point in points], marker="o", label=tex_text(series["label"]))
        ax.fill_between(x, [point["low"] for point in points], [point["high"] for point in points], color=line.get_color(), alpha=0.2)
    expert = float(np.mean([point["expert_mean"] for series in summary["series"] for point in series["points"]]))
    ax.axhline(expert, color="black", linestyle=":", label="Expert")
    ax.set(xlim=(-0.02, 1.02), ylabel=tex_text(summary["performance_label"]))
    ax.set_xlabel(tex_text(xlabel), fontsize=X_LABEL_FONTSIZE)
    set_composition_xticks(ax)
    if show_legend:
        ax.legend(loc="lower right")
    fig.tight_layout()
    save_plot(fig, out / "performance_vs_composition.png")
    plt.close(fig)
    for index, fraction in enumerate(sorted(summary["cohort"]["ablation"]["values"])):
        fig, ax = plt.subplots(figsize=(8, height))
        tick_values = []
        for series in summary["series"]:
            history = series["points"][index]["history"]
            # A final checkpoint can have separate diagnostic and final evaluations.
            # Retain the final evaluation once in the plotted trajectory.
            unique = {record["epoch"]: record for record in history}
            records = [unique[epoch] for epoch in sorted(unique)]
            x = [record["epoch"] for record in records]
            tick_values.extend(x)
            line, = ax.plot(x, [record["mean"] for record in records], marker="o", label=tex_text(series["label"]))
            ax.fill_between(x, [record["low"] for record in records], [record["high"] for record in records], color=line.get_color(), alpha=0.2)
        ax.axhline(expert, color="black", linestyle=":", label="Expert")
        ax.set_ylabel(tex_text(summary["performance_label"]))
        ax.set_xlabel("Policy training epoch", fontsize=X_LABEL_FONTSIZE)
        if show_legend:
            ax.legend(loc="lower right")
        fig.tight_layout()
        set_data_xticks(ax, tick_values)
        fig.tight_layout()
        save_plot(fig, out / f"performance_history_fraction_{fraction:g}.png")
        plt.close(fig)
    (out / "plot_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Composition evaluation directory")
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--metric", choices=tuple(METRICS), default="performance")
    args = parser.parse_args()
    output = args.out or args.root / "plots" / args.cohort.stem / args.metric
    summary = summarize(args.root, read_json(args.cohort), args.metric)
    render(summary, output)
    print(f"Saved reference composition plots and statistical summary to {output}")


if __name__ == "__main__":
    main()

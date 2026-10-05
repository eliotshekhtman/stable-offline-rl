"""Reference composition selection/statistics tests; no training or MuJoCo."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import matplotlib as mpl
from matplotlib.legend import Legend
import numpy as np

from plot import bootstrap_mean
import walker_composition_plot as plotting


class CompositionPlotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cohort = {
            "version": 2, "setting": "noise0.5", "seeds": [10000, 10100],
            "ablation": {"name": "other_fraction", "values": [0, 0.5]},
            "match": {"dataset_spec.test_fraction": 0.2},
            "series": [{"algo": "mobile", "match": {"training_config.real_ratio": 0.05}}],
        }
        self.paths = {}
        for fraction in [0, 0.5]:
            for seed in [10000, 10100]:
                self.paths[fraction, seed] = self.write_run(fraction, seed)

    def write_run(self, fraction, seed, suffix="", samples=None):
        run_id = f"mobile-{fraction}-{seed}{suffix}"
        path = self.root / run_id
        path.mkdir()
        samples = samples if samples is not None else ([1, 3, 1, 3] if seed == 10000 else [5, 7, 9, 11])
        environment = {"model_xml_sha256": "physics-a", "timestep": 0.008}
        manifest = {
            "protocol": "walker-composition-v1", "version": 1, "status": "complete",
            "algorithm": "mobile", "seed": seed, "training_id": run_id,
            "dataset_id": f"distinct-data-{seed}-{fraction}",
            "dataset_metadata_path": str(path / "metadata.json"),
            "environment": environment,
            "dataset_spec": {
                "protocol": "walker-composition-v1", "setting": "clean" if fraction == 0 else "noise0.5",
                "seed": seed, "clean_fraction": 1 - fraction, "other_fraction": fraction,
                "num_samples": 999995, "test_fraction": 0.2,
            },
            "training_config": {"epoch": 3000, "real_ratio": 0.05},
        }
        self.write(path / "run_manifest.json", manifest)
        self.write(path / "metadata.json", {
            "actual_other_episode_fraction": fraction + (0.1 if fraction and seed == 10100 else 0),
            **{key: manifest[key] for key in ("dataset_id", "dataset_spec", "environment")},
        })
        def record(epoch, final):
            return {
                "epoch": epoch, "step": epoch * 1000, "final": final, "episodes": len(samples),
                "performance": samples, "returns": [10 * x for x in samples],
                "normalized_scores": [2 * x for x in samples], "lengths": [1000] * len(samples),
                "policy_sha256": f"policy-{seed}-{epoch}-{fraction}", "reset_seeds": list(range(seed, seed + len(samples))),
            }
        records = [record(300, False), record(3000, True)]
        evaluation = {
            "protocol": "walker-composition-eval-v1", "version": 1,
            "run_manifest_path": str(path / "run_manifest.json"), "environment": environment,
            "evaluation_config": {
                "protocol": "walker-composition-eval-v1", "metric_version": 1,
                "seed": seed, "checkpoint_eval_episodes": len(samples), "final_eval_episodes": len(samples),
                "environment_sha256": plotting.identity(environment),
                "expert": {"file_sha256": "expert-file", "actor_sha256": "expert-actor", "episodes": 2,
                           "reset_seeds": [seed + 1000000, seed + 1000001]},
                "checkpoints": [{key: value for key, value in item.items() if key in {"epoch", "step", "final", "policy_sha256", "reset_seeds", "episodes"}} for item in records],
            },
            "records": records, "expert": {"performance": [15, 15], "returns": [150, 150], "normalized_scores": [30, 30]},
        }
        evaluation["expert"].update(evaluation["evaluation_config"]["expert"])
        evaluation["expert"]["lengths"] = [1000, 1000]
        evaluation["config_id"] = plotting.identity(evaluation["evaluation_config"])
        self.write(path / "evaluation.json", evaluation)
        return path

    @staticmethod
    def write(path, obj):
        path.write_text(json.dumps(obj))

    def edit(self, path, callback):
        obj = plotting.read_json(path)
        callback(obj)
        if path.name == "evaluation.json":
            obj["config_id"] = plotting.identity(obj["evaluation_config"])
        self.write(path, obj)

    def summarize(self, metric="performance"):
        return plotting.summarize(self.root, self.cohort, metric)

    def test_statistics_exactly_reuse_seed_balanced_bootstrap(self):
        result = self.summarize()
        expected = bootstrap_mean([np.array([1, 3, 1, 3]), np.array([5, 7, 9, 11])])
        point = result["series"][0]["points"][0]
        self.assertEqual((point["mean"], point["low"], point["high"]), expected)
        self.assertEqual(point["mean"], 5)
        self.assertIs(plotting.bootstrap_mean, bootstrap_mean)
        self.assertEqual(plotting.bootstrap_mean([np.array([1, 3]), np.array([5, 7, 9, 11])])[0], 5)
        self.assertNotEqual(5, np.mean([1, 3, 5, 7, 9, 11]))
        self.assertEqual(result["bootstrap_replicates"], 10000)
        self.assertEqual(result["bootstrap_percentiles"], [10.0, 90.0])

    def test_uses_actual_episode_fraction_and_requested_selection(self):
        self.assertAlmostEqual(self.summarize()["series"][0]["points"][1]["actual_fraction"], 0.55)

    def test_distinct_dataset_hashes_do_not_split_seed_group(self):
        self.assertEqual(len(self.summarize()["series"][0]["points"][0]["seeds"]), 2)

    def test_clean_endpoint_shared_between_families(self):
        self.cohort["setting"] = "clean-medium-v2"
        self.cohort["ablation"]["values"] = [0]
        self.assertEqual(self.summarize()["series"][0]["points"][0]["actual_fraction"], 0)

    def test_wrong_family_is_not_accidentally_selected(self):
        self.cohort["setting"] = "clean-medium-v2"
        with self.assertRaisesRegex(ValueError, "Missing evaluated run"):
            self.summarize()

    def test_missing_seed_fails(self):
        (self.paths[0.5, 10100] / "evaluation.json").unlink()
        with self.assertRaisesRegex(ValueError, "Missing evaluated run"):
            self.summarize()

    def test_lower_seed_is_not_loaded(self):
        self.write_run(0.5, 5, samples=[10000])
        self.assertEqual(self.summarize()["series"][0]["points"][1]["mean"], 5)

    def test_duplicate_training_replica_fails(self):
        self.write_run(0.5, 10000, suffix="-duplicate")
        with self.assertRaisesRegex(ValueError, "Ambiguous training"):
            self.summarize()

    def test_identical_evaluation_copy_does_not_double_count(self):
        duplicate = self.root / "copy"
        duplicate.mkdir()
        self.write(duplicate / "evaluation.json", plotting.read_json(self.paths[0, 10000] / "evaluation.json"))
        self.assertEqual(self.summarize()["series"][0]["points"][0]["mean"], 5)

    def test_different_eval_configs_require_filter(self):
        duplicate = self.root / "copy"
        duplicate.mkdir()
        evaluation = plotting.read_json(self.paths[0, 10000] / "evaluation.json")
        evaluation["evaluation_config"]["checkpoint_eval_episodes"] = 3
        evaluation["evaluation_config"]["checkpoints"][0]["episodes"] = 3
        evaluation["evaluation_config"]["checkpoints"][0]["reset_seeds"] = evaluation["evaluation_config"]["checkpoints"][0]["reset_seeds"][:3]
        checkpoint = evaluation["records"][0]
        checkpoint["episodes"] = 3
        for key in ("performance", "returns", "normalized_scores", "lengths", "reset_seeds"):
            checkpoint[key] = checkpoint[key][:3]
        evaluation["config_id"] = plotting.identity(evaluation["evaluation_config"])
        self.write(duplicate / "evaluation.json", evaluation)
        with self.assertRaisesRegex(ValueError, "Ambiguous evaluations"):
            self.summarize()
        self.cohort["eval_match"] = {"checkpoint_eval_episodes": 4}
        self.assertEqual(self.summarize()["series"][0]["points"][0]["mean"], 5)

    def test_changed_checkpoint_hash_is_not_duplicate(self):
        duplicate = self.root / "copy"
        duplicate.mkdir()
        evaluation = plotting.read_json(self.paths[0, 10000] / "evaluation.json")
        evaluation["records"][-1]["policy_sha256"] = "different-policy"
        evaluation["evaluation_config"]["checkpoints"][-1]["policy_sha256"] = "different-policy"
        evaluation["config_id"] = plotting.identity(evaluation["evaluation_config"])
        self.write(duplicate / "evaluation.json", evaluation)
        with self.assertRaisesRegex(ValueError, "Ambiguous evaluations"):
            self.summarize()

    def test_mixed_epochs_rejected(self):
        self.edit(self.paths[0.5, 10100] / "run_manifest.json", lambda obj: obj["training_config"].update(epoch=1000))
        with self.assertRaisesRegex(ValueError, "Mixed dataset protocols or training"):
            self.summarize()

    def test_unmatched_real_ratio_is_missing(self):
        self.edit(self.paths[0.5, 10100] / "run_manifest.json", lambda obj: obj["training_config"].update(real_ratio=0))
        with self.assertRaisesRegex(ValueError, "Missing evaluated run"):
            self.summarize()

    def test_different_algorithms_can_use_different_training_epochs(self):
        self.cohort["series"].append({"algo": "mopo"})
        for fraction in [0, 0.5]:
            for seed in [10000, 10100]:
                path = self.write_run(fraction, seed, suffix="-mopo")
                def change_manifest(obj):
                    obj["algorithm"] = "mopo"
                    obj["training_config"]["epoch"] = 1000
                self.edit(path / "run_manifest.json", change_manifest)
                def change_evaluation(obj):
                    for records in (obj["records"], obj["evaluation_config"]["checkpoints"]):
                        for record in records:
                            record["epoch"] //= 3
                            record["step"] = record["epoch"] * 1000
                self.edit(path / "evaluation.json", change_evaluation)
        result = self.summarize()
        self.assertEqual([series["label"] for series in result["series"]], ["MOBILE", "MOPO"])
        self.assertEqual(result["series"][1]["points"][0]["history"][-1]["epoch"], 1000)

    def test_mixed_unfiltered_real_ratios_rejected(self):
        self.cohort["series"][0]["match"] = {}
        self.edit(self.paths[0.5, 10100] / "run_manifest.json", lambda obj: obj["training_config"].update(real_ratio=0))
        with self.assertRaisesRegex(ValueError, "Mixed dataset protocols or training"):
            self.summarize()

    def test_mixed_physics_rejected(self):
        for name in ["run_manifest.json", "evaluation.json", "metadata.json"]:
            self.edit(self.paths[0.5, 10100] / name, lambda obj: obj["environment"].update(model_xml_sha256="physics-b"))
        self.edit(self.paths[0.5, 10100] / "evaluation.json", lambda obj: obj["evaluation_config"].update(environment_sha256=plotting.identity(obj["environment"])))
        with self.assertRaisesRegex(ValueError, "Mixed reference physics"):
            self.summarize()

    def test_eval_training_physics_mismatch_rejected(self):
        self.edit(self.paths[0.5, 10100] / "evaluation.json", lambda obj: obj["environment"].update(timestep=0.01))
        with self.assertRaisesRegex(ValueError, "Evaluation and training physics differ"):
            self.summarize()

    def test_missing_displacement_requires_reevaluation(self):
        self.edit(self.paths[0.5, 10100] / "evaluation.json", lambda obj: obj["records"][-1].pop("performance"))
        with self.assertRaisesRegex(ValueError, "Invalid cached evaluation array: performance"):
            self.summarize()

    def test_raw_and_normalized_metrics_are_explicit(self):
        self.assertEqual(self.summarize("raw_return")["series"][0]["points"][0]["mean"], 50)
        self.assertEqual(self.summarize("normalized_return")["series"][0]["points"][0]["mean"], 10)

    def test_missing_history_checkpoint_rejected(self):
        self.edit(self.paths[0.5, 10100] / "evaluation.json", lambda obj: obj["records"].pop(0))
        with self.assertRaisesRegex(ValueError, "missing checkpoint records"):
            self.summarize()

    def test_mixed_evaluation_seed_conventions_rejected(self):
        self.edit(self.paths[0.5, 10100] / "evaluation.json", lambda obj: obj["evaluation_config"].update(seed=303))
        with self.assertRaisesRegex(ValueError, "Mixed evaluation seed conventions"):
            self.summarize()

    def test_mixed_expert_policy_rejected(self):
        def different_expert(obj):
            obj["evaluation_config"]["expert"]["actor_sha256"] = "different-expert"
            obj["expert"]["actor_sha256"] = "different-expert"
        self.edit(self.paths[0.5, 10100] / "evaluation.json", different_expert)
        with self.assertRaisesRegex(ValueError, "Mixed evaluation configurations"):
            self.summarize()

    def test_metadata_must_match_dataset_spec_identity_and_physics(self):
        path = self.paths[0.5, 10100] / "metadata.json"
        original = plotting.read_json(path)
        for key in ("dataset_spec", "dataset_id", "environment"):
            modified = copy.deepcopy(original)
            modified[key] = "wrong"
            self.write(path, modified)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "Dataset metadata does not match"):
                self.summarize()
        self.write(path, original)

    def test_records_must_match_declared_episode_count(self):
        self.edit(self.paths[0.5, 10100] / "evaluation.json", lambda obj: obj["evaluation_config"]["checkpoints"][-1].update(episodes=5))
        with self.assertRaisesRegex(ValueError, "identity mismatch: episodes"):
            self.summarize()

    def test_checkpoint_counts_must_match_overall_eval_config(self):
        self.edit(self.paths[0.5, 10100] / "evaluation.json", lambda obj: obj["evaluation_config"].update(final_eval_episodes=100))
        with self.assertRaisesRegex(ValueError, "Checkpoint episodes disagree"):
            self.summarize()

    def test_corrupt_evaluation_config_hash_rejected(self):
        path = self.paths[0.5, 10100] / "evaluation.json"
        report = plotting.read_json(path)
        report["config_id"] = "wrong"
        self.write(path, report)
        with self.assertRaisesRegex(ValueError, "provenance does not match"):
            self.summarize()

    def test_invalid_seed_and_fraction_inputs_rejected(self):
        for key, value in [("seeds", [10000, 10000]), ("seeds", [True]), ("ablation", {"name": "other_fraction", "values": [float("nan")]})]:
            bad = copy.deepcopy(self.cohort)
            bad[key] = value
            with self.assertRaises(ValueError):
                plotting.validate_cohort(bad)

    def test_plot_options_preserve_cohort_and_statistics(self):
        baseline = copy.deepcopy(self.summarize())
        for options in ({}, {"figure_height": 4, "show_legend": True}, {"figure_height": 2.5, "show_legend": False}):
            with self.subTest(options=options):
                cohort = {**self.cohort, **options}
                original = copy.deepcopy(cohort)
                summary = plotting.summarize(self.root, cohort)
                self.assertEqual(cohort, original)
                self.assertEqual(summary["cohort"], original)
                self.assertEqual(
                    {key: value for key, value in summary.items() if key != "cohort"},
                    {key: value for key, value in baseline.items() if key != "cohort"},
                )

    def test_invalid_plot_options_rejected_before_output_creation(self):
        summary = self.summarize()
        invalid = {
            "figure_height": [0, -1, float("nan"), float("inf"), -float("inf"), True, False, "4", None, [], {}],
            "show_legend": [0, 1, "false", "true", None, [], {}],
        }
        out = self.root / "invalid-options"
        for key, values in invalid.items():
            for value in values:
                with self.subTest(key=key, value=value):
                    cohort = {**self.cohort, key: value}
                    with self.assertRaisesRegex(ValueError, key):
                        plotting.validate_cohort(cohort)
                    with self.assertRaisesRegex(ValueError, key):
                        plotting.render.__wrapped__({**summary, "cohort": cohort}, out)
                    self.assertFalse(out.exists())

    def test_constant_parameters_hidden_and_algorithm_uppercase(self):
        self.assertEqual(plotting.series_labels(self.cohort), ["MOBILE"])

    def test_variant_parameter_labels_shown(self):
        self.cohort["series"].append({"algo": "mobile", "match": {"training_config.real_ratio": 0}})
        self.assertEqual(plotting.series_labels(self.cohort), ["real ratio=0.05", "real ratio=0.00"])

    def test_mobile_labels_follow_plot_context(self):
        cases = [
            ([('mobile', 0), ('mobile', .25), ('mobile', .5)],
             ["real ratio=0.00", "real ratio=0.25", "real ratio=0.50"]),
            ([('mobile', 0), ('mobile', .5), ('iql', None)],
             ["MB-MOBILE", "Hybrid-MOBILE", "IQL"]),
            ([('mobile', 0), ('mobile', .25), ('mobile', .5), ('mopo', None)],
             ["MB-MOBILE", "Hybrid-MOBILE (real ratio=0.25)", "Hybrid-MOBILE (real ratio=0.50)", "MOPO"]),
            ([('mobile', 0)], ["MOBILE"]),
            ([('mobile', .5), ('iql', None)], ["MOBILE", "IQL"]),
            ([('mobile', None), ('iql', None)], ["MOBILE", "IQL"]),
        ]
        for variants, expected in cases:
            with self.subTest(variants=variants):
                self.cohort["series"] = [
                    {"algo": algo, "match": {"training_config.real_ratio": ratio} if ratio is not None else {}}
                    for algo, ratio in variants
                ]
                self.assertEqual(plotting.series_labels(self.cohort), expected)

    def test_mobile_labels_include_common_ratio_filters(self):
        self.cohort["match"]["training_config.real_ratio"] = 0
        self.cohort["series"] = [{"algo": "mobile"}, {"algo": "mobile", "match": {"training_config.real_ratio": .5}}]
        self.assertEqual(plotting.series_labels(self.cohort), ["real ratio=0.00", "real ratio=0.50"])

    def test_custom_mobile_labels_and_other_qualifiers_are_preserved(self):
        self.cohort["series"] = [
            {"algo": "mobile", "label": "My baseline", "match": {"training_config.real_ratio": 0, "training_config.epoch": 1000}},
            {"algo": "mobile", "match": {"training_config.real_ratio": .5, "training_config.epoch": 3000}},
            {"algo": "iql"},
        ]
        self.assertEqual(plotting.series_labels(self.cohort), ["My baseline", "Hybrid-MOBILE (epoch=3000)", "IQL"])

    def test_ambiguous_mobile_labels_still_require_explicit_labels(self):
        self.cohort["series"] = [{"algo": "mobile"}, {"algo": "mobile"}]
        with self.assertRaisesRegex(ValueError, "Series labels are ambiguous"):
            plotting.series_labels(self.cohort)

    def test_shipped_cohorts_validate(self):
        folder = Path(__file__).parent / "walker_composition_cohorts"
        for path in folder.glob("*.json"):
            cohort = plotting.validate_cohort(plotting.read_json(path))
            self.assertEqual(cohort["seeds"], [10000, 10100, 10200, 10300])
            self.assertEqual(cohort["match"]["dataset_spec.num_samples"], 999995)

    def test_render_smoke(self):
        subplots = plotting.plt.subplots
        for metric in plotting.METRICS:
            summary = copy.deepcopy(self.summarize(metric))
            for setting in ("noise0.5", "noise1.0", "clean-medium-v2"):
                with self.subTest(metric=metric, setting=setting):
                    summary["cohort"]["setting"] = setting
                    original = copy.deepcopy(summary)
                    figures = []

                    def capture(*args, **kwargs):
                        fig, ax = subplots(*args, **kwargs)
                        figures.append((fig, ax))
                        return fig, ax

                    out = self.root / "plots" / setting / metric
                    with mpl.rc_context({"legend.loc": "upper left"}), patch.object(plotting.plt, "subplots", side_effect=capture):
                        plotting.render(summary, out)
                    stems = {
                        "performance_vs_composition",
                        "performance_history_fraction_0",
                        "performance_history_fraction_0.5",
                    }
                    for suffix, signature in ((".png", b"\x89PNG\r\n\x1a\n"), (".pdf", b"%PDF-")):
                        self.assertEqual({path.name for path in out.glob(f"*{suffix}")}, {
                            f"{stem}{suffix}" for stem in stems
                        })
                        for stem in stems:
                            self.assertTrue((out / f"{stem}{suffix}").read_bytes().startswith(signature))
                    self.assertEqual(plotting.read_json(out / "plot_summary.json"), original)
                    self.assertEqual(summary, original)
                    self.assertEqual(len(figures), 3)
                    points = summary["series"][0]["points"]
                    for index, (fig, ax) in enumerate(figures):
                        np.testing.assert_allclose(fig.get_size_inches(), [8, 5])
                        self.assertIsNone(fig._suptitle)
                        for location in ("left", "center", "right"):
                            self.assertEqual(ax.get_title(loc=location), "")
                        xlabel = "Policy training epoch"
                        if index == 0:
                            xlabel = "Fraction of trajectories from D4RL medium-v2" if setting == "clean-medium-v2" else "Fraction of data from the noisy expert"
                            records, xkey = points, "actual_fraction"
                            np.testing.assert_allclose(ax.get_xlim(), [-0.02, 1.02])
                            np.testing.assert_allclose(ax.get_xticks(), [0, 0.25, 0.5, 0.75, 1])
                        else:
                            records, xkey = points[index - 1]["history"], "epoch"
                            np.testing.assert_allclose(ax.get_xticks(), [record["epoch"] for record in records])
                        self.assertEqual(ax.get_xlabel(), xlabel)
                        self.assertEqual(ax.get_ylabel(), summary["performance_label"])
                        self.assertEqual([text.get_text() for text in ax.get_legend().get_texts()], ["MOBILE", "Expert"])
                        self.assertEqual(ax.get_legend()._loc, Legend.codes["lower right"])
                        texts = [ax.xaxis.label, ax.yaxis.label, *ax.get_xticklabels(), *ax.get_yticklabels(), *ax.get_legend().get_texts()]
                        self.assertAlmostEqual(ax.xaxis.label.get_fontsize(), 24.3)
                        self.assertTrue(all(text.get_fontsize() == 18 for text in texts[1:]))
                        self.assertTrue(all(text.get_fontfamily() == ["monospace"] for text in texts))
                        self.assertTrue(all(text.get_usetex() for text in texts))
                        for axis in (ax.xaxis, ax.yaxis):
                            self.assertTrue(all(tick.gridline.get_visible() for tick in axis.get_major_ticks()))
                            self.assertLess(axis.get_zorder(), min(artist.get_zorder() for artist in [*ax.lines, *ax.collections]))
                        self.assertEqual(len(ax.lines), 2)
                        np.testing.assert_allclose(ax.lines[0].get_xdata(), [record[xkey] for record in records])
                        np.testing.assert_allclose(ax.lines[0].get_ydata(), [record["mean"] for record in records])
                        self.assertEqual(ax.lines[0].get_marker(), "o")
                        np.testing.assert_allclose(ax.lines[1].get_ydata(), points[0]["expert_mean"])
                        self.assertEqual(ax.lines[1].get_linestyle(), ":")
                        self.assertEqual(ax.lines[1].get_color(), "black")
                        self.assertEqual(len(ax.collections), 1)
                        self.assertEqual(ax.collections[0].get_alpha(), 0.2)
                        vertices = ax.collections[0].get_paths()[0].vertices
                        for record in records:
                            bounds = vertices[vertices[:, 0] == record[xkey], 1]
                            np.testing.assert_allclose([bounds.min(), bounds.max()], [record["low"], record["high"]])

    def test_render_custom_height_and_legend_visibility(self):
        subplots = plotting.plt.subplots
        for height, show_legend in ((2.5, False), (4, True), (3.5, True), (4.25, True)):
            with self.subTest(height=height, show_legend=show_legend):
                self.cohort.update(figure_height=height, show_legend=show_legend)
                summary = self.summarize()
                original = copy.deepcopy(summary)
                figures = []

                def capture(*args, **kwargs):
                    fig, ax = subplots(*args, **kwargs)
                    figures.append((fig, ax))
                    if not show_legend:
                        ax.legend = Mock(side_effect=AssertionError("Hidden legends must not be created"))
                    return fig, ax

                out = self.root / f"plot-options-{show_legend}"
                with patch.object(plotting.plt, "subplots", side_effect=capture):
                    plotting.render(summary, out)
                self.assertEqual(summary, original)
                self.assertEqual(plotting.read_json(out / "plot_summary.json"), original)
                self.assertEqual(len(figures), 3)
                for suffix, signature in ((".png", b"\x89PNG\r\n\x1a\n"), (".pdf", b"%PDF-")):
                    paths = list(out.glob(f"*{suffix}"))
                    self.assertEqual(len(paths), 3)
                    self.assertTrue(all(path.read_bytes().startswith(signature) for path in paths))
                for index, (fig, ax) in enumerate(figures):
                    np.testing.assert_allclose(fig.get_size_inches(), [8, height])
                    self.assertAlmostEqual(ax.xaxis.label.get_fontsize(), 24.3)
                    self.assertEqual(ax.yaxis.label.get_fontsize(), 18)
                    self.assertTrue(all(line.get_linewidth() == 1.5 for line in ax.lines))
                    self.assertEqual(ax.lines[0].get_marker(), "o")
                    self.assertEqual(ax.collections[0].get_alpha(), 0.2)
                    if show_legend:
                        self.assertEqual([text.get_text() for text in ax.get_legend().get_texts()], ["MOBILE", "Expert"])
                        self.assertEqual(ax.get_legend()._loc, Legend.codes["lower right"])
                    else:
                        self.assertIsNone(ax.get_legend())
                        self.assertFalse(fig.findobj(Legend))
                    points = original["series"][0]["points"]
                    records = points if index == 0 else points[index - 1]["history"]
                    xkey = "actual_fraction" if index == 0 else "epoch"
                    self.assertEqual([line.get_label() for line in ax.lines], ["MOBILE", "Expert"])
                    np.testing.assert_allclose(ax.lines[0].get_xdata(), [record[xkey] for record in records])
                    np.testing.assert_allclose(ax.lines[0].get_ydata(), [record["mean"] for record in records])
                    self.assertEqual(len(ax.collections), 1)
                    vertices = ax.collections[0].get_paths()[0].vertices
                    for record in records:
                        bounds = vertices[vertices[:, 0] == record[xkey], 1]
                        np.testing.assert_allclose([bounds.min(), bounds.max()], [record["low"], record["high"]])

    @plotting.latex_plot
    def test_render_preserves_distinct_series_coordinates_and_history_epochs(self):
        summary = self.summarize()
        second = copy.deepcopy(summary["series"][0])
        second["label"] = "MOPO"
        for point, actual_fraction in zip(second["points"], [0.04, 0.62]):
            point["actual_fraction"] = actual_fraction
            for record in point["history"]:
                record["epoch"] //= 3
                record["step"] = record["epoch"] * 1000
        summary["series"].append(second)
        original = copy.deepcopy(summary)
        figures = []
        subplots = plotting.plt.subplots

        def capture(*args, **kwargs):
            fig, ax = subplots(*args, **kwargs)
            figures.append((fig, ax))
            return fig, ax

        out = self.root / "distinct-series-plots"
        with patch.object(plotting.plt, "subplots", side_effect=capture):
            plotting.render(summary, out)
        self.assertEqual(summary, original)
        self.assertEqual(plotting.read_json(out / "plot_summary.json"), original)
        self.assertEqual(len(figures), 3)
        np.testing.assert_allclose(figures[0][1].get_xticks(), [0, 0.25, 0.5, 0.75, 1])
        for index, (fig, ax) in enumerate(figures):
            if index:
                major, minor = ax.get_xticks(), ax.get_xticks(minor=True)
                np.testing.assert_allclose(np.union1d(major, minor), [100, 300, 1000, 3000])
                np.testing.assert_allclose(major[[0, -1]], [100, 3000])
                boxes = [text.get_window_extent(fig.canvas.get_renderer()) for text in ax.get_xticklabels()]
                self.assertTrue(all(first.x1 < second.x0 for first, second in zip(boxes, boxes[1:])))
            self.assertEqual(len(ax.lines), 3)
            self.assertEqual(len(ax.collections), 2)
            self.assertEqual([text.get_text() for text in ax.get_legend().get_texts()], ["MOBILE", "MOPO", "Expert"])
            for series, line, band in zip(original["series"], ax.lines, ax.collections):
                records = series["points"] if index == 0 else series["points"][index - 1]["history"]
                xkey = "actual_fraction" if index == 0 else "epoch"
                np.testing.assert_allclose(line.get_xdata(), [record[xkey] for record in records])
                np.testing.assert_allclose(line.get_ydata(), [record["mean"] for record in records])
                vertices = band.get_paths()[0].vertices
                for record in records:
                    bounds = vertices[vertices[:, 0] == record[xkey], 1]
                    np.testing.assert_allclose([bounds.min(), bounds.max()], [record["low"], record["high"]])


if __name__ == "__main__":
    unittest.main()

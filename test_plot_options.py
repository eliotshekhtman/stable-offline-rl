from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import plot
import plot_rendering


def cohort_config(**options):
    return {
        "version": 2,
        "seeds": [0],
        "ablation": {"name": "chunk_length", "values": [1, 4]},
        "match": {},
        "series": [{"algo": "iql", "match": {}}],
        **options,
    }


class PlotOptionValidationTests(unittest.TestCase):
    def test_optional_fields_are_preserved_without_inserting_defaults_or_mutating_input(self):
        for options in (
            {}, {"figure_height": 3}, {"figure_height": 2.9},
            {"show_legend": True}, {"show_legend": False},
            {"figure_height": 4.6, "show_legend": False},
        ):
            with self.subTest(options=options):
                config = cohort_config(**options)
                original = deepcopy(config)
                self.assertEqual(plot_rendering.validate_plot_options(config), options)
                validated = plot.validate_plot_cohort(config)
                self.assertEqual(validated, original)
                self.assertEqual(validated["version"], 2)
                self.assertEqual(config, original)

    def test_invalid_values_are_rejected_by_shared_and_cohort_validation(self):
        invalid_options = {
            "figure_height": [0, -1, float("nan"), float("inf"), -float("inf"),
                              True, False, "3.5", None, [], {}],
            "show_legend": [0, 1, -1, 1.0, "true", "false", None, [], {}],
        }
        for key, values in invalid_options.items():
            for value in values:
                for validate in (plot_rendering.validate_plot_options, plot.validate_plot_cohort):
                    with self.subTest(key=key, value=value, validator=validate.__name__):
                        with self.assertRaisesRegex(ValueError, key):
                            validate(cohort_config(**{key: value}))

    def test_options_do_not_relax_unknown_field_or_version_validation(self):
        for changes in ({"version": 3}, {"figure_width": 8}, {"plot_options": {}}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                plot.validate_plot_cohort(cohort_config(figure_height=2.9, **changes))

    def test_shipped_cohort_heights_follow_the_requested_ablation_groups(self):
        import walker_composition_plot

        root = Path(__file__).parent
        cohorts = {
            3.5: (
                "scripts/halfcheetah_noisy_fraction.json",
                "scripts/halfcheetah_real_ratio.json",
                "scripts/reacher_noisy_fraction.json",
                "scripts/reacher_real_ratio.json",
                "walker_composition_cohorts/all_algorithms_clean_noisy.json",
                "walker_composition_cohorts/mobile_clean_noisy.json",
            ),
            4.25: (
                "scripts/halfcheetah_expert_chunks.json",
                "scripts/lift_chunks.json",
                "scripts/lift_real_ratio.json",
            ),
            None: (
                "scripts/halfcheetah_clean_medium.json",
                "scripts/reacher_noise_scale.json",
                "walker_composition_cohorts/all_algorithms_clean_medium.json",
                "walker_composition_cohorts/mobile_clean_medium.json",
            ),
        }
        for height, paths in cohorts.items():
            for path in paths:
                with self.subTest(cohort=path, height=height):
                    config = plot.load_json(root / path)
                    original = deepcopy(config)
                    validate = (walker_composition_plot.validate_cohort
                                if path.startswith("walker_composition_cohorts/")
                                else plot.validate_plot_cohort)
                    validated = validate(config)
                    expected = {} if height is None else {"figure_height": height}
                    self.assertEqual(plot_rendering.validate_plot_options(validated), expected)
                    self.assertEqual(config, original)


class PlotOptionRenderingTests(unittest.TestCase):
    def test_all_renderers_change_only_height_and_legend(self):
        try:
            plot_rendering._find_tex_tools()
        except RuntimeError as error:
            self.skipTest(str(error))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "rollouts").mkdir()
            samples = np.asarray([0.2, 0.4, 0.7])
            curves = np.asarray([[1.0, 0.5, 0.2], [1.2, 0.7, 0.3]])
            np.savez(root / "returns_last.npz", policy_episode_performance=samples)
            np.savez(root / "contraction_last.npz", distance_curves=curves)
            records = [{"step": step, "actual_percent": step * 50,
                        "state_ood_ratio": 0.5 * step,
                        "state_action_ood_ratio": 0.75 * step} for step in (1, 2)]
            for record in records:
                np.savez(root / "rollouts" / f"step_{record['step']}.npz",
                         performance=samples * record["step"])
            rows = [{
                "algo": algo, "chunk_length": chunk,
                "training_schema": {"algo": algo, "chunk_length": chunk},
                "seed_rows": [{"eval_dir": str(root)}], "env_name": "Reacher-v5",
                "expert_performance_mean": 1.0, "performance_label": "success rate",
            } for algo in ("iql", "mopo") for chunk in (1, 4)]
            ablation_rows = [{
                **row, "fraction": fraction, "chunk_length": 1,
                "training_schema": {"algo": row["algo"], "chunk_length": 1},
            } for row, fraction in zip(rows, (0.25, 0.75, 0.25, 0.75))]
            histories = [{
                **row, "label": row["algo"].upper(), "records": records,
                "seed_histories": [{"eval_dir": str(root), "records": records}],
            } for row in rows[::2]]
            original = deepcopy((rows, ablation_rows, histories))
            calls = (
                (plot.plot_performance_vs_chunk_length, (rows, root), (8, 5)),
                (plot.performance_ablation_plot,
                 (ablation_rows, root / "ablation.png", "fraction", "fraction"), (8, 5)),
                (plot.contraction_curve_plot,
                 (rows, "chunk_length", "chunk length", root / "contraction.png"), (10, 4)),
                (plot.performance_history_plot, (histories, root / "history.png"), (8, 5)),
                (plot.history_line_plot,
                 (histories, ("state_ood_ratio", "state_action_ood_ratio"),
                  ("state OOD", "state-action OOD"), root / "ood.png"), (8, 6)),
            )
            for function, args, default_size in calls:
                with self.subTest(renderer=function.__name__):
                    with patch.object(plot, "save_plot") as save:
                        function(*args)
                    baseline = save.call_args.args[0]
                    np.testing.assert_array_equal(baseline.get_size_inches(), default_size)
                    self.assertTrue(any(axis.get_legend() is not None for axis in baseline.axes))
                    height = 4.6 if function is plot.history_line_plot else 2.9
                    for options in (
                        {"figure_height": None, "show_legend": True},
                        {"figure_height": 3.5}, {"figure_height": 4.25},
                        {"figure_height": height}, {"show_legend": False},
                        {"figure_height": height, "show_legend": False},
                    ):
                        with self.subTest(options=options), patch.object(plot, "save_plot") as save:
                            function(*args, **options)
                        figure = save.call_args.args[0]
                        expected_height = options.get("figure_height") or default_size[1]
                        np.testing.assert_array_equal(figure.get_size_inches(), (default_size[0], expected_height))
                        self.assertEqual(len(figure.axes), len(baseline.axes))
                        self.assertEqual(figure._suptitle, baseline._suptitle)
                        for axis, before in zip(figure.axes, baseline.axes):
                            expected_legend = before.get_legend() is not None and options.get("show_legend", True)
                            self.assertEqual(axis.get_legend() is not None, expected_legend)
                            self.assertEqual(axis.get_xlabel(), before.get_xlabel())
                            self.assertEqual(axis.get_ylabel(), before.get_ylabel())
                            self.assertEqual(axis.get_title(), before.get_title())
                            for text, old_text in zip(
                                (axis.xaxis.label, axis.yaxis.label, axis.title),
                                (before.xaxis.label, before.yaxis.label, before.title),
                            ):
                                self.assertEqual(text.get_fontproperties(), old_text.get_fontproperties())
                                self.assertEqual(text.get_usetex(), old_text.get_usetex())
                            np.testing.assert_allclose(axis.get_xlim(), before.get_xlim(), rtol=0, atol=1e-12)
                            np.testing.assert_allclose(axis.get_ylim(), before.get_ylim(), rtol=0, atol=1e-12)
                            self.assertEqual(axis.get_xscale(), before.get_xscale())
                            self.assertEqual(len(axis.lines), len(before.lines))
                            self.assertEqual(len(axis.collections), len(before.collections))
                            for line, old_line in zip(axis.lines, before.lines):
                                np.testing.assert_array_equal(line.get_xydata(), old_line.get_xydata())
                                self.assertEqual(line.get_label(), old_line.get_label())
                                self.assertEqual(line.get_color(), old_line.get_color())
                                self.assertEqual(line.get_linewidth(), old_line.get_linewidth())
                                self.assertEqual(line.get_linestyle(), old_line.get_linestyle())
                                self.assertEqual(line.get_marker(), old_line.get_marker())
                            for band, old_band in zip(axis.collections, before.collections):
                                np.testing.assert_array_equal(band.get_facecolor(), old_band.get_facecolor())
                                self.assertEqual(band.get_alpha(), old_band.get_alpha())
                                self.assertEqual(len(band.get_paths()), len(old_band.get_paths()))
                                for path, old_path in zip(band.get_paths(), old_band.get_paths()):
                                    np.testing.assert_array_equal(path.vertices, old_path.vertices)
            self.assertEqual((rows, ablation_rows, histories), original)
            with np.load(root / "returns_last.npz") as data:
                np.testing.assert_array_equal(data["policy_episode_performance"], samples)
            with np.load(root / "contraction_last.npz") as data:
                np.testing.assert_array_equal(data["distance_curves"], curves)


class PlotOptionForwardingTests(unittest.TestCase):
    def test_root_and_all_wrappers_forward_cohort_options_without_attaching_them_to_rows(self):
        ablations = {
            "chunk_length": ("chunk_length", [1, 4]),
            "noisy_fraction": ("noisy_trajectory_fraction", [0.25, 0.75]),
            "noise_scale": ("noise_scale", [0.1, 0.3]),
            "minari_fraction": ("minari_trajectory_fraction", [0.25, 0.75]),
        }
        renderer_names = (
            "plot_performance_vs_chunk_length", "performance_ablation_plot",
            "contraction_curve_plot", "performance_history_plot", "history_line_plot",
        )
        for ablation, (key, values) in ablations.items():
            for options in ({}, {"figure_height": 4.6, "show_legend": False}):
                with self.subTest(ablation=ablation, options=options), ExitStack() as stack:
                    config = cohort_config(**options)
                    config["ablation"] = {"name": ablation, "values": values}
                    rows = [{
                        "algo": "iql", "chunk_length": 1,
                        "training_schema": {"algo": "iql", "chunk_length": 1},
                        "plot_dataset_tag": "example", "dataset_source": "generated",
                        "num_samples": 100, "noise_scale": 0.2,
                        "requested_prop_clean_expert": 0.0,
                        "requested_prop_noisy_expert": 1.0,
                        "requested_prop_random": 0.0,
                        "minari_dataset_id": "test/medium-v0", key: value,
                    } for value in values]
                    if ablation == "minari_fraction":
                        for row in rows:
                            row["dataset_source"] = "clean-minari"
                    histories = [{**rows[0], "records": []}]
                    original = deepcopy((config, rows, histories))
                    mocks = {
                        "load_json": config, "load_rows": rows, "load_histories": histories,
                        "filter_cohort_runs": (rows, histories), "average_seed_rows": rows,
                        "average_seed_histories": histories, "select_plot_cohort": rows,
                        "validate_cohort_grid": None, "select_cohort_histories": histories,
                    }
                    for name, result in mocks.items():
                        stack.enter_context(patch.object(plot, name, return_value=result))
                    stack.enter_context(patch.object(Path, "mkdir"))
                    renderers = {name: stack.enter_context(patch.object(plot, name))
                                 for name in renderer_names}
                    source = Path("cohort.json") if options else config
                    plot.plot_root(Path("evaluation"), eval_dirs=[], cohort=source)
                    for name, renderer in renderers.items():
                        expected = name != (
                            "performance_ablation_plot" if ablation == "chunk_length"
                            else "plot_performance_vs_chunk_length"
                        )
                        self.assertEqual(renderer.call_count, int(expected), name)
                        if expected:
                            kwargs = dict(renderer.call_args.kwargs)
                            if name == "performance_ablation_plot" and ablation == "noise_scale":
                                self.assertIs(kwargs.pop("fraction_axis"), False)
                            self.assertEqual(kwargs, options, name)
                            records = renderer.call_args.args[0]
                            self.assertEqual(records, histories if "history" in name else rows)
                    self.assertEqual((config, rows, histories), original)


if __name__ == "__main__":
    unittest.main()

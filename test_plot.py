import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import plot


class BootstrapTests(unittest.TestCase):
    def test_center_is_exact_seed_mean_and_constant_data_has_zero_width(self):
        center, low, high = plot.bootstrap_mean([
            np.zeros(20), np.full(20, 2.0),
        ])
        self.assertEqual(center, 1.0)
        self.assertLessEqual(low, center)
        self.assertGreaterEqual(high, center)

        center, low, high = plot.bootstrap_mean([np.ones(100)])
        self.assertEqual((center, low, high), (1.0, 1.0, 1.0))

    def test_curve_center_averages_pairs_then_seeds(self):
        curves = [
            np.asarray([[1.0, 2.0, np.nan], [3.0, 4.0, np.nan]]),
            np.asarray([[5.0, 6.0, 7.0], [7.0, 8.0, 9.0]]),
        ]
        center, low, high = plot.bootstrap_curve(curves)
        np.testing.assert_allclose(center, [4.0, 5.0, 8.0])
        self.assertTrue(np.all(low <= center))
        self.assertTrue(np.all(center <= high))


class EvaluationDiscoveryTests(unittest.TestCase):
    def test_latest_eval_dirs_keeps_each_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = []
            for seed, variants in ((0, ("20260101", "20260102")), (1, ("20260101",))):
                for variant in variants:
                    eval_dir = root / f"run_seed{seed}" / variant
                    eval_dir.mkdir(parents=True)
                    manifest_path = eval_dir / "run_manifest.json"
                    manifest_path.write_text(json.dumps({
                        "training_schema": {"algo": "cql", "seed": seed},
                    }))
                    (eval_dir / "results.json").write_text(json.dumps({
                        "run_manifest_path": str(manifest_path),
                    }))
                    if variant == variants[-1]:
                        expected.append(eval_dir)

            self.assertEqual(plot.latest_eval_dirs(root), sorted(expected))

    def test_load_rows_rejects_historical_unsupported_task(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            eval_dir = root / "eval"
            eval_dir.mkdir()
            metadata_path = root / "metadata.json"
            metadata_path.write_text(json.dumps({
                "source": "robomimic",
                "task": "Square",
                "dataset_type": "ph",
            }))
            manifest_path = root / "run_manifest.json"
            manifest_path.write_text(json.dumps({
                "algo": "bc",
                "chunk_length": 1,
                "dataset_source": "robomimic",
                "dataset_tag": "robomimic_square_ph",
                "dataset_metadata_path": str(metadata_path),
                "env_name": "Square",
                "training_schema": {
                    "algo": "bc",
                    "chunk_length": 1,
                    "dataset": {"source": "robomimic", "seed": 0},
                    "seed": 0,
                },
            }))
            (eval_dir / "results.json").write_text(json.dumps({
                "run_manifest_path": str(manifest_path),
                "evaluation_config": {
                    "schema_version": plot.EVALUATION_SCHEMA_VERSION,
                },
            }))

            with self.assertRaisesRegex(ValueError, "Unsupported task 'Square'"):
                plot.load_rows([eval_dir])


class DynamicsChunkModePlotTests(unittest.TestCase):
    @staticmethod
    def record(algo="mopo", chunk_length=4, mode=None):
        model_based = {}
        if mode is not None:
            model_based["chunk_dynamics"] = {"version": 1, "mode": mode}
        return {
            "algo": algo,
            "chunk_length": chunk_length,
            "training_schema": {
                "algo": algo,
                "chunk_length": chunk_length,
                "model_based": model_based,
            },
        }

    def test_old_model_based_schema_is_direct_and_keeps_old_labels(self):
        record = self.record()

        self.assertEqual(plot.dynamics_chunk_mode(record), "direct")
        self.assertEqual(plot.algorithm_label(record), "mopo")
        self.assertEqual(plot.policy_label(record), "mopo (l=4)")

    def test_recursive_mode_has_distinct_algorithm_and_policy_labels(self):
        record = self.record(mode="recursive")

        self.assertEqual(plot.dynamics_chunk_mode(record), "recursive")
        self.assertEqual(plot.algorithm_label(record), "mopo (recursive dynamics)")
        self.assertEqual(
            plot.policy_label(record), "mopo (l=4, recursive dynamics)"
        )

    def test_chunk_length_one_is_always_labeled_as_direct(self):
        record = self.record(chunk_length=1, mode="recursive")

        self.assertEqual(plot.dynamics_chunk_mode(record), "direct")
        self.assertEqual(plot.algorithm_label(record), "mopo")
        self.assertEqual(plot.policy_label(record), "mopo (l=1)")

    def test_model_free_label_is_unchanged(self):
        record = {
            "algo": "iql",
            "chunk_length": 4,
            "training_schema": {"algo": "iql", "chunk_length": 4},
        }

        self.assertIsNone(plot.dynamics_chunk_mode(record))
        self.assertEqual(plot.algorithm_label(record), "iql")
        self.assertEqual(plot.policy_label(record), "iql (l=4)")

    def test_algorithm_groups_separate_direct_and_recursive(self):
        direct = self.record(mode="direct")
        recursive = self.record(mode="recursive")

        groups = plot.algorithm_groups([recursive, direct])

        self.assertEqual(
            [(label, rows) for label, rows in groups],
            [("mopo", [direct]), ("mopo (recursive dynamics)", [recursive])],
        )

    def test_seed_averaging_keeps_modes_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata_path = root / "metadata.json"
            metadata_path.write_text(json.dumps({
                "source": "robomimic",
                "task": "Lift",
                "dataset_type": "mg_dense",
            }))
            eval_dirs = []
            for mode in ("direct", "recursive"):
                for seed in (1, 2):
                    eval_dir = root / mode / str(seed)
                    eval_dir.mkdir(parents=True)
                    model_based = {}
                    if mode == "recursive":
                        model_based["chunk_dynamics"] = {
                            "version": 1,
                            "mode": "recursive",
                        }
                    manifest_path = eval_dir / "run_manifest.json"
                    manifest_path.write_text(json.dumps({
                        "algo": "mobile",
                        "chunk_length": 4,
                        "dataset_source": "robomimic",
                        "dataset_tag": f"lift_seed{seed}",
                        "dataset_metadata_path": str(metadata_path),
                        "env_name": "Lift",
                        "training_schema": {
                            "algo": "mobile",
                            "chunk_length": 4,
                            "seed": seed,
                            "dataset": {"seed": seed, "source": "robomimic"},
                            "model_based": model_based,
                        },
                    }))
                    (eval_dir / "results.json").write_text(json.dumps({
                        "run_manifest_path": str(manifest_path),
                        "evaluation_config": {
                            "schema_version": plot.EVALUATION_SCHEMA_VERSION,
                        },
                        "last_policy_performance_mean": float(seed),
                        "expert_performance_mean": 1.0,
                    }))
                    eval_dirs.append(eval_dir)

            averaged = plot.average_seed_rows(plot.load_rows(eval_dirs))

        self.assertEqual(len(averaged), 2)
        self.assertEqual(sorted(len(row["seed_rows"]) for row in averaged), [2, 2])
        self.assertEqual(
            {row["label"] for row in averaged},
            {"mobile (l=4)", "mobile (l=4, recursive dynamics)"},
        )

    @patch("plot.final_performance_samples", return_value=[np.ones(2)])
    def test_performance_plot_renders_separate_mode_series(self, _samples):
        direct = {
            **self.record(mode="direct"),
            "fraction": 0.5,
            "expert_performance_mean": 1.0,
            "performance_label": "success rate",
        }
        recursive = {**direct, **self.record(mode="recursive")}
        figure, axis = plot.plt.subplots()
        with tempfile.TemporaryDirectory() as directory, patch(
            "plot.plt.subplots", return_value=(figure, axis)
        ), patch("plot.plt.close"):
            plot.performance_ablation_plot(
                [recursive, direct],
                "mode separation",
                Path(directory) / "plot.png",
                "fraction",
                "fraction",
            )

        self.assertEqual(
            axis.get_legend_handles_labels()[1],
            ["mopo", "mopo (recursive dynamics)", "expert"],
        )
        plot.plt.close(figure)


class ChunkLengthAxisTests(unittest.TestCase):
    @patch("plot.final_performance_samples", return_value=[np.ones(2)])
    def test_performance_plot_uses_log2_axis_and_labels(self, _samples):
        chunk_lengths = (1, 2, 4, 6, 8)
        rows = [
            {
                "algo": "iql",
                "chunk_length": chunk_length,
                "training_schema": {
                    "algo": "iql",
                    "chunk_length": chunk_length,
                },
                "expert_performance_mean": 1.0,
                "performance_label": "success rate",
            }
            for chunk_length in chunk_lengths
        ]
        figure, axis = plot.plt.subplots()
        with tempfile.TemporaryDirectory() as directory, patch(
            "plot.plt.subplots", return_value=(figure, axis)
        ), patch("plot.plt.close"):
            plot.plot_performance_vs_chunk_length(rows, Path(directory))

        self.assertEqual(axis.get_xscale(), "log")
        np.testing.assert_array_equal(axis.get_xticks(), chunk_lengths)
        self.assertEqual(
            [tick.get_text() for tick in axis.get_xticklabels()],
            [r"$2^{0}$", r"$2^{1}$", r"$2^{2}$", "6", r"$2^{3}$"],
        )
        np.testing.assert_array_equal(axis.lines[0].get_xdata(), chunk_lengths)
        np.testing.assert_allclose(
            axis.xaxis.get_transform().transform(np.asarray([1, 2, 4, 8])),
            [0, 1, 2, 3],
        )
        self.assertEqual(len(axis.xaxis.get_minorticklocs()), 0)
        plot.plt.close(figure)


class PlotCohortTests(unittest.TestCase):
    @staticmethod
    def record(
        algo="mobile", chunk_length=4, real_ratio=0.5, epoch=300,
        dynamics_mode=None,
    ):
        model_based = {
            "real_ratio": real_ratio,
        }
        if dynamics_mode is not None:
            model_based["chunk_dynamics"] = {
                "version": 1,
                "mode": dynamics_mode,
            }
        return {
            "algo": algo,
            "chunk_length": chunk_length,
            "training_schema": {
                "algo": algo,
                "chunk_length": chunk_length,
                "epoch": epoch,
                "model_based": model_based,
            },
        }

    @staticmethod
    def cohort(*series):
        return plot.validate_plot_cohort({
            "version": plot.PLOT_COHORT_VERSION,
            "series": list(series),
        })

    def test_mobile_real_ratio_variants_are_separate_labeled_series(self):
        rows = [
            self.record(chunk_length=chunk_length, real_ratio=real_ratio)
            for real_ratio in (0.0, 0.5)
            for chunk_length in (2, 4)
        ]
        cohort = self.cohort(
            {"algo": "mobile", "match": {"model_based.real_ratio": 0.0}},
            {"algo": "mobile", "match": {"model_based.real_ratio": 0.5}},
        )

        selected = plot.select_plot_cohort(rows, cohort)
        groups = plot.algorithm_groups(selected, "chunk_length")

        self.assertEqual(
            [label for label, _ in groups],
            ["mobile (real ratio=0.00)", "mobile (real ratio=0.50)"],
        )
        self.assertEqual([len(group) for _, group in groups], [2, 2])

    def test_automatic_grouping_rejects_two_configs_at_one_x_value(self):
        rows = [
            self.record(real_ratio=0.0),
            self.record(real_ratio=0.5),
        ]

        with self.assertRaisesRegex(
            ValueError, "multiple seed-averaged configurations"
        ):
            plot.algorithm_groups(rows, "chunk_length")

    def test_coherent_unselected_sweep_keeps_one_legacy_series(self):
        rows = [
            self.record(chunk_length=chunk_length, real_ratio=0.5)
            for chunk_length in (2, 4)
        ]

        groups = plot.algorithm_groups(rows, "chunk_length")

        self.assertEqual(groups, [("mobile", rows)])

    def test_underspecified_cohort_still_rejects_ambiguous_series(self):
        rows = [
            self.record(real_ratio=0.0),
            self.record(real_ratio=0.5),
        ]
        selected = plot.select_plot_cohort(
            rows, self.cohort({"algo": "mobile"})
        )

        with self.assertRaisesRegex(ValueError, "--cohort series"):
            plot.algorithm_groups(selected, "chunk_length")

    def test_declared_series_does_not_silently_split_dynamics_modes(self):
        rows = [
            self.record(dynamics_mode=None),
            self.record(dynamics_mode="recursive"),
        ]
        selected = plot.select_plot_cohort(
            rows, self.cohort({"algo": "mobile"})
        )

        with self.assertRaisesRegex(
            ValueError, "multiple seed-averaged configurations"
        ):
            plot.algorithm_groups(selected, "chunk_length")

    def test_separate_series_can_explicitly_select_dynamics_modes(self):
        direct = self.record(dynamics_mode=None)
        recursive = self.record(dynamics_mode="recursive")
        cohort = self.cohort(
            {
                "algo": "mobile",
                "label": "mobile direct",
                "match": {"model_based.chunk_dynamics.mode": None},
            },
            {
                "algo": "mobile",
                "label": "mobile recursive",
                "match": {"model_based.chunk_dynamics.mode": "recursive"},
            },
        )

        selected = plot.select_plot_cohort([direct, recursive], cohort)
        groups = plot.algorithm_groups(selected, "chunk_length")

        self.assertEqual(len(groups), 2)

    def test_selected_series_rejects_training_parameter_changes_across_x(self):
        rows = [
            self.record(chunk_length=2, epoch=100),
            self.record(chunk_length=4, epoch=300),
        ]
        selected = plot.select_plot_cohort(
            rows,
            self.cohort({
                "algo": "mobile",
                "match": {"model_based.real_ratio": 0.5},
            }),
        )

        with self.assertRaisesRegex(
            ValueError, "non-axis training parameters.*epoch"
        ):
            plot.algorithm_groups(selected, "chunk_length")

    def test_chunk_and_generated_dataset_axis_fields_may_change(self):
        chunk_rows = [
            {
                **self.record(chunk_length=chunk_length),
                "training_schema": {
                    **self.record(chunk_length=chunk_length)["training_schema"],
                    "macro_discount": 0.99**chunk_length,
                },
            }
            for chunk_length in (2, 4)
        ]
        self.assertEqual(
            len(plot.algorithm_groups(chunk_rows, "chunk_length")), 1
        )

        dataset_rows = []
        for fraction in (0.0, 0.5):
            row = self.record()
            row["noisy_trajectory_fraction"] = fraction
            row["training_schema"]["dataset"] = {
                "source": "generated",
                "num_samples": 1000,
                "noise_scale": 0.5,
                "prop_clean_expert": 1.0 - fraction,
                "prop_noisy_expert": fraction,
                "prop_random": 0.0,
                "prop_expert": 1.0,
            }
            dataset_rows.append(row)
        self.assertEqual(
            len(plot.algorithm_groups(
                dataset_rows, "noisy_trajectory_fraction"
            )),
            1,
        )

    def test_different_algorithms_may_have_different_parameters(self):
        mobile = self.record(algo="mobile", epoch=100)
        mopo = self.record(algo="mopo", epoch=300)

        groups = plot.algorithm_groups([mobile, mopo], "chunk_length")

        self.assertEqual(
            [label for label, _ in groups], ["mobile", "mopo"]
        )

    def test_overlapping_cohort_series_are_rejected(self):
        row = self.record(real_ratio=0.0)
        cohort = self.cohort(
            {"algo": "mobile"},
            {"algo": "mobile", "match": {"model_based.real_ratio": 0.0}},
        )

        with self.assertRaisesRegex(ValueError, "select the same"):
            plot.select_plot_cohort([row], cohort)

    def test_cohort_cannot_select_individual_seeds(self):
        with self.assertRaisesRegex(ValueError, "seed-averaged"):
            self.cohort({
                "algo": "mobile",
                "match": {"dataset.seed": 1},
            })

    def test_cohort_requires_every_declared_series_to_match(self):
        cohort = self.cohort({
            "algo": "mobile",
            "match": {"model_based.real_ratio": 0.0},
        })

        with self.assertRaisesRegex(ValueError, "matched no"):
            plot.select_plot_cohort([self.record(real_ratio=0.5)], cohort)

    def test_selected_history_label_includes_variant_parameters(self):
        row = {
            **self.record(real_ratio=0.0),
            "seed_group": "selected",
        }
        selected_rows = plot.select_plot_cohort(
            [row],
            self.cohort({
                "algo": "mobile",
                "match": {"model_based.real_ratio": 0.0},
            }),
        )
        histories = plot.select_cohort_histories(
            [{**row, "label": "mobile (l=4)"}], selected_rows
        )

        self.assertEqual(
            histories[0]["label"],
            "mobile (l=4, real ratio=0.00)",
        )


class NoiseScalePlotTests(unittest.TestCase):
    @staticmethod
    def row(noise_scale, chunk_length=1, noisy=1.0, epoch=300):
        clean = 1.0 - noisy
        return {
            "dataset_source": "generated",
            "algo": "iql",
            "chunk_length": chunk_length,
            "num_samples": 1000,
            "noise_scale": noise_scale,
            "requested_prop_clean_expert": clean,
            "requested_prop_noisy_expert": noisy,
            "requested_prop_random": 0.0,
            "training_schema": {
                "algo": "iql",
                "chunk_length": chunk_length,
                "epoch": epoch,
                "dataset": {
                    "source": "generated",
                    "num_samples": 1000,
                    "noise_scale": noise_scale,
                    "prop_clean_expert": clean,
                    "prop_noisy_expert": noisy,
                    "prop_random": 0.0,
                    "prop_expert": 1.0,
                },
            },
        }

    def test_noise_scale_is_the_only_training_schema_axis(self):
        rows = [self.row(scale) for scale in (0.0, 0.5, 1.0)]

        self.assertEqual(len(plot.algorithm_groups(rows, "noise_scale")), 1)

        rows[-1]["training_schema"]["epoch"] = 100
        with self.assertRaisesRegex(
            ValueError, "non-axis training parameters.*epoch"
        ):
            plot.algorithm_groups(rows, "noise_scale")

    @patch("plot.performance_ablation_plot")
    def test_noise_scale_plot_is_split_by_fixed_dataset_and_chunk(
        self, performance_plot
    ):
        rows = [
            self.row(scale, chunk_length=chunk_length)
            for chunk_length in (1, 4)
            for scale in (0.0, 0.5)
        ]

        with tempfile.TemporaryDirectory() as directory:
            plot.plot_noise_scale_ablation(rows, Path(directory))

        self.assertEqual(performance_plot.call_count, 2)
        paths = {call.args[2] for call in performance_plot.call_args_list}
        self.assertEqual(paths, {
            Path(directory)
            / "noise_scale/samples1000_clean0_noisy1_random0_chunk1/final"
            / "performance_vs_noise_scale.png",
            Path(directory)
            / "noise_scale/samples1000_clean0_noisy1_random0_chunk4/final"
            / "performance_vs_noise_scale.png",
        })
        for call in performance_plot.call_args_list:
            self.assertEqual(
                call.args[3:5],
                ("noise_scale", "Gaussian action-noise scale"),
            )
            self.assertFalse(call.kwargs["fraction_axis"])

    @patch("plot.performance_ablation_plot")
    def test_noise_scale_plot_requires_multiple_scales_and_a_noisy_component(
        self, performance_plot
    ):
        plot.plot_noise_scale_ablation([self.row(0.5)], Path("unused"))
        plot.plot_noise_scale_ablation(
            [self.row(scale, noisy=0.0) for scale in (0.0, 0.5)],
            Path("unused"),
        )

        performance_plot.assert_not_called()

    @patch("plot.final_performance_samples", return_value=[np.ones(2)])
    def test_noise_scale_plot_does_not_use_fraction_axis_limits(self, _samples):
        rows = []
        for scale in (0.0, 2.0):
            row = self.row(scale)
            row.update({
                "seed_rows": [{}],
                "expert_performance_mean": 1.0,
                "performance_label": "forward displacement",
            })
            rows.append(row)
        figure, axis = plot.plt.subplots()
        with tempfile.TemporaryDirectory() as directory, patch(
            "plot.plt.subplots", return_value=(figure, axis)
        ), patch("plot.plt.close"):
            plot.performance_ablation_plot(
                rows,
                "noise scale",
                Path(directory) / "plot.png",
                "noise_scale",
                "Gaussian action-noise scale",
                fraction_axis=False,
            )

        self.assertGreater(axis.get_xlim()[1], 2.0)
        plot.plt.close(figure)


class CleanMinariPlotTests(unittest.TestCase):
    @staticmethod
    def training_schema(source, fraction=None, noise_scale=0.5, epoch=300):
        dataset = {
            "version": 4,
            "source": source,
            "env_name": "Reacher-v5",
            "expert_path": "/experts/Reacher-v5.zip",
            "max_timesteps": 1000000,
            "num_samples": 500000,
            "deterministic": True,
            "seed": 0,
            "test_fraction": 0.2,
        }
        if source == "generated":
            dataset.update({
                "noise_scale": noise_scale,
                "prop_clean_expert": 1.0,
                "prop_noisy_expert": 0.0,
                "prop_random": 0.0,
                "prop_expert": 1.0,
            })
        else:
            dataset.update({
                "dataset_id": "mujoco/reacher/medium-v0",
                "minari_fraction": fraction,
            })
        return {
            "version": 3,
            "env_name": "Reacher-v5",
            "algo": "bc",
            "dataset": dataset,
            "chunk_length": 1,
            "base_discount": 0.99,
            "macro_discount": 0.99,
            "chunk_reward": "discounted_sum",
            "seed": 0,
            "epoch": epoch,
            "step_per_epoch": 1000,
            "batch_size": 256,
        }

    def test_dataset_fields_only_reads_clean_minari_markers_for_mixture(self):
        fields = plot.dataset_fields({
            "source": "clean-minari",
            "dataset_id": "mujoco/reacher/medium-v0",
            "actual_minari_trajectory_fraction": 0.5,
            "dataset_schema": {
                "num_samples": 500000,
                "minari_fraction": 0.5,
            },
        })

        self.assertEqual(fields["dataset_source"], "clean-minari")
        self.assertEqual(fields["minari_dataset"], "medium")
        self.assertEqual(fields["minari_trajectory_fraction"], 0.5)

    @patch("plot.contraction_curve_plot")
    @patch("plot.performance_ablation_plot")
    def test_mixture_plot_uses_only_matching_clean_baseline(
        self, performance_plot, contraction_plot
    ):
        mixed = [
            {
                "dataset_source": "clean-minari",
                "minari_dataset_id": "mujoco/reacher/medium-v0",
                "num_samples": 500000,
                "chunk_length": 1,
                "minari_trajectory_fraction": fraction,
            }
            for fraction in (0.5, 1.0)
        ]
        clean = {
            "dataset_source": "generated",
            "algo": "bc",
            "seed_rows": [{"seed": 0}],
            "training_schema": self.training_schema("generated"),
            "num_samples": 500000,
            "chunk_length": 1,
            "requested_prop_clean_expert": 1.0,
            "requested_prop_noisy_expert": 0.0,
            "requested_prop_random": 0.0,
        }
        wrong_size = {**clean, "num_samples": 1000000}
        noisy = {
            **clean,
            "requested_prop_clean_expert": 0.5,
            "requested_prop_noisy_expert": 0.5,
        }

        with tempfile.TemporaryDirectory() as directory:
            plot.plot_clean_minari_ablation(
                [*mixed, clean, wrong_size, noisy], Path(directory)
            )

        plotted_rows = performance_plot.call_args.args[0]
        self.assertEqual(
            sorted(row["minari_trajectory_fraction"] for row in plotted_rows),
            [0.0, 0.5, 1.0],
        )
        baseline = next(
            row for row in plotted_rows if row["minari_trajectory_fraction"] == 0.0
        )
        self.assertEqual(len(baseline["seed_rows"]), 1)
        self.assertEqual(contraction_plot.call_count, 1)

    def test_clean_baselines_coalesce_noise_variants_by_unique_seed(self):
        def baseline(noise_scale, created_at, seeds):
            return {
                "algo": "bc",
                "chunk_length": 1,
                "created_at": created_at,
                "training_schema": self.training_schema(
                    "generated", noise_scale=noise_scale
                ),
                "seed_rows": [
                    {
                        "seed": seed,
                        "created_at": created_at,
                        "eval_dir": f"{created_at}/seed{seed}",
                    }
                    for seed in seeds
                ],
            }

        old = baseline(0.1, "2026-01-01", [0, 1])
        new = baseline(0.5, "2026-02-01", [0, 2])

        coalesced = plot.coalesce_clean_minari_baselines([old, new])

        self.assertEqual(len(coalesced), 1)
        self.assertEqual(
            [row["seed"] for row in coalesced[0]["seed_rows"]],
            [0, 1, 2],
        )
        seed_zero = next(
            row for row in coalesced[0]["seed_rows"] if row["seed"] == 0
        )
        self.assertEqual(seed_zero["created_at"], "2026-02-01")

    def test_clean_baseline_and_mixtures_form_one_consistent_line(self):
        baseline = {
            "algo": "bc",
            "chunk_length": 1,
            "minari_trajectory_fraction": 0.0,
            "training_schema": self.training_schema("generated"),
        }
        mixtures = [
            {
                "algo": "bc",
                "chunk_length": 1,
                "minari_trajectory_fraction": fraction,
                "training_schema": self.training_schema(
                    "clean-minari", fraction=fraction
                ),
            }
            for fraction in (0.5, 1.0)
        ]

        groups = plot.algorithm_groups(
            [baseline, *mixtures], "minari_trajectory_fraction"
        )

        self.assertEqual(len(groups), 1)

    def test_clean_baselines_with_different_training_configs_do_not_merge(self):
        rows = [
            {
                "algo": "bc",
                "chunk_length": 1,
                "training_schema": self.training_schema(
                    "generated", epoch=epoch
                ),
                "seed_rows": [{"seed": 0}],
            }
            for epoch in (100, 300)
        ]

        self.assertEqual(
            len(plot.coalesce_clean_minari_baselines(rows)), 2
        )

    def test_plot_root_does_not_delete_unrelated_existing_plots(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out = root / "plots"
            out.mkdir()
            marker = out / "existing.png"
            marker.write_bytes(b"existing")

            plot.plot_root(root, out=out, eval_dirs=[])

            self.assertEqual(marker.read_bytes(), b"existing")


if __name__ == "__main__":
    unittest.main()

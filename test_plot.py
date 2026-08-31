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


class CleanMinariPlotTests(unittest.TestCase):
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
            "num_samples": 500000,
            "chunk_length": 1,
            "requested_prop_clean_expert": 1.0,
            "requested_prop_noisy_expert": 0.0,
            "requested_prop_random": 0.0,
        }
        historical_clean = {**clean, "seed_rows": [{"seed": 1}]}
        wrong_size = {**clean, "num_samples": 1000000}
        noisy = {
            **clean,
            "requested_prop_clean_expert": 0.5,
            "requested_prop_noisy_expert": 0.5,
        }

        with tempfile.TemporaryDirectory() as directory:
            plot.plot_clean_minari_ablation(
                [*mixed, clean, historical_clean, wrong_size, noisy], Path(directory)
            )

        plotted_rows = performance_plot.call_args.args[0]
        self.assertEqual(
            sorted(row["minari_trajectory_fraction"] for row in plotted_rows),
            [0.0, 0.5, 1.0],
        )
        baseline = next(
            row for row in plotted_rows if row["minari_trajectory_fraction"] == 0.0
        )
        self.assertEqual(len(baseline["seed_rows"]), 2)
        self.assertEqual(contraction_plot.call_count, 1)

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

import json
import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()

"""Chunk comparisons use upper-right legends; other plots keep task corners."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.legend import Legend
import numpy as np

import plot
from plot_rendering import legend_location


class PlotLegendTests(unittest.TestCase):
    def test_only_reacher_and_lift_use_the_upper_right_corner(self):
        for task in ("Reacher-v5", "Lift"):
            with self.subTest(task=task):
                self.assertEqual(legend_location(task), "upper right")
        for task in ("Can", "HalfCheetah-v5", "Walker2d-v5", None):
            with self.subTest(task=task):
                self.assertEqual(legend_location(task), "lower right")

    def test_chunk_comparisons_override_task_corner_without_affecting_other_plots(self):
        savefig = Figure.savefig
        for task, folder, metric, corner in (
            ("Reacher-v5", "walker", "success rate", "upper right"),
            ("Lift", "walker", "final distance to target", "upper right"),
            ("Can", "reacher", "final distance to target", "lower right"),
            ("HalfCheetah-v5", "reacher", "episode return", "lower right"),
        ):
            with self.subTest(task=task), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / folder
                root.mkdir()
                (root / "rollouts").mkdir()
                np.savez(root / "returns_last.npz", policy_episode_performance=[0.2, 0.4])
                np.savez(root / "contraction_last.npz", distance_curves=[[1.0, 0.5]])
                records = [{
                    "step": step, "actual_percent": step * 50,
                    "state_ood_ratio": 0.5, "state_action_ood_ratio": 0.75,
                } for step in (1, 2)]
                for record in records:
                    np.savez(root / "rollouts" / f"step_{record['step']}.npz", performance=[0.2, 0.4])

                rows = [{
                    "env_name": task, "algo": algo, "chunk_length": chunk_length,
                    "training_schema": {"algo": algo, "chunk_length": chunk_length},
                    "seed_rows": [{"eval_dir": str(root)}],
                    "expert_performance_mean": 1.0, "performance_label": metric,
                } for algo in ("iql", "mopo") for chunk_length in (1, 4)]
                ablation_rows = [{
                    **row, "chunk_length": 1,
                    "training_schema": {"algo": row["algo"], "chunk_length": 1},
                    "fraction": value, "noise_scale": value,
                } for row, value in zip(rows, (0.0, 0.5, 0.0, 0.5))]
                histories = [{
                    **row, "label": row["algo"].upper(), "records": records,
                    "seed_histories": [{"eval_dir": str(root), "records": records}],
                } for row in rows[::2]]
                saved = {}

                def save_and_inspect(figure, filename, **kwargs):
                    savefig(figure, filename, **kwargs)
                    path = Path(filename)
                    signature = {".png": b"\x89PNG\r\n\x1a\n", ".pdf": b"%PDF-"}[path.suffix]
                    self.assertTrue(path.read_bytes().startswith(signature))
                    if path.suffix == ".png":
                        self.assertEqual(kwargs["dpi"], 200)
                    legends = [axis.get_legend() for axis in figure.axes if axis.get_legend() is not None]
                    self.assertTrue(legends)
                    expected_corner = "upper right" if path.stem in {
                        "performance_vs_chunk_length", "contraction_vs_chunk_length",
                    } else corner
                    for legend in legends:
                        self.assertEqual(legend._loc, Legend.codes[expected_corner])
                    saved[path.name] = len(legends)

                calls = [
                    (plot.plot_performance_vs_chunk_length, (rows, root)),
                    (plot.performance_ablation_plot, (ablation_rows, root / "fraction.png", "fraction", "fraction", True)),
                    (plot.performance_ablation_plot, (ablation_rows, root / "noise.png", "noise_scale", "noise scale", False)),
                    (plot.plot_contraction_vs_chunk_length, (rows, root)),
                    (plot.contraction_curve_plot, (ablation_rows, "fraction", "fraction",
                                                   root / "contraction_fraction.png")),
                    (plot.contraction_curve_plot, (ablation_rows, "noise_scale", "noise scale",
                                                   root / "contraction_noise.png")),
                    (plot.performance_history_plot, (histories, root / "history.png")),
                    (plot.history_line_plot, (histories, ("state_ood_ratio", "state_action_ood_ratio"),
                                              ("state OOD", "state-action OOD"), root / "ood.png")),
                ]
                with mpl.rc_context({"legend.loc": "center left"}), patch.object(
                    Figure, "savefig", autospec=True, side_effect=save_and_inspect,
                ):
                    for function, args in calls:
                        with self.subTest(adapter=function.__name__, output=str(args[-1])):
                            function(*args)
                            self.assertEqual(mpl.rcParams["legend.loc"], "center left")
                expected = {
                    "performance_vs_chunk_length": 1,
                    "fraction": 1, "noise": 1,
                    "contraction_vs_chunk_length": 2,
                    "contraction_fraction": 2, "contraction_noise": 2,
                    "history": 1, "ood": 1,
                }
                self.assertEqual(saved, {
                    f"{stem}{suffix}": count
                    for stem, count in expected.items()
                    for suffix in (".png", ".pdf")
                })


if __name__ == "__main__":
    unittest.main()

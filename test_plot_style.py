from io import BytesIO
import os
from pathlib import Path
import tempfile
from unittest.mock import patch

import matplotlib as mpl
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np

import plot_rendering
import test_plot_rendering


class PlotStyleTests(test_plot_rendering.PlotAdapterRenderingTests):
    def assert_axis_style(self, axis, legend_size=18, xlabel_size=24.3):
        self.assertAlmostEqual(axis.xaxis.label.get_fontsize(), xlabel_size if axis.get_xlabel() else 18)
        self.assertEqual(axis.yaxis.label.get_fontsize(), 18)
        self.assertAlmostEqual(axis.title.get_fontsize(), 21.6)
        for tick in axis.get_xticklabels() + axis.get_yticklabels():
            self.assertEqual(tick.get_fontsize(), 18)
        legend = axis.get_legend()
        if legend is not None:
            for label in legend.get_texts():
                self.assertAlmostEqual(label.get_fontsize(), legend_size)
        self.assertIs(axis.get_axisbelow(), True)
        for coordinate in (axis.xaxis, axis.yaxis):
            grids = coordinate.get_gridlines()
            self.assertTrue(grids)
            for grid in grids:
                self.assertTrue(grid.get_visible())
                self.assertEqual(grid.get_color(), "#b0b0b0")
                self.assertEqual(grid.get_linewidth(), 0.6)
                self.assertEqual(grid.get_alpha(), 0.35)
                self.assertEqual(grid.get_linestyle(), "-")
            for tick in coordinate.get_minor_ticks():
                self.assertFalse(tick.gridline.get_visible())
            for artist in [*axis.lines, *axis.collections]:
                self.assertLess(coordinate.get_zorder(), artist.get_zorder())

    def test_all_plot_adapters_create_latex_text_and_restore_settings(self):
        savefig = Figure.savefig

        def save_and_inspect(figure, filename, **kwargs):
            savefig(figure, filename, **kwargs)
            path = Path(filename)
            legend_size = 14.4 if path.stem in {
                "contraction_vs_chunk_length", "history", "ood",
            } else 18
            for axis in figure.axes:
                self.assert_axis_style(axis, legend_size)
                for line in axis.lines:
                    self.assertEqual(line.get_linewidth(), 1.5)
            expected_size = {
                "contraction_vs_chunk_length": (5 * len(figure.axes), 4),
                "ood": (8, 3 * len(figure.axes)),
            }.get(path.stem, (8, 5))
            np.testing.assert_array_equal(figure.get_size_inches(), expected_size)
            if path.suffix == ".png":
                self.assertEqual(kwargs["dpi"], 200)
            self.assertEqual(kwargs["bbox_inches"], "tight")
            self.assertEqual(kwargs["pad_inches"], 0)

        with patch.object(Figure, "savefig", new=save_and_inspect):
            super().test_all_plot_adapters_create_latex_text_and_restore_settings()

    def test_shorter_plots_keep_crowded_legends_clear_of_x_axis_text(self):
        try:
            plot_rendering._find_tex_tools()
        except RuntimeError as error:
            self.skipTest(str(error))
        import plot

        saved = []

        def inspect(figure, path):
            self.addCleanup(plot.plt.close, figure)
            self.assertEqual(figure.get_figheight(), 3.5 if path.stem == "standard" else 3)
            figure.canvas.draw()
            renderer = figure.canvas.get_renderer()
            for axis in figure.axes:
                legend = axis.get_legend()
                self.assertEqual(len(legend.get_texts()), 6 if path.stem == "standard" else 5)
                bounds = legend.get_window_extent(renderer)
                lower, upper = axis.get_xlim()
                labels = [axis.xaxis.label] + [
                    tick.label1 for tick in axis.xaxis.get_major_ticks()
                    if lower <= tick.get_loc() <= upper
                ]
                for label in labels:
                    if label.get_visible() and label.get_text():
                        self.assertFalse(
                            bounds.overlaps(label.get_window_extent(renderer)),
                            f"{path.stem} legend overlaps {label.get_text()}",
                        )
            saved.append(path.stem)

        with tempfile.TemporaryDirectory() as directory, patch.object(plot, "save_plot", side_effect=inspect):
            root = Path(directory)
            np.savez(root / "returns_last.npz", policy_episode_performance=[0.5, 1.0])
            np.savez(root / "contraction_last.npz", distance_curves=[np.exp(-np.arange(50) / 10)])
            common = {
                "seed_rows": [{"eval_dir": str(root)}], "env_name": "Reacher-v5",
                "expert_performance_mean": 1.7, "performance_label": "return",
            }
            standard_rows = [{
                **common, "algo": algo, "fraction": fraction, "chunk_length": 1,
                "training_schema": {"algo": algo, "chunk_length": 1},
            } for algo in ("iql", "cql", "mobile", "mopo", "sac") for fraction in (0, 0.5, 1)]
            contraction_rows = [{
                **common, "algo": "iql", "chunk_length": chunk,
                "training_schema": {"algo": "iql", "chunk_length": chunk},
            } for chunk in (1, 2, 4, 8, 16)]
            for function, args, height in (
                (plot.performance_ablation_plot,
                 (standard_rows, root / "standard.png", "fraction", "Noisy ratio"), 3.5),
                (plot.contraction_curve_plot,
                 (contraction_rows, "chunk_length", "Chunk length", root / "contraction.png"), 3),
            ):
                with self.subTest(adapter=function.__name__):
                    function(*args, figure_height=height)
            self.assertEqual(saved, ["standard", "contraction"])

    def test_real_decorated_figure_pins_style_and_restores_custom_settings(self):
        try:
            plot_rendering._find_tex_tools()
        except RuntimeError as error:
            self.skipTest(str(error))
        custom_style = {
            "font.size": 7,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 6,
            "ytick.labelsize": 5,
            "legend.fontsize": 4,
            "axes.grid": False,
            "axes.grid.axis": "y",
            "axes.grid.which": "minor",
            "axes.axisbelow": False,
            "grid.color": "red",
            "grid.linewidth": 2,
            "grid.alpha": 0.8,
            "grid.linestyle": ":",
        }
        for fail in (False, True):
            with self.subTest(fail=fail), mpl.rc_context(custom_style):
                before = dict(mpl.rcParams)
                previous_path = os.environ.get("PATH")

                @plot_rendering.latex_plot
                def render():
                    figure = Figure(figsize=(5, 3), dpi=80)
                    FigureCanvasAgg(figure)
                    axis = figure.subplots()
                    axis.plot([0, 1], [1, 2], label="Series")
                    axis.fill_between([0, 1], [0.5, 1.5], [1.5, 2.5])
                    axis.set(xlabel="Input", ylabel="Output")
                    axis.set_xticks([0.25, 0.75], minor=True)
                    axis.set_yticks([1.25, 1.75], minor=True)
                    axis.legend()
                    note = axis.text(0.5, 1, "Note")
                    output = BytesIO()
                    figure.savefig(output, format="png")
                    self.assert_axis_style(axis, xlabel_size=18)
                    self.assertEqual(note.get_fontsize(), 18)
                    self.assertEqual(axis.get_title(), "")
                    self.assertIsNone(figure._suptitle)
                    self.assertEqual(figure.dpi, 80)
                    self.assertTrue(output.getvalue().startswith(b"\x89PNG\r\n\x1a\n"))
                    if fail:
                        raise ValueError("render failed")

                if fail:
                    with self.assertRaisesRegex(ValueError, "render failed"):
                        render()
                else:
                    render()
                self.assertEqual(mpl.rcParams, before)
                self.assertEqual(os.environ.get("PATH"), previous_path)

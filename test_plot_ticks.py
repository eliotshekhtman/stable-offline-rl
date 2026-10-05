"""Observed-value ticks must preserve geometry and remain legible."""

from io import BytesIO
import unittest

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np

from plot_rendering import latex_plot, set_composition_xticks, set_data_xticks


class PlotTickTests(unittest.TestCase):
    @latex_plot
    def test_close_observed_values_keep_minor_ticks_without_overlapping_labels(self):
        figure = Figure(figsize=(6, 4))
        FigureCanvasAgg(figure)
        axis = figure.subplots()
        values = np.array([0, 0.3001, 0.3002, 0.7, 0.9])
        line, = axis.plot(values, np.arange(len(values)))
        axis.axhline(0.5)
        limits = axis.get_xlim()
        figure.tight_layout()
        set_data_xticks(axis, values)
        figure.tight_layout()
        figure.canvas.draw()
        major = axis.get_xticks()
        minor = axis.get_xticks(minor=True)
        self.assertLess(len(major), len(values))
        np.testing.assert_array_equal(np.sort(np.concatenate((major, minor))), values)
        self.assertNotIn(1, major)
        np.testing.assert_array_equal(axis.get_xlim(), limits)
        np.testing.assert_array_equal(line.get_xdata(), values)
        np.testing.assert_array_equal(line.get_ydata(), np.arange(len(values)))
        self.assertTrue(all(not text.get_text() for text in axis.get_xticklabels(minor=True)))
        boxes = [text.get_window_extent(figure.canvas.get_renderer()) for text in axis.get_xticklabels()]
        self.assertTrue(all(first.x1 < second.x0 for first, second in zip(boxes, boxes[1:])))
        self.assertTrue(all(not tick.gridline.get_visible() for tick in axis.xaxis.get_minor_ticks()))

    @latex_plot
    def test_dense_timesteps_are_sparse_integer_ticks_without_dropping_curve_points(self):
        figure = Figure(figsize=(5, 4))
        FigureCanvasAgg(figure)
        axis = figure.subplots()
        curves = [np.arange(151), np.arange(301)]
        for values in curves:
            axis.plot(values, np.exp(-values / 50))
        limits = axis.get_xlim()
        set_data_xticks(axis, np.concatenate(curves), dense=True)
        figure.tight_layout()
        figure.savefig(BytesIO(), format="png")
        ticks = axis.get_xticks()
        self.assertGreaterEqual(len(ticks), 2)
        self.assertLessEqual(len(ticks), 8)
        self.assertEqual(ticks[0], 0)
        self.assertEqual(ticks[-1], 300)
        self.assertTrue(np.isin(ticks, curves[-1]).all())
        self.assertEqual(len(axis.get_xticks(minor=True)), 0)
        np.testing.assert_array_equal(axis.get_xlim(), limits)
        for line, values in zip(axis.lines, curves):
            np.testing.assert_array_equal(line.get_xdata(), values)
            np.testing.assert_array_equal(line.get_ydata(), np.exp(-values / 50))

    @latex_plot
    def test_small_distinct_values_have_distinct_labels_and_composition_clears_minors(self):
        figure = Figure(figsize=(8, 4))
        FigureCanvasAgg(figure)
        axis = figure.subplots()
        values = [0.30000001, 0.30000002]
        axis.plot(values, [0, 1])
        set_data_xticks(axis, values)
        self.assertEqual(len(set(text.get_text() for text in axis.get_xticklabels())), 2)
        np.testing.assert_array_equal(axis.get_xticks(), values)
        axis.set_xlim(-0.02, 1.02)
        axis.set_xticks([0.1, 0.6], minor=True)
        set_composition_xticks(axis)
        np.testing.assert_array_equal(axis.get_xticks(), [0, 0.25, 0.5, 0.75, 1])
        self.assertEqual(len(axis.get_xticks(minor=True)), 0)
        np.testing.assert_array_equal(axis.get_xlim(), [-0.02, 1.02])


if __name__ == "__main__":
    unittest.main()

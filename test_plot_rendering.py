from io import BytesIO
import importlib
import os
from pathlib import Path
import re
import struct
import tempfile
import unittest
from unittest.mock import patch

import matplotlib as mpl
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_pdf import FigureCanvasPdf
from matplotlib.figure import Figure
from matplotlib.image import imread
from matplotlib.text import Text
from matplotlib.transforms import Bbox
import numpy as np

import plot_rendering


class LatexRenderingTests(unittest.TestCase):
    toolchain = ("/tex/bin/latex", "/tex/bin/dvipng")

    def test_import_does_not_probe_tools_or_change_style_and_path(self):
        before = dict(mpl.rcParams)
        previous_path = os.environ.get("PATH")
        with patch.object(plot_rendering.shutil, "which") as which:
            importlib.reload(plot_rendering)
            which.assert_not_called()
        self.assertEqual(mpl.rcParams, before)
        self.assertEqual(os.environ.get("PATH"), previous_path)

    def test_style_and_path_restored_after_success_and_render_error(self):
        for fail in (False, True):
            with self.subTest(fail=fail), mpl.rc_context({
                "text.usetex": False, "font.family": "sans-serif",
            }), patch.dict(os.environ, {"PATH": "/before"}), patch.object(
                plot_rendering, "_find_tex_tools", return_value=self.toolchain,
            ), patch.object(plot_rendering, "_preflight") as preflight:
                before = dict(mpl.rcParams)

                @plot_rendering.latex_plot
                def render():
                    self.assertTrue(mpl.rcParams["text.usetex"])
                    self.assertEqual(mpl.rcParams["font.family"], ["monospace"])
                    self.assertEqual(mpl.rcParams["font.monospace"], ["Computer Modern Typewriter"])
                    self.assertEqual(os.environ["PATH"], "/tex/bin:/before")
                    if fail:
                        raise ValueError("render failed")
                    return "rendered"

                if fail:
                    with self.assertRaisesRegex(ValueError, "render failed"):
                        render()
                else:
                    self.assertEqual(render(), "rendered")
                self.assertEqual(render.__name__, "render")
                preflight.assert_called_once_with(self.toolchain)
                self.assertEqual(mpl.rcParams, before)
                self.assertEqual(os.environ["PATH"], "/before")

    def test_preflight_error_restores_style_and_absent_path(self):
        with patch.dict(os.environ), patch.object(
            plot_rendering, "_find_tex_tools", return_value=self.toolchain,
        ), patch.object(plot_rendering, "_preflight", side_effect=RuntimeError("preflight failed")):
            os.environ.pop("PATH", None)
            before = dict(mpl.rcParams)

            @plot_rendering.latex_plot
            def render():
                self.fail("Rendering must wait for a successful preflight")

            with self.assertRaisesRegex(RuntimeError, "preflight failed"):
                render()
            self.assertNotIn("PATH", os.environ)
            self.assertEqual(mpl.rcParams, before)

    def test_preflight_cached_only_after_success_and_per_toolchain(self):
        for backend in ("png", "pdf"):
            with self.subTest(backend=backend), patch.object(
                plot_rendering, "_VALIDATED_TOOLCHAINS", set(),
            ), patch.object(
                FigureCanvasAgg, "print_png",
                side_effect=[OSError("missing.sty"), None, None] if backend == "png" else None,
            ) as print_png, patch.object(
                FigureCanvasPdf, "print_pdf",
                side_effect=[OSError("missing.sty"), None, None] if backend == "pdf" else None,
            ) as print_pdf, patch.object(Figure, "savefig") as savefig:
                with self.assertRaisesRegex(
                    RuntimeError, "(?:PNG/PDF|PNG|PDF) preflight.*cm-super.*\nOriginal error: missing.sty",
                ):
                    plot_rendering._preflight(self.toolchain)
                self.assertNotIn(self.toolchain, plot_rendering._VALIDATED_TOOLCHAINS)
                plot_rendering._preflight(self.toolchain)
                self.assertIn(self.toolchain, plot_rendering._VALIDATED_TOOLCHAINS)
                plot_rendering._preflight(self.toolchain)
                plot_rendering._preflight(("/other/latex", "/other/dvipng"))
                self.assertEqual(print_png.call_count, 3)
                self.assertEqual(print_pdf.call_count, 2 if backend == "png" else 3)
                for call in print_png.call_args_list + print_pdf.call_args_list:
                    self.assertIsInstance(call.args[0], BytesIO)
                savefig.assert_not_called()

    def test_local_toolchain_discovered_without_changing_path(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            Path, "home", return_value=Path(directory),
        ), patch.dict(os.environ, {"PATH": "/no-tex-tools"}):
            binaries = Path(directory) / ".local/texlive/bin/test-platform"
            binaries.mkdir(parents=True)
            for name in ("latex", "dvipng"):
                executable = binaries / name
                executable.touch()
                executable.chmod(0o755)
            self.assertEqual(plot_rendering._find_tex_tools(), (
                str(binaries / "latex"), str(binaries / "dvipng"),
            ))
            self.assertEqual(os.environ["PATH"], "/no-tex-tools")

    def test_missing_tools_explain_local_installation(self):
        with patch.object(plot_rendering.shutil, "which", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "latex and dvipng.*~/.local/texlive"):
                plot_rendering._find_tex_tools()

    def test_escape_plain_text_in_one_pass(self):
        self.assertEqual(
            plot_rendering.tex_text(r"a_b 50% & # $ {x} \ ~ ^"),
            r"a\_b 50\% \& \# \$ \{x\} \textbackslash{} \textasciitilde{} \textasciicircum{}",
        )
        self.assertEqual(plot_rendering.tex_text("MOPO (recursive dynamics)"), "MOPO (recursive dynamics)")

    def test_real_png_and_pdf_with_special_text_negative_ticks_and_exponent(self):
        try:
            plot_rendering._find_tex_tools()
        except RuntimeError as error:
            self.skipTest(str(error))
        previous_path = os.environ.get("PATH")
        before = dict(mpl.rcParams)

        @plot_rendering.latex_plot
        def render():
            figure = Figure(figsize=(5, 3))
            FigureCanvasAgg(figure)
            axis = figure.subplots()
            axis.plot([1, 2, 4], [-2, 0, 2])
            axis.set_xticks([1, 2, 4], [r"$2^{0}$", r"$2^{1}$", r"$2^{2}$"])
            axis.set_yticks([-2, 0, 2])
            axis.set_title(plot_rendering.tex_text(r"a_b 50% & # $ {x} \ ~ ^"))
            png_output, pdf_output = BytesIO(), BytesIO()
            figure.savefig(png_output, format="png")
            figure.savefig(pdf_output, format="pdf")
            self.assertTrue(axis.get_yticklabels()[0].get_usetex())
            self.assertEqual(axis.get_xticklabels()[1].get_text(), r"$2^{1}$")
            return png_output.getvalue(), pdf_output.getvalue()

        png, pdf = render()
        self.assertTrue(png.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertGreater(len(png), 1000)
        self.assertTrue(pdf.startswith(b"%PDF-"))
        self.assertGreater(len(pdf), 1000)
        self.assertIn(b"/Font", pdf)
        self.assertNotIn(b"/Subtype /Image", pdf)
        self.assertEqual(os.environ.get("PATH"), previous_path)
        self.assertEqual(mpl.rcParams, before)


class SavePlotTests(unittest.TestCase):
    def test_save_plot_writes_png_and_vector_pdf_without_changing_figure(self):
        with tempfile.TemporaryDirectory() as directory, mpl.rc_context({
            "text.usetex": False, "savefig.bbox": None,
        }):
            figure = Figure(figsize=(8, 4.25), dpi=80)
            canvas = FigureCanvasAgg(figure)
            axis = figure.subplots()
            line, = axis.plot([1, 2, 4], [-2, 0, 2], label="Series")
            axis.set(xlabel="Input", ylabel="Output", xlim=(0, 5), ylim=(-3, 3))
            axis.legend()
            canvas.draw()
            original_position = axis.get_position().bounds
            original_data = line.get_xydata().copy()
            original_coordinates = axis.transData.transform(original_data).copy()
            path = Path(directory) / "plot.with.dots.png"

            plot_rendering.save_plot(figure, path)

            self.assertEqual({item.name for item in path.parent.iterdir()}, {
                "plot.with.dots.png", "plot.with.dots.pdf",
            })
            png = path.read_bytes()
            pdf = path.with_suffix(".pdf").read_bytes()
            self.assertTrue(png.startswith(b"\x89PNG\r\n\x1a\n"))
            png_size = struct.unpack(">II", png[16:24])
            self.assertLess(png_size[0], 1600)
            self.assertLess(png_size[1], 850)
            self.assertGreater(png_size[0], 1000)
            self.assertGreater(png_size[1], 500)
            self.assertTrue(pdf.startswith(b"%PDF-"))
            media_box = re.search(rb"/MediaBox\s*\[\s*0\s+0\s+([\d.]+)\s+([\d.]+)\s*\]", pdf)
            self.assertIsNotNone(media_box)
            pdf_size = tuple(map(float, media_box.groups()))
            self.assertLess(pdf_size[0], 576)
            self.assertLess(pdf_size[1], 306)
            np.testing.assert_allclose(pdf_size, np.asarray(png_size) * 72 / 200, atol=2)
            self.assertIn(b"/Font", pdf)
            self.assertNotIn(b"/Subtype /Image", pdf)
            np.testing.assert_array_equal(figure.get_size_inches(), (8, 4.25))
            self.assertEqual(figure.dpi, 80)
            self.assertIs(figure.canvas, canvas)
            self.assertEqual(axis.get_position().bounds, original_position)
            self.assertEqual(axis.get_xlim(), (0, 5))
            self.assertEqual(axis.get_ylim(), (-3, 3))
            self.assertEqual(axis.get_xlabel(), "Input")
            self.assertEqual(axis.get_ylabel(), "Output")
            self.assertEqual(axis.get_legend().get_texts()[0].get_text(), "Series")
            np.testing.assert_array_equal(line.get_xydata(), original_data)
            np.testing.assert_array_equal(axis.transData.transform(original_data), original_coordinates)

    def test_tight_export_preserves_labels_and_legend_without_outer_padding(self):
        with tempfile.TemporaryDirectory() as directory, mpl.rc_context({"text.usetex": False}):
            figure = Figure(figsize=(5, 3), dpi=200)
            canvas = FigureCanvasAgg(figure)
            axis = figure.subplots()
            figure.subplots_adjust(left=0.25, right=0.75, bottom=0.3, top=0.8)
            axis.plot([0, 1, 2], [0, 1, 2], label="Series")
            axis.set_xlabel("An intentionally oversized horizontal axis label for export",
                            fontsize=plot_rendering.X_LABEL_FONTSIZE)
            axis.set_ylabel("Normalized evaluation return for the expert policy", fontsize=18)
            axis.set_xticks([0, 1, 2])
            axis.set_yticks([0, 1, 2])
            legend = axis.legend(loc="center left", bbox_to_anchor=(3.0, 0.5))
            canvas.draw()
            renderer = canvas.get_renderer()
            xlabel_bounds = axis.xaxis.label.get_window_extent(renderer)
            ylabel_bounds = axis.yaxis.label.get_window_extent(renderer)
            self.assertGreater(xlabel_bounds.width, figure.bbox.width)
            self.assertGreater(ylabel_bounds.height, figure.bbox.height)
            self.assertGreater(legend.get_window_extent(renderer).x1, xlabel_bounds.x1)
            crop = Bbox.union([
                figure.get_tightbbox(renderer).transformed(figure.dpi_scale_trans),
                xlabel_bounds, ylabel_bounds,
            ])
            label_bounds = [artist.get_window_extent(renderer).extents.copy() for artist in [
                axis.xaxis.label, axis.yaxis.label, legend,
                *axis.get_xticklabels(), *axis.get_yticklabels(),
            ]]
            path = Path(directory) / "cropped.png"

            plot_rendering.save_plot(figure, path)

            pixels = imread(path)
            height, width = pixels.shape[:2]
            self.assertEqual((width, height), (int(crop.width), int(crop.height)))
            pdf = path.with_suffix(".pdf").read_bytes()
            media_box = re.search(rb"/MediaBox\s*\[\s*0\s+0\s+([\d.]+)\s+([\d.]+)\s*\]", pdf)
            self.assertIsNotNone(media_box)
            np.testing.assert_allclose(tuple(map(float, media_box.groups())),
                                       np.asarray((width, height)) * 72 / 200, atol=2)
            for left, bottom, right, top in label_bounds:
                self.assertGreaterEqual(left - crop.x0, 0)
                self.assertGreaterEqual(bottom - crop.y0, 0)
                self.assertLess(right - crop.x0, width + 1)
                self.assertLess(top - crop.y0, height + 1)
            ink_rows, ink_columns = np.nonzero(np.any(pixels[:, :, :3] < 0.95, axis=2))
            margins = (ink_columns.min(), width - 1 - ink_columns.max(),
                       ink_rows.min(), height - 1 - ink_rows.max())
            for margin in margins:
                self.assertGreaterEqual(margin, 0)
                self.assertLess(margin, 25)


class PlotAdapterRenderingTests(unittest.TestCase):
    def test_all_plot_adapters_create_latex_text_and_restore_settings(self):
        try:
            plot_rendering._find_tex_tools()
        except RuntimeError as error:
            self.skipTest(str(error))
        import plot
        import walker_composition_plot

        plain_label = r"Trial_50% & # $ {x} \ ~ ^"
        metric = "Reward_50%"
        saved = {}
        savefig = Figure.savefig

        def save_and_inspect(figure, filename, **kwargs):
            savefig(figure, filename, **kwargs)
            self.assertEqual(mpl.rcParams["font.monospace"], ["Computer Modern Typewriter"])
            texts = [text for text in figure.findobj(Text) if text.get_visible() and text.get_text()]
            self.assertTrue(texts)
            for text in texts:
                self.assertTrue(text.get_usetex(), text.get_text())
                self.assertEqual(text.get_fontfamily(), ["monospace"], text.get_text())
            self.assertIsNone(figure._suptitle)
            suffix = Path(filename).suffix
            self.assertIn(suffix, {".png", ".pdf"})
            signature = b"\x89PNG\r\n\x1a\n" if suffix == ".png" else b"%PDF-"
            self.assertTrue(Path(filename).read_bytes().startswith(signature))
            if suffix == ".pdf":
                self.assertNotIn(b"/Subtype /Image", Path(filename).read_bytes())
            saved[Path(filename).name] = figure

        with tempfile.TemporaryDirectory() as directory, mpl.rc_context({
            "text.usetex": False, "font.family": "sans-serif", "font.monospace": ["DejaVu Sans Mono"],
        }), patch.object(Figure, "savefig", autospec=True, side_effect=save_and_inspect):
            root = Path(directory)
            (root / "rollouts").mkdir()
            np.savez(root / "returns_last.npz", policy_episode_performance=[-4.0, -2.0, 0.0])
            np.savez(root / "contraction_last.npz", distance_curves=[[1.0, 0.5], [0.8, 0.4]])
            records = [{
                "step": step, "actual_percent": step * 50,
                "state_ood_ratio": step * 0.5, "state_action_ood_ratio": step * 0.75,
            } for step in (1, 2)]
            for record in records:
                np.savez(root / "rollouts" / f"step_{record['step']}.npz", performance=[-4.0, -2.0, 0.0])
            rows = [{
                "algo": "iql", "chunk_length": chunk_length,
                "training_schema": {"algo": "iql", "chunk_length": chunk_length},
                "seed_rows": [{"eval_dir": str(root)}],
                "expert_performance_mean": 2.0, "performance_label": metric,
                "_plot_cohort_series": 0,
                "_plot_cohort_spec": {"algo": "iql", "label": plain_label, "match": {}},
            } for chunk_length in (1, 4)]
            histories = [{
                **rows[0], "label": plain_label, "records": records,
                "seed_histories": [{"eval_dir": str(root), "records": records}],
            }]
            summary = {
                "cohort": {"setting": "noise0.5", "ablation": {"values": [0, 0.5]}},
                "performance_label": metric,
                "series": [{"label": plain_label, "points": [{
                    "actual_fraction": fraction, "mean": -2.0, "low": -3.0,
                    "high": -1.0, "expert_mean": 2.0,
                    "history": [{"epoch": epoch, "mean": -2.0, "low": -3.0, "high": -1.0}
                                for epoch in (1, 2)],
                } for fraction in (0, 0.5)]}],
            }
            before = dict(mpl.rcParams)
            previous_path = os.environ.get("PATH")
            calls = [
                (plot.plot_performance_vs_chunk_length, (rows, root)),
                (plot.performance_ablation_plot, (rows, root / "ablation.png", "chunk_length", "Chunk_50%", False)),
                (plot.plot_contraction_vs_chunk_length, (rows, root)),
                (plot.performance_history_plot, (histories, root / "history.png")),
                (plot.history_line_plot, (histories, ("state_ood_ratio", "state_action_ood_ratio"),
                                          ("state OOD", "state-action OOD"), root / "ood.png")),
                (walker_composition_plot.render, (summary, root)),
            ]
            for function, args in calls:
                with self.subTest(adapter=function.__name__):
                    function(*args)
                    self.assertEqual(mpl.rcParams, before)
                    self.assertEqual(os.environ.get("PATH"), previous_path)

            expected_stems = {
                "performance_vs_chunk_length", "ablation", "contraction_vs_chunk_length",
                "history", "ood", "performance_vs_composition",
                "performance_history_fraction_0", "performance_history_fraction_0.5",
            }
            self.assertEqual(set(saved), {
                stem + suffix for stem in expected_stems for suffix in (".png", ".pdf")
            })
            for stem in expected_stems:
                self.assertIs(saved[stem + ".png"], saved[stem + ".pdf"])
            for name, figure in saved.items():
                stem = Path(name).stem
                for axis in figure.axes:
                    if stem == "contraction_vs_chunk_length":
                        self.assertEqual(axis.get_title(), plot_rendering.tex_text(plain_label))
                    else:
                        self.assertEqual(axis.get_title(), "")
                if stem not in {"ood", "contraction_vs_chunk_length"}:
                    self.assertEqual(figure.axes[0].get_ylabel(), plot_rendering.tex_text(metric))
            chunk_axis = saved["performance_vs_chunk_length.png"].axes[0]
            self.assertEqual([tick.get_text() for tick in chunk_axis.get_xticklabels()], [r"$2^{0}$", r"$2^{2}$"])
            self.assertTrue(any("-" in tick.get_text() or "−" in tick.get_text() for tick in chunk_axis.get_yticklabels()))
            self.assertEqual(saved["ablation.png"].axes[0].get_xlabel(), r"Chunk\_50\%")
            self.assertEqual(saved["history.png"].axes[0].get_xlabel(), r"Training completed (\%)")
            self.assertEqual(saved["ood.png"].axes[-1].get_xlabel(), r"Training completed (\%)")
            self.assertEqual(saved["contraction_vs_chunk_length.png"].axes[0].get_ylabel(), "Distance (m)")
            self.assertEqual(saved["performance_vs_composition.png"].axes[0].get_xlabel(),
                             "Fraction of data from the noisy expert")


if __name__ == "__main__":
    unittest.main()

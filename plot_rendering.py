"""Scoped LaTeX rendering for the repository's plotting entry points."""

from functools import wraps
from io import BytesIO
import os
from pathlib import Path
import shutil
from uuid import uuid4

import matplotlib as mpl
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_pdf import FigureCanvasPdf
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MaxNLocator, NullFormatter, NullLocator
import numpy as np


TEXT_SCALE = 1.8
SMALL_LEGEND_FONTSIZE = 8 * TEXT_SCALE
X_LABEL_FONTSIZE = 13.5 * TEXT_SCALE

_LATEX_STYLE = {
    "text.usetex": True,
    "font.family": "monospace",
    "font.serif": ["Computer Modern Roman"],
    "font.sans-serif": ["Computer Modern Sans Serif"],
    "font.monospace": ["Computer Modern Typewriter"],
    "font.size": 10 * TEXT_SCALE,
    "axes.labelsize": 10 * TEXT_SCALE,
    "axes.titlesize": 12 * TEXT_SCALE,
    "xtick.labelsize": 10 * TEXT_SCALE,
    "ytick.labelsize": 10 * TEXT_SCALE,
    "legend.fontsize": 10 * TEXT_SCALE,
    "axes.grid": True,
    "axes.grid.axis": "both",
    "axes.grid.which": "major",
    "axes.axisbelow": True,
    "grid.color": "#b0b0b0",
    "grid.linewidth": 0.6,
    "grid.alpha": 0.35,
    "grid.linestyle": "-",
    "text.latex.preamble": "",
}
_VALIDATED_TOOLCHAINS = set()
_INSTALL_HELP = (
    "Install TeX Live with latex, dvipng, type1cm, and cm-super, either on "
    "PATH or under ~/.local/texlive/bin/<platform>."
)


def validate_plot_options(cohort: dict) -> dict:
    """Extract optional presentation settings without changing cohort selection."""
    options = {}
    if "figure_height" in cohort:
        height = cohort["figure_height"]
        if type(height) not in (int, float) or not np.isfinite(height) or height <= 0:
            raise ValueError("Plot cohort 'figure_height' must be a finite positive number in inches")
        options["figure_height"] = height
    if "show_legend" in cohort:
        if type(cohort["show_legend"]) is not bool:
            raise ValueError("Plot cohort 'show_legend' must be a boolean")
        options["show_legend"] = cohort["show_legend"]
    return options


def _find_tex_tools() -> tuple[str, str]:
    latex, dvipng = shutil.which("latex"), shutil.which("dvipng")
    if latex and dvipng:
        return latex, dvipng
    for directory in sorted((Path.home() / ".local/texlive/bin").glob("*")):
        latex = shutil.which("latex", path=str(directory))
        dvipng = shutil.which("dvipng", path=str(directory))
        if latex and dvipng:
            return latex, dvipng
    raise RuntimeError("LaTeX plot rendering requires latex and dvipng. " + _INSTALL_HELP)


def _preflight(toolchain: tuple[str, str]) -> None:
    if toolchain in _VALIDATED_TOOLCHAINS:
        return
    figure = Figure(figsize=(1, 0.3), dpi=50)
    canvas = FigureCanvasAgg(figure)
    # Unique text prevents the disk cache from hiding a broken TeX installation.
    figure.text(0, 0, r"$-2^{3}$ LaTeX " + uuid4().hex, fontsize=6)
    try:
        canvas.print_png(BytesIO())
        FigureCanvasPdf(figure).print_pdf(BytesIO())
    except (RuntimeError, OSError) as error:
        raise RuntimeError(
            "LaTeX plot rendering failed its PNG/PDF preflight. " + _INSTALL_HELP
            + f"\nOriginal error: {error}"
        ) from error
    _VALIDATED_TOOLCHAINS.add(toolchain)


def latex_plot(function):
    """Render a complete plot using LaTeX, restoring settings on every exit."""
    @wraps(function)
    def render(*args, **kwargs):
        toolchain = _find_tex_tools()
        previous_path = os.environ.get("PATH")
        directories = list(dict.fromkeys(str(Path(tool).parent) for tool in toolchain))
        if previous_path:
            directories.append(previous_path)
        try:
            os.environ["PATH"] = os.pathsep.join(directories)
            with mpl.rc_context(_LATEX_STYLE):
                _preflight(toolchain)
                return function(*args, **kwargs)
        finally:
            if previous_path is None:
                os.environ.pop("PATH", None)
            else:
                os.environ["PATH"] = previous_path
    return render


def save_plot(figure: Figure, path: Path) -> None:
    """Save PNG and vector PDF cropped to visible content with no added padding."""
    # Layout-only bounds can ignore the full extent of long axis labels.
    extra_artists = figure.get_default_bbox_extra_artists() + [
        label for axis in figure.axes if axis.get_visible()
        for label in (axis.xaxis.label, axis.yaxis.label)
        if label.get_visible() and label.get_in_layout()
    ]
    save_kwargs = {"bbox_inches": "tight", "pad_inches": 0, "bbox_extra_artists": extra_artists}
    figure.savefig(path, dpi=200, **save_kwargs)
    figure.savefig(path.with_suffix(".pdf"), **save_kwargs)


def tex_text(plain: str) -> str:
    """Escape ordinary text; intentionally authored LaTeX must bypass this."""
    replacements = {
        "\\": r"\textbackslash{}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        **{character: "\\" + character for character in "%_&#${}"},
    }
    return "".join(replacements.get(character, character) for character in plain)


def legend_location(env_name: str | None) -> str:
    """Keep legend corners fixed by task, independent of plotted values."""
    return "upper right" if env_name in {"Reacher-v5", "Lift"} else "lower right"


def set_composition_xticks(axis) -> None:
    """Keep composition ticks independent of realized trajectory fractions."""
    axis.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    axis.xaxis.set_minor_locator(NullLocator())


def set_data_xticks(axis, values, *, dense=False) -> None:
    """Label observed X values that fit; never move data or change axis limits."""
    values = np.unique(np.asarray(values, dtype=float))
    values = values[np.isfinite(values)]
    if not len(values):
        return
    limits = axis.get_xlim()
    candidates = values
    if dense and len(values) > 8:
        suggested = MaxNLocator(nbins=6, integer=True).tick_values(values[0], values[-1])
        candidates = np.unique(np.concatenate((
            values[[0, -1]], values[np.isin(values, suggested)],
        )))
    # Distinct labeled coordinates must not acquire identical rounded labels.
    for precision in range(6, 18):
        labels = [f"{value:.{precision}g}" for value in candidates]
        if len(set(labels)) == len(labels):
            break
    renderer = axis.figure.canvas.get_renderer()
    font = FontProperties(size=mpl.rcParams["xtick.labelsize"])
    widths = np.asarray([
        renderer.get_text_width_height_descent(
            label, font, ismath="TeX" if mpl.rcParams["text.usetex"] else False,
        )[0]
        for label in labels
    ])
    pixels = axis.get_xaxis_transform().transform(
        np.column_stack((candidates, np.zeros(len(candidates))))
    )[:, 0]
    gap = 6 * axis.figure.dpi / 72
    left, right = pixels - widths / 2, pixels + widths / 2
    selected = [0]
    last = len(candidates) - 1
    keep_last = last > 0 and left[last] >= right[0] + gap
    boundary = left[last] - gap if keep_last else float("inf")
    for index in range(1, last):
        if left[index] >= right[selected[-1]] + gap and right[index] <= boundary:
            selected.append(index)
    if keep_last:
        selected.append(last)
    major = candidates[selected]
    axis.set_xticks(major, [labels[index] for index in selected])
    if dense:
        axis.xaxis.set_minor_locator(NullLocator())
    else:
        axis.set_xticks(values[~np.isin(values, major)], minor=True)
        axis.xaxis.set_minor_formatter(NullFormatter())
    axis.set_xlim(limits, auto=None)

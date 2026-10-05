"""Display-only MOBILE names shared by the ordinary and reference plotters."""

import math


def real_ratio_label(value: float) -> str:
    # Keep the usual two decimals without rounding distinct ratios together.
    rendered = f"{value:.2f}"
    if float(rendered) != value:
        rendered = str(value)
    return f"real ratio={rendered}"


def mobile_names(algorithms: list[str], ratios: list[float | None]) -> list[str | None]:
    if len(algorithms) != len(ratios):
        raise ValueError("Expected one real ratio per plotted series")
    values = set()
    for algo, ratio in zip(algorithms, ratios):
        if algo != "mobile" or ratio is None:
            continue
        if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or not math.isfinite(ratio) or not 0 <= ratio <= 1:
            raise ValueError(f"Invalid MOBILE real ratio: {ratio!r}")
        values.add(ratio)
    ratio_only = set(algorithms) == {"mobile"} and len(values) > 1
    positive = {ratio for ratio in values if ratio > 0}
    hybrid_comparison = not ratio_only and 0 in values and bool(positive)
    names = []
    for algo, ratio in zip(algorithms, ratios):
        name = None
        if algo == "mobile" and ratio is not None:
            if ratio_only:
                name = real_ratio_label(ratio)
            elif hybrid_comparison:
                name = "MB-MOBILE" if ratio == 0 else "Hybrid-MOBILE"
                if ratio > 0 and len(positive) > 1:
                    name += f" ({real_ratio_label(ratio)})"
        names.append(name)
    return names

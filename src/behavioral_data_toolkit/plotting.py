"""Optional Matplotlib helpers (install with ``pip install behavioral-data-toolkit[plot]``).

Matplotlib is imported lazily, so importing this module never requires it. The
functions use the object-oriented API and do not touch global pyplot state, so
they are safe in headless environments and scripts.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._checks import apply_missing_policy, as_numeric_series
from .correlation import CorrelationMatrix
from .exceptions import InputValidationError, StatisticsError

__all__ = ["plot_correlation_heatmap", "plot_distribution"]


def _new_axes(figsize: tuple[float, float]) -> Any:
    try:
        from matplotlib.figure import Figure
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "Plotting needs Matplotlib: pip install 'behavioral-data-toolkit[plot]'"
        ) from exc
    return Figure(figsize=figsize, layout="constrained").subplots()


def plot_distribution(
    data: Any,
    *,
    ax: Any = None,
    bins: int | str = "auto",
    title: str | None = None,
    missing: str = "omit",
) -> Any:
    """Histogram of a numeric variable with the mean and median marked.

    Args:
        data: Numeric 1-D data.
        ax: Existing Matplotlib Axes; a new figure is created if omitted.
        bins: Passed to ``Axes.hist``.
        title: Optional axes title.
        missing: ``"omit"`` or ``"raise"``.

    Returns:
        The Matplotlib Axes (``ax.figure.savefig(...)`` saves it).
    """
    clean, n_missing = apply_missing_policy(as_numeric_series(data, "data"), missing)
    if clean.empty:
        raise StatisticsError("No non-missing values to plot.")
    axes = ax if ax is not None else _new_axes((6.0, 4.0))
    values = clean.to_numpy(dtype=float)
    axes.hist(values, bins=bins, color="#4C72B0", edgecolor="white")
    axes.axvline(float(np.mean(values)), color="#C44E52", linestyle="--", label="mean")
    axes.axvline(float(np.median(values)), color="#55A868", linestyle=":", label="median")
    axes.set_xlabel(str(clean.name) if clean.name is not None else "value")
    axes.set_ylabel("count")
    axes.legend()
    suffix = f" (n={values.size}, {n_missing} missing)"
    axes.set_title((title or "Distribution") + suffix)
    return axes


def plot_correlation_heatmap(
    matrix: CorrelationMatrix,
    *,
    ax: Any = None,
    annotate: bool = True,
    title: str | None = None,
) -> Any:
    """Heatmap of a :class:`~behavioral_data_toolkit.correlation.CorrelationMatrix`.

    The colour scale is fixed to [-1, 1]. Undefined (NaN) cells are left blank.

    Returns:
        The Matplotlib Axes.
    """
    if not isinstance(matrix, CorrelationMatrix):
        raise InputValidationError("matrix must be a CorrelationMatrix.")
    axes = ax if ax is not None else _new_axes((6.0, 5.0))
    values = matrix.r.to_numpy(dtype=float)
    k = values.shape[0]
    image = axes.imshow(np.ma.masked_invalid(values), vmin=-1.0, vmax=1.0, cmap="coolwarm")
    axes.set_xticks(range(k), labels=list(matrix.r.columns), rotation=45, ha="right")
    axes.set_yticks(range(k), labels=list(matrix.r.index))
    if annotate:
        for i in range(k):
            for j in range(k):
                if np.isfinite(values[i, j]):
                    color = "white" if abs(values[i, j]) > 0.6 else "black"
                    axes.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", color=color)
    axes.figure.colorbar(image, ax=axes, label=f"{matrix.method.capitalize()} r")
    axes.set_title(title or f"{matrix.method.capitalize()} correlation matrix")
    return axes

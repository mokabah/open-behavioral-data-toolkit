"""Pearson and Spearman correlation, and correlation matrices.

Reminder for interpretation: correlation does not imply causation; p-values
reported here are **unadjusted** for multiple comparisons; with many pairs
(e.g. a correlation matrix) some small p-values are expected by chance alone.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sps

from ._checks import as_numeric_frame, as_numeric_series, check_choice, is_real_number
from ._serialize import frame_to_records, to_json_safe
from .exceptions import InputValidationError, StatisticsError

__all__ = ["CorrelationMatrix", "CorrelationResult", "correlation_matrix", "pearson", "spearman"]

_MIN_N = 3


@dataclass(frozen=True)
class CorrelationResult:
    """Result of a bivariate correlation.

    Attributes:
        method: ``"pearson"`` or ``"spearman"``.
        r: Correlation coefficient in [-1, 1].
        p_value: Two-sided p-value (unadjusted).
        n: Number of complete pairs used.
        n_excluded: Pairs dropped because at least one value was missing.
        ci_low, ci_high: Fisher-z confidence limits (Pearson only; ``None`` for
            Spearman, if ``n <= 3`` or if ``|r| == 1``).
        confidence: Confidence level of the interval (``None`` if no interval).
    """

    method: str
    r: float
    p_value: float
    n: int
    n_excluded: int
    ci_low: float | None
    ci_high: float | None
    confidence: float | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dictionary."""
        return to_json_safe(asdict(self))  # type: ignore[no-any-return]


def _paired(x: Any, y: Any, missing: str) -> tuple[np.ndarray, np.ndarray, int]:
    check_choice(missing, ("omit", "raise"), "missing")
    sx = as_numeric_series(x, "x")
    sy = as_numeric_series(y, "y")
    if len(sx) != len(sy):
        raise InputValidationError(
            f"x and y must have the same length (paired by position); got {len(sx)} and {len(sy)}."
        )
    ax = sx.to_numpy(dtype=float)
    ay = sy.to_numpy(dtype=float)
    keep = ~(np.isnan(ax) | np.isnan(ay))
    n_excluded = int((~keep).sum())
    if n_excluded and missing == "raise":
        raise InputValidationError(
            f"{n_excluded} pair(s) contain a missing value and missing='raise'."
        )
    return ax[keep], ay[keep], n_excluded


def _require_defined(ax: np.ndarray, ay: np.ndarray) -> None:
    if ax.size < _MIN_N:
        raise StatisticsError(f"At least {_MIN_N} complete pairs are required; found {ax.size}.")
    if float(ax.max()) == float(ax.min()) or float(ay.max()) == float(ay.min()):
        raise StatisticsError("Correlation is undefined when a variable is constant.")


def _fisher_ci(r: float, n: int, confidence: float) -> tuple[float, float] | None:
    if n <= 3 or abs(r) >= 1.0:
        return None
    z = math.atanh(r)
    half = float(sps.norm.ppf(1 - (1 - confidence) / 2)) / math.sqrt(n - 3)
    return math.tanh(z - half), math.tanh(z + half)


def pearson(
    x: Any, y: Any, *, missing: str = "omit", confidence: float = 0.95
) -> CorrelationResult:
    r"""Pearson product-moment correlation with a Fisher-z confidence interval.

    Mathematical basis:
        ``r = sum((x - mean_x)(y - mean_y)) / sqrt(sum((x-mean_x)^2) * sum((y-mean_y)^2))``.
        The p-value tests H0: rho = 0 using ``t = r*sqrt((n-2)/(1-r^2))`` with
        ``n-2`` degrees of freedom. The interval is
        ``tanh(atanh(r) +/- z_{1-alpha/2} / sqrt(n-3))``.

    Assumptions:
        Linear association; interval-level, roughly continuous variables;
        independent observations; for the p-value and interval, approximately
        bivariate normal data.

    Limitations:
        Sensitive to outliers and restricted range; says nothing about causation;
        the interval is approximate and omitted when ``n <= 3`` or ``|r| = 1``.

    Missing data:
        ``missing="omit"`` uses pairs where both values are present (and reports
        ``n_excluded``); ``"raise"`` errors if any pair is incomplete.

    Args:
        x, y: Numeric 1-D data of equal length, paired by position.
        missing: ``"omit"`` or ``"raise"``.
        confidence: Interval level in (0, 1).

    Raises:
        InputValidationError: Bad arguments, non-numeric data, unequal lengths.
        StatisticsError: Fewer than 3 complete pairs, or a constant variable.
    """
    if not is_real_number(confidence):
        raise InputValidationError("confidence must be a number in (0, 1).")
    if not 0.0 < confidence < 1.0:
        raise InputValidationError("confidence must be strictly between 0 and 1.")
    ax, ay, n_excluded = _paired(x, y, missing)
    _require_defined(ax, ay)
    res = sps.pearsonr(ax, ay)
    r = float(np.clip(res.statistic, -1.0, 1.0))
    ci = _fisher_ci(r, int(ax.size), float(confidence))
    return CorrelationResult(
        method="pearson",
        r=r,
        p_value=float(res.pvalue),
        n=int(ax.size),
        n_excluded=n_excluded,
        ci_low=ci[0] if ci else None,
        ci_high=ci[1] if ci else None,
        confidence=float(confidence) if ci else None,
    )


def spearman(x: Any, y: Any, *, missing: str = "omit") -> CorrelationResult:
    r"""Spearman rank correlation.

    Mathematical basis:
        Pearson's ``r`` computed on the ranks of ``x`` and ``y``; tied values
        receive their average rank. Without ties this equals
        ``1 - 6 * sum(d^2) / (n (n^2 - 1))`` with rank differences ``d``. The
        p-value uses the t approximation with ``n-2`` degrees of freedom (as in
        SciPy); it is approximate for small ``n``.

    Assumptions:
        Monotonic (not necessarily linear) association; at least ordinal data;
        independent observations.

    Limitations:
        Many ties reduce its resolution; no confidence interval is provided in
        this version; says nothing about causation.

    Missing data and errors: as for :func:`pearson`.
    """
    ax, ay, n_excluded = _paired(x, y, missing)
    _require_defined(ax, ay)
    res = sps.spearmanr(ax, ay)
    return CorrelationResult(
        method="spearman",
        r=float(np.clip(res.statistic, -1.0, 1.0)),
        p_value=float(res.pvalue),
        n=int(ax.size),
        n_excluded=n_excluded,
        ci_low=None,
        ci_high=None,
        confidence=None,
    )


@dataclass(frozen=True, eq=False)
class CorrelationMatrix:
    """Correlation matrix with matching p-value and sample-size matrices.

    Entries are NaN where a correlation is undefined (fewer than 3 complete
    pairs, or a constant variable). The diagonal of ``p_values`` is NaN.

    Attributes:
        method: ``"pearson"`` or ``"spearman"``.
        missing: Missing-data policy used.
        r: Coefficients (symmetric DataFrame).
        p_values: Two-sided unadjusted p-values.
        n: Complete pairs used for each cell (diagonal: non-missing count).
    """

    method: str
    missing: str
    r: pd.DataFrame
    p_values: pd.DataFrame
    n: pd.DataFrame

    def to_long(self) -> pd.DataFrame:
        """Upper-triangle pairs as a tidy table: ``var1, var2, r, p_value, n``."""
        names = list(self.r.columns)
        rows = [
            {
                "var1": names[i],
                "var2": names[j],
                "r": self.r.iloc[i, j],
                "p_value": self.p_values.iloc[i, j],
                "n": int(self.n.iloc[i, j]),
            }
            for i in range(len(names))
            for j in range(i + 1, len(names))
        ]
        return pd.DataFrame(rows, columns=["var1", "var2", "r", "p_value", "n"])

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dictionary."""
        return {
            "method": self.method,
            "missing": self.missing,
            "r": frame_to_records(self.r),
            "p_values": frame_to_records(self.p_values),
            "n": frame_to_records(self.n),
        }


def correlation_matrix(
    data: pd.DataFrame,
    columns: list[str] | None = None,
    *,
    method: str = "pearson",
    missing: str = "pairwise",
) -> CorrelationMatrix:
    """Correlation matrix of numeric columns, with p-values and pair counts.

    Missing data (explicit policy):
        * ``"pairwise"`` (default): each cell uses the rows complete for *that
          pair*; ``n`` can differ between cells. The resulting matrix is not
          guaranteed to be positive semi-definite.
        * ``"listwise"``: only rows complete on *all* selected columns are used.
        * ``"raise"``: error if any value is missing.

    Assumptions and limitations: see :func:`pearson` / :func:`spearman`. In
    addition, p-values are **not** corrected for multiple comparisons.

    Args:
        data: DataFrame.
        columns: Columns to include (default: all; all must be numeric).
        method: ``"pearson"`` or ``"spearman"``.
        missing: ``"pairwise"``, ``"listwise"`` or ``"raise"``.

    Raises:
        InputValidationError: Bad arguments or fewer than 2 columns.
        ColumnError: Columns missing or non-numeric.
    """
    check_choice(method, ("pearson", "spearman"), "method")
    check_choice(missing, ("pairwise", "listwise", "raise"), "missing")
    frame = as_numeric_frame(data, columns, "data")
    names = list(frame.columns)
    k = len(names)
    if k < 2:
        raise InputValidationError("A correlation matrix needs at least 2 columns.")
    n_missing = int(frame.isna().sum().sum())
    if n_missing and missing == "raise":
        raise InputValidationError(
            f"Data contain {n_missing} missing value(s) and missing='raise'."
        )
    if missing == "listwise":
        frame = frame.dropna()
    values = frame.to_numpy(dtype=float)
    func = sps.pearsonr if method == "pearson" else sps.spearmanr

    r = np.full((k, k), np.nan)
    p = np.full((k, k), np.nan)
    n = np.zeros((k, k), dtype=int)

    for i in range(k):
        col = values[:, i]
        valid = ~np.isnan(col)
        n[i, i] = int(valid.sum())
        if n[i, i] >= _MIN_N and float(col[valid].max()) != float(col[valid].min()):
            r[i, i] = 1.0
        for j in range(i + 1, k):
            keep = ~(np.isnan(values[:, i]) | np.isnan(values[:, j]))
            ax, ay = values[keep, i], values[keep, j]
            n[i, j] = n[j, i] = int(ax.size)
            if ax.size < _MIN_N or ax.max() == ax.min() or ay.max() == ay.min():
                continue
            res = func(ax, ay)
            r[i, j] = r[j, i] = float(np.clip(res.statistic, -1.0, 1.0))
            p[i, j] = p[j, i] = float(res.pvalue)

    index = pd.Index(names, name="variable")
    return CorrelationMatrix(
        method=method,
        missing=missing,
        r=pd.DataFrame(r, index=index, columns=index),
        p_values=pd.DataFrame(p, index=index, columns=index),
        n=pd.DataFrame(n, index=index, columns=index),
    )

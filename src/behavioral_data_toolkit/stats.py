"""Descriptive statistics and Cronbach's alpha.

All functions accept numeric 1-D data (lists, arrays, Series) and make the
missing-data policy explicit through a ``missing`` argument. Non-numeric input
raises an error; nothing is coerced or imputed.
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sps

from ._checks import (
    apply_missing_policy,
    as_numeric_frame,
    as_numeric_series,
    check_choice,
)
from ._serialize import to_json_safe
from .exceptions import InputValidationError, StatisticsError

__all__ = [
    "CronbachAlphaResult",
    "DistributionSummary",
    "cronbach_alpha",
    "distribution_summary",
    "mean",
    "median",
    "std",
]


def _prepare(data: Any, missing: str, min_n: int, name: str = "data") -> tuple[np.ndarray, int]:
    series = as_numeric_series(data, name)
    clean, n_missing = apply_missing_policy(series, missing, name)
    if len(clean) < min_n:
        raise StatisticsError(
            f"At least {min_n} non-missing value(s) required; found {len(clean)}."
        )
    return clean.to_numpy(dtype=float), n_missing


def mean(data: Any, *, missing: str = "omit") -> float:
    """Arithmetic mean, ``sum(x) / n``.

    Assumptions/limitations: sensitive to outliers; meaningful for interval-level
    data (commonly applied to summed/averaged questionnaire scores).

    Args:
        data: Numeric 1-D data.
        missing: ``"omit"`` drops missing values; ``"raise"`` errors if any exist.

    Raises:
        StatisticsError: No non-missing values.
    """
    arr, _ = _prepare(data, missing, 1)
    return float(np.mean(arr))


def median(data: Any, *, missing: str = "omit") -> float:
    """Median (middle value; mean of the two middle values when ``n`` is even).

    Robust to outliers; appropriate for ordinal data.

    Raises:
        StatisticsError: No non-missing values.
    """
    arr, _ = _prepare(data, missing, 1)
    return float(np.median(arr))


def std(data: Any, *, ddof: int = 1, missing: str = "omit") -> float:
    """Standard deviation, ``sqrt(sum((x - mean)^2) / (n - ddof))``.

    Args:
        data: Numeric 1-D data.
        ddof: ``1`` (default) gives the *sample* standard deviation (unbiased
            variance estimator); ``0`` gives the population value.
        missing: ``"omit"`` or ``"raise"``.

    Raises:
        InputValidationError: ``ddof`` not 0 or 1.
        StatisticsError: Too few values (``n >= 2`` needed when ``ddof=1``).
    """
    if ddof not in (0, 1) or isinstance(ddof, bool):
        raise InputValidationError("ddof must be 0 or 1.")
    arr, _ = _prepare(data, missing, 1 + ddof)
    return float(np.std(arr, ddof=ddof))


def _none_if_nan(value: float) -> float | None:
    return None if np.isnan(value) else float(value)


@dataclass(frozen=True)
class DistributionSummary:
    """Descriptive summary of one variable.

    ``None`` means "undefined for these data" (e.g. ``std`` with n=1, or
    skewness for constant data).

    Attributes:
        n: Non-missing observations used.
        n_missing: Missing observations excluded.
        mean, std (sample, ddof=1), minimum, q1, median, q3, maximum, iqr.
        skewness: Adjusted Fisher-Pearson ``G1`` (needs n >= 3).
        excess_kurtosis: Bias-corrected excess kurtosis ``G2`` (needs n >= 4).
    """

    n: int
    n_missing: int
    mean: float
    std: float | None
    minimum: float
    q1: float
    median: float
    q3: float
    maximum: float
    iqr: float
    skewness: float | None
    excess_kurtosis: float | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dictionary."""
        return to_json_safe(asdict(self))  # type: ignore[no-any-return]


def distribution_summary(data: Any, *, missing: str = "omit") -> DistributionSummary:
    """Summarise a distribution: location, spread, quartiles, shape.

    Mathematical basis:
        * Quartiles use linear interpolation between order statistics (NumPy
          default, "type 7"); other software may use different definitions.
        * Skewness is the bias-adjusted Fisher-Pearson coefficient
          ``G1 = n / ((n-1)(n-2)) * sum(((x - mean)/s)^3)`` with sample SD ``s``.
        * Excess kurtosis is the bias-adjusted ``G2`` (0 for a normal
          distribution).

    Assumptions/limitations:
        Shape statistics are unstable in small samples and are not tests of
        normality. Missing values are handled per ``missing`` and counted in
        ``n_missing``.

    Raises:
        StatisticsError: No non-missing values.
    """
    arr, n_missing = _prepare(data, missing, 1)
    n = int(arr.size)
    q1, med, q3 = (float(v) for v in np.percentile(arr, [25, 50, 75]))
    sd = float(np.std(arr, ddof=1)) if n >= 2 else None
    constant = float(arr.max()) == float(arr.min())
    skew = None
    kurt = None
    if not constant:
        if n >= 3:
            skew = _none_if_nan(float(sps.skew(arr, bias=False)))
        if n >= 4:
            kurt = _none_if_nan(float(sps.kurtosis(arr, fisher=True, bias=False)))
    return DistributionSummary(
        n=n,
        n_missing=n_missing,
        mean=float(np.mean(arr)),
        std=sd,
        minimum=float(arr.min()),
        q1=q1,
        median=med,
        q3=q3,
        maximum=float(arr.max()),
        iqr=q3 - q1,
        skewness=skew,
        excess_kurtosis=kurt,
    )


@dataclass(frozen=True)
class CronbachAlphaResult:
    """Result of :func:`cronbach_alpha`.

    Attributes:
        alpha: Cronbach's alpha (raw, unstandardised).
        n_items: Number of items (k).
        n_observations: Rows used (after the missing-data policy).
        n_excluded: Rows excluded for missing responses.
        mean_inter_item_correlation: Mean off-diagonal Pearson correlation among
            items, or ``None`` if any item is constant. Reported because alpha
            alone confounds inter-item correlation with the number of items.
        missing: Policy used (``"listwise"`` or ``"raise"``).
    """

    alpha: float
    n_items: int
    n_observations: int
    n_excluded: int
    mean_inter_item_correlation: float | None
    missing: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dictionary."""
        return to_json_safe(asdict(self))  # type: ignore[no-any-return]


def cronbach_alpha(
    items: pd.DataFrame,
    columns: list[str] | None = None,
    *,
    missing: str = "listwise",
) -> CronbachAlphaResult:
    r"""Cronbach's alpha for a set of items.

    Mathematical basis:
        For ``k`` items with sample variances ``s_i^2`` and total-score variance
        ``s_T^2`` (variance of the row sums, ``ddof=1``)::

            alpha = k / (k - 1) * (1 - sum_i(s_i^2) / s_T^2)

        Equivalently, with the item covariance matrix ``C``:
        ``alpha = k/(k-1) * (1 - trace(C) / sum(C))``.

    Assumptions (all must be considered by the analyst):
        * Items are scored in the same direction (reverse-score first).
        * Essentially tau-equivalence: equal true-score loadings. If violated,
          alpha is a *lower bound* of reliability, not an estimate of it.
        * Uncorrelated item errors.
        * Items are treated as interval-level; for few-category ordinal items
          alpha can understate reliability.

    Limitations - how NOT to interpret alpha:
        * A high alpha is **not** evidence of unidimensionality, and does not
          establish construct validity. Many items, or redundant items, inflate
          alpha.
        * Alpha is a property of a *sample and a score*, not of the instrument.
        * It can be negative (typically from unreversed items or very weak
          correlations); a warning is emitted in that case.
        * Small samples give unstable estimates; no confidence interval is
          computed in this version.
        * No threshold (such as ".70 is acceptable") is applied here; such
          conventions are context-dependent and contested.

    Missing data:
        ``missing="listwise"`` (default) uses only rows complete on all items
        and reports how many rows were excluded; ``"raise"`` errors if any value
        is missing. Pairwise deletion is not offered because it can yield
        inconsistent covariance matrices.

    Args:
        items: DataFrame of numeric item responses (rows = respondents).
        columns: Item columns to use (default: all columns of ``items``).
        missing: ``"listwise"`` or ``"raise"``.

    Raises:
        InputValidationError: Fewer than 2 items, non-numeric input, bad policy.
        StatisticsError: Fewer than 2 complete rows, or zero variance of total score.
    """
    check_choice(missing, ("listwise", "raise"), "missing")
    frame = as_numeric_frame(items, columns, "items")
    k = frame.shape[1]
    if k < 2:
        raise InputValidationError("Cronbach's alpha needs at least 2 items.")
    complete = frame.dropna()
    n_excluded = int(frame.shape[0] - complete.shape[0])
    if n_excluded and missing == "raise":
        raise InputValidationError(
            f"{n_excluded} row(s) have missing responses and missing='raise'."
        )
    n = int(complete.shape[0])
    if n < 2:
        raise StatisticsError(f"At least 2 complete rows required; found {n}.")
    totals = complete.sum(axis=1)
    if float(totals.max()) == float(totals.min()):
        raise StatisticsError(
            "Total score has zero variance (identical sums in all rows); alpha is undefined."
        )
    item_var_sum = float(complete.var(axis=0, ddof=1).sum())
    total_var = float(totals.var(ddof=1))
    alpha = k / (k - 1) * (1.0 - item_var_sum / total_var)

    mean_r: float | None = None
    if float(complete.std(axis=0, ddof=1).min()) > 0.0:
        corr = np.corrcoef(complete.to_numpy(dtype=float), rowvar=False)
        mean_r = float(corr[np.triu_indices(k, k=1)].mean())

    if alpha < 0:
        warnings.warn(
            "Cronbach's alpha is negative. This usually indicates items scored in "
            "different directions (check reverse-keyed items) or negligible "
            "inter-item correlation; the value should not be interpreted as reliability.",
            UserWarning,
            stacklevel=2,
        )
    return CronbachAlphaResult(
        alpha=float(alpha),
        n_items=k,
        n_observations=n,
        n_excluded=n_excluded,
        mean_inter_item_correlation=mean_r,
        missing=missing,
    )

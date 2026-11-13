"""Reverse scoring, composite scores and questionnaire scoring.

This module performs arithmetic on item responses. It does **not** interpret
scores: no cut-offs, norms, severity bands or diagnostic labels are provided, and
none should be inferred from the output.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

import pandas as pd

from ._checks import (
    as_numeric_frame,
    as_numeric_series,
    check_choice,
    column_list,
    is_integer,
    is_real_number,
    require_dataframe,
)
from ._serialize import to_json_safe
from .exceptions import InputValidationError, ScoringError

__all__ = [
    "ScaleDefinition",
    "composite_score",
    "reverse_score",
    "score_questionnaire",
    "score_questionnaires",
]

_T = TypeVar("_T", pd.Series, pd.DataFrame)


def _check_scale_bounds(scale_min: float, scale_max: float) -> None:
    for label, value in (("scale_min", scale_min), ("scale_max", scale_max)):
        if not is_real_number(value):
            raise InputValidationError(f"{label} must be a number; got {value!r}.")
        if not math.isfinite(value):
            raise InputValidationError(f"{label} must be finite; got {value!r}.")
    if scale_min >= scale_max:
        raise InputValidationError(
            f"scale_min ({scale_min}) must be smaller than scale_max ({scale_max})."
        )


def _check_range(frame: pd.DataFrame, lo: float, hi: float) -> None:
    """Raise ScoringError if any non-missing value lies outside ``[lo, hi]``."""
    outside = ((frame < lo) | (frame > hi)).sum()
    bad = {str(c): int(n) for c, n in outside.items() if n > 0}
    if bad:
        raise ScoringError(
            f"Responses outside the scale range [{lo}, {hi}] (column: count): {bad}. "
            "Inspect with validator.find_invalid_values and clean the data first."
        )


def reverse_score(
    data: _T,
    scale_min: float,
    scale_max: float,
    *,
    columns: Sequence[str] | None = None,
) -> _T:
    """Reverse-score responses on a bounded scale.

    Mathematical basis:
        ``x' = scale_min + scale_max - x``. On a 1-5 scale, 1 <-> 5 and 2 <-> 4.
        This is correct for any scale whose points are equally spaced and
        bounded by ``scale_min`` and ``scale_max`` (including 0-based scales).

    Assumptions and limitations:
        * Responses must lie within ``[scale_min, scale_max]``; out-of-range
          values raise :class:`ScoringError` rather than being clipped.
        * Missing values stay missing.
        * The function does not decide *which* items are reverse-keyed; that is
          defined by the instrument's scoring manual.

    Args:
        data: A Series, or a DataFrame.
        scale_min: Lowest possible response.
        scale_max: Highest possible response.
        columns: For a DataFrame, the columns to reverse (default: all). Other
            columns are returned unchanged. Not allowed for a Series.

    Returns:
        A new object of the same type; the input is not modified. Reversed
        columns have dtype float64.
    """
    _check_scale_bounds(scale_min, scale_max)
    if isinstance(data, pd.Series):
        if columns is not None:
            raise InputValidationError("'columns' applies to DataFrames only.")
        series = as_numeric_series(data, "data")
        _check_range(series.to_frame(), scale_min, scale_max)
        return (scale_min + scale_max - series).rename(data.name)
    require_dataframe(data, "data")
    cols = column_list(columns)
    if cols is None:
        cols = list(data.columns)
    frame = as_numeric_frame(data, cols)
    _check_range(frame, scale_min, scale_max)
    out = data.copy()
    for col in cols:
        out[col] = scale_min + scale_max - frame[col]
    return out


def composite_score(
    data: pd.DataFrame,
    items: Sequence[str],
    *,
    method: str = "mean",
    min_valid: int | None = None,
    name: str = "composite",
) -> pd.Series:
    """Compute a per-row composite (mean or sum) over item columns.

    Mathematical basis:
        mean: ``(1/m) * sum(x_i)`` over the ``m`` answered items.
        sum: ``sum(x_i)`` over all ``k`` items.

    Missing-data handling (explicit):
        ``min_valid`` is the minimum number of answered items required for a row
        to receive a score; rows below it get NaN. The default (``None``) is
        ``k``, i.e. complete responses only. With ``method="mean"`` and
        ``min_valid < k`` the mean of the answered items is used, which assumes
        the unanswered items would have been answered similarly (check the
        instrument's manual before using partial scoring). Partial *sums* are
        not offered because they are biased downward; use ``method="mean"`` or
        prorate explicitly.

    Args:
        data: DataFrame of numeric item responses.
        items: Item column names.
        method: ``"mean"`` or ``"sum"``.
        min_valid: Minimum number of answered items (1..k).
        name: Name of the returned Series.

    Returns:
        Series aligned with ``data.index``.
    """
    check_choice(method, ("mean", "sum"), "method")
    cols = column_list(items, "items")
    if not cols:
        raise InputValidationError("items must contain at least one column name.")
    frame = as_numeric_frame(data, cols)
    k = len(cols)
    if min_valid is None:
        min_valid = k
    if not is_integer(min_valid) or not 1 <= min_valid <= k:
        raise InputValidationError(f"min_valid must be an integer between 1 and {k}.")
    min_valid = int(min_valid)
    if method == "sum" and min_valid < k:
        raise InputValidationError(
            "Partial sums are not supported (they are biased downward). "
            "Use method='mean' for partial scoring, or require all items."
        )
    answered = frame.notna().sum(axis=1)
    if method == "mean":
        result = frame.mean(axis=1, skipna=True)
    else:
        result = frame.sum(axis=1, skipna=False)
    result = result.where(answered >= min_valid)
    return result.rename(name)


@dataclass(frozen=True)
class ScaleDefinition:
    """Declarative definition of a questionnaire scale.

    Attributes:
        name: Name of the resulting score column.
        items: Item column names.
        min_value: Lowest possible item response.
        max_value: Highest possible item response.
        reverse_items: Items (subset of ``items``) to reverse-score first.
        method: ``"mean"`` or ``"sum"``.
        min_valid: Minimum answered items (see :func:`composite_score`).
    """

    name: str
    items: tuple[str, ...]
    min_value: float
    max_value: float
    reverse_items: tuple[str, ...] = ()
    method: str = "mean"
    min_valid: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise InputValidationError("ScaleDefinition.name must be a non-empty string.")
        items = tuple(self.items)
        reverse = tuple(self.reverse_items)
        if not items:
            raise InputValidationError("ScaleDefinition.items must not be empty.")
        if len(set(items)) != len(items):
            raise InputValidationError("ScaleDefinition.items contains duplicates.")
        if len(set(reverse)) != len(reverse):
            raise InputValidationError("ScaleDefinition.reverse_items contains duplicates.")
        unknown = [r for r in reverse if r not in items]
        if unknown:
            raise InputValidationError(f"reverse_items not among items: {unknown}.")
        _check_scale_bounds(self.min_value, self.max_value)
        check_choice(self.method, ("mean", "sum"), "method")
        object.__setattr__(self, "items", items)
        object.__setattr__(self, "reverse_items", reverse)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dictionary."""
        return to_json_safe(  # type: ignore[no-any-return]
            {
                "name": self.name,
                "items": list(self.items),
                "min_value": self.min_value,
                "max_value": self.max_value,
                "reverse_items": list(self.reverse_items),
                "method": self.method,
                "min_valid": self.min_valid,
            }
        )


def score_questionnaire(data: pd.DataFrame, scale: ScaleDefinition) -> pd.Series:
    """Score one questionnaire scale.

    Steps: (1) check items exist and are numeric; (2) require every non-missing
    response to lie within ``[min_value, max_value]``; (3) reverse-score the
    ``reverse_items``; (4) compute the composite with the scale's missing-data
    rule.

    Returns:
        Series named ``scale.name``.

    Raises:
        ScoringError: Responses outside the scale range.
        ColumnError: Items missing or non-numeric.
    """
    if not isinstance(scale, ScaleDefinition):
        raise InputValidationError("scale must be a ScaleDefinition.")
    frame = as_numeric_frame(data, list(scale.items))
    _check_range(frame, scale.min_value, scale.max_value)
    if scale.reverse_items:
        for col in scale.reverse_items:
            frame[col] = scale.min_value + scale.max_value - frame[col]
    return composite_score(
        frame,
        list(scale.items),
        method=scale.method,
        min_valid=scale.min_valid,
        name=scale.name,
    )


def score_questionnaires(data: pd.DataFrame, scales: Sequence[ScaleDefinition]) -> pd.DataFrame:
    """Score several scales and return them as columns of one DataFrame."""
    scale_list = list(scales)
    if not scale_list:
        raise InputValidationError("scales must contain at least one ScaleDefinition.")
    names = [s.name for s in scale_list if isinstance(s, ScaleDefinition)]
    if len(names) != len(scale_list):
        raise InputValidationError("Every element of scales must be a ScaleDefinition.")
    if len(set(names)) != len(names):
        raise InputValidationError("Scale names must be unique.")
    return pd.concat([score_questionnaire(data, s) for s in scale_list], axis=1)

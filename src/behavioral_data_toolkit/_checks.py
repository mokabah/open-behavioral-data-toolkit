"""Internal input-validation helpers (private API; may change without notice)."""

from __future__ import annotations

import numbers
from collections.abc import Collection, Sequence
from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

from .exceptions import ColumnError, InputValidationError


def is_real_number(value: Any) -> bool:
    """True for Python and NumPy real scalars; booleans are not numbers here."""
    return isinstance(value, numbers.Real) and not isinstance(value, (bool, np.bool_))


def is_integer(value: Any) -> bool:
    """True for Python and NumPy integer scalars; booleans are excluded."""
    return isinstance(value, numbers.Integral) and not isinstance(value, (bool, np.bool_))


def check_choice(value: str, allowed: Collection[str], name: str) -> str:
    """Return ``value`` if it is one of ``allowed``; otherwise raise."""
    if value not in allowed:
        options = ", ".join(repr(a) for a in allowed)
        raise InputValidationError(f"{name} must be one of {options}; got {value!r}.")
    return value


def require_dataframe(obj: Any, name: str = "data") -> pd.DataFrame:
    """Raise unless ``obj`` is a :class:`pandas.DataFrame`."""
    if not isinstance(obj, pd.DataFrame):
        raise InputValidationError(f"{name} must be a pandas DataFrame; got {type(obj).__name__}.")
    return obj


def column_list(columns: Sequence[str] | None, name: str = "columns") -> list[str] | None:
    """Validate a column-name list (a bare string is rejected on purpose)."""
    if columns is None:
        return None
    if isinstance(columns, str):
        raise InputValidationError(
            f"{name} must be a list of column names, not a single string "
            f"({columns!r}); wrap it in a list."
        )
    cols = list(columns)
    if len(set(cols)) != len(cols):
        raise InputValidationError(f"{name} contains duplicate entries.")
    return cols


def require_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    """Raise :class:`ColumnError` if any of ``columns`` is absent from ``df``."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ColumnError(f"Column(s) not found in data: {missing}.")


def as_numeric_series(data: Any, name: str = "data") -> pd.Series:
    """Convert 1-D input to a float Series, refusing non-numeric data.

    Strings are never coerced to numbers. Booleans are rejected. Infinite
    values are rejected. Missing values (NaN/None/NA) are kept.
    """
    if isinstance(data, pd.DataFrame):
        raise InputValidationError(f"{name} must be 1-dimensional, not a DataFrame.")
    if isinstance(data, pd.Series):
        series = data
    else:
        try:
            series = pd.Series(data)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise InputValidationError(f"{name} could not be read as 1-D data: {exc}") from exc
    if series.dtype == object and series.isna().all():
        # Empty input or only None/NA: no values to judge, so treat as all-missing
        # numeric data (the statistic functions then report "too few values").
        series = series.astype("float64")
    if ptypes.is_bool_dtype(series) or not ptypes.is_numeric_dtype(series):
        raise InputValidationError(
            f"{name} must contain numeric values (dtype {series.dtype}); "
            "convert or clean the data explicitly before analysis."
        )
    series = series.astype("float64")
    if np.isinf(series.to_numpy()).any():
        raise InputValidationError(f"{name} contains infinite values.")
    return series


def as_numeric_frame(
    df: Any, columns: Sequence[str] | None = None, name: str = "data"
) -> pd.DataFrame:
    """Return a float64 copy of the selected numeric columns of ``df``."""
    require_dataframe(df, name)
    cols = column_list(columns)
    if cols is None:
        cols = list(df.columns)
    if not cols:
        raise InputValidationError(f"{name} has no columns to analyse.")
    require_columns(df, cols)
    repeated = [c for c in cols if int((df.columns == c).sum()) > 1]
    if repeated:
        raise ColumnError(
            f"Column name(s) {repeated} appear more than once in the data; rename them first."
        )
    sub = df.loc[:, cols]
    bad = [c for c in cols if ptypes.is_bool_dtype(sub[c]) or not ptypes.is_numeric_dtype(sub[c])]
    if bad:
        raise ColumnError(f"Column(s) {bad} are not numeric; clean or convert them first.")
    out = sub.astype("float64")
    if np.isinf(out.to_numpy()).any():
        raise InputValidationError(f"{name} contains infinite values.")
    return out


def apply_missing_policy(
    series: pd.Series, missing: str, name: str = "data"
) -> tuple[pd.Series, int]:
    """Apply a univariate missing-data policy (``'omit'`` or ``'raise'``).

    Returns the cleaned series and the number of missing values found.
    """
    check_choice(missing, ("omit", "raise"), "missing")
    n_missing = int(series.isna().sum())
    if n_missing and missing == "raise":
        raise InputValidationError(
            f"{name} contains {n_missing} missing value(s) and missing='raise'."
        )
    return series.dropna(), n_missing

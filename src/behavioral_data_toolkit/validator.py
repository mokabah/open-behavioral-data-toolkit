"""Data-quality checks: missing values, duplicates, invalid values, summaries.

Definition of *missing*: values for which :func:`pandas.isna` is true (NaN,
``None``, ``NaT``, ``pd.NA``). Empty strings and whitespace-only strings are
**not** missing unless the loader converted them; ``load_csv`` does convert empty
fields to NaN via pandas' defaults.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ._checks import check_choice, column_list, require_columns, require_dataframe
from ._serialize import frame_to_records
from .exceptions import InputValidationError

__all__ = [
    "DataQualitySummary",
    "ValueRule",
    "data_quality_summary",
    "find_duplicates",
    "find_invalid_values",
    "missing_summary",
]


@dataclass(frozen=True)
class ValueRule:
    """Validity constraints for one column.

    A value is valid only if it satisfies *every* constraint that is set.
    Missing values are never reported by this rule (see :func:`missing_summary`).

    Attributes:
        minimum: Smallest allowed numeric value (inclusive).
        maximum: Largest allowed numeric value (inclusive).
        allowed: Explicit set of allowed values.
    """

    minimum: float | None = None
    maximum: float | None = None
    allowed: Collection[Any] | None = None

    def __post_init__(self) -> None:
        if self.minimum is None and self.maximum is None and self.allowed is None:
            raise InputValidationError("ValueRule needs at least one constraint.")
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise InputValidationError("ValueRule minimum cannot exceed maximum.")
        if self.allowed is not None and len(self.allowed) == 0:
            raise InputValidationError("ValueRule.allowed must not be empty.")


def missing_summary(df: pd.DataFrame, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Count missing values per column.

    Returns:
        DataFrame indexed by column with ``n_missing``, ``n_total`` and
        ``pct_missing`` (0-100; NaN if the frame has zero rows).
    """
    require_dataframe(df, "df")
    cols = column_list(columns) or list(df.columns)
    require_columns(df, cols)
    n_total = len(df)
    n_missing = df[cols].isna().sum()
    pct = n_missing / n_total * 100.0 if n_total else pd.Series(np.nan, index=n_missing.index)
    out = pd.DataFrame(
        {
            "n_missing": n_missing.astype(int).to_numpy(),
            "n_total": n_total,
            "pct_missing": pct.to_numpy(dtype=float),
        },
        index=pd.Index(cols, name="column"),
    )
    return out


def find_duplicates(
    df: pd.DataFrame,
    subset: Sequence[str] | None = None,
    keep: str | bool = False,
) -> pd.DataFrame:
    """Return the rows that are duplicates of other rows.

    Args:
        df: Data to check.
        subset: Columns that define a duplicate (e.g. a participant ID).
            ``None`` compares entire rows.
        keep: ``False`` (default) returns *all* members of each duplicate group,
            which is best for inspection; ``"first"``/``"last"`` returns only the
            redundant copies, i.e. the rows that would be dropped.

    Notes:
        Missing values compare as equal to each other (pandas behaviour).
    """
    require_dataframe(df, "df")
    cols = column_list(subset, "subset")
    if cols is not None:
        require_columns(df, cols)
    if keep is not False:
        check_choice(keep, ("first", "last"), "keep")  # type: ignore[arg-type]
    mask = df.duplicated(subset=cols, keep=keep)
    return df.loc[mask]


def find_invalid_values(df: pd.DataFrame, rules: Mapping[str, ValueRule]) -> pd.DataFrame:
    """List values that violate per-column rules.

    Missing values are skipped. For rules with ``minimum``/``maximum``, entries
    that cannot be read as numbers are reported as ``non-numeric value`` and
    infinite values as ``non-finite value``.

    Returns:
        DataFrame with columns ``row`` (index label of ``df``), ``column``,
        ``value`` and ``reason``, ordered by column (in the order of ``rules``)
        and then by row position. One value may produce several reasons.
    """
    require_dataframe(df, "df")
    if not isinstance(rules, Mapping):
        raise InputValidationError("rules must be a mapping of column name -> ValueRule.")
    require_columns(df, list(rules))
    records: list[dict[str, Any]] = []

    def add(mask: pd.Series, column: str, reason: str) -> None:
        for label, value in df.loc[mask, column].items():
            records.append({"row": label, "column": column, "value": value, "reason": reason})

    for column, rule in rules.items():
        if not isinstance(rule, ValueRule):
            raise InputValidationError(f"rules[{column!r}] must be a ValueRule.")
        series = df[column]
        present = series.notna()
        if rule.minimum is not None or rule.maximum is not None:
            numeric = pd.to_numeric(series, errors="coerce")
            add(present & numeric.isna(), column, "non-numeric value")
            finite = numeric.notna() & np.isfinite(numeric.to_numpy(dtype=float, na_value=np.nan))
            add(numeric.notna() & ~finite, column, "non-finite value")
            if rule.minimum is not None:
                add(finite & (numeric < rule.minimum), column, f"below minimum {rule.minimum}")
            if rule.maximum is not None:
                add(finite & (numeric > rule.maximum), column, f"above maximum {rule.maximum}")
        if rule.allowed is not None:
            ok = series.isin(list(rule.allowed))
            add(present & ~ok, column, "not in allowed values")

    return pd.DataFrame(records, columns=["row", "column", "value", "reason"])


@dataclass(frozen=True, eq=False)
class DataQualitySummary:
    """Aggregate data-quality findings.

    Attributes:
        n_rows: Number of rows.
        n_columns: Number of columns.
        n_complete_rows: Rows with no missing value in any column.
        n_duplicate_rows: Redundant rows beyond the first occurrence of each
            duplicate group.
        duplicate_subset: Columns used to define duplicates (``None`` = all).
        missing: Output of :func:`missing_summary`.
        invalid: Output of :func:`find_invalid_values` (empty if no rules).
    """

    n_rows: int
    n_columns: int
    n_complete_rows: int
    n_duplicate_rows: int
    duplicate_subset: tuple[str, ...] | None
    missing: pd.DataFrame
    invalid: pd.DataFrame

    @property
    def n_invalid_values(self) -> int:
        """Number of rule violations found."""
        return int(len(self.invalid))

    @property
    def is_clean(self) -> bool:
        """True if there are no missing values, duplicates or rule violations."""
        return (
            int(self.missing["n_missing"].sum()) == 0
            and self.n_duplicate_rows == 0
            and self.n_invalid_values == 0
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dictionary."""
        return {
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "n_complete_rows": self.n_complete_rows,
            "n_duplicate_rows": self.n_duplicate_rows,
            "duplicate_subset": list(self.duplicate_subset) if self.duplicate_subset else None,
            "n_invalid_values": self.n_invalid_values,
            "is_clean": self.is_clean,
            "missing": frame_to_records(self.missing),
            "invalid": frame_to_records(self.invalid.reset_index(drop=True)),
        }


def data_quality_summary(
    df: pd.DataFrame,
    *,
    duplicate_subset: Sequence[str] | None = None,
    rules: Mapping[str, ValueRule] | None = None,
) -> DataQualitySummary:
    """Run missing/duplicate/invalid-value checks and collect the results.

    Args:
        df: Data to check.
        duplicate_subset: Columns defining a duplicate; ``None`` = whole row.
        rules: Optional per-column validity rules.
    """
    require_dataframe(df, "df")
    subset = column_list(duplicate_subset, "duplicate_subset")
    if subset is not None:
        require_columns(df, subset)
    n_dup = int(df.duplicated(subset=subset, keep="first").sum())
    invalid = (
        find_invalid_values(df, rules)
        if rules
        else pd.DataFrame(columns=["row", "column", "value", "reason"])
    )
    n_complete = int(df.notna().all(axis=1).sum())
    return DataQualitySummary(
        n_rows=int(df.shape[0]),
        n_columns=int(df.shape[1]),
        n_complete_rows=n_complete,
        n_duplicate_rows=n_dup,
        duplicate_subset=tuple(subset) if subset is not None else None,
        missing=missing_summary(df),
        invalid=invalid,
    )

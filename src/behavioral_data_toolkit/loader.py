"""CSV ingestion, data-structure inspection, and column validation."""

from __future__ import annotations

import csv
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd
from pandas.api import types as ptypes
from pandas.errors import EmptyDataError, ParserError

from ._checks import column_list, require_dataframe
from ._serialize import frame_to_records
from .exceptions import ColumnError, DataLoadError, InputValidationError

__all__ = ["DataStructure", "inspect_dataframe", "load_csv", "validate_columns"]


@dataclass(frozen=True, eq=False)
class DataStructure:
    """Description of a DataFrame's shape and columns.

    Attributes:
        n_rows: Number of rows.
        n_columns: Number of columns.
        columns: DataFrame indexed by column name with the columns ``dtype``,
            ``n_non_null``, ``n_missing`` and ``n_unique`` (unique non-missing
            values).
    """

    n_rows: int
    n_columns: int
    columns: pd.DataFrame

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dictionary."""
        return {
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "columns": frame_to_records(self.columns),
        }


def load_csv(
    path: str | os.PathLike[str],
    *,
    required_columns: Sequence[str] | None = None,
    sep: str = ",",
    encoding: str = "utf-8",
    na_values: Sequence[str] | None = None,
    strip_column_names: bool = True,
    **read_csv_kwargs: Any,
) -> pd.DataFrame:
    """Read a CSV file into a DataFrame with defensive checks.

    The first row must be a header. The function refuses files whose header
    contains empty or duplicate column names (pandas would otherwise silently
    rename duplicates), and files without any data rows. Values are **not**
    coerced or cleaned: use :mod:`behavioral_data_toolkit.validator` to inspect
    quality.

    Args:
        path: File path.
        required_columns: Columns that must be present.
        sep: Field delimiter.
        encoding: File encoding.
        na_values: Additional strings to treat as missing (pandas defaults,
            which include the empty string, still apply).
        strip_column_names: Strip surrounding whitespace from column names.
        **read_csv_kwargs: Passed to :func:`pandas.read_csv`. ``header`` is not
            allowed.

    Raises:
        DataLoadError: File missing/unreadable/empty/malformed, or header invalid.
        ColumnError: A required column is absent.
        InputValidationError: ``header`` was passed in ``read_csv_kwargs``.
    """
    if "header" in read_csv_kwargs:
        raise InputValidationError(
            "'header' cannot be overridden; the first row must be the header."
        )
    cols_required = column_list(required_columns, "required_columns")

    try:
        with open(path, newline="", encoding=encoding) as handle:
            header = next(csv.reader(handle, delimiter=sep), None)
    except FileNotFoundError as exc:
        raise DataLoadError(f"File not found: {path}") from exc
    except (OSError, UnicodeDecodeError, LookupError, csv.Error) as exc:
        raise DataLoadError(f"Could not read {path}: {exc}") from exc
    if not header:
        raise DataLoadError(f"{path} is empty (no header row).")

    names = [h.strip() if strip_column_names else h for h in header]
    if any(n == "" for n in names):
        raise DataLoadError(f"{path} has an empty column name in its header.")
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise DataLoadError(f"{path} has duplicate column names: {duplicates}.")

    try:
        df = pd.read_csv(path, sep=sep, encoding=encoding, na_values=na_values, **read_csv_kwargs)
    except EmptyDataError as exc:
        raise DataLoadError(f"{path} contains no data.") from exc
    except (ParserError, UnicodeDecodeError, ValueError) as exc:
        raise DataLoadError(f"Could not parse {path}: {exc}") from exc

    if strip_column_names:
        df.columns = [str(c).strip() for c in df.columns]
    if df.empty:
        raise DataLoadError(f"{path} has a header but no data rows.")
    if cols_required:
        validate_columns(df, required=cols_required)
    return df


def inspect_dataframe(df: pd.DataFrame) -> DataStructure:
    """Summarise the structure of ``df`` (shape, dtypes, missing, unique counts)."""
    require_dataframe(df, "df")
    info = pd.DataFrame(
        {
            "dtype": [str(t) for t in df.dtypes],
            "n_non_null": df.notna().sum().to_numpy(),
            "n_missing": df.isna().sum().to_numpy(),
            "n_unique": [int(df[c].nunique(dropna=True)) for c in df.columns],
        },
        index=pd.Index(list(df.columns), name="column"),
    )
    return DataStructure(n_rows=int(df.shape[0]), n_columns=int(df.shape[1]), columns=info)


def validate_columns(
    df: pd.DataFrame,
    required: Sequence[str] | None = None,
    *,
    numeric: Sequence[str] | None = None,
) -> None:
    """Check that columns exist, are unique, and (optionally) numeric.

    Args:
        df: Data to check.
        required: Columns that must exist.
        numeric: Columns that must exist *and* have a numeric (non-boolean) dtype.

    Raises:
        ColumnError: Duplicate column names, absent columns, or non-numeric columns.
    """
    require_dataframe(df, "df")
    req = column_list(required, "required") or []
    num = column_list(numeric, "numeric") or []

    dup = sorted({str(c) for c in df.columns[df.columns.duplicated()]})
    if dup:
        raise ColumnError(f"DataFrame has duplicate column names: {dup}.")
    absent = [c for c in dict.fromkeys([*req, *num]) if c not in df.columns]
    if absent:
        raise ColumnError(f"Missing required column(s): {absent}.")
    non_numeric = [
        c for c in num if ptypes.is_bool_dtype(df[c]) or not ptypes.is_numeric_dtype(df[c])
    ]
    if non_numeric:
        raise ColumnError(f"Column(s) expected to be numeric but are not: {non_numeric}.")

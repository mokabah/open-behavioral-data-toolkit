from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from behavioral_data_toolkit.exceptions import (
    ColumnError,
    DataLoadError,
    InputValidationError,
)
from behavioral_data_toolkit.loader import inspect_dataframe, load_csv, validate_columns


def write(tmp_path, text: str, name: str = "d.csv"):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def test_load_csv_basic(tmp_path):
    p = write(tmp_path, "id,q1,q2\n1,3,4\n2,5,\n")
    df = load_csv(p, required_columns=["id", "q1"])
    assert list(df.columns) == ["id", "q1", "q2"]
    assert df.shape == (2, 3)
    assert df["q2"].isna().sum() == 1  # empty field -> missing, not coerced to 0


def test_load_csv_strips_column_names(tmp_path):
    p = write(tmp_path, " id , q1\n1,2\n")
    assert list(load_csv(p).columns) == ["id", "q1"]


def test_load_csv_missing_file(tmp_path):
    with pytest.raises(DataLoadError, match="not found"):
        load_csv(tmp_path / "nope.csv")


def test_load_csv_empty_file(tmp_path):
    with pytest.raises(DataLoadError, match="empty"):
        load_csv(write(tmp_path, ""))


def test_load_csv_header_only(tmp_path):
    with pytest.raises(DataLoadError, match="no data rows"):
        load_csv(write(tmp_path, "a,b\n"))


def test_load_csv_duplicate_header_rejected(tmp_path):
    with pytest.raises(DataLoadError, match="duplicate"):
        load_csv(write(tmp_path, "a,a\n1,2\n"))


def test_load_csv_duplicate_after_stripping_rejected(tmp_path):
    with pytest.raises(DataLoadError, match="duplicate"):
        load_csv(write(tmp_path, "a, a\n1,2\n"))


def test_load_csv_empty_column_name_rejected(tmp_path):
    with pytest.raises(DataLoadError, match="empty column name"):
        load_csv(write(tmp_path, "a,\n1,2\n"))


def test_load_csv_bad_encoding(tmp_path):
    p = tmp_path / "latin.csv"
    p.write_bytes("a,b\n\xe9,1\n".encode("latin-1"))
    with pytest.raises(DataLoadError):
        load_csv(p)  # utf-8 default cannot decode 0xE9


def test_load_csv_header_kwarg_forbidden(tmp_path):
    with pytest.raises(InputValidationError):
        load_csv(write(tmp_path, "a\n1\n"), header=None)


def test_load_csv_required_columns_missing(tmp_path):
    with pytest.raises(ColumnError, match="q9"):
        load_csv(write(tmp_path, "a,b\n1,2\n"), required_columns=["a", "q9"])


def test_load_csv_custom_na_values(tmp_path):
    df = load_csv(write(tmp_path, "a,b\n1,-99\n2,3\n"), na_values=["-99"])
    assert df["b"].isna().sum() == 1


def test_load_csv_semicolon(tmp_path):
    df = load_csv(write(tmp_path, "a;b\n1;2\n"), sep=";")
    assert df.shape == (1, 2)


def test_inspect_dataframe():
    df = pd.DataFrame({"a": [1, 2, np.nan, 2], "b": ["x", "y", "y", None]})
    s = inspect_dataframe(df)
    assert (s.n_rows, s.n_columns) == (4, 2)
    assert s.columns.loc["a", "n_missing"] == 1
    assert s.columns.loc["a", "n_unique"] == 2
    assert s.columns.loc["b", "n_non_null"] == 3
    d = s.to_dict()
    assert d["columns"]["columns"] == ["column", "dtype", "n_non_null", "n_missing", "n_unique"]
    by_name = {row["column"]: row for row in d["columns"]["rows"]}
    assert by_name["a"]["n_missing"] == 1 and by_name["b"]["n_unique"] == 2


def test_inspect_requires_dataframe():
    with pytest.raises(InputValidationError):
        inspect_dataframe([1, 2, 3])  # type: ignore[arg-type]


def test_validate_columns_ok():
    df = pd.DataFrame({"a": [1], "b": ["x"]})
    validate_columns(df, required=["a", "b"], numeric=["a"])


def test_validate_columns_missing():
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(ColumnError, match="zzz"):
        validate_columns(df, required=["a", "zzz"])


def test_validate_columns_non_numeric():
    df = pd.DataFrame({"a": ["1", "2"], "flag": [True, False]})
    with pytest.raises(ColumnError, match="not numeric|numeric"):
        validate_columns(df, numeric=["a"])
    with pytest.raises(ColumnError):
        validate_columns(df, numeric=["flag"])  # booleans are not scale scores


def test_validate_columns_duplicate_names():
    df = pd.DataFrame([[1, 2]], columns=["a", "a"])
    with pytest.raises(ColumnError, match="duplicate"):
        validate_columns(df)


def test_string_instead_of_list_rejected():
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(InputValidationError):
        validate_columns(df, required="a")  # type: ignore[arg-type]

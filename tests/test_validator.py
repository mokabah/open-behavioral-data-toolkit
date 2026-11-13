from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from behavioral_data_toolkit.exceptions import ColumnError, InputValidationError
from behavioral_data_toolkit.validator import (
    ValueRule,
    data_quality_summary,
    find_duplicates,
    find_invalid_values,
    missing_summary,
)


@pytest.fixture
def df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1, 2, 2, 3, 4],
            "q1": [1, 5, 5, 9, np.nan],
            "q2": [2, 3, 3, np.nan, np.nan],
            "grp": ["a", "b", "b", "z", "a"],
        }
    )


def test_missing_summary(df):
    m = missing_summary(df)
    assert m.loc["q1", "n_missing"] == 1
    assert m.loc["q2", "n_missing"] == 2
    assert m.loc["q2", "pct_missing"] == pytest.approx(40.0)
    assert m.loc["id", "n_missing"] == 0
    assert (m["n_total"] == 5).all()


def test_missing_summary_columns_subset_and_errors(df):
    assert list(missing_summary(df, ["q1"]).index) == ["q1"]
    with pytest.raises(ColumnError):
        missing_summary(df, ["nope"])


def test_missing_summary_empty_frame():
    m = missing_summary(pd.DataFrame({"a": []}))
    assert m.loc["a", "n_missing"] == 0
    assert np.isnan(m.loc["a", "pct_missing"])


def test_find_duplicates_all_members_by_default(df):
    dup = find_duplicates(df, subset=["id"])
    assert list(dup.index) == [1, 2]


def test_find_duplicates_keep_first_returns_only_redundant(df):
    dup = find_duplicates(df, subset=["id"], keep="first")
    assert list(dup.index) == [2]


def test_find_duplicates_whole_row(df):
    assert list(find_duplicates(df).index) == [1, 2]


def test_find_duplicates_bad_args(df):
    with pytest.raises(InputValidationError):
        find_duplicates(df, keep="middle")  # type: ignore[arg-type]
    with pytest.raises(ColumnError):
        find_duplicates(df, subset=["nope"])


def test_find_invalid_values_range_and_allowed(df):
    rules = {
        "q1": ValueRule(minimum=1, maximum=5),
        "grp": ValueRule(allowed=["a", "b"]),
    }
    bad = find_invalid_values(df, rules)
    assert list(bad.columns) == ["row", "column", "value", "reason"]
    got = {(r.row, r.column, r.reason) for r in bad.itertuples()}
    assert got == {
        (3, "q1", "above maximum 5"),
        (3, "grp", "not in allowed values"),
    }  # NaN in q1 is *not* reported as invalid


def test_find_invalid_values_below_minimum():
    d = pd.DataFrame({"x": [0, 1, 2]})
    bad = find_invalid_values(d, {"x": ValueRule(minimum=1)})
    assert list(bad["row"]) == [0]
    assert bad["reason"].iloc[0] == "below minimum 1"


def test_find_invalid_values_non_numeric_and_inf():
    d = pd.DataFrame({"x": ["1", "two", "3", None]})
    bad = find_invalid_values(d, {"x": ValueRule(minimum=0, maximum=5)})
    assert list(bad["reason"]) == ["non-numeric value"]
    assert list(bad["row"]) == [1]
    d2 = pd.DataFrame({"x": [1.0, np.inf, -np.inf]})
    bad2 = find_invalid_values(d2, {"x": ValueRule(minimum=0, maximum=5)})
    assert list(bad2["reason"]) == ["non-finite value", "non-finite value"]


def test_find_invalid_values_none_found():
    d = pd.DataFrame({"x": [1, 2, 3]})
    bad = find_invalid_values(d, {"x": ValueRule(minimum=1, maximum=3)})
    assert bad.empty and list(bad.columns) == ["row", "column", "value", "reason"]


def test_find_invalid_values_argument_errors(df):
    with pytest.raises(ColumnError):
        find_invalid_values(df, {"nope": ValueRule(minimum=0)})
    with pytest.raises(InputValidationError):
        find_invalid_values(df, {"q1": (1, 5)})  # type: ignore[dict-item]
    with pytest.raises(InputValidationError):
        find_invalid_values(df, [("q1", ValueRule(minimum=0))])  # type: ignore[arg-type]


def test_value_rule_validation():
    with pytest.raises(InputValidationError):
        ValueRule()
    with pytest.raises(InputValidationError):
        ValueRule(minimum=5, maximum=1)
    with pytest.raises(InputValidationError):
        ValueRule(allowed=[])


def test_data_quality_summary(df):
    s = data_quality_summary(
        df, duplicate_subset=["id"], rules={"q1": ValueRule(minimum=1, maximum=5)}
    )
    assert (s.n_rows, s.n_columns) == (5, 4)
    assert s.n_complete_rows == 3  # rows 0,1,2 have no NaN
    assert s.n_duplicate_rows == 1
    assert s.n_invalid_values == 1
    assert not s.is_clean
    d = s.to_dict()
    json.dumps(d, allow_nan=False)  # strictly valid JSON
    assert d["duplicate_subset"] == ["id"]


def test_data_quality_summary_clean():
    d = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    s = data_quality_summary(d)
    assert s.is_clean and s.n_invalid_values == 0

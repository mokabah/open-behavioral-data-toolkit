from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from behavioral_data_toolkit.exceptions import (
    ColumnError,
    InputValidationError,
    ScoringError,
)
from behavioral_data_toolkit.scoring import (
    ScaleDefinition,
    composite_score,
    reverse_score,
    score_questionnaire,
    score_questionnaires,
)

# ---------------------------------------------------------------- reverse_score


def test_reverse_score_series_1_to_5():
    s = pd.Series([1, 2, 3, 4, 5, np.nan], name="q")
    out = reverse_score(s, 1, 5)
    assert out.name == "q"
    np.testing.assert_array_equal(out.to_numpy()[:5], [5, 4, 3, 2, 1])
    assert np.isnan(out.iloc[5])  # missing stays missing


def test_reverse_score_zero_based_scale():
    out = reverse_score(pd.Series([0, 1, 4]), 0, 4)
    np.testing.assert_array_equal(out.to_numpy(), [4, 3, 0])


def test_reverse_score_is_involution():
    s = pd.Series([1.0, 2.5, 4.0, 5.0])
    twice = reverse_score(reverse_score(s, 1, 5), 1, 5)
    np.testing.assert_allclose(twice.to_numpy(), s.to_numpy())


def test_reverse_score_dataframe_columns_and_no_mutation():
    df = pd.DataFrame({"a": [1, 5], "b": [2, 4], "id": ["x", "y"]})
    before = df.copy()
    out = reverse_score(df, 1, 5, columns=["a"])
    assert list(out["a"]) == [5, 1]
    assert list(out["b"]) == [2, 4]  # untouched
    assert list(out["id"]) == ["x", "y"]  # non-numeric column untouched
    pd.testing.assert_frame_equal(df, before)  # input not modified


def test_reverse_score_out_of_range_raises():
    with pytest.raises(ScoringError, match="outside the scale range"):
        reverse_score(pd.Series([1, 6]), 1, 5)
    with pytest.raises(ScoringError):
        reverse_score(pd.DataFrame({"a": [0, 3]}), 1, 5)


def test_reverse_score_argument_errors():
    with pytest.raises(InputValidationError):
        reverse_score(pd.Series([1]), 5, 1)
    with pytest.raises(InputValidationError):
        reverse_score(pd.Series([1]), 1, float("inf"))
    with pytest.raises(InputValidationError):
        reverse_score(pd.Series([1]), "1", 5)  # type: ignore[arg-type]
    with pytest.raises(InputValidationError):
        reverse_score(pd.Series([1]), 1, 5, columns=["a"])
    with pytest.raises(InputValidationError):
        reverse_score(pd.Series(["a", "b"]), 1, 5)
    with pytest.raises(ColumnError):
        reverse_score(pd.DataFrame({"a": [1]}), 1, 5, columns=["zzz"])


# -------------------------------------------------------------- composite_score


@pytest.fixture
def items() -> pd.DataFrame:
    return pd.DataFrame(
        {"i1": [1, 2, 3], "i2": [3, np.nan, 5], "i3": [5, 4, np.nan]},
        index=["p", "q", "r"],
    )


def test_composite_mean_complete_only_by_default(items):
    out = composite_score(items, ["i1", "i2", "i3"])
    assert out["p"] == pytest.approx(3.0)
    assert np.isnan(out["q"]) and np.isnan(out["r"])
    assert out.name == "composite"
    assert list(out.index) == ["p", "q", "r"]


def test_composite_mean_partial(items):
    out = composite_score(items, ["i1", "i2", "i3"], min_valid=2, name="s")
    assert out.name == "s"
    assert out["q"] == pytest.approx(3.0)  # (2 + 4) / 2
    assert out["r"] == pytest.approx(4.0)  # (3 + 5) / 2


def test_composite_min_valid_threshold(items):
    out = composite_score(items[["i1", "i2"]], ["i1", "i2"], min_valid=1)
    assert out["q"] == pytest.approx(2.0)


def test_composite_sum(items):
    out = composite_score(items, ["i1", "i2", "i3"], method="sum")
    assert out["p"] == 9
    assert np.isnan(out["q"])


def test_composite_partial_sum_refused(items):
    with pytest.raises(InputValidationError, match="Partial sums"):
        composite_score(items, ["i1", "i2", "i3"], method="sum", min_valid=2)


@pytest.mark.parametrize("bad", [0, 4, True, 1.5, "2"])
def test_composite_min_valid_validation(items, bad):
    with pytest.raises(InputValidationError):
        composite_score(items, ["i1", "i2", "i3"], min_valid=bad)


def test_composite_argument_errors(items):
    with pytest.raises(InputValidationError):
        composite_score(items, [])
    with pytest.raises(InputValidationError):
        composite_score(items, ["i1"], method="median")
    with pytest.raises(ColumnError):
        composite_score(items, ["i1", "nope"])
    with pytest.raises(ColumnError):
        composite_score(pd.DataFrame({"a": ["x", "y"]}), ["a"])


# ------------------------------------------------------- questionnaire scoring


def test_scale_definition_validation():
    with pytest.raises(InputValidationError):
        ScaleDefinition("", ("a",), 1, 5)
    with pytest.raises(InputValidationError):
        ScaleDefinition("s", (), 1, 5)
    with pytest.raises(InputValidationError):
        ScaleDefinition("s", ("a", "a"), 1, 5)
    with pytest.raises(InputValidationError):
        ScaleDefinition("s", ("a",), 1, 5, reverse_items=("b",))
    with pytest.raises(InputValidationError):
        ScaleDefinition("s", ("a",), 5, 1)
    with pytest.raises(InputValidationError):
        ScaleDefinition("s", ("a",), 1, 5, method="max")
    s = ScaleDefinition("s", ["a", "b"], 1, 5, reverse_items=["b"])  # type: ignore[arg-type]
    assert s.items == ("a", "b") and s.reverse_items == ("b",)
    assert s.to_dict()["reverse_items"] == ["b"]


def test_score_questionnaire_hand_computed():
    df = pd.DataFrame({"a": [1, 5], "b": [2, 4], "c": [5, 1]})
    scale = ScaleDefinition("demo", ("a", "b", "c"), 1, 5, reverse_items=("c",))
    out = score_questionnaire(df, scale)
    # row 0: a=1, b=2, c reversed 5 -> 1  => mean (1+2+1)/3
    # row 1: a=5, b=4, c reversed 1 -> 5  => mean (5+4+5)/3
    np.testing.assert_allclose(out.to_numpy(), [4 / 3, 14 / 3])
    assert out.name == "demo"


def test_score_questionnaire_sum_and_no_mutation():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    before = df.copy()
    out = score_questionnaire(df, ScaleDefinition("t", ("a", "b"), 1, 5, method="sum"))
    assert list(out) == [4, 6]
    pd.testing.assert_frame_equal(df, before)


def test_score_questionnaire_out_of_range_raises():
    df = pd.DataFrame({"a": [1, 9], "b": [2, 3]})
    with pytest.raises(ScoringError):
        score_questionnaire(df, ScaleDefinition("s", ("a", "b"), 1, 5))


def test_score_questionnaire_bad_inputs():
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(InputValidationError):
        score_questionnaire(df, {"name": "s"})  # type: ignore[arg-type]
    with pytest.raises(ColumnError):
        score_questionnaire(df, ScaleDefinition("s", ("a", "z"), 1, 5))


def test_score_questionnaires_multiple():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 5]})
    out = score_questionnaires(
        df,
        [
            ScaleDefinition("s1", ("a", "b"), 1, 5, method="sum"),
            ScaleDefinition("s2", ("c",), 1, 5),
        ],
    )
    assert list(out.columns) == ["s1", "s2"]
    assert list(out["s1"]) == [4, 6]


def test_score_questionnaires_validation():
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(InputValidationError):
        score_questionnaires(df, [])
    s = ScaleDefinition("x", ("a",), 1, 5)
    with pytest.raises(InputValidationError, match="unique"):
        score_questionnaires(df, [s, s])
    with pytest.raises(InputValidationError):
        score_questionnaires(df, [s, "nope"])  # type: ignore[list-item]


# ------------------------------------------------ regression: NumPy scalar arguments


def test_numpy_scalars_accepted_as_scale_bounds_and_min_valid():
    out = reverse_score(pd.Series([1, 2]), np.int64(1), np.int64(5))
    assert list(out) == [5, 4]
    out = reverse_score(pd.Series([1.0]), np.float32(1), np.float64(5))
    assert list(out) == [5.0]
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [1.0, np.nan]})
    comp = composite_score(df, ["a", "b"], min_valid=np.int64(1))
    assert list(comp) == [1.0, 2.0]
    scale = ScaleDefinition("s", ("a",), np.int64(1), np.int64(5))
    assert scale.to_dict()["min_value"] == 1 and isinstance(scale.to_dict()["min_value"], int)


def test_numpy_bool_is_not_a_number_for_bounds():
    with pytest.raises(InputValidationError):
        reverse_score(pd.Series([1]), np.bool_(True), 5)


def test_duplicate_column_names_get_a_clear_error():
    dup = pd.DataFrame([[1, 2, 3], [2, 3, 5]], columns=["a", "a", "b"])
    with pytest.raises(ColumnError, match="more than once"):
        composite_score(dup, ["a", "b"])
    # duplicates among columns that are NOT selected are harmless
    assert list(composite_score(dup, ["b"])) == [3.0, 5.0]

"""Correlation tests with independent reference calculations.

References: hand-derived values, ``numpy.corrcoef``, rank-then-Pearson,
the textbook Spearman formula (no ties), a manual Fisher-z interval, and the
t-distribution identity for the p-value.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

from behavioral_data_toolkit.correlation import (
    correlation_matrix,
    pearson,
    spearman,
)
from behavioral_data_toolkit.exceptions import (
    ColumnError,
    InputValidationError,
    StatisticsError,
)

X = [1, 2, 3, 4, 5]
Y = [2, 4, 5, 4, 5]
Z_975 = 1.959963984540054  # standard-normal 97.5th percentile (constant)


def test_pearson_hand_derived():
    # mean_x=3, mean_y=4; Sxy=6, Sxx=10, Syy=6  ->  r = 6/sqrt(60)
    res = pearson(X, Y)
    assert res.r == pytest.approx(6 / math.sqrt(60), abs=1e-12)
    assert res.n == 5 and res.n_excluded == 0 and res.method == "pearson"


def test_pearson_pvalue_matches_t_identity():
    res = pearson(X, Y)
    r, n = res.r, res.n
    t = r * math.sqrt((n - 2) / (1 - r**2))
    expected_p = 2 * sps.t.sf(abs(t), n - 2)
    assert res.p_value == pytest.approx(expected_p, rel=1e-10)


def test_pearson_ci_matches_manual_fisher_and_scipy():
    res = pearson(X, Y)
    z = math.atanh(res.r)
    se = 1 / math.sqrt(res.n - 3)
    assert res.ci_low == pytest.approx(math.tanh(z - Z_975 * se), abs=1e-9)
    assert res.ci_high == pytest.approx(math.tanh(z + Z_975 * se), abs=1e-9)
    ref = sps.pearsonr(X, Y).confidence_interval(0.95)
    assert res.ci_low == pytest.approx(ref.low, abs=1e-9)
    assert res.ci_high == pytest.approx(ref.high, abs=1e-9)
    assert res.confidence == 0.95


def test_pearson_matches_numpy_on_random_data():
    rng = np.random.default_rng(3)
    a = rng.normal(size=80)
    b = 0.5 * a + rng.normal(size=80)
    assert pearson(a, b).r == pytest.approx(np.corrcoef(a, b)[0, 1], abs=1e-12)


def test_pearson_properties():
    rng = np.random.default_rng(4)
    a, b = rng.normal(size=50), rng.normal(size=50)
    base = pearson(a, b).r
    assert pearson(b, a).r == pytest.approx(base)  # symmetry
    assert pearson(3 * a + 7, b).r == pytest.approx(base)  # positive affine invariance
    assert pearson(-a, b).r == pytest.approx(-base)  # sign flip


def test_pearson_perfect_correlation_has_no_ci():
    res = pearson([1, 2, 3, 4], [2, 4, 6, 8])
    assert res.r == pytest.approx(1.0)
    assert res.ci_low is None and res.ci_high is None and res.confidence is None


def test_pearson_n3_has_no_ci():
    res = pearson([1, 2, 3], [1, 3, 2])
    assert res.n == 3 and res.ci_low is None


def test_pearson_confidence_level_changes_interval():
    narrow = pearson(X, Y, confidence=0.80)
    wide = pearson(X, Y, confidence=0.99)
    assert (wide.ci_high - wide.ci_low) > (narrow.ci_high - narrow.ci_low)


@pytest.mark.parametrize("bad", [0, 1, 1.5, -0.2, True, "0.95"])
def test_pearson_confidence_validation(bad):
    with pytest.raises(InputValidationError):
        pearson(X, Y, confidence=bad)


def test_spearman_matches_textbook_formula_without_ties():
    x = [1, 2, 3, 4, 5, 6]
    y = [2, 1, 4, 3, 6, 5]  # rank differences: 1,-1,1,-1,1,-1 -> sum d^2 = 6
    n = 6
    expected = 1 - 6 * 6 / (n * (n**2 - 1))
    res = spearman(x, y)
    assert res.r == pytest.approx(expected, abs=1e-12)
    assert res.method == "spearman" and res.ci_low is None


def test_spearman_with_ties_equals_pearson_on_average_ranks():
    x = [1, 2, 2, 3, 4, 4, 4, 9]
    y = [3, 1, 2, 2, 5, 4, 6, 8]
    ref = np.corrcoef(sps.rankdata(x), sps.rankdata(y))[0, 1]
    assert spearman(x, y).r == pytest.approx(ref, abs=1e-12)


def test_spearman_monotonic_nonlinear_is_one():
    x = np.arange(1, 11)
    assert spearman(x, x**3).r == pytest.approx(1.0)
    assert pearson(x, x**3).r < 1.0


def test_missing_pairs_omitted_and_counted():
    x = [1, 2, 3, np.nan, 5, 6]
    y = [2, np.nan, 5, 7, 4, 9]
    res = pearson(x, y)
    assert res.n == 4 and res.n_excluded == 2
    keep = [0, 2, 4, 5]
    ref = np.corrcoef(np.array(x)[keep], np.array(y)[keep])[0, 1]
    assert res.r == pytest.approx(ref, abs=1e-12)
    with pytest.raises(InputValidationError, match="missing"):
        pearson(x, y, missing="raise")
    with pytest.raises(InputValidationError):
        spearman(x, y, missing="drop")


@pytest.mark.parametrize("func", [pearson, spearman])
def test_error_conditions(func):
    with pytest.raises(InputValidationError, match="same length"):
        func([1, 2, 3], [1, 2])
    with pytest.raises(StatisticsError, match="At least 3"):
        func([1, 2], [1, 2])
    with pytest.raises(StatisticsError, match="constant"):
        func([1, 1, 1, 1], [1, 2, 3, 4])
    with pytest.raises(StatisticsError, match="constant"):
        func([1, 2, 3, 4], [5, 5, 5, 5])
    with pytest.raises(InputValidationError):
        func(["a", "b", "c"], [1, 2, 3])
    with pytest.raises(StatisticsError):
        func([1, np.nan, 3, 4], [1, 2, np.nan, np.nan])  # only 1 complete pair


def test_result_to_dict_is_strict_json():
    json.dumps(pearson(X, Y).to_dict(), allow_nan=False)
    d = spearman(X, Y).to_dict()
    assert d["ci_low"] is None
    json.dumps(d, allow_nan=False)


# --------------------------------------------------------------------- matrix


@pytest.fixture
def frame() -> pd.DataFrame:
    rng = np.random.default_rng(5)
    a = rng.normal(size=60)
    return pd.DataFrame(
        {
            "a": a,
            "b": a + rng.normal(size=60),
            "c": rng.normal(size=60),
            "d": -a + rng.normal(size=60),
        }
    )


def test_matrix_matches_numpy_and_pairwise_function(frame):
    m = correlation_matrix(frame)
    np.testing.assert_allclose(
        m.r.to_numpy(), np.corrcoef(frame.to_numpy(), rowvar=False), atol=1e-12
    )
    single = pearson(frame["a"], frame["c"])
    assert m.r.loc["a", "c"] == pytest.approx(single.r)
    assert m.p_values.loc["a", "c"] == pytest.approx(single.p_value)
    assert m.n.loc["a", "c"] == 60
    assert np.isnan(m.p_values.loc["a", "a"]) and m.r.loc["a", "a"] == 1.0


def test_matrix_symmetric_and_labelled(frame):
    m = correlation_matrix(frame, ["a", "b", "c"])
    assert list(m.r.columns) == ["a", "b", "c"] == list(m.r.index)
    np.testing.assert_allclose(m.r.to_numpy(), m.r.to_numpy().T)
    np.testing.assert_array_equal(m.n.to_numpy(), m.n.to_numpy().T)


def test_matrix_spearman_matches_rank_pearson(frame):
    m = correlation_matrix(frame, method="spearman")
    ranks = frame.apply(sps.rankdata)
    np.testing.assert_allclose(
        m.r.to_numpy(), np.corrcoef(ranks.to_numpy(), rowvar=False), atol=1e-12
    )


def test_matrix_pairwise_vs_listwise_missing(frame):
    df = frame.copy()
    df.loc[0:4, "a"] = np.nan
    df.loc[5:9, "b"] = np.nan
    pair = correlation_matrix(df, missing="pairwise")
    lst = correlation_matrix(df, missing="listwise")
    assert pair.n.loc["a", "b"] == 50 and pair.n.loc["a", "c"] == 55
    assert (lst.n.to_numpy()[np.triu_indices(4, 1)] == 50).all()
    complete = df.dropna()
    assert lst.r.loc["a", "c"] == pytest.approx(np.corrcoef(complete["a"], complete["c"])[0, 1])
    with pytest.raises(InputValidationError, match="missing"):
        correlation_matrix(df, missing="raise")


def test_matrix_undefined_cells_are_nan():
    df = pd.DataFrame({"a": [1, 2, 3, 4], "const": [1, 1, 1, 1], "c": [2, 1, 4, 3]})
    m = correlation_matrix(df)
    assert np.isnan(m.r.loc["a", "const"]) and np.isnan(m.p_values.loc["a", "const"])
    assert np.isnan(m.r.loc["const", "const"])
    assert not np.isnan(m.r.loc["a", "c"])


def test_matrix_to_long_and_dict(frame):
    m = correlation_matrix(frame, ["a", "b", "c"])
    long = m.to_long()
    assert list(long.columns) == ["var1", "var2", "r", "p_value", "n"]
    assert len(long) == 3 and set(long["var1"]) == {"a", "b"}
    json.dumps(m.to_dict(), allow_nan=False)


def test_matrix_argument_errors(frame):
    with pytest.raises(InputValidationError):
        correlation_matrix(frame, ["a"])
    with pytest.raises(InputValidationError):
        correlation_matrix(frame, method="kendall")
    with pytest.raises(InputValidationError):
        correlation_matrix(frame, missing="mean")
    with pytest.raises(ColumnError):
        correlation_matrix(frame, ["a", "zzz"])
    with pytest.raises(ColumnError):
        correlation_matrix(frame.assign(s=["x"] * 60), ["a", "s"])
    with pytest.raises(InputValidationError):
        correlation_matrix(frame.to_numpy())  # type: ignore[arg-type]


def test_confidence_accepts_numpy_float():
    res = pearson(X, Y, confidence=np.float32(0.9))
    assert res.confidence == pytest.approx(0.9)
    with pytest.raises(InputValidationError):
        pearson(X, Y, confidence=np.bool_(True))


def test_matrix_and_alpha_reject_duplicate_column_names_clearly():
    from behavioral_data_toolkit.stats import cronbach_alpha

    dup = pd.DataFrame([[1, 2, 3], [2, 3, 5], [3, 5, 4], [4, 4, 6]], columns=["a", "a", "b"])
    with pytest.raises(ColumnError, match="more than once"):
        correlation_matrix(dup)
    with pytest.raises(ColumnError, match="more than once"):
        cronbach_alpha(dup)

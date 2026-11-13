"""Tests for descriptive statistics and Cronbach's alpha.

Reference calculations are deliberately independent of the implementation:
hand-derived values, Python's ``statistics`` module, exact rational arithmetic
(``fractions.Fraction``), and the covariance-matrix form of alpha.
"""

from __future__ import annotations

import json
import math
import statistics
import warnings
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from behavioral_data_toolkit.exceptions import (
    ColumnError,
    InputValidationError,
    StatisticsError,
)
from behavioral_data_toolkit.stats import (
    cronbach_alpha,
    distribution_summary,
    mean,
    median,
    std,
)

RNG_DATA = np.random.default_rng(20240607).normal(50, 10, size=40).tolist()

# ------------------------------------------------------------------ mean etc.


def test_mean_median_std_hand_values():
    x = [1, 2, 3, 4, 10]
    assert mean(x) == pytest.approx(4.0)
    assert median(x) == pytest.approx(3.0)
    assert std(x) == pytest.approx(math.sqrt(12.5))  # sum sq dev = 50, / (5-1)
    assert std(x, ddof=0) == pytest.approx(math.sqrt(10.0))  # 50 / 5


def test_against_statistics_module():
    assert mean(RNG_DATA) == pytest.approx(statistics.fmean(RNG_DATA), rel=1e-12)
    assert median(RNG_DATA) == pytest.approx(statistics.median(RNG_DATA), rel=1e-12)
    assert std(RNG_DATA) == pytest.approx(statistics.stdev(RNG_DATA), rel=1e-12)
    assert std(RNG_DATA, ddof=0) == pytest.approx(statistics.pstdev(RNG_DATA), rel=1e-12)


def test_median_even_n():
    assert median([1, 2, 3, 4]) == 2.5


def test_missing_omit_and_raise():
    x = [1, 2, None, 4, np.nan]
    assert mean(x) == pytest.approx(7 / 3)
    assert median(pd.Series(x)) == 2.0
    with pytest.raises(InputValidationError, match="missing"):
        mean(x, missing="raise")
    with pytest.raises(InputValidationError):
        mean(x, missing="drop")  # unknown policy


def test_all_missing_or_empty_raises():
    with pytest.raises(StatisticsError):
        mean([np.nan, np.nan])
    with pytest.raises(StatisticsError):
        median([])
    with pytest.raises(StatisticsError):
        mean([None, None])
    with pytest.raises(StatisticsError):
        std([5.0])  # sample SD needs n >= 2
    assert std([5.0], ddof=0) == 0.0


def test_input_validation():
    with pytest.raises(InputValidationError):
        mean(["a", "b"])
    with pytest.raises(InputValidationError):
        mean([True, False])  # booleans are not measurements
    with pytest.raises(InputValidationError):
        mean([1.0, np.inf])
    with pytest.raises(InputValidationError):
        mean(pd.DataFrame({"a": [1, 2]}))
    with pytest.raises(InputValidationError):
        std([1, 2, 3], ddof=2)
    with pytest.raises(InputValidationError):
        std([1, 2, 3], ddof=True)  # type: ignore[arg-type]


def test_nullable_integer_dtype_supported():
    s = pd.Series([1, 2, pd.NA, 4], dtype="Int64")
    assert mean(s) == pytest.approx(7 / 3)


# ---------------------------------------------------------- distribution_summary


def _pure_python_g1_g2(x: list[float]) -> tuple[float, float]:
    n = len(x)
    m = sum(x) / n
    s = math.sqrt(sum((v - m) ** 2 for v in x) / (n - 1))
    z = [(v - m) / s for v in x]
    g1 = n / ((n - 1) * (n - 2)) * sum(v**3 for v in z)
    g2 = n * (n + 1) / ((n - 1) * (n - 2) * (n - 3)) * sum(v**4 for v in z) - 3 * (n - 1) ** 2 / (
        (n - 2) * (n - 3)
    )
    return g1, g2


def test_distribution_summary_hand_values():
    d = distribution_summary([1, 2, 3, 4, 10, np.nan])
    assert (d.n, d.n_missing) == (5, 1)
    assert d.mean == pytest.approx(4.0)
    assert d.std == pytest.approx(math.sqrt(12.5))
    assert (d.minimum, d.q1, d.median, d.q3, d.maximum, d.iqr) == (1, 2, 3, 4, 10, 2)
    # hand derivation: G1 = 5/(4*3) * 180 / 12.5**1.5
    assert d.skewness == pytest.approx(5 / 12 * 180 / 12.5**1.5)


def test_distribution_shape_against_pure_python_formulas():
    d = distribution_summary(RNG_DATA)
    g1, g2 = _pure_python_g1_g2(RNG_DATA)
    assert d.skewness == pytest.approx(g1, rel=1e-9)
    assert d.excess_kurtosis == pytest.approx(g2, rel=1e-9)
    assert d.q1 == pytest.approx(float(np.percentile(RNG_DATA, 25)))


def test_distribution_summary_undefined_values_are_none():
    one = distribution_summary([3.0])
    assert one.std is None and one.skewness is None and one.excess_kurtosis is None
    two = distribution_summary([1.0, 2.0])
    assert two.std is not None and two.skewness is None
    three = distribution_summary([1.0, 2.0, 4.0])
    assert three.skewness is not None and three.excess_kurtosis is None
    const = distribution_summary([2.0, 2.0, 2.0, 2.0, 2.0])
    assert const.std == 0.0 and const.skewness is None and const.excess_kurtosis is None


def test_distribution_summary_missing_policy_and_json():
    with pytest.raises(InputValidationError):
        distribution_summary([1, np.nan, 3], missing="raise")
    d = distribution_summary([1.0])
    json.dumps(d.to_dict(), allow_nan=False)  # None, never NaN
    assert d.to_dict()["std"] is None


# -------------------------------------------------------------- Cronbach alpha

HAND = pd.DataFrame({"i1": [2, 4, 3, 5, 1], "i2": [3, 4, 3, 4, 2], "i3": [3, 5, 4, 5, 2]})


def test_alpha_hand_derived_value():
    # Derivation: item variances 2.5, 0.7, 1.7 (sum 4.9); totals 8,13,10,14,5
    # -> variance 13.5; alpha = 3/2 * (1 - 4.9/13.5) = 43/45.
    r = cronbach_alpha(HAND)
    assert r.alpha == pytest.approx(43 / 45, abs=1e-12)
    assert (r.n_items, r.n_observations, r.n_excluded) == (3, 5, 0)


def _exact_alpha(df: pd.DataFrame) -> Fraction:
    """Exact rational alpha from integer data (no floating point involved)."""
    rows = [[Fraction(int(v)) for v in row] for row in df.to_numpy()]
    n, k = len(rows), len(rows[0])

    def var(vals: list[Fraction]) -> Fraction:
        m = sum(vals) / len(vals)
        return sum((v - m) ** 2 for v in vals) / (len(vals) - 1)

    item_vars = sum(var([r[j] for r in rows]) for j in range(k))
    total_var = var([sum(r) for r in rows])
    assert n > 1
    return Fraction(k, k - 1) * (1 - item_vars / total_var)


def test_alpha_matches_exact_rational_arithmetic():
    rng = np.random.default_rng(7)
    data = pd.DataFrame(rng.integers(1, 6, size=(60, 6)), columns=list("abcdef"))
    expected = float(_exact_alpha(data))
    assert cronbach_alpha(data).alpha == pytest.approx(expected, rel=1e-12)


def test_alpha_matches_covariance_matrix_form():
    rng = np.random.default_rng(11)
    latent = rng.normal(size=(120, 1))
    data = pd.DataFrame(latent + rng.normal(scale=0.8, size=(120, 5)))
    cov = np.cov(data.to_numpy(), rowvar=False)
    k = cov.shape[0]
    expected = k / (k - 1) * (1 - np.trace(cov) / cov.sum())
    assert cronbach_alpha(data).alpha == pytest.approx(expected, rel=1e-12)


def test_alpha_perfectly_correlated_items_is_one():
    d = pd.DataFrame({"a": [1, 2, 3, 4], "b": [1, 2, 3, 4], "c": [1, 2, 3, 4]})
    assert cronbach_alpha(d).alpha == pytest.approx(1.0)


def test_alpha_invariant_to_shift_and_common_scaling():
    base = cronbach_alpha(HAND).alpha
    shifted = HAND + pd.Series({"i1": 10, "i2": -3, "i3": 0.5})
    assert cronbach_alpha(shifted).alpha == pytest.approx(base, rel=1e-12)
    assert cronbach_alpha(HAND * 7.0).alpha == pytest.approx(base, rel=1e-12)


def test_alpha_unreversed_items_negative_with_warning():
    # b is (almost) the reverse of a. Hand check: item variances 2.5 + 2.5 = 5;
    # totals 6,5,7,6,6 -> variance 0.5; alpha = 2 * (1 - 5/0.5) = -18.
    d = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [5, 3, 4, 2, 1]})
    with pytest.warns(UserWarning, match="negative"):
        r = cronbach_alpha(d)
    assert r.alpha == pytest.approx(-18.0)


def test_alpha_no_warning_when_positive():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cronbach_alpha(HAND)


def test_alpha_missing_listwise_default_and_raise():
    d = HAND.astype(float).copy()
    d.loc[len(d)] = [3.0, np.nan, 3.0]
    r = cronbach_alpha(d)
    assert r.n_excluded == 1 and r.n_observations == 5
    assert r.alpha == pytest.approx(43 / 45, abs=1e-12)  # same as complete-case data
    with pytest.raises(InputValidationError, match="missing"):
        cronbach_alpha(d, missing="raise")
    with pytest.raises(InputValidationError):
        cronbach_alpha(d, missing="pairwise")  # not offered


def test_alpha_columns_subset():
    d = HAND.assign(other=["x", "y", "z", "w", "v"])
    assert cronbach_alpha(d, columns=["i1", "i2", "i3"]).alpha == pytest.approx(43 / 45)
    with pytest.raises(ColumnError):
        cronbach_alpha(d)  # non-numeric column included
    with pytest.raises(ColumnError):
        cronbach_alpha(d, columns=["i1", "nope"])


def test_alpha_mean_inter_item_correlation_matches_numpy():
    r = cronbach_alpha(HAND)
    c = np.corrcoef(HAND.to_numpy(dtype=float), rowvar=False)
    expected = (c[0, 1] + c[0, 2] + c[1, 2]) / 3
    assert r.mean_inter_item_correlation == pytest.approx(expected, rel=1e-12)


def test_alpha_constant_item_still_computable_but_no_mean_r():
    d = pd.DataFrame({"a": [1, 2, 3, 4], "b": [2, 2, 2, 2], "c": [1, 3, 2, 4]})
    r = cronbach_alpha(d)
    assert r.mean_inter_item_correlation is None
    assert np.isfinite(r.alpha)


def test_alpha_error_conditions():
    with pytest.raises(InputValidationError, match="at least 2 items"):
        cronbach_alpha(HAND[["i1"]])
    with pytest.raises(StatisticsError, match="2 complete rows"):
        cronbach_alpha(HAND.iloc[:1])
    const_total = pd.DataFrame({"a": [1, 2, 3], "b": [3, 2, 1]})  # every row sums to 4
    with pytest.raises(StatisticsError, match="zero variance"):
        cronbach_alpha(const_total)
    with pytest.raises(InputValidationError):
        cronbach_alpha(np.ones((4, 3)))  # type: ignore[arg-type]
    with pytest.raises(InputValidationError):
        cronbach_alpha(pd.DataFrame({"a": [1.0, np.inf, 2.0], "b": [1.0, 2.0, 3.0]}))


def test_alpha_result_json_and_no_input_mutation():
    before = HAND.copy()
    r = cronbach_alpha(HAND)
    pd.testing.assert_frame_equal(HAND, before)
    d = r.to_dict()
    json.dumps(d, allow_nan=False)
    assert d["missing"] == "listwise"

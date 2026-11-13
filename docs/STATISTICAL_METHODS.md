# Statistical Methods

This document states, for every statistical function, the mathematical basis, the
assumptions, the limitations, how missing data are handled, and how the implementation is
verified. The docstrings in the source contain the same information.

## Conventions

- **No imputation, no silent cleaning.** Missing values are handled by an explicit `missing=`
  argument. Non-numeric input, booleans and infinite values raise errors.
- **Missing-data policies**

  | Function(s) | Allowed values | Meaning |
  |---|---|---|
  | `mean`, `median`, `std`, `distribution_summary`, `pearson`, `spearman` | `"omit"` (default), `"raise"` | `omit`: use available values (complete pairs for bivariate). `raise`: error if any value is missing |
  | `cronbach_alpha` | `"listwise"` (default), `"raise"` | rows with any missing item are excluded and counted in `n_excluded` |
  | `correlation_matrix` | `"pairwise"` (default), `"listwise"`, `"raise"` | pairwise uses each pair's complete rows (n varies by cell) |
  | `composite_score` | `min_valid` | minimum number of answered items per row (default: all) |

- **Undefined results** are reported by raising `StatisticsError` (single statistics) or by
  `None`/NaN entries (`DistributionSummary` fields, correlation-matrix cells), never by a
  silent number.
- **Sample standard deviation** uses `ddof=1` by default.

## Descriptive statistics

### `mean`, `median`, `std`

- Mean: `sum(x)/n`. Median: middle order statistic (average of the two middle values for even `n`).
- Standard deviation: `sqrt(sum((x - mean)^2) / (n - ddof))`; `ddof=1` is the sample SD.
- *Assumptions/limitations*: the mean and SD are sensitive to outliers and presuppose
  interval-level measurement; the median suits ordinal data. The sample SD needs `n >= 2`.

### `distribution_summary`

- Quartiles: linear interpolation between order statistics (NumPy default, Hyndman-Fan type 7).
  Other software may use a different definition and give slightly different quartiles.
- Skewness: adjusted Fisher-Pearson `G1 = n/((n-1)(n-2)) * sum(((x-mean)/s)^3)`; needs `n >= 3`.
- Excess kurtosis: bias-corrected `G2`; needs `n >= 4`. Both are `None` for constant data.
- *Limitations*: shape statistics are unstable in small samples and are not normality tests.

## Scoring

### `reverse_score`

`x' = scale_min + scale_max - x`. Responses outside `[scale_min, scale_max]` raise `ScoringError`
(no clipping). Missing stays missing. Which items are reverse-keyed comes from the
instrument's scoring manual, not from the data.

### `composite_score` / `score_questionnaire`

Row-wise mean `(1/m) * sum(x_i)` over the `m` answered items, or the sum over all `k` items.
With `min_valid < k`, the mean of answered items is used, which assumes unanswered items would
have been answered similarly; check the instrument's manual. Partial **sums** are refused
because they are biased downward.

The toolkit applies **no** cut-offs, norms or interpretive labels to scores.

## Cronbach's alpha (`cronbach_alpha`)

**Formula** (`k` items, item sample variances `s_i^2`, variance of row sums `s_T^2`, both `ddof=1`):

```
alpha = k/(k-1) * (1 - sum_i(s_i^2) / s_T^2)
      = k/(k-1) * (1 - trace(C) / sum(C))          # C = item covariance matrix
```

**Assumptions**

1. All items are scored in the same direction (reverse-score first).
2. Essential tau-equivalence (equal loadings on one common true score). Without it, alpha is a
   *lower bound* of reliability rather than an estimate of it.
3. Uncorrelated item errors.
4. Items are treated as interval-level; with few response categories alpha can understate
   reliability.

**Limitations and how not to interpret it**

- A high alpha is **not** evidence of unidimensionality and does **not** establish construct
  validity. It can be high for multidimensional item sets and rises with the number of items;
  `mean_inter_item_correlation` is reported to help separate those effects.
- Alpha describes a *score in a sample*, not an instrument in general.
- It can be negative (typically from unreversed items); the function then warns.
- Small samples give unstable estimates; **no confidence interval** is provided in v0.1.0.
- No threshold (e.g. ".70") is applied. Such conventions are context-dependent and contested.
- Alternatives to consider when assumptions are doubtful (not implemented here): McDonald's
  omega, ordinal alpha, factor-analytic reliability.

**Missing data**: `"listwise"` or `"raise"`. Pairwise deletion is deliberately not offered
because it can produce inconsistent covariance matrices.

**Errors**: fewer than 2 items; fewer than 2 complete rows; zero variance of the total score.

## Correlation

### `pearson`

- `r = sum((x-mean_x)(y-mean_y)) / sqrt(sum((x-mean_x)^2) * sum((y-mean_y)^2))`.
- p-value: `t = r*sqrt((n-2)/(1-r^2))` with `n-2` degrees of freedom (H0: rho = 0).
- Interval: Fisher z, `tanh(atanh(r) +/- z_{1-alpha/2}/sqrt(n-3))`; omitted when `n <= 3` or `|r| = 1`.
- *Assumptions*: linear association, interval-level data, independent observations; roughly
  bivariate-normal data for the p-value and interval.
- *Limitations*: sensitive to outliers and restricted range; not evidence of causation; the
  interval is approximate.

### `spearman`

- Pearson's `r` on average ranks (ties receive the mean rank). Without ties:
  `1 - 6*sum(d^2)/(n(n^2-1))`.
- p-value from the t approximation with `n-2` df (as in SciPy); approximate for small `n`.
- *Assumptions*: monotonic association, at least ordinal data, independent observations.
- *Limitations*: no confidence interval in v0.1.0; many ties reduce resolution.

### `correlation_matrix`

Computes the pairwise results above for every column pair, returning `r`, `p_values` and `n`
matrices. Cells with fewer than 3 complete pairs or a constant variable are NaN.
**p-values are not corrected for multiple comparisons**; with many pairs, some small p-values
are expected by chance. Pairwise deletion can give a matrix that is not positive semi-definite.

## Verification

Reference calculations are independent of the implementation. Tests live in `tests/`.

| Function | Independent reference used in tests |
|---|---|
| `mean`, `median`, `std` | Python `statistics` module (`fmean`, `median`, `stdev`, `pstdev`); hand values |
| `distribution_summary` | hand-derived quartiles and `G1`; pure-Python `G1`/`G2` formulas |
| `cronbach_alpha` | hand-derived `43/45` on a 5x3 table; exact rational arithmetic with `fractions.Fraction`; covariance-matrix form with NumPy; shift/scale invariance; `-18` hand case for unreversed items |
| `pearson` | hand-derived `6/sqrt(60)`; `numpy.corrcoef`; t-distribution identity for the p-value; manual Fisher-z interval and SciPy's `confidence_interval` |
| `spearman` | textbook `1 - 6*sum(d^2)/(n(n^2-1))` (no ties); rank-then-Pearson (ties) |
| `correlation_matrix` | `numpy.corrcoef`; rank-then-Pearson; bivariate functions cell by cell |
| `reverse_score`, `composite_score`, `score_questionnaire` | hand-computed rows; involution property |

## References

- Cronbach, L. J. (1951). Coefficient alpha and the internal structure of tests. *Psychometrika, 16*(3), 297-334.
- Cortina, J. M. (1993). What is coefficient alpha? An examination of theory and applications. *Journal of Applied Psychology, 78*(1), 98-104.
- Sijtsma, K. (2009). On the use, the misuse, and the very limited usefulness of Cronbach's alpha. *Psychometrika, 74*(1), 107-120.
- McNeish, D. (2018). Thanks coefficient alpha, we'll take it from here. *Psychological Methods, 23*(3), 412-433.
- Fisher, R. A. (1921). On the "probable error" of a coefficient of correlation deduced from a small sample. *Metron, 1*, 3-32.
- Spearman, C. (1904). The proof and measurement of association between two things. *American Journal of Psychology, 15*(1), 72-101.
- Joanes, D. N., & Gill, C. A. (1998). Comparing measures of sample skewness and kurtosis. *Journal of the Royal Statistical Society: Series D, 47*(1), 183-189.
- Hyndman, R. J., & Fan, Y. (1996). Sample quantiles in statistical packages. *The American Statistician, 50*(4), 361-365.

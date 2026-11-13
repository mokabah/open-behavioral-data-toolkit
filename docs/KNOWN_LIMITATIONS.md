# Known Limitations (v0.1.0)

Stated plainly so that users and reviewers can judge fitness for their purpose.

## Maturity

- **Alpha software.** No external researcher has reviewed or used it yet. No adoption, download
  or community metrics exist or are claimed.
- It is **not** claimed to be production-ready. Verify important results against a tool you trust.
- The public API may change in 0.x releases.

## Scope

- No diagnostic, screening or clinical functionality, by design. No cut-offs, norms or severity labels.
- Input is CSV only (via pandas). No Excel/SPSS/Stata readers, no codebook/metadata support.
- No imputation, no outlier handling, no data transformation beyond reverse scoring.
- No inferential modelling (regression, ANOVA, mixed models), factor analysis, IRT, or effect sizes.

## Statistical

- Cronbach's alpha: no confidence interval; no alpha-if-item-deleted or item-total correlations;
  no omega or ordinal alpha. Alpha is not evidence of unidimensionality or construct validity.
- Correlations: p-values are unadjusted for multiple comparisons. Spearman has no confidence
  interval and its p-value is the t approximation. Kendall's tau is not provided.
  Pairwise-deleted matrices may not be positive semi-definite.
- Quartiles use one definition (linear interpolation, type 7); other packages may differ.
- Skewness/kurtosis are unstable in small samples and are descriptive only.
- Observations are assumed independent; clustered/repeated-measures data are not handled.

## Engineering

- Local verification was done with Python 3.13, pandas 3.0, NumPy 2.5 and SciPy 1.18 only.
  CI is configured for Python 3.11-3.13 on Linux and Windows and for the minimum declared
  dependency versions, but **those CI runs have not been executed yet** (the repository had not
  been pushed when this was written). Declared minimum versions are derived from API needs and
  are untested until CI runs.
- Line/branch coverage was not measured locally (the coverage plugin was unavailable offline).
  CI reports coverage once the repository is pushed.
- `mypy --strict` passes on the source package; tests are not type-checked.
- `load_csv` assumes a header in the first row and does not detect ambiguous encodings.
- Report fingerprints identify content, but missing values and empty strings serialise
  identically.
- Large datasets are untested; everything runs in memory with pandas.

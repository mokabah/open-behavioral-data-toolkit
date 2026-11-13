# Technical Architecture

## Layout (src-layout)

```
src/behavioral_data_toolkit/
    __init__.py        public API + __version__
    _version.py        single source of the version string
    exceptions.py      exception hierarchy
    _checks.py         internal input validation helpers (private)
    loader.py          CSV ingestion, structure inspection, column validation
    validator.py       missing / duplicate / invalid-value detection, quality summary
    scoring.py         reverse scoring, composite scores, questionnaire scoring
    stats.py           mean, median, std, distribution summary, Cronbach's alpha
    correlation.py     Pearson, Spearman, correlation matrices
    reporting.py       structured reports, JSON/Markdown export, reproducibility metadata
    plotting.py        optional Matplotlib helpers (lazy import)
    py.typed           PEP 561 marker
```

## Design principles

1. **Pure functions over frameworks.** Functions take pandas/NumPy inputs and
   return plain values, frozen dataclasses, or DataFrames. No global state, no
   hidden configuration.
2. **Explicit missing-data policy.** Every function that can see missing values
   has a `missing=` argument whose allowed values are documented and validated.
   Nothing is imputed.
3. **Fail loudly.** Invalid input raises a subclass of
   `BehavioralDataToolkitError` (input errors also subclass `ValueError`).
   No silent coercion of strings to numbers, no silent clipping.
4. **No mutation.** Inputs are never modified; functions return new objects.
5. **Layering (one-directional dependencies).**
   `exceptions` <- `_checks` <- `{loader, validator, scoring, stats}` <-
   `correlation` <- `reporting`. `plotting` depends only on result types and is
   optional. `reporting` never computes statistics; it only serialises results.
6. **Reproducibility.** Reports carry toolkit/Python/NumPy/pandas/SciPy versions,
   a SHA-256 fingerprint of the analysed data, user-declared parameters, and an
   optional timestamp (can be disabled for byte-identical output).
7. **Minimal dependencies.** Runtime: pandas, NumPy, SciPy. Matplotlib is an
   optional extra. Markdown tables are rendered without `tabulate`.

## Result types

| Function | Returns |
|---|---|
| `inspect_dataframe` | `DataStructure` (dataclass) |
| `data_quality_summary` | `DataQualitySummary` (dataclass, holds DataFrames) |
| `distribution_summary` | `DistributionSummary` (dataclass) |
| `cronbach_alpha` | `CronbachAlphaResult` (dataclass) |
| `pearson` / `spearman` | `CorrelationResult` (dataclass) |
| `correlation_matrix` | `CorrelationMatrix` (dataclass of DataFrames) |

All result types expose `to_dict()` returning JSON-safe values (NaN -> `None`).

## Testing strategy

- `pytest`, one test module per source module.
- Every statistical function is verified against an independent reference:
  hand-derived values, Python's `statistics` module, alternative algebraic
  formulas (e.g. covariance-matrix form of alpha), and SciPy.
- Input-validation and missing-data branches are tested explicitly.
- Reports are tested for valid JSON, determinism, and metadata.

## Public API stability

v0.1.x is pre-1.0: the API may change between minor versions. Changes are
recorded in `CHANGELOG.md`.

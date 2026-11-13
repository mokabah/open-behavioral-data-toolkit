# Usage Guide

A step-by-step walk through a typical analysis. Run the snippets from the **repository root**,
in order, in one Python session. All data are **synthetic**
(`examples/data/synthetic_survey.csv`); nothing here is an empirical finding.

These snippets are executed by the test suite (`tests/test_docs.py`), so they stay correct.

## 1. Load and inspect

`load_csv` refuses files with empty or duplicate column names and files without data rows,
but it does not clean values.

```python
import behavioral_data_toolkit as bdt

df = bdt.load_csv("examples/data/synthetic_survey.csv", required_columns=["participant_id"])
info = bdt.inspect_dataframe(df)
print(info.n_rows, info.n_columns)
print(info.columns.head())
```

## 2. Validate

Validation only reports; it never changes the data.

```python
items = [f"sw{i}" for i in range(1, 7)] + [f"sf{i}" for i in range(1, 5)]
rules = {c: bdt.ValueRule(minimum=1, maximum=5) for c in items}
quality = bdt.data_quality_summary(df, duplicate_subset=["participant_id"], rules=rules)
print(quality.n_duplicate_rows, quality.n_invalid_values, quality.is_clean)
print(bdt.find_invalid_values(df, rules))
print(bdt.missing_summary(df, items).head())
```

## 3. Clean (your decisions)

What to do with duplicates, out-of-range codes and missing values is a research decision.
Make it explicit in code, so it is visible and reproducible.

```python
clean = df.drop_duplicates(subset="participant_id", keep="first").copy()
clean[items] = clean[items].astype(float)
for row in quality.invalid.itertuples():
    clean.loc[row.row, row.column] = float("nan")
clean = clean.reset_index(drop=True)
```

## 4. Score a questionnaire

`ScaleDefinition` states the scale once: items, response range, reverse-keyed items, and the
missing-data rule. Responses outside the range raise `ScoringError`.

```python
wellbeing = bdt.ScaleDefinition(
    name="synthetic_wellbeing",
    items=("sw1", "sw2", "sw3", "sw4", "sw5", "sw6"),
    min_value=1,
    max_value=5,
    reverse_items=("sw2", "sw5"),
    min_valid=5,  # illustrative: allow one unanswered item; real instruments define their own rule
)
scores = bdt.score_questionnaire(clean, wellbeing)
print(scores.describe())
```

`composite_score` and `reverse_score` are also available separately.

## 5. Descriptive statistics

```python
summary = bdt.distribution_summary(scores, missing="omit")
print(summary.n, summary.n_missing, round(summary.mean, 3), round(summary.std, 3))
print(bdt.mean(scores), bdt.median(scores), bdt.std(scores))
```

`missing="raise"` turns any missing value into an error instead.

## 6. Cronbach's alpha

Reverse-score first. Read the assumptions and limitations in
[STATISTICAL_METHODS.md](STATISTICAL_METHODS.md): alpha is **not** evidence of unidimensionality
or construct validity.

```python
reversed_items = bdt.reverse_score(clean, 1, 5, columns=["sw2", "sw5"])
alpha = bdt.cronbach_alpha(reversed_items, list(wellbeing.items))
print(round(alpha.alpha, 3), alpha.n_observations, alpha.n_excluded)
print(alpha.mean_inter_item_correlation)
```

## 7. Correlation

```python
result = bdt.pearson(clean["age"], scores)
print(result.r, result.p_value, result.n, result.n_excluded)

analysis = clean[["age", "screen_hours"]].assign(wellbeing=scores)
matrix = bdt.correlation_matrix(analysis, method="spearman", missing="pairwise")
print(matrix.r)
print(matrix.to_long())
```

p-values are not adjusted for multiple comparisons.

## 8. Report

`AnalysisReport` stores results plus reproducibility metadata: library versions, a SHA-256
fingerprint of the data, the parameters you declare, and an optional timestamp.

```python
report = bdt.AnalysisReport(
    "Demo analysis",
    data=clean,
    parameters={"data_source": "synthetic", "alpha_missing": "listwise"},
    include_timestamp=False,  # byte-identical output across runs
)
report.add_text("Note", "Synthetic data; descriptive results only.")
report.add_result("Cronbach's alpha", alpha)
report.add_result("Correlations", matrix)
markdown = report.to_markdown()  # pass a path to write a file: report.to_markdown("report.md")
print(markdown[:80])
```

`report.to_json(path)` writes strict JSON (no NaN/Infinity; undefined values are `null`).

## 9. Optional plots

Requires Matplotlib (`pip install -e ".[plot]"`).

```python
ax = bdt.plot_distribution(scores, title="Synthetic wellbeing score")
ax.figure.savefig("distribution.png", dpi=150)

ax = bdt.plot_correlation_heatmap(matrix)
ax.figure.savefig("correlations.png", dpi=150)
```

## Error handling

All deliberate errors derive from `BehavioralDataToolkitError`; invalid-argument errors also
derive from `ValueError`:

| Exception | Raised when |
|---|---|
| `DataLoadError` | file missing/unreadable/empty/malformed |
| `InputValidationError` | wrong type, bad option value, unsupported missing-data request |
| `ColumnError` | missing or non-numeric columns |
| `ScoringError` | responses outside the declared scale range |
| `StatisticsError` | statistic undefined (too few values, constant variable, zero total variance) |

## Full workflow

[`examples/example_workflow.py`](../examples/example_workflow.py) runs all of the above and writes
a report plus two figures to `examples/output/`.

# Roadmap

A transparent, non-binding list of candidate work. Items are **candidates, not commitments**;
priorities follow contributor interest and scientific review capacity. Open an issue to discuss
or claim an item (see `scripts/seed_github.sh` for the starter issue set).

## v0.1.0 (current)

Loader, validator, scoring, descriptive statistics, Cronbach's alpha, Pearson/Spearman,
correlation matrices, reporting. See [CHANGELOG.md](../CHANGELOG.md).

## v0.2.0 candidates (psychometrics and robustness)

- Alpha-if-item-deleted and corrected item-total correlations.
- Confidence interval for Cronbach's alpha (with a clearly stated method and reference test).
- Multiple-comparison adjustment (Holm, Bonferroni, Benjamini-Hochberg) for correlation matrices.
- Kendall's tau-b.
- Jupyter notebook version of the example workflow.
- Property-based tests (Hypothesis) for scoring invariants.

## v0.3.0 candidates (breadth)

- McDonald's omega (requires careful scientific review).
- Readers for Excel (`.xlsx`) and optionally SPSS `.sav` via optional extras.
- Codebook support: declare scale definitions and valid ranges in a YAML/JSON file.
- Effect sizes for group comparisons (Cohen's d, Hedges' g) with confidence intervals.

## Later / under discussion

- Command-line interface for validation and reporting.
- Internationalised report templates.
- Documentation site.

## Not planned

Diagnostic, screening or clinical-interpretation features. The toolkit is for descriptive
research workflows only.

## How releases are decided

A release requires: all tests passing in CI, the QA checklist in `docs/QA_REPORT.md` repeated,
an updated `CHANGELOG.md` and `docs/KNOWN_LIMITATIONS.md`, and statistical changes reviewed by
someone other than the author where possible.

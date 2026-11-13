# Feasibility Assessment (v0.1.0)

## Verdict

Feasible within the stated scope, provided v0.1.0 stays small and honest:
a *foundation* toolkit for data loading, validation, scoring, descriptive
statistics, correlation, and reporting. It is not a replacement for mature
packages (`pingouin`, `statsmodels`, `scipy.stats`, `factor_analyzer`); its value
is a transparent, well-tested, reproducible **workflow layer** with explicit
missing-data handling and psychometric-aware defaults.

## Why the scope is realistic

| Area | Risk | Assessment |
|---|---|---|
| Data loading / validation | Low | Thin, well-defined wrappers around pandas |
| Scoring (reverse, composite) | Low | Simple arithmetic; main risk is silent misuse (out-of-range values, partial responses) - handled by explicit checks |
| Descriptive statistics | Low | Delegates to NumPy/SciPy; the work is input validation and missing-data policy |
| Cronbach's alpha | Medium | Formula is simple; risk is *over-interpretation*. Mitigated by documentation, warnings, no "reliability is acceptable" verdicts |
| Pearson / Spearman | Low-Medium | Delegates to SciPy; Fisher-z CI implemented independently and cross-checked |
| Reporting / reproducibility | Medium | Needs deterministic output, data fingerprint, environment metadata, valid JSON (no NaN) |

## Principal risks and mitigations

1. **Scientific misuse.** Alpha treated as validity; p-values without multiplicity
   control; scores read as diagnoses.
   *Mitigation:* documented assumptions/limitations per function
   (`docs/STATISTICAL_METHODS.md`), warnings, explicit non-clinical notice in
   README and every generated report.
2. **Silent missing-data handling.** *Mitigation:* every statistical function
   takes an explicit `missing=` policy; there is no hidden default imputation.
3. **Numerical correctness.** *Mitigation:* tests compare against independent
   references (hand-derived values, pure-Python `statistics` module, alternative
   algebraic formulas, SciPy).
4. **Dependency drift** (pandas 3.x changes). *Mitigation:* CI matrix over
   Python 3.11-3.13; minimal dependencies.
5. **Scope creep.** *Mitigation:* roadmap defers factor analysis, effect sizes,
   multiple-comparison correction, and a CLI to later versions.

## Out of scope for v0.1.0

Inferential modelling (regression, ANOVA), factor analysis / IRT, imputation,
multiple-comparison correction, clinical scoring manuals, any diagnostic output.

## Success criteria

- All tests pass in a clean environment (actually executed, results recorded in
  `docs/QA_REPORT.md`).
- Every public statistical function documents assumptions and is verified
  against an independent reference.
- Every example in the documentation runs.
- Known limitations are listed, not hidden.

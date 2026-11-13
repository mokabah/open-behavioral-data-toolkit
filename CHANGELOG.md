# Changelog

All notable changes are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to follow
[Semantic Versioning](https://semver.org/) (pre-1.0: minor releases may break the API).

## [0.1.0] - unreleased

Initial version, prepared for release. Set the release date when the tag is created.

### Added

- `loader`: `load_csv` (rejects empty/duplicate headers and empty files), `inspect_dataframe`,
  `validate_columns`.
- `validator`: `missing_summary`, `find_duplicates`, `find_invalid_values` (`ValueRule`),
  `data_quality_summary`.
- `scoring`: `reverse_score`, `composite_score`, `ScaleDefinition`, `score_questionnaire`,
  `score_questionnaires`.
- `stats`: `mean`, `median`, `std`, `distribution_summary`, `cronbach_alpha` (with documented
  assumptions and limitations; negative-alpha warning).
- `correlation`: `pearson` (Fisher-z confidence interval), `spearman`, `correlation_matrix`
  (pairwise/listwise/raise missing policies, p-value and sample-size matrices).
- `reporting`: `AnalysisReport` with strict-JSON and Markdown export, data fingerprint
  (SHA-256), environment metadata, optional timestamp.
- `plotting` (optional extra): `plot_distribution`, `plot_correlation_heatmap`.
- Synthetic example dataset, generator and end-to-end example workflow.
- Test suite with independent reference calculations; CI workflow; issue and PR templates.
- Documentation: usage guide, statistical methods, architecture, roadmap, known limitations.

### Known limitations

See [docs/KNOWN_LIMITATIONS.md](docs/KNOWN_LIMITATIONS.md).

# QA Report (v0.1.0)

Records what was **actually executed** on 2026-11-13 in a single local environment. Nothing
below is claimed beyond these runs. Repeat this checklist before every release.

## Environment

Python 3.13.18, pandas 3.0.5, NumPy 2.5.3, SciPy 1.18.1, Matplotlib 3.11.2, pytest 9.1.1,
ruff 0.16.9, mypy 2.3.1. Linux. No network (packages could not be installed).

## Results

| Check | Command / method | Result |
|---|---|---|
| Full test suite | `pytest` | 143 passed |
| Tests against the *installed* package | wheel installed into a fresh venv; `pytest` run with `src/` not on the path | 143 passed |
| Deprecation/future warnings as errors | `python -W error::FutureWarning -W error::DeprecationWarning -m pytest` | 143 passed |
| Lint | `ruff check .`, `ruff format --check .` | clean |
| Types | `mypy` (strict) on `src/behavioral_data_toolkit` | no issues (12 files) |
| Imports | every submodule imported; every name in `__all__` resolved | 11 modules, 39 names, 0 unresolved |
| Package metadata | inspected wheel `METADATA` | name, version 0.1.0, MIT, `>=3.11`, dependencies, extras, `py.typed` present |
| Wheel / sdist contents | inspected | wheel has all modules + LICENSE; sdist includes docs, examples, tests, CITATION.cff, community files |
| Documentation examples | `tests/test_docs.py` executes every ```python block in README.md and docs/USAGE.md; checks internal links | pass |
| Example workflow | runs end to end; wellbeing alpha recomputed independently with NumPy (covariance form) | matches to 1e-10; output reproducible byte for byte |
| Statistical outputs | independent references (see STATISTICAL_METHODS.md > Verification) | pass |
| License / citation | LICENSE is MIT; CITATION.cff parses as YAML | present; author and repository URL need maintainer confirmation (below) |
| GitHub config | all YAML files parse; `scripts/seed_github.sh` syntax-checked and dry-run | valid; script **not** run against GitHub |

Tests per module: correlation 30, scoring 27, stats 24, loader 20, validator 15, reporting 14,
plotting 5, examples 5, docs 3 (total 143, from test collection).

## Bugs found and fixed during QA

1. `[]` and `[None, None]` were reported as "non-numeric" (pandas gives object dtype); now treated as all-missing and raise `StatisticsError`.
2. NumPy integer/float scalars (e.g. `np.int64`) were rejected for scale bounds, `min_valid`, `confidence`; now accepted.
3. DataFrames with duplicate column names produced a misleading "not numeric" error; now a clear `ColumnError`.
4. Empty tables/lists rendered confusingly in Markdown reports; now `*(no rows)*` / `(empty)`.

## Not verified (honest gaps)

- Python 3.11 and 3.12, Windows/macOS, and the minimum declared dependency versions
  (numpy 1.26, pandas 2.1, scipy 1.11): configured in CI but **never run**.
- Line/branch coverage: not measured (coverage tooling unavailable offline).
- `python -m build` and `twine check` were not available; the wheel/sdist were built with
  setuptools directly.
- GitHub Actions workflow and issue forms were validated as YAML only, not executed by GitHub.
- No external scientific review has taken place.

## Unresolved items and known limitations

See [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md). No known failing tests or open bugs at the
time of writing; this does not mean the absence of bugs.

## Not production-ready

No evidence supports a production-readiness claim. v0.1.0 is alpha software.

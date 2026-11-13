# Contributing

Thank you for considering a contribution. This project values **correctness, transparency
and reproducibility** over feature count, so small, well-tested changes are preferred.

By participating you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Ways to help

- Report a bug or a statistical-correctness concern (use the issue templates).
- Add tests or independent reference calculations for existing functions.
- Improve documentation, examples and error messages.
- Pick up an issue labelled `good first issue` or `help wanted`.
- Review pull requests, particularly the mathematics.

## Development setup

```bash
git clone https://github.com/meworo/open-behavioral-data-toolkit.git
cd open-behavioral-data-toolkit
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Run the checks that CI runs:

```bash
pytest                 # test suite
ruff check .           # lint
ruff format .          # formatting
mypy                   # type checking (use Python 3.12+ for mypy with current NumPy stubs)
```

## Standards for code

- Python 3.11+, type hints on public functions, PEP 8 (enforced by ruff), docstrings in Google style.
- Keep modules small and dependencies minimal (runtime: pandas, NumPy, SciPy). Discuss new
  dependencies in an issue first.
- Never modify inputs in place. Raise `BehavioralDataToolkitError` subclasses for invalid input;
  do not silently coerce, clip or impute.

## Standards for statistical contributions

A new or changed statistical function must include:

1. **Mathematical basis** in the docstring (formula, estimator, degrees of freedom, etc.).
2. **Assumptions and limitations**, including what the result must *not* be interpreted as.
3. **Explicit missing-data handling** through a documented `missing=` argument.
4. **Input validation** with clear error messages.
5. **Tests against an independent reference**: a hand-derived value (show the derivation in a
   comment), exact rational arithmetic, an alternative algebraic form, or another trusted
   implementation. A test that only re-runs the same formula is not sufficient.
6. An entry in `docs/STATISTICAL_METHODS.md` and in `CHANGELOG.md`.

## Data policy

- **Never commit real participant data**, even anonymised, and never data you do not have the
  right to share. Use synthetic data only (see `examples/generate_synthetic_data.py`).
- Synthetic data must be labelled as synthetic everywhere it appears. Do not present synthetic
  results as empirical findings.
- The toolkit must not gain diagnostic, screening or clinical-interpretation features (no
  cut-offs, severity bands or labels).

## Pull request process

1. Open or comment on an issue first for anything beyond a small fix.
2. Create a branch, make focused commits, and add tests.
3. Make sure `pytest`, `ruff check .` and `mypy` pass locally.
4. Open a pull request using the template. Describe what changed and how you verified it.
5. A maintainer reviews; please allow reasonable time. Be ready to revise.

## Reporting bugs

Use the **Bug report** issue template. Include the smallest data (synthetic or invented) that
reproduces the problem, the exact code, the full traceback, and the output of:

```python
import behavioral_data_toolkit as bdt

print(bdt.collect_environment_metadata())
```

For a suspected statistical error, use the **Statistical correctness** template and, where
possible, give the reference value and where it came from.

## Releases

Maintainers follow semantic versioning and keep `CHANGELOG.md` current. Versions before 1.0
may include breaking changes in minor releases, which will be listed in the changelog.

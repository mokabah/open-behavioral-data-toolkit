#!/usr/bin/env bash
# Create the project's labels and a starter set of clearly scoped issues.
#
# Safe by default: prints what it WOULD do. Nothing is sent to GitHub unless you pass --apply.
# Requires the GitHub CLI (`gh`) authenticated with permission to manage issues on the repo.
#
# Usage:
#   scripts/seed_github.sh OWNER/REPO            # dry run
#   scripts/seed_github.sh OWNER/REPO --apply    # create labels and issues
set -euo pipefail

REPO="${1:-}"
MODE="${2:-}"
if [[ -z "$REPO" || "$REPO" != */* ]]; then
  echo "usage: $0 OWNER/REPO [--apply]" >&2
  exit 2
fi
APPLY=false
[[ "$MODE" == "--apply" ]] && APPLY=true

run() {
  if $APPLY; then "$@"; else printf 'DRY RUN: '; printf '%q ' "$@"; printf '\n'; fi
}

label() { run gh label create "$1" --color "$2" --description "$3" --force --repo "$REPO"; }
issue() { run gh issue create --repo "$REPO" --title "$1" --label "$2" --body "$3"; }

label "good first issue"          7057ff "Small, clearly scoped; suitable for a first contribution"
label "help wanted"               008672 "Maintainers would welcome a contribution"
label "bug"                       d73a4a "Something does not work as documented"
label "enhancement"               a2eeef "New feature or improvement"
label "documentation"             0075ca "Documentation and examples"
label "statistics"                fbca04 "Involves a statistical method"
label "scientific-correctness"    b60205 "Mathematical or methodological correctness"
label "testing"                   c5def5 "Tests and reference calculations"
label "needs-review"              e4e669 "Needs scientific or maintainer review"

issue "Docs: add a glossary of terms" "good first issue,documentation" \
"Add \`docs/GLOSSARY.md\` defining the terms used in the toolkit: missing value, listwise vs pairwise deletion, reverse scoring, composite score, tau-equivalence, Fisher z, and ddof.

**Acceptance criteria**
- Each term has a one- or two-sentence plain-language definition and, where relevant, a link to the function that uses it.
- Linked from README.md and docs/USAGE.md.
- No claims beyond what docs/STATISTICAL_METHODS.md already states."

issue "Docs: Windows setup instructions" "good first issue,documentation" \
"The README and CONTRIBUTING.md show a macOS/Linux virtual-environment activation command. Verify the full setup on Windows (PowerShell and cmd) and document any differences.

**Acceptance criteria**
- Setup steps tested on Windows and written down.
- \`pytest\`, \`ruff check .\` and \`mypy\` instructions confirmed to work."

issue "Tests: property-based tests for scoring invariants" "good first issue,testing" \
"Add Hypothesis tests (dev dependency only) for: reverse_score applied twice returns the input; composite mean lies between the minimum and maximum item value; composite_score never returns a value when fewer than min_valid items are answered.

**Acceptance criteria**
- Tests in tests/test_scoring_properties.py; hypothesis added to the dev extra in pyproject.toml.
- No change to runtime dependencies."

issue "Examples: Jupyter notebook version of the example workflow" "good first issue,documentation" \
"Create examples/example_workflow.ipynb that mirrors examples/example_workflow.py with short explanations. Use the synthetic dataset only and label it as synthetic in the first cell.

**Acceptance criteria**
- Notebook runs top to bottom in a clean environment with the outputs cleared before commit.
- Mentions that cleaning decisions are the analyst's, not the toolkit's."

issue "Stats: alpha-if-item-deleted and corrected item-total correlations" "help wanted,statistics,needs-review" \
"Add a function returning, for each item, (a) Cronbach's alpha with that item removed and (b) the corrected item-total correlation (item vs sum of the other items).

**Acceptance criteria**
- Docstring with formula, assumptions and limitations; explicitly warn against deleting items on statistical grounds alone.
- Explicit missing= policy (listwise or raise), consistent with cronbach_alpha.
- Tests against an independent reference: hand-derived values on a small table and brute-force recomputation by dropping each column.
- Entry in docs/STATISTICAL_METHODS.md and CHANGELOG.md."

issue "Stats: confidence interval for Cronbach's alpha" "help wanted,statistics,scientific-correctness,needs-review" \
"Propose and implement a confidence interval for alpha. Please discuss the method (e.g. Feldt's F-based interval vs bootstrap) in this issue before coding, citing sources and stating assumptions.

**Acceptance criteria**
- Agreed method with a reference.
- Verified against an independent implementation or published worked example.
- Limitations documented (normality, sample size)."

issue "Correlation: multiple-comparison adjustment for correlation_matrix" "help wanted,statistics" \
"Add an optional adjust= argument (none, holm, bonferroni, bh) to correlation_matrix that returns adjusted p-values alongside the unadjusted ones.

**Acceptance criteria**
- Default stays unadjusted; adjusted values returned in an additional matrix.
- Tests against hand-computed adjustments on a small p-value vector.
- Documentation explains what each adjustment controls (family-wise error vs false discovery rate)."

issue "Correlation: Kendall's tau-b" "help wanted,statistics" \
"Add kendall(x, y, missing=...) returning CorrelationResult with method='kendall', and allow method='kendall' in correlation_matrix.

**Acceptance criteria**
- Documented formula, tie handling and assumptions.
- Tests: hand-derived value on a small table without ties, and a brute-force concordant/discordant pair count.
- Docs and changelog updated."

echo
$APPLY && echo "Done." || echo "Dry run only. Re-run with --apply to create labels and issues."

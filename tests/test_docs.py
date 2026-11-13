"""Execute every ```python block in the documentation so examples cannot rot.

Blocks of one document run in order in a shared namespace (they form a tutorial).
They run in a temporary working directory; repository-relative data paths are rewritten
to absolute paths.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = ["README.md", "docs/USAGE.md"]
BLOCK = re.compile(r"^```python\n(.*?)^```", re.DOTALL | re.MULTILINE)


@pytest.mark.parametrize("doc", DOCS)
def test_python_blocks_execute(doc, tmp_path, monkeypatch):
    text = (ROOT / doc).read_text(encoding="utf-8")
    blocks = BLOCK.findall(text)
    assert blocks, f"no python blocks found in {doc}"
    if any("plot_" in b for b in blocks):
        pytest.importorskip("matplotlib")
    monkeypatch.chdir(tmp_path)
    namespace: dict[str, object] = {"__name__": "__docs__"}
    data_path = (ROOT / "examples" / "data" / "synthetic_survey.csv").as_posix()
    for number, block in enumerate(blocks, start=1):
        code = block.replace("examples/data/synthetic_survey.csv", data_path)
        try:
            exec(compile(code, f"{doc}[block {number}]", "exec"), namespace)  # noqa: S102
        except Exception as exc:  # pragma: no cover - only on failure
            pytest.fail(f"{doc} python block {number} failed: {exc!r}\n---\n{block}")


def test_internal_markdown_links_resolve():
    link = re.compile(r"\]\((?!https?://|#|mailto:)([^)#\s]+)")
    for doc in [
        "README.md",
        "CONTRIBUTING.md",
        "CHANGELOG.md",
        "docs/USAGE.md",
        "docs/ROADMAP.md",
        "docs/ARCHITECTURE.md",
        "docs/STATISTICAL_METHODS.md",
        "docs/KNOWN_LIMITATIONS.md",
        "docs/FEASIBILITY.md",
        "docs/QA_REPORT.md",
    ]:
        path = ROOT / doc
        if not path.exists():
            continue
        for target in link.findall(path.read_text(encoding="utf-8")):
            assert (path.parent / target).resolve().exists(), f"{doc}: broken link -> {target}"

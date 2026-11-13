"""Structured analysis reports with reproducibility metadata.

A report collects already-computed results (it never computes statistics
itself), records the software environment, a fingerprint of the analysed data and
the parameters the analyst declares, and exports to strict JSON or Markdown.

Reports are descriptive statistical summaries. They contain no diagnostic or
clinical conclusions and must not be read as such.
"""

from __future__ import annotations

import hashlib
import json
import platform
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy

from ._checks import require_dataframe
from ._serialize import frame_to_records, to_json_safe
from ._version import __version__
from .exceptions import InputValidationError

__all__ = ["AnalysisReport", "collect_environment_metadata", "dataframe_fingerprint"]

NOTICE = (
    "This report is a descriptive statistical summary produced by Open Behavioral "
    "Data Toolkit. It does not contain diagnoses or clinical conclusions and must "
    "not be used as a basis for them."
)


def collect_environment_metadata() -> dict[str, str]:
    """Return versions of Python and the numerical libraries in use."""
    return {
        "toolkit_version": __version__,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scipy_version": scipy.__version__,
    }


def dataframe_fingerprint(df: pd.DataFrame) -> str:
    """SHA-256 hex digest of a canonical CSV serialisation of ``df``.

    The digest depends on column names, column order, row order and values (floats
    are written with their shortest round-trip representation). It does not
    depend on the index. Missing values and empty strings serialise identically.
    It identifies *this content*, so any change to the data changes the digest.
    """
    require_dataframe(df, "df")
    payload = df.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _is_table(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == {"columns", "rows"}
        and isinstance(value["columns"], list)
        and isinstance(value["rows"], list)
    )


def _table_lines(table: Mapping[str, Any]) -> list[str]:
    columns = table["columns"]
    lines = [
        "| " + " | ".join(_fmt(c) for c in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in table["rows"]:
        lines.append("| " + " | ".join(_fmt(row.get(c)) for c in columns) + " |")
    if not table["rows"]:
        lines.append("| " + " | ".join(["*(no rows)*", *[""] * (len(columns) - 1)]) + " |")
    return lines


def _mapping_lines(mapping: Mapping[str, Any], level: int = 0) -> list[str]:
    pad = "  " * level
    lines: list[str] = []
    for key, value in mapping.items():
        if _is_table(value):
            lines += [f"{pad}- **{key}**:", ""]
            lines += [f"{pad}  {line}" for line in _table_lines(value)]
            lines.append("")
        elif isinstance(value, dict):
            lines.append(f"{pad}- **{key}**:")
            lines += _mapping_lines(value, level + 1)
        elif isinstance(value, list):
            joined = ", ".join(_fmt(v) for v in value) or "(empty)"
            lines.append(f"{pad}- **{key}**: {joined}")
        else:
            lines.append(f"{pad}- **{key}**: {_fmt(value)}")
    return lines


class AnalysisReport:
    """Builder for a structured, exportable analysis report.

    Args:
        title: Report title.
        data: The analysed DataFrame (optional). Its shape and a SHA-256
            fingerprint are recorded so that readers can verify which data
            produced the results. The data itself is never stored in the report.
        parameters: Analysis settings the analyst wants on record (e.g. missing
            data policies, random seed, data source such as ``"synthetic"``).
        include_timestamp: Record the creation time (UTC). Disable it for
            byte-identical output across runs.
        timestamp: Override the recorded time with a fixed ISO-8601 string.

    Example:
        >>> report = AnalysisReport("Demo", parameters={"data_source": "synthetic"},
        ...                         include_timestamp=False)
        >>> _ = report.add_text("Note", "Synthetic data only.")
        >>> "Demo" in report.to_markdown()
        True
    """

    def __init__(
        self,
        title: str,
        *,
        data: pd.DataFrame | None = None,
        parameters: Mapping[str, Any] | None = None,
        include_timestamp: bool = True,
        timestamp: str | None = None,
    ) -> None:
        if not isinstance(title, str) or not title.strip():
            raise InputValidationError("title must be a non-empty string.")
        if parameters is not None and not isinstance(parameters, Mapping):
            raise InputValidationError("parameters must be a mapping.")
        self.title = title
        self._sections: list[dict[str, Any]] = []
        metadata: dict[str, Any] = {}
        if include_timestamp:
            metadata["created_utc"] = timestamp or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        metadata.update(collect_environment_metadata())
        if data is not None:
            require_dataframe(data, "data")
            metadata["data_shape"] = {"rows": int(data.shape[0]), "columns": int(data.shape[1])}
            metadata["data_fingerprint_sha256"] = dataframe_fingerprint(data)
        metadata["parameters"] = to_json_safe(dict(parameters or {}))
        self._metadata = metadata

    # ------------------------------------------------------------- building

    def _add(self, title: str, kind: str, content: Any) -> AnalysisReport:
        if not isinstance(title, str) or not title.strip():
            raise InputValidationError("Section title must be a non-empty string.")
        self._sections.append({"title": title, "kind": kind, "content": content})
        return self

    def add_text(self, title: str, text: str) -> AnalysisReport:
        """Append a free-text section."""
        if not isinstance(text, str):
            raise InputValidationError("text must be a string.")
        return self._add(title, "text", text)

    def add_table(self, title: str, table: pd.DataFrame) -> AnalysisReport:
        """Append a table (the index is included unless it is a default RangeIndex)."""
        require_dataframe(table, "table")
        return self._add(title, "table", frame_to_records(table))

    def add_mapping(self, title: str, mapping: Mapping[str, Any]) -> AnalysisReport:
        """Append key-value results (values are converted to strict-JSON types)."""
        if not isinstance(mapping, Mapping):
            raise InputValidationError("mapping must be a Mapping.")
        return self._add(title, "mapping", to_json_safe(dict(mapping)))

    def add_result(self, title: str, result: Any) -> AnalysisReport:
        """Append a result object: a DataFrame (table) or anything with ``to_dict()``."""
        if isinstance(result, pd.DataFrame):
            return self.add_table(title, result)
        if hasattr(result, "to_dict") and callable(result.to_dict):
            payload = to_json_safe(result.to_dict())
            if isinstance(payload, dict):
                return self._add(title, "mapping", payload)
        raise InputValidationError(
            "add_result needs a DataFrame or an object whose to_dict() returns a dict."
        )

    # -------------------------------------------------------------- export

    @property
    def metadata(self) -> dict[str, Any]:
        """A copy of the reproducibility metadata."""
        return json.loads(json.dumps(self._metadata))  # type: ignore[no-any-return]

    def to_dict(self) -> dict[str, Any]:
        """Return the report as a strict-JSON-compatible dictionary."""
        return {
            "title": self.title,
            "notice": NOTICE,
            "sections": json.loads(json.dumps(self._sections)),
            "metadata": self.metadata,
        }

    def to_json(self, path: str | Path | None = None, *, indent: int = 2) -> str:
        """Serialise to strict JSON (no NaN/Infinity); optionally write to ``path``."""
        text = json.dumps(self.to_dict(), indent=indent, allow_nan=False, ensure_ascii=False) + "\n"
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    def to_markdown(self, path: str | Path | None = None) -> str:
        """Render as Markdown; optionally write to ``path``."""
        lines = [f"# {self.title}", "", f"> {NOTICE}", ""]
        for section in self._sections:
            lines += [f"## {section['title']}", ""]
            kind, content = section["kind"], section["content"]
            if kind == "text":
                lines += [content, ""]
            elif kind == "table":
                lines += _table_lines(content) + [""]
            else:
                lines += _mapping_lines(content) + [""]
        lines += ["## Reproducibility metadata", ""]
        lines += _mapping_lines(self._metadata)
        text = "\n".join(lines).rstrip() + "\n"
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

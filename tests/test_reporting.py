from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

import behavioral_data_toolkit as bdt
from behavioral_data_toolkit.exceptions import InputValidationError
from behavioral_data_toolkit.reporting import (
    NOTICE,
    AnalysisReport,
    collect_environment_metadata,
    dataframe_fingerprint,
)

DF = pd.DataFrame({"a": [1, 2, 3, 4], "b": [2.5, 1.0, 4.0, 3.5]})


def make_report(**kw) -> AnalysisReport:
    kw.setdefault("include_timestamp", False)
    return AnalysisReport(
        "Demo report", data=DF, parameters={"seed": 1, "source": "synthetic"}, **kw
    )


def test_fingerprint_matches_independent_sha256():
    expected = hashlib.sha256(b"a,b\n1,2.5\n2,1.0\n3,4.0\n4,3.5\n").hexdigest()
    assert dataframe_fingerprint(DF) == expected


def test_fingerprint_sensitive_to_content_and_order():
    changed = DF.copy()
    changed.loc[0, "a"] = 99
    assert dataframe_fingerprint(changed) != dataframe_fingerprint(DF)
    assert dataframe_fingerprint(DF.iloc[::-1]) != dataframe_fingerprint(DF)
    assert dataframe_fingerprint(DF.set_axis([10, 11, 12, 13])) == dataframe_fingerprint(DF)


def test_environment_metadata_contents():
    meta = collect_environment_metadata()
    assert meta["toolkit_version"] == bdt.__version__
    assert meta["numpy_version"] == np.__version__
    assert meta["pandas_version"] == pd.__version__
    assert set(meta) >= {"python_version", "scipy_version", "python_implementation"}


def test_metadata_in_report():
    md = make_report().metadata
    assert md["data_shape"] == {"rows": 4, "columns": 2}
    assert md["data_fingerprint_sha256"] == dataframe_fingerprint(DF)
    assert md["parameters"] == {"seed": 1, "source": "synthetic"}
    assert "created_utc" not in md


def test_timestamp_options():
    assert "created_utc" in AnalysisReport("t").metadata
    fixed = AnalysisReport("t", timestamp="2026-01-01T00:00:00Z").metadata
    assert fixed["created_utc"] == "2026-01-01T00:00:00Z"


def test_report_is_deterministic_without_timestamp():
    def build() -> AnalysisReport:
        r = make_report()
        r.add_text("Intro", "Synthetic demonstration.")
        r.add_result("Alpha", bdt.cronbach_alpha(DF))
        r.add_result("Matrix", bdt.correlation_matrix(DF))
        return r

    a, b = build(), build()
    assert a.to_json() == b.to_json()
    assert a.to_markdown() == b.to_markdown()


def test_json_is_strict_and_contains_notice():
    r = make_report()
    r.add_mapping(
        "m", {"x": float("nan"), "y": np.float64(1.5), "z": np.int64(3), "arr": np.array([1, 2])}
    )
    r.add_result("dist", bdt.distribution_summary([1.0]))  # contains None fields
    text = r.to_json()
    parsed = json.loads(text)  # strict parse
    assert "NaN" not in text and "Infinity" not in text
    assert parsed["notice"] == NOTICE
    assert parsed["sections"][0]["content"] == {"x": None, "y": 1.5, "z": 3, "arr": [1, 2]}
    assert text.endswith("\n")


def test_markdown_structure_and_notice():
    r = make_report()
    r.add_text("Intro", "Hello")
    r.add_table("Tab", pd.DataFrame({"k": ["a|b", None], "v": [0.123456, 2]}))
    r.add_result("Alpha", bdt.cronbach_alpha(DF))
    md = r.to_markdown()
    assert md.startswith("# Demo report\n")
    assert NOTICE in md
    assert "## Intro" in md and "Hello" in md
    assert "| k | v |" in md and "a\\|b" in md  # pipe escaped
    assert "0.1235" in md and "NA" in md  # float formatting / missing marker
    assert "**alpha**" in md
    assert "## Reproducibility metadata" in md
    assert md.index("## Intro") < md.index("## Reproducibility metadata")
    assert md.endswith("\n") and not md.endswith("\n\n")


def test_matrix_result_renders_tables():
    r = make_report()
    r.add_result("Corr", bdt.correlation_matrix(DF))
    md = r.to_markdown()
    assert "| variable | a | b |" in md


def test_empty_table_renders_placeholder():
    r = make_report()
    r.add_table("Empty", pd.DataFrame(columns=["x", "y"]))
    assert "*(no rows)*" in r.to_markdown()


def test_write_to_files(tmp_path):
    r = make_report()
    r.add_text("t", "x")
    jp, mp = tmp_path / "r.json", tmp_path / "r.md"
    text_j, text_m = r.to_json(jp), r.to_markdown(mp)
    assert jp.read_text(encoding="utf-8") == text_j
    assert mp.read_text(encoding="utf-8") == text_m


def test_sections_are_snapshots():
    t = pd.DataFrame({"x": [1, 2]})
    r = make_report()
    r.add_table("t", t)
    t.loc[0, "x"] = 100  # later mutation must not alter the report
    assert r.to_dict()["sections"][0]["content"]["rows"][0]["x"] == 1


def test_chaining_and_validation():
    r = make_report()
    assert r.add_text("a", "b") is r
    with pytest.raises(InputValidationError):
        AnalysisReport("  ")
    with pytest.raises(InputValidationError):
        AnalysisReport("t", parameters=[("a", 1)])  # type: ignore[arg-type]
    with pytest.raises(InputValidationError):
        AnalysisReport("t", data=[1, 2])  # type: ignore[arg-type]
    with pytest.raises(InputValidationError):
        r.add_text("", "x")
    with pytest.raises(InputValidationError):
        r.add_text("t", 5)  # type: ignore[arg-type]
    with pytest.raises(InputValidationError):
        r.add_table("t", [1, 2])  # type: ignore[arg-type]
    with pytest.raises(InputValidationError):
        r.add_mapping("t", [1, 2])  # type: ignore[arg-type]
    with pytest.raises(InputValidationError):
        r.add_result("t", 42)


def test_metadata_property_returns_copy():
    r = make_report()
    r.metadata["parameters"]["seed"] = 999
    assert r.metadata["parameters"]["seed"] == 1

"""The examples must keep working and must stay reproducible."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

import behavioral_data_toolkit as bdt

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
DATA = EXAMPLES / "data" / "synthetic_survey.csv"


def load_example(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"example_{name}", EXAMPLES / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gen() -> ModuleType:
    return load_example("generate_synthetic_data")


@pytest.fixture(scope="module")
def workflow() -> ModuleType:
    return load_example("example_workflow")


def test_generator_is_deterministic_and_matches_committed_csv(gen, tmp_path):
    first, second = tmp_path / "a.csv", tmp_path / "b.csv"
    gen.main(["prog", str(first)])
    gen.main(["prog", str(second)])
    assert first.read_text() == second.read_text()
    assert first.read_text() == DATA.read_text(), "committed CSV is out of date; regenerate it"


def test_synthetic_data_contains_the_documented_problems(gen):
    df = bdt.load_csv(DATA)
    items = gen.ITEMS
    rules = {c: bdt.ValueRule(minimum=1, maximum=5) for c in items}
    q = bdt.data_quality_summary(df, duplicate_subset=["participant_id"], rules=rules)
    assert (df.shape[0], q.n_duplicate_rows, q.n_invalid_values) == (201, 1, 2)
    assert sorted(q.invalid["value"].astype(float)) == [7.0, 9.0]
    assert int(df[items].isna().sum().sum()) == 12


def test_hand_verified_participant_score():
    # P001 answered sw1..sw6 = 4,2,5,4,1,4 (see CSV). sw2 and sw5 are reverse-keyed on 1-5:
    # 2 -> 4 and 1 -> 5, so the items become 4,4,5,4,5,4 and the mean is 26/6.
    df = bdt.load_csv(DATA)
    scale = bdt.ScaleDefinition(
        "w", tuple(f"sw{i}" for i in range(1, 7)), 1, 5, reverse_items=("sw2", "sw5")
    )
    first = df[df["participant_id"] == "P001"].iloc[0]
    assert [first[f"sw{i}"] for i in range(1, 7)] == [4, 2, 5, 4, 1, 4]
    scores = bdt.score_questionnaire(df.iloc[[0]].astype({c: float for c in scale.items}), scale)
    assert scores.iloc[0] == pytest.approx(26 / 6)


def test_workflow_runs_and_alpha_matches_independent_computation(workflow, tmp_path):
    report = workflow.run(DATA, tmp_path)
    assert (tmp_path / "example_report.json").exists()
    assert (tmp_path / "example_report.md").exists()
    parsed = json.loads((tmp_path / "example_report.json").read_text(encoding="utf-8"))
    assert parsed["notice"]
    assert parsed["metadata"]["parameters"]["data_source"].startswith("synthetic")

    # Independent recomputation of the wellbeing alpha straight from the CSV with NumPy only.
    raw = pd.read_csv(DATA).drop_duplicates(subset="participant_id", keep="first")
    cols = [f"sw{i}" for i in range(1, 7)]
    block = raw[cols].astype(float)
    block = block.where((block >= 1) & (block <= 5))  # out-of-range -> missing
    for col in ("sw2", "sw5"):
        block[col] = 6 - block[col]
    complete = block.dropna().to_numpy()
    cov = np.cov(complete, rowvar=False)
    k = cov.shape[0]
    expected = k / (k - 1) * (1 - np.trace(cov) / cov.sum())

    section = next(
        s
        for s in report.to_dict()["sections"]
        if "synthetic_wellbeing" in s["title"] and "alpha" in s["title"]
    )
    assert section["content"]["alpha"] == pytest.approx(expected, rel=1e-10)
    assert section["content"]["n_observations"] == complete.shape[0]


def test_workflow_report_is_reproducible(workflow, tmp_path):
    a = workflow.run(DATA, tmp_path / "a")
    b = workflow.run(DATA, tmp_path / "b")
    assert a.to_json() == b.to_json()
    assert (tmp_path / "a" / "example_report.md").read_text() == (
        tmp_path / "b" / "example_report.md"
    ).read_text()

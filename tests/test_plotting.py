from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")

import behavioral_data_toolkit as bdt  # noqa: E402
from behavioral_data_toolkit.exceptions import InputValidationError, StatisticsError  # noqa: E402


def test_plot_distribution_returns_axes_and_saves(tmp_path):
    ax = bdt.plot_distribution([1, 2, 2, 3, 4, np.nan], title="T")
    assert "T" in ax.get_title() and "n=5" in ax.get_title() and "1 missing" in ax.get_title()
    out = tmp_path / "d.png"
    ax.figure.savefig(out)
    assert out.stat().st_size > 0


def test_plot_distribution_errors():
    with pytest.raises(StatisticsError):
        bdt.plot_distribution([np.nan, np.nan])
    with pytest.raises(InputValidationError):
        bdt.plot_distribution(["a", "b"])
    with pytest.raises(InputValidationError):
        bdt.plot_distribution([1, np.nan], missing="raise")


def test_plot_heatmap(tmp_path):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(30, 3)), columns=list("abc"))
    ax = bdt.plot_correlation_heatmap(bdt.correlation_matrix(df))
    assert "Pearson" in ax.get_title()
    out = tmp_path / "h.png"
    ax.figure.savefig(out)
    assert out.stat().st_size > 0


def test_plot_heatmap_validation_and_nan_cells():
    with pytest.raises(InputValidationError):
        bdt.plot_correlation_heatmap(pd.DataFrame())  # type: ignore[arg-type]
    df = pd.DataFrame({"a": [1, 2, 3, 4], "const": [1, 1, 1, 1], "c": [2, 1, 4, 3]})
    bdt.plot_correlation_heatmap(bdt.correlation_matrix(df), annotate=True)  # must not raise


def test_package_exports():
    for name in bdt.__all__:
        assert hasattr(bdt, name), name
    assert bdt.__version__ == "0.1.0"

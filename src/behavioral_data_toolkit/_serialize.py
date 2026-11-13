"""Internal helpers that turn results into JSON-safe structures (private API)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def frame_to_records(df: pd.DataFrame) -> dict[str, Any]:
    """Represent a DataFrame as ``{"columns": [...], "rows": [{...}, ...]}``.

    The index is included as a column unless it is a default ``RangeIndex``.
    """
    frame = df
    if not isinstance(df.index, pd.RangeIndex):
        label = df.index.name if df.index.name is not None else "index"
        if label in df.columns:
            label = f"{label}_index"
        frame = df.copy()
        frame.insert(0, label, df.index.to_numpy())
    columns = [str(c) for c in frame.columns]
    rows = [
        {col: to_json_safe(val) for col, val in zip(columns, row, strict=True)}
        for row in frame.itertuples(index=False, name=None)
    ]
    return {"columns": columns, "rows": rows}


def to_json_safe(obj: Any) -> Any:
    """Recursively convert ``obj`` into values accepted by strict JSON.

    Non-finite floats (NaN, inf) become ``None``; NumPy scalars become Python
    scalars; DataFrames/Series/dataclass results are expanded.
    """
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if obj is pd.NA or obj is pd.NaT:
        return None
    if isinstance(obj, pd.DataFrame):
        return frame_to_records(obj)
    if isinstance(obj, pd.Series):
        return {str(k): to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, dict):
        return {str(k): to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [to_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [to_json_safe(v) for v in obj.tolist()]
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return to_json_safe(obj.to_dict())
    return str(obj)

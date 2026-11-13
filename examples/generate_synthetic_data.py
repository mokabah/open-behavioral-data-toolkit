"""Generate a SYNTHETIC questionnaire dataset for demonstrations and tests.

Every value is produced by a seeded pseudo-random number generator. No
participant exists, and no pattern in this file is an empirical finding. The two
"scales" are invented for the demonstration; they are not real instruments.

Deliberate data-quality problems are injected so that the validator has something
to find: missing responses, two out-of-range codes, and one duplicated participant.

Usage:
    python examples/generate_synthetic_data.py [output.csv]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20260101
N_PARTICIPANTS = 200

WELLBEING_ITEMS = [f"sw{i}" for i in range(1, 7)]  # synthetic scale A, 6 items
REVERSE_KEYED = ["sw2", "sw5"]  # stored reverse-keyed; must be reverse-scored
FOCUS_ITEMS = [f"sf{i}" for i in range(1, 5)]  # synthetic scale B, 4 items
ITEMS = WELLBEING_ITEMS + FOCUS_ITEMS
DEFAULT_PATH = Path(__file__).resolve().parent / "data" / "synthetic_survey.csv"


def generate(n: int = N_PARTICIPANTS, seed: int = SEED) -> pd.DataFrame:
    """Return the synthetic dataset (deterministic for a given ``n`` and ``seed``)."""
    rng = np.random.default_rng(seed)
    trait_a = rng.normal(size=n)
    trait_b = 0.3 * trait_a + np.sqrt(1 - 0.3**2) * rng.normal(size=n)

    def likert(trait: np.ndarray) -> np.ndarray:
        raw = 3 + 0.9 * trait + rng.normal(0.0, 0.9, size=n)
        return np.clip(np.rint(raw), 1, 5).astype(int)

    data: dict[str, object] = {"participant_id": [f"P{i:03d}" for i in range(1, n + 1)]}
    for item in WELLBEING_ITEMS:
        values = likert(trait_a)
        data[item] = 6 - values if item in REVERSE_KEYED else values
    for item in FOCUS_ITEMS:
        data[item] = likert(trait_b)
    data["age"] = rng.integers(18, 66, size=n)
    data["group"] = rng.choice(["A", "B"], size=n)
    data["screen_hours"] = np.round(
        np.clip(rng.normal(4.0, 1.5, size=n) - 0.3 * trait_a, 0, None), 1
    )
    df = pd.DataFrame(data)

    # --- injected data-quality problems (positions are fixed for reproducibility)
    df[ITEMS] = df[ITEMS].astype("Int64")
    cells = rng.choice(n * len(ITEMS), size=12, replace=False)
    for cell in cells:
        df.iloc[cell // len(ITEMS), df.columns.get_loc(ITEMS[cell % len(ITEMS)])] = pd.NA
    df.loc[16, "sw3"] = 7  # out of range (scale is 1-5)
    df.loc[79, "sf2"] = 9  # out of range (e.g. a "refused" code)
    df = pd.concat([df, df.iloc[[4]]], ignore_index=True)  # duplicated participant
    return df


def main(argv: list[str]) -> None:
    path = Path(argv[1]) if len(argv) > 1 else DEFAULT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    generate().to_csv(path, index=False, lineterminator="\n")
    print(f"Wrote synthetic dataset to {path}")


if __name__ == "__main__":
    main(sys.argv)

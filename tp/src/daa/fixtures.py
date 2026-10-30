"""Synthetic data used by tests and the primer notebook.

It mirrors the raw schema loosely: ID columns, numeric, nominal, ordinal, and
the ``AGE_RANGE`` target. It exists so the skeleton can run without ``data/``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SEED
from .data import TARGET


def make_synthetic_frame(n_samples: int = 240, seed: int = SEED) -> pd.DataFrame:
    """Return a small tabular frame with a learnable target and missing values."""
    rng = np.random.default_rng(seed)
    age_signal = rng.normal(size=n_samples)
    target = pd.cut(
        age_signal,
        bins=[-np.inf, -0.7, 0.0, 0.7, np.inf],
        labels=False,
    ).astype(int)

    frame = pd.DataFrame(
        {
            "PID": np.arange(1, n_samples + 1),
            "HHID": np.arange(1, n_samples + 1) * 100 + 1,
            "num_height": 170 - 4 * target + rng.normal(0, 5, n_samples),
            "num_weight": 70 + 2 * target + rng.normal(0, 8, n_samples),
            "num_income": 3 + target + rng.normal(0, 1, n_samples),
            "num_noise": rng.normal(size=n_samples),
            "cat_region": rng.choice(["Norte", "Centro", "Sul"], size=n_samples),
            "cat_sex": rng.choice(["F", "M"], size=n_samples),
            "cat_smoke": rng.choice(["Nunca", "Sim", "Não aplicável"], size=n_samples),
            "ord_health": rng.integers(1, 6, size=n_samples),
            TARGET: target,
        }
    )

    frame.loc[rng.random(n_samples) < 0.1, "num_height"] = np.nan
    frame.loc[rng.random(n_samples) < 0.1, "cat_region"] = np.nan
    return frame


def make_synthetic_xy(n_samples: int = 240, seed: int = SEED) -> tuple[pd.DataFrame, pd.Series]:
    """Convenience wrapper returning ``(X, y)``."""
    frame = make_synthetic_frame(n_samples=n_samples, seed=seed)
    return frame.drop(columns=[TARGET]), frame[TARGET].astype(int)

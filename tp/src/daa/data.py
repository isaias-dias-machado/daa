"""Data loading and the column-role contract.

Only this module reads the raw CSVs. ``ColumnSpec`` is the single source of
truth for which columns are numeric, ordinal, nominal, or dropped.

Design rule: a column is not "ordinal" because its dtype looks ordered. The
order is a domain claim, so ``ordinal`` is empty until someone who has read the
codebook promotes a column explicitly.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import TEST_PATH, TRAIN_PATH

TARGET = "AGE_RANGE"
ID_COLUMNS = ("PID", "HHID")
WEIGHT_COLUMN = "WGT"


@dataclass(frozen=True)
class ColumnSpec:
    """Column roles for the preprocessor.

    Invariants:
        The four groups are disjoint and never contain the target column.

    Failure modes:
        ``__post_init__`` raises ``ValueError`` when a column is assigned twice
        or when the target is listed as an input.
        ``validate`` raises ``ValueError`` when a column is missing from a frame.
    """

    numeric: tuple[str, ...]
    ordinal: tuple[str, ...]
    nominal: tuple[str, ...]
    drop: tuple[str, ...]

    def __post_init__(self) -> None:
        groups = {
            "numeric": self.numeric,
            "ordinal": self.ordinal,
            "nominal": self.nominal,
            "drop": self.drop,
        }
        seen: set[str] = set()
        for columns in groups.values():
            for column in columns:
                if column in seen:
                    raise ValueError(f"column {column!r} is assigned to more than one role")
                seen.add(column)
        if TARGET in seen:
            raise ValueError(f"{TARGET!r} is the target and must not be an input")

    @property
    def input_columns(self) -> tuple[str, ...]:
        """Columns the model may see, in a stable order."""
        return self.numeric + self.ordinal + self.nominal

    def validate(self, frame: pd.DataFrame) -> None:
        """Raise if any input column is absent from ``frame``."""
        missing = [column for column in self.input_columns if column not in frame.columns]
        if missing:
            raise ValueError(f"frame is missing columns: {missing}")

    def counts(self) -> dict[str, int]:
        """Small summary used in notebooks and logs."""
        return {
            "numeric": len(self.numeric),
            "ordinal": len(self.ordinal),
            "nominal": len(self.nominal),
            "drop": len(self.drop),
        }


def infer_column_spec(frame: pd.DataFrame) -> ColumnSpec:
    """Infer column roles from dtypes and constants.

    Rules:
        - IDs, the survey weight, the target, and columns that are constant in
          this frame are dropped.
        - Numeric dtypes are numeric.
        - Everything else is nominal.
        - Ordinal stays empty on purpose; see the module docstring.

    The inference is a starting point, not a codebook. Change the returned spec
    explicitly when the domain says otherwise.
    """
    drop: set[str] = set()
    for column in (*ID_COLUMNS, WEIGHT_COLUMN, TARGET):
        if column in frame.columns:
            drop.add(column)

    for column in frame.columns:
        if column in drop:
            continue
        if frame[column].nunique(dropna=False) <= 1:
            drop.add(column)

    numeric = tuple(
        column for column in frame.select_dtypes(include="number").columns if column not in drop
    )
    nominal = tuple(
        column for column in frame.columns if column not in drop and column not in numeric
    )
    return ColumnSpec(
        numeric=numeric,
        ordinal=(),
        nominal=nominal,
        drop=tuple(sorted(drop)),
    )


def load_train(path: str | Path | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Load the training CSV and split it into ``X`` and ``y``.

    ``X`` keeps the ID columns so out-of-fold predictions can be joined back to
    a person for anomaly analysis. The preprocessor drops them; they never
    reach a model.
    """
    frame = pd.read_csv(path or TRAIN_PATH, low_memory=False)
    if TARGET not in frame.columns:
        raise ValueError(f"{TARGET!r} is missing from the training data")
    target = frame[TARGET].astype(int).rename(TARGET)
    features = frame.drop(columns=[TARGET])
    return features, target


def load_test(path: str | Path | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Load the test CSV and return features plus the submission row ids.

    The Kaggle file has no ``RowId`` column; the row order is the id.
    """
    frame = pd.read_csv(path or TEST_PATH, low_memory=False)
    row_ids = pd.Series(np.arange(len(frame), dtype=int), name="RowId")
    return frame, row_ids


def target_fingerprint(target: pd.Series | np.ndarray) -> str:
    """Stable fingerprint of the target vector, used to validate frozen folds."""
    values = np.asarray(pd.Series(target).to_numpy(), dtype=np.int64)
    return hashlib.sha256(values.tobytes()).hexdigest()

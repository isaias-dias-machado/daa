"""Submission writer for the Kaggle task.

The competition expects exactly ``RowId,AGE_RANGE`` with ``RowId`` equal to the
row order of ``dataset_test_X.csv``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SUBMISSION_COLUMNS = ("RowId", "AGE_RANGE")


def write_submission(estimator, X_test: pd.DataFrame, row_ids, path: str | Path) -> Path:
    """Predict and write a submission CSV.

    Failure modes:
        ``ValueError`` when the ids are duplicated or when the prediction length
        does not match the test frame.
    """
    predictions = np.asarray(estimator.predict(X_test), dtype=int)
    ids = np.asarray(row_ids)

    if len(predictions) != len(X_test):
        raise ValueError("prediction length does not match the test frame")
    if len(ids) != len(predictions):
        raise ValueError("row id length does not match the predictions")
    if pd.Series(ids).duplicated().any():
        raise ValueError("row ids must be unique")

    submission = pd.DataFrame({"RowId": ids, "AGE_RANGE": predictions})
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(destination, index=False)
    return destination


def write_submission_from_result(result, X_test: pd.DataFrame, row_ids, path: str | Path) -> Path:
    """Same as ``write_submission`` but uses the estimator stored in a run."""
    if result.estimator is None:
        raise ValueError("run_cv(..., fit_full=True) is required for a submission")
    return write_submission(result.estimator, X_test, row_ids, path)


def validate_submission(path: str | Path, expected_row_ids) -> pd.DataFrame:
    """Load a submission and check the competition contract."""
    submission = pd.read_csv(path)
    if tuple(submission.columns) != SUBMISSION_COLUMNS:
        raise ValueError(f"expected columns {SUBMISSION_COLUMNS}, got {tuple(submission.columns)}")
    expected = np.asarray(expected_row_ids)
    if len(submission) != len(expected):
        raise ValueError("submission has the wrong number of rows")
    if not np.array_equal(submission["RowId"].to_numpy(), expected):
        raise ValueError("submission RowId does not match the test row order")
    if submission["AGE_RANGE"].isna().any():
        raise ValueError("submission contains missing predictions")
    return submission

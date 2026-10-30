from __future__ import annotations

import pytest
from sklearn.metrics import f1_score

from daa.evaluation.metrics import (
    METRICS,
    SCORERS,
    accuracy,
    classification_summary,
    confusion_table,
    macro_f1,
    mean_absolute_band_error,
    quadratic_weighted_kappa,
)


def test_macro_f1_matches_sklearn():
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 1, 1, 1, 2, 0]
    expected = f1_score(y_true, y_pred, average="macro", zero_division=0)
    assert macro_f1(y_true, y_pred) == pytest.approx(expected)


def test_perfect_predictions():
    y = [0, 1, 2, 3]
    assert macro_f1(y, y) == 1.0
    assert accuracy(y, y) == 1.0
    assert quadratic_weighted_kappa(y, y) == 1.0
    assert mean_absolute_band_error(y, y) == 0.0


def test_metric_registry_names():
    assert set(METRICS) == {"macro_f1", "accuracy", "qwk", "mae_bands"}
    assert set(SCORERS) == set(METRICS)


def test_classification_summary_has_one_row_per_label():
    summary = classification_summary([0, 0, 1, 1], [0, 1, 1, 1], labels=(0, 1))
    assert list(summary["label"]) == [0, 1]
    assert set(summary.columns) == {"label", "precision", "recall", "f1", "support"}


def test_confusion_table_shape():
    table = confusion_table([0, 0, 1, 1], [0, 1, 1, 1], labels=(0, 1))
    assert table.shape == (2, 2)
    assert table.loc[0, 0] == 1
    assert table.loc[0, 1] == 1

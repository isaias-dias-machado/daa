"""Frozen metric definitions.

Change these only by group agreement: every historical number in
``outputs/leaderboard.csv`` is comparable because these definitions stay fixed.

Primary metric: macro F1 (the Kaggle metric). Secondary metrics: accuracy,
quadratic weighted kappa (``qwk``), and mean absolute error in age-band units
(``mae_bands``).
"""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    mean_absolute_error,
)


def macro_f1(y_true, y_pred) -> float:
    """Macro-averaged F1 with ``zero_division=0``."""
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def accuracy(y_true, y_pred) -> float:
    """Plain accuracy."""
    return float(accuracy_score(y_true, y_pred))


def quadratic_weighted_kappa(y_true, y_pred) -> float:
    """QWK treats a neighbouring-band error as less bad than a distant one."""
    return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))


def mean_absolute_band_error(y_true, y_pred) -> float:
    """Average distance in age-band index; lower is better."""
    return float(mean_absolute_error(y_true, y_pred))


METRICS = {
    "macro_f1": macro_f1,
    "accuracy": accuracy,
    "qwk": quadratic_weighted_kappa,
    "mae_bands": mean_absolute_band_error,
}

# For GridSearchCV/RandomizedSearchCV. ``mae_bands`` is a loss, so sklearn
# reports it negated; higher (closer to zero) is still better.
SCORERS = {
    "macro_f1": make_scorer(f1_score, average="macro", zero_division=0),
    "accuracy": "accuracy",
    "qwk": make_scorer(cohen_kappa_score, weights="quadratic"),
    "mae_bands": make_scorer(mean_absolute_error, greater_is_better=False),
}


def classification_summary(y_true, y_pred, labels=None) -> pd.DataFrame:
    """Per-label precision, recall, F1 and support."""
    label_list = sorted(pd.unique(y_true)) if labels is None else list(labels)
    report = classification_report(
        y_true,
        y_pred,
        labels=label_list,
        output_dict=True,
        zero_division=0,
    )
    rows = []
    for label in label_list:
        entry = report[str(label)]
        rows.append(
            {
                "label": label,
                "precision": entry["precision"],
                "recall": entry["recall"],
                "f1": entry["f1-score"],
                "support": entry["support"],
            }
        )
    return pd.DataFrame(rows)


def confusion_table(y_true, y_pred, labels=None) -> pd.DataFrame:
    """Confusion matrix as a labelled DataFrame."""
    label_list = sorted(pd.unique(y_true)) if labels is None else list(labels)
    matrix = confusion_matrix(y_true, y_pred, labels=label_list)
    return pd.DataFrame(
        matrix,
        index=pd.Index(label_list, name="true"),
        columns=pd.Index(label_list, name="pred"),
    )

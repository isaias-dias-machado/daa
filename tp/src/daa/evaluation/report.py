"""Reporting helpers: confusion matrix, feature importance, permutation importance."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import ConfusionMatrixDisplay

from ..config import SEED
from .cv import RunResult
from .metrics import SCORERS


def plot_confusion(result: RunResult, path: str | Path | None = None):
    """Plot the global confusion matrix; save to ``path`` when given."""
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 7))
    display = ConfusionMatrixDisplay(
        confusion_matrix=result.confusion.to_numpy(),
        display_labels=result.labels,
    )
    display.plot(ax=axis, xticks_rotation=90, colorbar=False, values_format="d")
    axis.set_title(f"{result.name} — confusion matrix (first repeat)")
    figure.tight_layout()
    if path is not None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=150, bbox_inches="tight")
        plt.close(figure)
        return destination
    return figure


def feature_importance_table(result: RunResult, top_n: int = 30) -> pd.DataFrame:
    """Importance of the *transformed* features for a tree or linear model.

    Failure mode:
        ``ValueError`` when the estimator was not fitted on the full data or
        exposes neither ``feature_importances_`` nor ``coef_``.
    """
    if result.estimator is None:
        raise ValueError("run_cv(..., fit_full=True) is required for feature importance")

    steps = getattr(result.estimator, "named_steps", {})
    preprocessor = steps.get("prep")
    classifier = steps.get("clf")
    if preprocessor is None or classifier is None:
        raise ValueError("expected a Pipeline with steps named 'prep' and 'clf'")

    names = preprocessor.get_feature_names_out()
    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        importances = np.abs(classifier.coef_).mean(axis=0)
    else:
        raise ValueError("classifier exposes neither feature_importances_ nor coef_")

    if len(names) != len(importances):
        raise ValueError("feature names and importances have different lengths")

    table = pd.DataFrame({"feature": names, "importance": importances})
    return table.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)


def permutation_importance_table(
    result: RunResult,
    X: pd.DataFrame,
    y: pd.Series,
    n_repeats: int = 5,
) -> pd.DataFrame:
    """Permutation importance on the raw columns, scored with macro F1.

    This refits nothing, but it is expensive: it scores the fitted pipeline
    ``n_repeats * n_features`` times. Use it on the final model, not in a loop.
    """
    if result.estimator is None:
        raise ValueError("run_cv(..., fit_full=True) is required for permutation importance")

    importance = permutation_importance(
        result.estimator,
        X,
        y,
        scoring=SCORERS["macro_f1"],
        n_repeats=n_repeats,
        random_state=SEED,
        n_jobs=-1,
    )
    table = pd.DataFrame(
        {
            "feature": X.columns,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }
    )
    return table.sort_values("importance_mean", ascending=False).reset_index(drop=True)

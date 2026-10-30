"""The evaluation boundary: fold loop, run result, and artifact writer.

``run_cv`` is deliberately a manual loop instead of ``cross_validate``: it
gives us out-of-fold predictions and probabilities, which the error analysis
and anomaly detection need. The splits and metric definitions are frozen, so
every run is comparable.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

from ..config import RUNS_DIR
from ..splits import Fold, get_folds
from ..utils import git_dirty, git_sha, slugify
from .metrics import METRICS, classification_summary, confusion_table

SCORE_COLUMNS = ("macro_f1", "accuracy", "qwk", "mae_bands", "fit_time", "score_time")


@dataclass
class RunResult:
    """Everything one experiment produced.

    ``oof`` has one row per sample per repeat, including ``proba_*`` columns;
    filter ``repeat == 0`` for a non-duplicated view. ``estimator`` is only
    populated when ``run_cv(..., fit_full=True)`` is used.
    """

    name: str
    run_id: str
    timestamp: str
    fold_scores: pd.DataFrame
    metrics: dict[str, float]
    oof: pd.DataFrame
    per_class: pd.DataFrame
    confusion: pd.DataFrame
    labels: tuple[int, ...]
    n_features: int
    params: dict[str, Any]
    estimator: Any = None

    def summary(self) -> pd.Series:
        """Compact metric table for notebooks."""
        return pd.Series(self.metrics, name=self.name)

    def __repr__(self) -> str:
        return (
            f"RunResult(name={self.name!r}, "
            f"macro_f1={self.metrics['macro_f1_mean']:.4f}"
            f"±{self.metrics['macro_f1_std']:.4f})"
        )


def _feature_count(fitted_estimator) -> int:
    """Number of columns produced by the ``prep`` step, or 0 if unknown."""
    steps = getattr(fitted_estimator, "named_steps", {})
    preprocessor = steps.get("prep")
    if preprocessor is not None and hasattr(preprocessor, "get_feature_names_out"):
        return len(preprocessor.get_feature_names_out())
    return 0


def run_cv(
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    folds: list[Fold] | None = None,
    *,
    name: str = "run",
    labels: tuple[int, ...] | None = None,
    params: dict[str, Any] | None = None,
    fit_full: bool = False,
) -> RunResult:
    """Fit ``estimator`` on every fold and collect scores and OOF predictions.

    The estimator is cloned per fold. If it is a ``Pipeline``, the whole
    preprocessor is refitted on each training fold; that is the point.

    Failure modes:
        ``TypeError`` when ``X`` is not a DataFrame (column names matter for
        interpretation).
        ``ValueError`` when ``X`` and ``y`` have different lengths.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame; column names are part of the contract")
    if len(X) != len(y):
        raise ValueError(f"X and y have different lengths: {len(X)} != {len(y)}")
    if folds is None:
        folds = get_folds(y)

    label_list = tuple(sorted(pd.unique(y))) if labels is None else tuple(labels)
    fold_rows: list[dict[str, float]] = []
    oof_parts: list[pd.DataFrame] = []
    n_features = 0

    for fold in folds:
        X_train = X.iloc[fold.train_idx]
        X_validation = X.iloc[fold.val_idx]
        y_train = y.iloc[fold.train_idx]
        y_validation = y.iloc[fold.val_idx]

        model = clone(estimator)
        started = time.perf_counter()
        model.fit(X_train, y_train)
        fit_time = time.perf_counter() - started

        started = time.perf_counter()
        predictions = model.predict(X_validation)
        score_time = time.perf_counter() - started

        n_features = n_features or _feature_count(model)
        row: dict[str, float] = {
            "repeat": fold.repeat,
            "fold": fold.fold,
            "fit_time": fit_time,
            "score_time": score_time,
        }
        for metric_name, metric_function in METRICS.items():
            row[metric_name] = metric_function(y_validation, predictions)
        fold_rows.append(row)

        oof = pd.DataFrame(
            {
                "row_index": X_validation.index,
                "repeat": fold.repeat,
                "fold": fold.fold,
                "y_true": np.asarray(y_validation),
                "y_pred": np.asarray(predictions),
            }
        ).reset_index(drop=True)

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X_validation)
            probability_columns = [f"proba_{label}" for label in model.classes_]
            oof = pd.concat(
                [
                    oof,
                    pd.DataFrame(probabilities, columns=probability_columns),
                ],
                axis=1,
            )
        oof_parts.append(oof)

    oof = pd.concat(oof_parts, ignore_index=True)
    probability_columns = [f"proba_{label}" for label in label_list]
    for column in probability_columns:
        if column not in oof.columns:
            oof[column] = 0.0
    oof = oof[["row_index", "repeat", "fold", "y_true", "y_pred", *probability_columns]]

    fold_scores = pd.DataFrame(fold_rows)
    metrics: dict[str, float] = {}
    for column in SCORE_COLUMNS:
        metrics[f"{column}_mean"] = float(fold_scores[column].mean())
        metrics[f"{column}_std"] = float(fold_scores[column].std(ddof=0))
    metrics["n_features"] = float(n_features)

    first_repeat = oof[oof["repeat"] == oof["repeat"].min()]
    per_class = classification_summary(
        first_repeat["y_true"],
        first_repeat["y_pred"],
        labels=label_list,
    )
    confusion = confusion_table(
        first_repeat["y_true"],
        first_repeat["y_pred"],
        labels=label_list,
    )

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")
    run_id = f"{timestamp}_{slugify(name)}"

    fitted_estimator = None
    if fit_full:
        fitted_estimator = clone(estimator).fit(X, y)

    return RunResult(
        name=name,
        run_id=run_id,
        timestamp=timestamp,
        fold_scores=fold_scores,
        metrics=metrics,
        oof=oof,
        per_class=per_class,
        confusion=confusion,
        labels=label_list,
        n_features=n_features,
        params=params or {"estimator": type(estimator).__name__},
        estimator=fitted_estimator,
    )


def _json_default(value):
    """Make numpy/pandas objects JSON-friendly without losing their text form."""
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def write_run(result: RunResult, output_dir: str | Path | None = None) -> Path:
    """Write one run directory and append a row to the leaderboard.

    Layout:
        ``<output_dir>/<run_id>/metrics.json``
        ``<output_dir>/<run_id>/params.json``
        ``<output_dir>/<run_id>/fold_scores.csv``
        ``<output_dir>/<run_id>/oof_predictions.csv``
        ``<output_dir>/<run_id>/per_class.csv``
        ``<output_dir>/<run_id>/confusion_matrix.csv``
        ``<output_dir>/<run_id>/estimator.joblib`` (when ``fit_full=True``)
        ``<output_dir>/../leaderboard.csv``

    Failure mode:
        ``FileExistsError`` when the run id already exists; run ids include
        microseconds, so this is effectively a collision guard.
    """
    runs_directory = Path(output_dir) if output_dir is not None else RUNS_DIR
    run_directory = runs_directory / result.run_id
    run_directory.mkdir(parents=True, exist_ok=False)

    metadata = {
        "run_id": result.run_id,
        "timestamp": result.timestamp,
        "name": result.name,
        "metrics": result.metrics,
        "labels": list(result.labels),
        "n_features": result.n_features,
        "params": result.params,
        "git_sha": git_sha(),
        "git_dirty": git_dirty(),
    }
    (run_directory / "metrics.json").write_text(
        json.dumps(metadata, indent=2, default=_json_default)
    )
    (run_directory / "params.json").write_text(
        json.dumps(result.params, indent=2, default=_json_default)
    )
    result.fold_scores.to_csv(run_directory / "fold_scores.csv", index=False)
    result.oof.to_csv(run_directory / "oof_predictions.csv", index=False)
    result.per_class.to_csv(run_directory / "per_class.csv", index=False)
    result.confusion.to_csv(run_directory / "confusion_matrix.csv")

    if result.estimator is not None:
        joblib.dump(result.estimator, run_directory / "estimator.joblib")

    _append_leaderboard(result, runs_directory.parent / "leaderboard.csv")
    return run_directory


def _append_leaderboard(result: RunResult, path: Path) -> None:
    """Append (or replace) one row for this run id."""
    row = {
        "run_id": result.run_id,
        "timestamp": result.timestamp,
        "name": result.name,
        "n_features": result.n_features,
        "macro_f1_mean": result.metrics["macro_f1_mean"],
        "macro_f1_std": result.metrics["macro_f1_std"],
        "accuracy_mean": result.metrics["accuracy_mean"],
        "qwk_mean": result.metrics["qwk_mean"],
        "mae_bands_mean": result.metrics["mae_bands_mean"],
        "git_sha": git_sha(),
        "git_dirty": git_dirty(),
        "params": json.dumps(result.params, sort_keys=True, default=_json_default),
    }
    if path.exists():
        leaderboard = pd.read_csv(path)
        leaderboard = leaderboard[leaderboard["run_id"] != result.run_id]
        leaderboard = pd.concat([leaderboard, pd.DataFrame([row])], ignore_index=True)
    else:
        leaderboard = pd.DataFrame([row])
    path.parent.mkdir(parents=True, exist_ok=True)
    leaderboard.to_csv(path, index=False)

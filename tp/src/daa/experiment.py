"""The integration layer: compose preprocessor + model + CV in one call.

Notebooks should call ``run_experiment`` (or compose the pieces themselves and
call ``run_cv``). The point is that the notebook decides *what* to run while
this module decides *how* a run is built and recorded.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.pipeline import Pipeline

from .data import load_train
from .evaluation.cv import RunResult, run_cv, write_run
from .features.preprocess import build_preprocessor
from .models.registry import make_model
from .splits import Fold


def build_pipeline(
    preprocessor,
    model_name: str,
    **model_params: Any,
) -> Pipeline:
    """Return ``Pipeline([("prep", preprocessor), ("clf", model)])``.

    The step names are part of the contract: search spaces use the ``clf__``
    prefix and reporting looks for ``prep`` and ``clf``.
    """
    return Pipeline(
        [
            ("prep", preprocessor),
            ("clf", make_model(model_name, **model_params)),
        ]
    )


def run_experiment(
    model_name: str,
    model_params: dict[str, Any] | None = None,
    *,
    name: str,
    preprocessor: str = "impute",
    X: pd.DataFrame | None = None,
    y: pd.Series | None = None,
    folds: list[Fold] | None = None,
    fit_full: bool = False,
    save: bool = True,
    output_dir: str | Path | None = None,
) -> RunResult:
    """Build, evaluate, and optionally persist one experiment.

    Parameters:
        model_name: a key from ``daa.models.registry.MODELS``.
        model_params: constructor arguments for that model.
        name: human-readable run name; becomes part of the run directory.
        preprocessor: ``"impute"`` or ``"passthrough"``.
        X, y: optional; loaded from ``data/dataset_train.csv`` when omitted.
        folds: optional; the frozen splits are loaded when omitted.
        fit_full: also fit on all rows, which is required for a submission.
        save: write the run directory and leaderboard row.
        output_dir: override ``outputs/runs`` (useful in tests).

    Failure modes:
        ``KeyError`` for an unknown model, ``ValueError`` for an unknown
        preprocessor variant or a missing target.
    """
    if X is None or y is None:
        X, y = load_train()

    # Imported here to keep the module import graph easy to follow.
    from .data import infer_column_spec

    spec = infer_column_spec(X)
    pipeline = build_pipeline(
        build_preprocessor(spec, variant=preprocessor),
        model_name,
        **(model_params or {}),
    )
    result = run_cv(
        pipeline,
        X,
        y,
        folds,
        name=name,
        params={
            "preprocessor": preprocessor,
            "model": model_name,
            "model_params": model_params or {},
        },
        fit_full=fit_full,
    )
    if save:
        write_run(result, output_dir=output_dir)
    return result

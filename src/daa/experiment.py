"""The integration layer: compose preprocessor + model + CV in one call.

Notebooks should call ``run_experiment`` (or compose the pieces themselves and
call ``run_cv``). The point is that the notebook decides *what* to run while
this module decides *how* a run is built and recorded.

This module only *composes*: it resolves names, it never defines them. Named
feature sets are defined by ``features.selection`` (from the shortlist
artifact), the default preprocessing variant is pinned by the encoding owner
in ``features.preprocess.DEFAULT_VARIANT``, and models plus search spaces live
in ``models/``. Nothing feature- or encoding-related is hardcoded here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.pipeline import Pipeline

from .data import load_train
from .evaluation.cv import RunResult, run_cv, write_run
from .features.preprocess import DEFAULT_VARIANT, build_preprocessor
from .features.selection import columns_fingerprint, resolve_feature_set, subset_spec
from .models.registry import make_model
from .models.spaces import SEARCH_SPACES, build_search
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
    model_name: str | None = None,
    model_params: dict[str, Any] | None = None,
    *,
    name: str,
    preprocessor: str | None = None,
    X: pd.DataFrame | None = None,
    y: pd.Series | None = None,
    folds: list[Fold] | None = None,
    columns: list[str] | None = None,
    feature_set: str | None = "all",
    search: str | dict[str, Any] | None = None,
    repeats: int | None = None,
    estimator=None,
    fit_full: bool = False,
    save: bool = True,
    output_dir: str | Path | None = None,
) -> RunResult:
    """Build, evaluate, and optionally persist one experiment.

    Parameters:
        model_name: a key from ``daa.models.registry.MODELS``. May be omitted
            only when ``estimator`` is given.
        model_params: constructor arguments for that model.
        name: human-readable run name; becomes part of the run directory.
        preprocessor: a variant from ``daa.features.preprocess.VARIANTS``.
            ``None`` (the default) means the encoding owner's
            ``features.preprocess.DEFAULT_VARIANT``; the resolved variant is
            recorded in ``params``.
        X, y: optional; loaded from ``data/dataset_train.csv`` when omitted.
        folds: optional; the frozen splits are loaded when omitted.
        columns: optional explicit shortlist. When given, it wins over name
            resolution; every name must be an input of the inferred spec (see
            ``daa.features.selection.subset_spec``). ``None`` means: resolve
            from ``feature_set``.
        feature_set: a name from
            ``daa.features.selection.FEATURE_SET_NAMES``. The default
            ``"all"`` means all inputs. Any other name (e.g. ``"selected"``)
            is resolved by ``resolve_feature_set`` against the shortlist
            artifact, so callers name a set instead of listing columns.
            Pass ``None`` explicitly to keep the historical behaviour of
            deriving the recorded label from ``columns`` (``"all"`` /
            ``"subset"``).
        search: ``None``, ``"grid"``, ``"random"``, or a dict
            ``{"kind", "space", "n_iter", "scoring", "n_jobs"}``. The pipeline
            is wrapped by ``models.spaces.build_search`` before ``run_cv``;
            because ``run_cv`` clones per fold, this is honest nested CV. The
            string forms use ``SEARCH_SPACES[model_name]``, so they need
            ``model_name`` (with ``estimator=``, pass a dict with an explicit
            ``space``). The search configuration is recorded in ``params``.
        repeats: when given, only folds with ``fold.repeat < repeats`` run.
            For fast iteration only — combine with ``save=False``, and note
            that a saved partial run records ``repeats`` in ``params``.
        estimator: escape hatch — an already-constructed (unfitted) estimator
            is used as the ``clf`` step directly, skipping ``make_model``
            (calibrated / wrapped / ensemble estimators). ``params["model"]``
            then records ``type(estimator).__name__``.
        fit_full: also fit on all rows, which is required for a submission.
        save: write the run directory and leaderboard row.
        output_dir: override ``outputs/runs`` (useful in tests).

    Invariants:
        The recorded ``params`` always carry ``feature_set``, the number of
        input columns, and a short ``feature_hash``, so two runs with different
        columns are distinguishable even when the names are similar.

    Failure modes:
        ``KeyError`` for an unknown model, ``ValueError`` for an unknown
        preprocessor variant, feature set, search kind/space/scoring, a
        missing target, a column that is not an input, or ``model_name=None``
        without ``estimator=``.
    """
    if X is None or y is None:
        X, y = load_train()

    # Imported here to keep the module import graph easy to follow.
    from .data import explicit_spec, infer_column_spec

    # Infer from dtypes, then apply the hand-written drops (admin metadata,
    # near-constant columns). Both steps are deterministic and learnt from no
    # data, so they are safe outside the CV folds.
    spec = explicit_spec(infer_column_spec(X))

    # Feature resolution: an explicit column list always wins; otherwise a
    # named feature set resolves to the owner's validated artifact.
    if columns is None and feature_set not in (None, "all"):
        columns = list(resolve_feature_set(feature_set, spec, target=y))
    if columns is not None:
        spec = subset_spec(spec, columns)

    variant = preprocessor if preprocessor is not None else DEFAULT_VARIANT
    preprocessor_obj = build_preprocessor(spec, variant=variant)

    if estimator is not None:
        pipeline = Pipeline([("prep", preprocessor_obj), ("clf", estimator)])
        recorded_model: str = type(estimator).__name__
    else:
        if model_name is None:
            raise ValueError("model_name is required unless estimator= is given")
        pipeline = build_pipeline(preprocessor_obj, model_name, **(model_params or {}))
        recorded_model = model_name

    search_record: dict[str, Any] | None = None
    if search is not None:
        search_options = {"kind": search} if isinstance(search, str) else dict(search)
        kind = search_options.pop("kind", "grid")
        space = search_options.pop("space", None)
        n_iter = search_options.pop("n_iter", 10)
        scoring = search_options.pop("scoring", "macro_f1")
        n_jobs = search_options.pop("n_jobs", -1)
        if search_options:
            raise ValueError(f"unknown search options: {sorted(search_options)}")
        if space is None and model_name is None:
            raise ValueError(
                "search= needs model_name for the default space; "
                "with estimator= pass search as a dict with an explicit space"
            )
        pipeline = build_search(
            pipeline,
            model_name,
            kind=kind,
            space=space,
            n_iter=n_iter,
            scoring=scoring,
            n_jobs=n_jobs,
        )
        effective_space = space if space is not None else SEARCH_SPACES[model_name]
        search_record = {
            "kind": kind,
            "n_iter": n_iter,
            "scoring": scoring,
            "space": sorted(effective_space),
        }

    params: dict[str, Any] = {
        "preprocessor": variant,
        "model": recorded_model,
        "model_params": model_params or {},
        "feature_set": feature_set or ("all" if columns is None else "subset"),
        "n_input_columns": len(spec.input_columns),
        "feature_hash": columns_fingerprint(spec.input_columns)[:12],
    }
    if search_record is not None:
        params["search"] = search_record
    if repeats is not None:
        params["repeats"] = repeats

    result = run_cv(
        pipeline,
        X,
        y,
        folds,
        name=name,
        params=params,
        repeats=repeats,
        fit_full=fit_full,
    )
    if save:
        write_run(result, output_dir=output_dir)
    return result

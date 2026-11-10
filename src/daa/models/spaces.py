"""Search spaces for ``GridSearchCV`` / ``RandomizedSearchCV``.

Keys use the ``clf__`` prefix because the project pipeline names its final step
``clf``. Keep these small enough to run on a laptop; the tuner owns the larger
spaces.
"""

from __future__ import annotations

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from ..config import SEED
from ..evaluation.metrics import SCORERS

SEARCH_SPACES: dict[str, dict[str, list[object]]] = {
    "decision_tree": {
        "clf__max_depth": [None, 8, 16, 32],
        "clf__min_samples_leaf": [1, 5, 20, 50],
    },
    "logreg": {
        "clf__C": [0.1, 1.0, 10.0],
        "clf__class_weight": [None, "balanced"],
    },
    "rf": {
        "clf__n_estimators": [200, 400],
        "clf__max_depth": [None, 20, 40],
        "clf__min_samples_leaf": [1, 5, 20],
        "clf__max_features": ["sqrt", 0.3],
    },
    "extra_trees": {
        "clf__n_estimators": [200, 400],
        "clf__max_depth": [None, 20, 40],
        "clf__min_samples_leaf": [1, 5, 20],
    },
    "hist_gb": {
        "clf__learning_rate": [0.05, 0.1],
        "clf__max_iter": [200, 400],
        "clf__max_leaf_nodes": [15, 31],
        "clf__l2_regularization": [0.0, 1.0],
    },
}

SEARCH_KINDS = ("grid", "random")


def build_search(
    pipeline,
    model_name: str | None = None,
    *,
    kind: str = "grid",
    space: dict[str, list[object]] | None = None,
    n_iter: int = 10,
    scoring: str = "macro_f1",
    n_jobs: int = -1,
):
    """Wrap ``pipeline`` in a search object over a named model's space.

    Purpose: give ``run_experiment`` one place that knows how a search is
    built, so callers pass ``search="grid"`` instead of constructing sklearn
    objects. The pipeline step names are part of the contract: space keys use
    the ``clf__`` prefix.

    Parameters:
        pipeline: the unfitted ``Pipeline`` to search over.
        model_name: registry name whose ``SEARCH_SPACES`` entry is used when
            ``space`` is omitted. May be ``None`` only when ``space`` is given
            (the ``estimator=`` escape hatch in ``run_experiment``).
        kind: ``"grid"`` for ``GridSearchCV`` or ``"random"`` for
            ``RandomizedSearchCV``.
        space: explicit parameter space; overrides ``SEARCH_SPACES[model_name]``.
        n_iter: number of sampled candidates for ``kind="random"``.
        scoring: a key of ``daa.evaluation.metrics.SCORERS``.
        n_jobs: passed through to sklearn.

    Invariants:
        ``refit=True`` always, so the returned object (and the estimator stored
        by ``run_cv(fit_full=True)``) can ``predict`` directly for submissions.
        Random search is deterministic via ``SEED``.

    Failure modes:
        ``ValueError`` for an unknown ``kind``, an unknown ``scoring``, or a
        ``model_name`` without a registered space when ``space`` is omitted.
    """
    if kind not in SEARCH_KINDS:
        raise ValueError(f"unknown search kind {kind!r}; choose from {SEARCH_KINDS}")
    if scoring not in SCORERS:
        raise ValueError(f"unknown scoring {scoring!r}; choose from {tuple(SCORERS)}")
    if space is None:
        if model_name not in SEARCH_SPACES:
            raise ValueError(
                f"no search space for model {model_name!r}; "
                f"choose from {tuple(SEARCH_SPACES)} or pass an explicit space"
            )
        space = SEARCH_SPACES[model_name]

    scorer = SCORERS[scoring]
    if kind == "grid":
        return GridSearchCV(
            pipeline,
            param_grid=space,
            scoring=scorer,
            refit=True,
            n_jobs=n_jobs,
        )
    return RandomizedSearchCV(
        pipeline,
        param_distributions=space,
        n_iter=n_iter,
        scoring=scorer,
        refit=True,
        random_state=SEED,
        n_jobs=n_jobs,
    )

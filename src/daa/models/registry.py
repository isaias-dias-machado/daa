"""The model boundary: a name maps to a fresh, unfitted classifier.

A factory must never return a shared estimator instance. ``Pipeline`` and
``GridSearchCV`` clone estimators internally, but returning a fresh object keeps
the contract obvious and makes notebook code safe.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

from sklearn.base import ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from ..config import SEED

ModelFactory = Callable[..., ClassifierMixin]

MODELS: dict[str, ModelFactory] = {
    "dummy": partial(DummyClassifier, strategy="most_frequent"),
    "decision_tree": partial(DecisionTreeClassifier, random_state=SEED),
    "logreg": partial(LogisticRegression, max_iter=2000),
    "rf": partial(
        RandomForestClassifier,
        n_estimators=400,
        n_jobs=-1,
        random_state=SEED,
    ),
    "extra_trees": partial(
        ExtraTreesClassifier,
        n_estimators=400,
        n_jobs=-1,
        random_state=SEED,
    ),
    "hist_gb": partial(HistGradientBoostingClassifier, random_state=SEED),
}

MODEL_NAMES = tuple(MODELS)


def make_model(name: str, **params: object) -> ClassifierMixin:
    """Return an unfitted classifier from the registry.

    Failure mode:
        ``KeyError`` when ``name`` is not registered.
    """
    try:
        factory = MODELS[name]
    except KeyError as error:
        available = ", ".join(MODEL_NAMES)
        raise KeyError(f"unknown model {name!r}; available: {available}") from error
    return factory(**params)

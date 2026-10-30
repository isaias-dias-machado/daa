"""Search spaces for ``GridSearchCV`` / ``RandomizedSearchCV``.

Keys use the ``clf__`` prefix because the project pipeline names its final step
``clf``. Keep these small enough to run on a laptop; the tuner owns the larger
spaces.
"""

from __future__ import annotations

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

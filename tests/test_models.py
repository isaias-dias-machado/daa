from __future__ import annotations

import pytest
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline

from daa.evaluation.metrics import SCORERS
from daa.experiment import build_pipeline
from daa.features.preprocess import build_preprocessor
from daa.models.registry import MODEL_NAMES, MODELS, make_model
from daa.models.spaces import SEARCH_SPACES, build_search

FAST_PARAMS = {
    "rf": {"n_estimators": 20},
    "extra_trees": {"n_estimators": 20},
    "hist_gb": {"max_iter": 20},
    "logreg": {"max_iter": 500},
}


def test_registry_is_not_empty():
    assert MODEL_NAMES
    assert set(MODEL_NAMES) == set(MODELS)


def test_unknown_model_raises():
    with pytest.raises(KeyError, match="unknown model"):
        make_model("does_not_exist")


def test_models_are_cloneable():
    for name in MODEL_NAMES:
        estimator = make_model(name)
        cloned = clone(estimator)
        assert type(cloned) is type(estimator)


def test_search_space_keys_are_valid():
    for name, space in SEARCH_SPACES.items():
        pipeline = Pipeline([("clf", make_model(name))])
        values = {key: options[0] for key, options in space.items()}
        pipeline.set_params(**values)


def test_every_search_space_has_a_model():
    assert set(SEARCH_SPACES).issubset(set(MODEL_NAMES))


def test_every_model_fits_a_pipeline(synthetic_xy, spec):
    features, target = synthetic_xy
    preprocessor = build_preprocessor(spec, variant="impute")

    for name in MODEL_NAMES:
        pipeline = build_pipeline(preprocessor, name, **FAST_PARAMS.get(name, {}))
        pipeline.fit(features, target)
        predictions = pipeline.predict(features)
        assert len(predictions) == len(features)
        assert hasattr(pipeline, "predict_proba")


def test_build_search_uses_the_models_space_by_default(spec):
    pipeline = Pipeline([("clf", make_model("decision_tree"))])
    search = build_search(pipeline, "decision_tree", n_jobs=1)

    assert isinstance(search, GridSearchCV)
    assert search.param_grid == SEARCH_SPACES["decision_tree"]
    assert search.scoring == SCORERS["macro_f1"]
    assert search.refit is True


def test_build_search_grid_and_random_predict_after_fit(synthetic_xy, spec):
    features, target = synthetic_xy
    tiny_space = {"clf__max_depth": [1, 3]}

    grid = build_search(
        build_pipeline(build_preprocessor(spec), "decision_tree", max_depth=3),
        "decision_tree",
        kind="grid",
        space=tiny_space,
        n_jobs=1,
    )
    random = build_search(
        build_pipeline(build_preprocessor(spec), "decision_tree", max_depth=3),
        "decision_tree",
        kind="random",
        space=tiny_space,
        n_iter=2,
        scoring="accuracy",
        n_jobs=1,
    )

    assert isinstance(grid, GridSearchCV)
    assert isinstance(random, RandomizedSearchCV)
    assert random.scoring == SCORERS["accuracy"]
    for search in (grid, random):
        search.fit(features, target)
        assert len(search.predict(features)) == len(features)
        assert len(search.predict_proba(features)) == len(features)


def test_build_search_rejects_bad_arguments(spec):
    pipeline = Pipeline([("clf", make_model("decision_tree"))])
    with pytest.raises(ValueError, match="unknown search kind"):
        build_search(pipeline, "decision_tree", kind="bogus")
    with pytest.raises(ValueError, match="unknown scoring"):
        build_search(pipeline, "decision_tree", scoring="bogus")
    with pytest.raises(ValueError, match="no search space"):
        build_search(pipeline, "dummy")
    with pytest.raises(ValueError, match="no search space"):
        build_search(pipeline, None)
    # an explicit space works without a model name (the estimator= escape hatch)
    search = build_search(pipeline, None, space={"clf__max_depth": [1]})
    assert isinstance(search, GridSearchCV)

from __future__ import annotations

import pytest
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from daa.experiment import build_pipeline
from daa.features.preprocess import build_preprocessor
from daa.models.registry import MODEL_NAMES, MODELS, make_model
from daa.models.spaces import SEARCH_SPACES

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

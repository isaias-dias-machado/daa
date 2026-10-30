from __future__ import annotations

import pandas as pd
import pytest

from daa.data import ColumnSpec
from daa.features.preprocess import build_preprocessor


def test_impute_variant_contract(synthetic_xy, spec):
    features, _ = synthetic_xy
    preprocessor = build_preprocessor(spec, variant="impute")

    transformed = preprocessor.fit_transform(features)
    names = preprocessor.get_feature_names_out()

    assert isinstance(transformed, pd.DataFrame)
    assert not transformed.isna().any().any()
    assert list(transformed.columns) == list(names)
    assert "PID" not in transformed.columns
    assert len(transformed) == len(features)


def test_unseen_category_is_ignored(synthetic_xy, spec):
    features, _ = synthetic_xy
    preprocessor = build_preprocessor(spec, variant="impute").fit(features)

    unseen = features.copy()
    unseen.loc[unseen.index[0], "cat_region"] = "Atlantida"
    transformed = preprocessor.transform(unseen)

    assert not transformed.isna().any().any()


def test_passthrough_leaves_numeric_nan(synthetic_xy, spec):
    features, _ = synthetic_xy
    preprocessor = build_preprocessor(spec, variant="passthrough")
    transformed = preprocessor.fit_transform(features)
    assert transformed.isna().any().any()


def test_unknown_variant_raises(spec):
    with pytest.raises(ValueError, match="unknown variant"):
        build_preprocessor(spec, variant="magic")


def test_ordinal_columns_are_encoded(synthetic_xy, spec):
    features, _ = synthetic_xy
    ordinal_spec = ColumnSpec(
        numeric=tuple(column for column in spec.numeric if column != "ord_health"),
        ordinal=("ord_health",),
        nominal=spec.nominal,
        drop=spec.drop,
    )
    preprocessor = build_preprocessor(ordinal_spec, variant="impute")
    transformed = preprocessor.fit_transform(features)
    names = preprocessor.get_feature_names_out()

    assert "ord_health" in names
    assert not transformed.isna().any().any()


def test_empty_spec_raises():
    empty = ColumnSpec(numeric=(), ordinal=(), nominal=(), drop=())
    with pytest.raises(ValueError, match="no input columns"):
        build_preprocessor(empty)

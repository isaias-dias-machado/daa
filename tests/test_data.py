from __future__ import annotations

import pandas as pd
import pytest

from daa.data import (
    TARGET,
    ColumnSpec,
    infer_column_spec,
    load_test,
    load_train,
    target_fingerprint,
)


def test_infer_column_spec_roles(synthetic_xy):
    features, _ = synthetic_xy
    spec = infer_column_spec(features)

    assert "num_height" in spec.numeric
    assert "ord_health" in spec.numeric  # numeric dtype until promoted by hand
    assert "cat_region" in spec.nominal
    assert spec.ordinal == ()
    assert "PID" in spec.drop
    assert "HHID" in spec.drop
    assert TARGET not in spec.input_columns
    assert set(spec.numeric).isdisjoint(spec.nominal)


def test_infer_column_spec_drops_constants(synthetic_xy):
    features, _ = synthetic_xy
    features = features.copy()
    features["constant_column"] = 1
    spec = infer_column_spec(features)
    assert "constant_column" in spec.drop


def test_spec_rejects_duplicates():
    with pytest.raises(ValueError, match="more than one role"):
        ColumnSpec(
            numeric=("a",),
            ordinal=("a",),
            nominal=(),
            drop=(),
        )


def test_spec_rejects_target_as_input():
    with pytest.raises(ValueError, match="target"):
        ColumnSpec(
            numeric=(TARGET,),
            ordinal=(),
            nominal=(),
            drop=(),
        )


def test_spec_validate_missing():
    spec = ColumnSpec(numeric=("a",), ordinal=(), nominal=(), drop=())
    with pytest.raises(ValueError, match="missing"):
        spec.validate(pd.DataFrame({"b": [1]}))


def test_load_train_roundtrip(tmp_path, synthetic_frame):
    path = tmp_path / "train.csv"
    synthetic_frame.to_csv(path, index=False)

    features, target = load_train(path)

    assert TARGET not in features.columns
    assert target.name == TARGET
    assert len(features) == len(target) == len(synthetic_frame)
    assert "PID" in features.columns


def test_load_test_row_ids(tmp_path, synthetic_xy):
    features, _ = synthetic_xy
    path = tmp_path / "test.csv"
    features.to_csv(path, index=False)

    loaded, row_ids = load_test(path)

    assert len(loaded) == len(row_ids)
    assert row_ids.name == "RowId"
    assert row_ids.tolist() == list(range(len(row_ids)))


def test_target_fingerprint_changes_with_values():
    first = pd.Series([0, 1, 2, 3])
    second = pd.Series([0, 1, 2, 4])
    assert target_fingerprint(first) != target_fingerprint(second)
    assert target_fingerprint(first) == target_fingerprint(first.copy())

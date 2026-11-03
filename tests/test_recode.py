"""Tests for the recoding rule-book.

Most tests use a tiny synthetic frame, so they need no ``data/``. The last test
runs on the real files and is skipped when ``data/`` is absent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from daa.config import TRAIN_PATH
from daa.data import (
    EXPLICIT_DROPS,
    explicit_spec,
    infer_column_spec,
    load_train,
)
from daa.features.recode import (
    BUCKET_COLUMNS,
    FLAG_SUFFIX,
    HHMM_COLUMNS,
    PARSED_COLUMNS,
    QUANTITY_COLUMNS,
    flag_column,
    recode_frame,
)


@pytest.fixture
def tiny_frame():
    """A frame that exercises every rule. Values are chosen to be easy to check."""
    return pd.DataFrame(
        {
            "PID": [1, 2, 3, 4],
            "PE7": ["0130", "Não aplicável", "Sem resposta", "3000"],
            "BM2": ["70.0", "Sem resposta", "80.5", "60"],
            "HHNBPERS_COD": [">=3", "1.0", "3 ou mais", "2.0"],
            "IN25": [">= 25", "Não aplicável", "4.0", "53 ou mais"],
            "cat_region": ["Norte", "Sul", "Norte", "Centro"],
        }
    )


def test_hhmm_is_a_duration(tiny_frame):
    recoded = recode_frame(tiny_frame)
    # 01:30 -> 90 minutes; 30:00 -> 1,800 minutes (no cap on hours).
    assert recoded["PE7"].tolist()[:1] == [90.0]
    assert recoded["PE7"].iloc[3] == 1800.0


def test_bucket_rules(tiny_frame):
    recoded = recode_frame(tiny_frame)
    assert recoded["HHNBPERS_COD"].tolist() == [3.0, 1.0, 3.0, 2.0]
    assert recoded["IN25"].iloc[[0, 2, 3]].tolist() == [25.0, 4.0, 53.0]
    assert np.isnan(recoded["IN25"].iloc[1])


def test_parsed_columns_are_float(tiny_frame):
    recoded = recode_frame(tiny_frame)
    for column in ("PE7", "BM2", "HHNBPERS_COD", "IN25"):
        assert recoded[column].dtype == "float64"


def test_sentinel_becomes_nan_and_keeps_a_flag(tiny_frame):
    recoded = recode_frame(tiny_frame)

    assert np.isnan(recoded["BM2"].iloc[1])
    assert recoded["BM2_FLAG"].tolist() == ["answer", "Sem resposta", "answer", "answer"]
    assert recoded["PE7_FLAG"].tolist() == ["answer", "Não aplicável", "Sem resposta", "answer"]


def test_real_nan_gets_the_missing_flag():
    frame = pd.DataFrame({"BM2": ["70.0", None]})
    recoded = recode_frame(frame)
    assert recoded["BM2_FLAG"].tolist() == ["answer", "missing"]
    assert np.isnan(recoded["BM2"].iloc[1])


def test_input_is_not_mutated(tiny_frame):
    before = tiny_frame.copy(deep=True)
    recode_frame(tiny_frame)
    pd.testing.assert_frame_equal(tiny_frame, before)


def test_original_columns_are_kept(tiny_frame):
    recoded = recode_frame(tiny_frame)
    assert list(recoded.columns[: len(tiny_frame.columns)]) == list(tiny_frame.columns)
    assert recoded["cat_region"].tolist() == tiny_frame["cat_region"].tolist()


def test_absent_known_columns_are_skipped():
    frame = pd.DataFrame({"cat_region": ["Norte", "Sul"]})
    recoded = recode_frame(frame)
    pd.testing.assert_frame_equal(recoded, frame)


def test_leftover_string_raises_with_the_value():
    frame = pd.DataFrame({"BM2": ["70.0", "nao sei"]})
    with pytest.raises(ValueError, match="nao sei"):
        recode_frame(frame)


def test_hhmm_minutes_above_59_raise():
    frame = pd.DataFrame({"PE9": ["0160"]})
    with pytest.raises(ValueError, match="minutes above 59"):
        recode_frame(frame)


def test_hhmm_wrong_width_raises():
    frame = pd.DataFrame({"PE9": ["130"]})
    with pytest.raises(ValueError, match="130"):
        recode_frame(frame)


def test_column_groups_are_disjoint_and_complete():
    groups = (HHMM_COLUMNS, QUANTITY_COLUMNS, BUCKET_COLUMNS)
    assert len(PARSED_COLUMNS) == sum(len(group) for group in groups) == 29
    assert len(set(PARSED_COLUMNS)) == 29
    assert flag_column("BM2") == f"BM2{FLAG_SUFFIX}"


def test_explicit_spec_drops_the_hand_written_columns(synthetic_frame):
    features = synthetic_frame.drop(columns=["AGE_RANGE"]).copy()
    # Vary the values so they are not constants: infer_column_spec drops
    # constants on its own, and this test is about the hand-written override.
    features["REFYEAR"] = [2019] * (len(features) - 1) + [2020]
    features["REFMONTH"] = np.arange(len(features)) % 12 + 1
    features["IN13"] = "Não aplicável"
    features.loc[features.index[0], "IN13"] = "Só com ajuda"
    features["AC1A"] = "Não"
    features.loc[features.index[0], "AC1A"] = "Sim"

    inferred = infer_column_spec(features)
    explicit = explicit_spec(inferred)

    for column in EXPLICIT_DROPS:
        assert column not in explicit.input_columns
        assert column in explicit.drop
    # The input spec is not changed, and the other roles are untouched.
    assert "REFYEAR" in inferred.input_columns
    for group in ("numeric", "ordinal", "nominal"):
        assert getattr(explicit, group) == tuple(
            column for column in getattr(inferred, group) if column not in EXPLICIT_DROPS
        )


def test_explicit_spec_skips_absent_columns(synthetic_xy):
    features, _ = synthetic_xy
    inferred = infer_column_spec(features)
    explicit = explicit_spec(inferred)
    assert explicit.numeric == inferred.numeric
    assert explicit.nominal == inferred.nominal
    assert set(inferred.drop) <= set(explicit.drop)
    assert set(explicit.drop) - set(inferred.drop) == set(EXPLICIT_DROPS)


@pytest.mark.skipif(not TRAIN_PATH.exists(), reason="data/ is absent")
def test_real_train_frame_is_recoded():
    features, target = load_train()

    assert len(features) == len(target) == 10_962
    for column in ("BM1", "BM2", "PE7", "PE9", "HHNBPERS_COD", "IN25", "HO12_COD"):
        assert features[column].dtype == "float64"
    for column in PARSED_COLUMNS:
        assert features[column].dtype == "float64"
        assert flag_column(column) in features.columns

    # 29 parsed columns, so 29 flag columns.
    assert sum(name.endswith(FLAG_SUFFIX) for name in features.columns) == 29
    # PE7 max is 30h00 = 1,800 minutes. Values above 23:59 are kept.
    assert features["PE7"].max() == 1800.0
    assert (features["PE7"] > 24 * 60).sum() == 13
    # IDs and the left-nominal columns are untouched.
    assert "PID" in features.columns
    assert features["PA1"].dtype != "float64"

"""Shared fixtures. Tests run on synthetic data, never on ``data/``."""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import pytest

from daa.data import TARGET, infer_column_spec
from daa.fixtures import make_synthetic_frame
from daa.splits import SplitConfig, make_folds


@pytest.fixture
def synthetic_frame():
    return make_synthetic_frame(n_samples=240, seed=0)


@pytest.fixture
def synthetic_xy(synthetic_frame):
    features = synthetic_frame.drop(columns=[TARGET])
    target = synthetic_frame[TARGET].astype(int)
    return features, target


@pytest.fixture
def spec(synthetic_xy):
    features, _ = synthetic_xy
    return infer_column_spec(features)


@pytest.fixture
def folds(synthetic_xy):
    _, target = synthetic_xy
    return make_folds(target, SplitConfig(n_splits=3, n_repeats=1, seed=0))

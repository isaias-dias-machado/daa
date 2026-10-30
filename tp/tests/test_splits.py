from __future__ import annotations

import numpy as np
import pytest

from daa.splits import (
    SplitConfig,
    get_folds,
    load_folds,
    make_folds,
    save_folds,
)


def test_folds_partition_every_sample(synthetic_xy):
    _, target = synthetic_xy
    config = SplitConfig(n_splits=3, n_repeats=2, seed=0)
    folds = make_folds(target, config)

    assert len(folds) == config.n_splits * config.n_repeats
    for repeat in range(config.n_repeats):
        validation = np.concatenate([fold.val_idx for fold in folds if fold.repeat == repeat])
        assert sorted(validation.tolist()) == list(range(len(target)))

    for fold in folds:
        assert set(fold.train_idx).isdisjoint(set(fold.val_idx))
        assert len(fold.train_idx) + len(fold.val_idx) == len(target)


def test_save_load_roundtrip(tmp_path, synthetic_xy):
    _, target = synthetic_xy
    folds = make_folds(target, SplitConfig(n_splits=3, n_repeats=1, seed=0))
    path = tmp_path / "folds.npz"
    save_folds(folds, target, path)

    loaded = load_folds(target, path)

    assert len(loaded) == len(folds)
    for original, restored in zip(folds, loaded, strict=True):
        assert original.repeat == restored.repeat
        assert original.fold == restored.fold
        assert np.array_equal(original.val_idx, restored.val_idx)


def test_load_rejects_changed_target(tmp_path, synthetic_xy):
    _, target = synthetic_xy
    folds = make_folds(target, SplitConfig(n_splits=3, n_repeats=1, seed=0))
    path = tmp_path / "folds.npz"
    save_folds(folds, target, path)

    with pytest.raises(ValueError, match="different target"):
        load_folds(target + 1, path)


def test_get_folds_creates_and_reuses(tmp_path, synthetic_xy):
    _, target = synthetic_xy
    path = tmp_path / "folds.npz"
    config = SplitConfig(n_splits=3, n_repeats=1, seed=0)

    first = get_folds(target, path, config)
    assert path.exists()
    second = get_folds(target, path, config)

    for original, restored in zip(first, second, strict=True):
        assert np.array_equal(original.val_idx, restored.val_idx)

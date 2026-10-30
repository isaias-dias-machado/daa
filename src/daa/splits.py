"""Frozen cross-validation folds.

The fold assignment is generated once from the target vector and persisted.
Every experiment loads the same file, so scores are comparable. The stored
target fingerprint makes a changed target vector fail loudly instead of
silently mixing experiments.

Regenerate deliberately:

    python -m daa.splits --force
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

from .config import SEED, SPLITS_PATH
from .data import load_train, target_fingerprint


@dataclass(frozen=True)
class SplitConfig:
    """Parameters of the one agreed cross-validation scheme."""

    n_splits: int = 5
    n_repeats: int = 3
    seed: int = SEED


@dataclass(frozen=True)
class Fold:
    """One train/validation split, stored as positional indices."""

    repeat: int
    fold: int
    train_idx: np.ndarray
    val_idx: np.ndarray


DEFAULT_SPLIT_CONFIG = SplitConfig()


def make_assignments(target: pd.Series, config: SplitConfig = DEFAULT_SPLIT_CONFIG) -> np.ndarray:
    """Return an ``int8`` matrix of shape ``(n_repeats, n_samples)``.

    Cell ``[repeat, sample]`` is the validation fold of that sample. Every
    sample belongs to exactly one validation fold per repeat.
    """
    values = np.asarray(pd.Series(target).to_numpy(), dtype=np.int64)
    if len(values) < config.n_splits:
        raise ValueError(f"need at least {config.n_splits} rows for {config.n_splits}-fold CV")

    assignments = np.full((config.n_repeats, len(values)), fill_value=-1, dtype=np.int8)
    splitter = RepeatedStratifiedKFold(
        n_splits=config.n_splits,
        n_repeats=config.n_repeats,
        random_state=config.seed,
    )
    for split_number, (_, validation_idx) in enumerate(
        splitter.split(np.zeros(len(values)), values)
    ):
        repeat, fold = divmod(split_number, config.n_splits)
        assignments[repeat, validation_idx] = fold

    if (assignments < 0).any():
        raise RuntimeError("some samples were never assigned to a validation fold")
    return assignments


def folds_from_assignments(assignments: np.ndarray) -> list[Fold]:
    """Expand the assignment matrix into ready-to-use train/validation indices."""
    folds: list[Fold] = []
    for repeat in range(assignments.shape[0]):
        for fold in range(int(assignments[repeat].max()) + 1):
            validation_idx = np.flatnonzero(assignments[repeat] == fold)
            train_idx = np.flatnonzero(assignments[repeat] != fold)
            folds.append(
                Fold(
                    repeat=repeat,
                    fold=fold,
                    train_idx=train_idx,
                    val_idx=validation_idx,
                )
            )
    return folds


def make_folds(target: pd.Series, config: SplitConfig = DEFAULT_SPLIT_CONFIG) -> list[Fold]:
    """Generate folds in memory without touching the persisted file."""
    return folds_from_assignments(make_assignments(target, config))


def save_folds(
    folds: list[Fold],
    target: pd.Series,
    path: str | Path = SPLITS_PATH,
    config: SplitConfig = DEFAULT_SPLIT_CONFIG,
) -> Path:
    """Persist the fold assignment plus a fingerprint of the target vector."""
    n_samples = len(pd.Series(target))
    n_repeats = max(fold.repeat for fold in folds) + 1
    n_splits = max(fold.fold for fold in folds) + 1
    assignments = np.full((n_repeats, n_samples), fill_value=-1, dtype=np.int8)
    for fold in folds:
        if (assignments[fold.repeat, fold.val_idx] != -1).any():
            raise ValueError("two folds claim the same sample")
        assignments[fold.repeat, fold.val_idx] = fold.fold

    if (assignments < 0).any():
        raise ValueError("fold list does not cover every sample")

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        assignments=assignments,
        target_fingerprint=target_fingerprint(target),
        n_splits=n_splits,
        n_repeats=n_repeats,
        seed=config.seed,
        n_samples=n_samples,
    )
    return destination


def load_folds(target: pd.Series, path: str | Path = SPLITS_PATH) -> list[Fold]:
    """Load persisted folds and verify that they belong to ``target``."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"no splits at {source}; run `python -m daa.splits` first")

    with np.load(source) as stored:
        assignments = stored["assignments"].copy()
        stored_fingerprint = str(stored["target_fingerprint"])
        stored_samples = int(stored["n_samples"])

    current_samples = len(pd.Series(target))
    if stored_fingerprint != target_fingerprint(target) or stored_samples != current_samples:
        raise ValueError(
            "these splits belong to a different target vector; "
            "regenerate with `python -m daa.splits --force`"
        )
    return folds_from_assignments(assignments)


def get_folds(
    target: pd.Series,
    path: str | Path = SPLITS_PATH,
    config: SplitConfig = DEFAULT_SPLIT_CONFIG,
) -> list[Fold]:
    """Load the frozen folds, or create and persist them on first use."""
    source = Path(path)
    if source.exists():
        return load_folds(target, source)
    folds = make_folds(target, config)
    save_folds(folds, target, source, config)
    return folds


def main(argv: list[str] | None = None) -> None:
    """Generate the shared split file from the training target."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=SPLITS_PATH)
    parser.add_argument("--splits", type=int, default=DEFAULT_SPLIT_CONFIG.n_splits)
    parser.add_argument("--repeats", type=int, default=DEFAULT_SPLIT_CONFIG.n_repeats)
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args(argv)

    if arguments.path.exists() and not arguments.force:
        raise SystemExit(f"{arguments.path} already exists; pass --force to regenerate")

    _, target = load_train()
    config = SplitConfig(n_splits=arguments.splits, n_repeats=arguments.repeats)
    folds = make_folds(target, config)
    save_folds(folds, target, arguments.path, config)
    print(f"wrote {len(folds)} folds for {len(target)} rows to {arguments.path}")


if __name__ == "__main__":
    main()

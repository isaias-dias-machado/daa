"""Feature selection: screen columns, rank them, persist the shortlist.

This module answers one narrow question: **which columns can be deleted and
still keep the macro F1 of the full model?** It does not answer "which factors
characterise age". That question is explanatory and belongs to a different
artifact, built after the model is fixed.

Two kinds of evidence are produced here:

- ``mutual_info_screen``: a cheap univariate filter. It scores one column at a
  time, so it cannot see redundancy between columns and cannot see two weak
  columns that only work together.
- ``rank_by_permutation``: the expensive ranker. It fits a model, then measures
  how much the score drops when one raw column is shuffled.

Both look at the labels, so both are optimistic when the same rows are used to
score the model afterwards. The notebook measures that optimism with a nested
check; this module only supplies the pieces.

Design rule: every function takes and returns *named* columns. Selecting by
position would make the ranking table unreadable and the shortlist impossible
to verify.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance

from ..config import SEED
from ..data import ID_COLUMNS, TARGET, ColumnSpec, target_fingerprint
from ..evaluation.metrics import SCORERS
from ..splits import Fold
from ..utils import git_dirty, git_sha, json_default


def columns_fingerprint(columns: Sequence[str]) -> str:
    """Stable fingerprint of an ordered column list.

    Purpose: detect a shortlist file whose column list was edited by hand. The
    order is part of the fingerprint because the shortlist is ordered by
    importance.

    Invariants:
        The same sequence always produces the same string, across processes
        and machines. The JSON encoding makes the boundary between names
        unambiguous, so no column name can imitate another list.

    Failure modes:
        None. Any sequence of strings is accepted.
    """
    payload = json.dumps(list(columns), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def subset_spec(spec: ColumnSpec, columns: Sequence[str]) -> ColumnSpec:
    """Return ``spec`` restricted to ``columns``, keeping every role.

    Purpose: build the preprocessor for a shortlist without writing a second
    spec by hand. A column keeps the role it had in ``spec``; nothing is
    promoted or demoted.

    Invariants:
        The result lists columns in ``spec`` order (numeric, ordinal, nominal),
        not in the order of ``columns``. Therefore reversing ``columns`` gives
        the same spec, and two runs that select the same set build the same
        pipeline.
        ``drop`` is copied unchanged: it describes the frame, not the model.

    Failure modes:
        ``ValueError`` when ``columns`` is empty, contains a duplicate, names
        the target, names an ID column, or names a column that is not an input
        of ``spec`` (including columns ``spec`` drops).
    """
    requested = list(columns)
    if not requested:
        raise ValueError("columns is empty; a model needs at least one input column")

    duplicates = sorted({column for column in requested if requested.count(column) > 1})
    if duplicates:
        raise ValueError(f"columns contains duplicates: {duplicates}")

    for column in requested:
        if column == TARGET:
            raise ValueError(f"{TARGET!r} is the target and cannot be selected")
        if column in ID_COLUMNS:
            raise ValueError(f"{column!r} is an ID column and cannot be selected")

    inputs = set(spec.input_columns)
    unknown = [column for column in requested if column not in inputs]
    if unknown:
        dropped = [column for column in unknown if column in spec.drop]
        absent = [column for column in unknown if column not in spec.drop]
        reasons = []
        if dropped:
            reasons.append(f"dropped by the spec: {dropped}")
        if absent:
            reasons.append(f"not in the spec: {absent}")
        raise ValueError("columns are not model inputs (" + "; ".join(reasons) + ")")

    keep = set(requested)
    return ColumnSpec(
        numeric=tuple(column for column in spec.numeric if column in keep),
        ordinal=tuple(column for column in spec.ordinal if column in keep),
        nominal=tuple(column for column in spec.nominal if column in keep),
        drop=spec.drop,
    )


def ordinal_codes(frame: pd.DataFrame) -> pd.DataFrame:
    """Encode every column as integer codes, with NaN as ``-1``.

    Purpose: give the tree-based ranker one column per raw feature, so the
    ranking table has one row per named column and no dummy aggregation.

    Why ``factorize`` and not ``OrdinalEncoder``: factorize needs no fit, so
    there is nothing to leak and no fitted object to keep. The codes are
    arbitrary. A tree only needs equality, so the false order is an
    approximation; the notebook measures whether it changes the ranking.

    Invariants:
        The output has the same shape and column names as ``frame``, and no
        NaN. A value that is missing in ``frame`` becomes ``-1``.

    Failure modes:
        ``TypeError`` when ``frame`` is not a DataFrame.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame; column names are part of the contract")
    return frame.apply(lambda column: pd.factorize(column, use_na_sentinel=True)[0])


def mutual_info_screen(
    frame: pd.DataFrame,
    target: pd.Series,
    top_n: int | None = None,
    n_shuffles: int = 10,
    seed: int = SEED,
) -> pd.DataFrame:
    """Rank columns by mutual information with the target, corrected for noise.

    Purpose: make the expensive ranker cheap by dropping columns with no
    marginal signal.

    The correction matters. A shuffled target still produces a small positive
    MI, and that noise floor is not negligible: measured on this data, the
    60th score was 0.088 while the noise floor reached 0.076. One shuffle is
    not enough to estimate it, so ``n_shuffles`` shuffles are averaged and
    subtracted.

    Invariants:
        The result has one row per input column, sorted by ``mi_adjusted``
        descending and then by name, so ties are broken deterministically.
        ``rank`` starts at 1. ``mi_adjusted`` may be negative; that is an
        honest "less than noise", not an error.
        The MI call is deterministic once every column is a discrete code.

    Failure modes:
        ``TypeError`` when ``frame`` is not a DataFrame.
        ``ValueError`` when the lengths differ or ``n_shuffles < 1``.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame; column names are part of the contract")
    if len(frame) != len(target):
        raise ValueError(f"frame and target have different lengths: {len(frame)} != {len(target)}")
    if n_shuffles < 1:
        raise ValueError(f"n_shuffles must be at least 1, got {n_shuffles}")

    codes = ordinal_codes(frame)
    values = np.asarray(pd.Series(target).reset_index(drop=True))
    scores = mutual_info_classif(codes, values, discrete_features=True, random_state=seed)

    generator = np.random.default_rng(seed)
    noise = np.zeros(len(frame.columns))
    for _ in range(n_shuffles):
        shuffled = generator.permutation(values)
        noise += mutual_info_classif(codes, shuffled, discrete_features=True, random_state=seed)
    noise /= n_shuffles

    table = pd.DataFrame(
        {
            "column": list(frame.columns),
            "mi": scores,
            "mi_noise": noise,
            "mi_adjusted": scores - noise,
        }
    ).sort_values(["mi_adjusted", "column"], ascending=[False, True], ignore_index=True)
    table["rank"] = np.arange(1, len(table) + 1)
    table = table[["rank", "column", "mi", "mi_noise", "mi_adjusted"]]

    if top_n is not None:
        if top_n < 1:
            raise ValueError(f"top_n must be at least 1, got {top_n}")
        table = table.head(top_n).reset_index(drop=True)
    return table


def rank_by_permutation(
    estimator,
    frame: pd.DataFrame,
    target: pd.Series,
    folds: Sequence[Fold],
    n_repeats: int = 5,
    scoring: str = "macro_f1",
    seed: int = SEED,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Rank raw columns by how much the score drops when each one is shuffled.

    Purpose: the ranking that the shortlist is built from. It sees interactions
    between columns, which mutual information cannot.

    ``folds`` are positional index pairs, not necessarily the frozen CV folds.
    That is deliberate: the nested check needs a fit set and an evaluation set
    carved out of one outer training fold, and it can pass that pair as a
    single ``Fold``.

    Invariants:
        One row per column of ``frame``. ``importance_mean`` is the mean over
        folds, ``importance_std`` the spread over folds (a stability signal,
        not a confidence interval). Rows are sorted by mean descending and
        then by name. The estimator is cloned per fold, so nothing is shared
        between folds.

    Why ``n_jobs=1`` by default: the forest already uses every core, and
    measured on this laptop, ``n_jobs=-1`` was slower.

    Failure modes:
        ``TypeError`` when ``frame`` is not a DataFrame.
        ``ValueError`` when lengths differ, ``folds`` is empty, the score name
        is unknown, or ``frame`` has duplicate column names (the permutation
        result would be ambiguous).
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame; column names are part of the contract")
    if len(frame) != len(target):
        raise ValueError(f"frame and target have different lengths: {len(frame)} != {len(target)}")
    if not folds:
        raise ValueError("folds is empty; there is nothing to rank on")
    if scoring not in SCORERS:
        raise ValueError(f"unknown score {scoring!r}; choose from {tuple(SCORERS)}")
    if frame.columns.duplicated().any():
        duplicates = list(frame.columns[frame.columns.duplicated()])
        raise ValueError(f"frame has duplicate column names: {duplicates}")

    values = pd.Series(target).reset_index(drop=True)
    per_fold: list[np.ndarray] = []
    for fold in folds:
        model = clone(estimator)
        model.fit(frame.iloc[fold.train_idx], values.iloc[fold.train_idx])
        importance = permutation_importance(
            model,
            frame.iloc[fold.val_idx],
            values.iloc[fold.val_idx],
            scoring=SCORERS[scoring],
            n_repeats=n_repeats,
            random_state=seed,
            n_jobs=n_jobs,
        )
        per_fold.append(np.asarray(importance.importances_mean, dtype=float))

    stacked = np.vstack(per_fold)
    table = pd.DataFrame(
        {
            "column": list(frame.columns),
            "importance_mean": stacked.mean(axis=0),
            "importance_std": stacked.std(axis=0, ddof=0),
        }
    ).sort_values(["importance_mean", "column"], ascending=[False, True], ignore_index=True)
    table["rank"] = np.arange(1, len(table) + 1)
    return table[["rank", "column", "importance_mean", "importance_std"]]


@dataclass(frozen=True)
class Selection:
    """A persisted shortlist plus the evidence that produced it.

    Invariants:
        ``columns`` is ordered (most important first), unique, and every name
        is an input column of the spec it was validated against.
        ``k`` equals ``len(columns)``; the loader rejects a file where the two
        disagree.
        ``fingerprint`` is ``columns_fingerprint(columns)``.

    Failure modes:
        Constructed by ``load_selected_features``, which raises instead of
        returning an inconsistent object.
    """

    columns: tuple[str, ...]
    k: int
    fingerprint: str
    target_fingerprint: str | None
    metadata: dict[str, Any]


def save_selection(
    path: str | Path,
    columns: Sequence[str],
    *,
    spec: ColumnSpec,
    target: pd.Series | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write the shortlist and its provenance to one JSON file.

    Purpose: notebook outputs are stripped at commit time, so a table kept in
    the notebook is not a durable artifact. This file is.

    Contents: the ordered column list, ``k``, the column fingerprint, the
    target fingerprint, the spec counts, the seed, the git state, a timestamp,
    and a free-form ``metadata`` block (method, run ids, baseline comparison).

    Invariants:
        The columns are validated by ``subset_spec`` before anything is
        written, so a bad shortlist never reaches disk.

    Failure modes:
        ``ValueError`` from ``subset_spec``. ``OSError`` when the path cannot
        be written.
    """
    ordered = tuple(columns)
    subset_spec(spec, ordered)

    payload = {
        "columns": list(ordered),
        "k": len(ordered),
        "fingerprint": columns_fingerprint(ordered),
        "target_fingerprint": None if target is None else target_fingerprint(target),
        "spec_counts": spec.counts(),
        "seed": SEED,
        "git_sha": git_sha(),
        "git_dirty": git_dirty(),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "metadata": dict(metadata or {}),
    }

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, default=json_default) + "\n")
    return destination


def load_selected_features(
    path: str | Path,
    spec: ColumnSpec,
    target: pd.Series | None = None,
) -> Selection:
    """Load a shortlist and check that it still fits the current data.

    Purpose: the same job ``load_folds`` does for the splits. A shortlist that
    names a column the frame no longer has must fail loudly, not silently drop
    it.

    Checks, in order: the file exists, the required keys are present, ``k``
    matches the list, the column fingerprint matches, every name is a unique
    input of ``spec``, and (when ``target`` is given) the target fingerprint
    matches.

    Failure modes:
        ``FileNotFoundError`` when the file is absent.
        ``ValueError`` for any inconsistency above.
    """
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"no selection at {source}; run the selection notebook first")

    payload = json.loads(source.read_text())
    for key in ("columns", "k", "fingerprint"):
        if key not in payload:
            raise ValueError(f"{source} is not a selection file: missing {key!r}")

    columns = tuple(payload["columns"])
    if payload["k"] != len(columns):
        raise ValueError(f"{source}: k={payload['k']} does not match {len(columns)} columns")
    if columns_fingerprint(columns) != payload["fingerprint"]:
        raise ValueError(f"{source}: the column list does not match its fingerprint")

    subset_spec(spec, columns)

    stored_target = payload.get("target_fingerprint")
    if target is not None:
        if stored_target is None:
            raise ValueError(f"{source} has no target fingerprint; cannot check the target")
        if stored_target != target_fingerprint(target):
            raise ValueError(f"{source} was built for a different target vector")

    return Selection(
        columns=columns,
        k=int(payload["k"]),
        fingerprint=str(payload["fingerprint"]),
        target_fingerprint=stored_target,
        metadata=dict(payload.get("metadata", {})),
    )

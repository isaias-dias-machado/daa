from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from daa.evaluation.cv import run_cv, write_run
from daa.evaluation.submit import (
    validate_submission,
    write_submission,
    write_submission_from_result,
)
from daa.experiment import build_pipeline, run_experiment
from daa.features.preprocess import build_preprocessor


def make_pipeline(spec):
    preprocessor = build_preprocessor(spec, variant="impute")
    return build_pipeline(preprocessor, "decision_tree", max_depth=3)


def test_run_cv_contract(synthetic_xy, spec, folds):
    features, target = synthetic_xy
    result = run_cv(make_pipeline(spec), features, target, folds, name="test")

    assert len(result.fold_scores) == len(folds)
    assert result.metrics["macro_f1_mean"] == pytest.approx(result.fold_scores["macro_f1"].mean())
    repeats = len({fold.repeat for fold in folds})
    assert len(result.oof) == len(features) * repeats
    assert set(result.per_class["label"]) == set(target.unique())
    assert result.confusion.shape == (target.nunique(), target.nunique())
    assert result.n_features > 0
    assert result.estimator is None


def test_run_cv_fit_full(synthetic_xy, spec, folds):
    features, target = synthetic_xy
    result = run_cv(
        make_pipeline(spec),
        features,
        target,
        folds,
        name="test_full",
        fit_full=True,
    )
    assert result.estimator is not None
    assert len(result.estimator.predict(features)) == len(features)


def test_run_cv_rejects_non_dataframe(synthetic_xy, folds):
    _, target = synthetic_xy
    with pytest.raises(TypeError, match="DataFrame"):
        run_cv("not an estimator", np.zeros((len(target), 2)), target, folds)


def test_write_run_artifacts(tmp_path, synthetic_xy, spec, folds):
    features, target = synthetic_xy
    output_dir = tmp_path / "runs"

    first = run_cv(make_pipeline(spec), features, target, folds, name="first")
    run_directory = write_run(first, output_dir=output_dir)

    assert (run_directory / "metrics.json").exists()
    assert (run_directory / "fold_scores.csv").exists()
    assert (run_directory / "oof_predictions.csv").exists()
    assert (run_directory / "per_class.csv").exists()
    assert (run_directory / "confusion_matrix.csv").exists()
    assert not (run_directory / "estimator.joblib").exists()

    leaderboard = pd.read_csv(tmp_path / "leaderboard.csv")
    assert len(leaderboard) == 1
    assert leaderboard.loc[0, "name"] == "first"

    second = run_cv(make_pipeline(spec), features, target, folds, name="second")
    write_run(second, output_dir=output_dir)
    leaderboard = pd.read_csv(tmp_path / "leaderboard.csv")
    assert len(leaderboard) == 2


def test_run_experiment_saves(tmp_path, synthetic_xy, folds):
    features, target = synthetic_xy
    result = run_experiment(
        "dummy",
        name="dummy_test",
        X=features,
        y=target,
        folds=folds,
        fit_full=True,
        output_dir=tmp_path / "runs",
    )

    assert result.estimator is not None
    assert (tmp_path / "leaderboard.csv").exists()
    assert (tmp_path / "runs" / result.run_id / "estimator.joblib").exists()


def test_run_experiment_can_skip_saving(tmp_path, synthetic_xy, folds):
    features, target = synthetic_xy
    run_experiment(
        "dummy",
        name="not_saved",
        X=features,
        y=target,
        folds=folds,
        save=False,
        output_dir=tmp_path / "runs",
    )
    assert not (tmp_path / "runs").exists()


def test_submission_roundtrip(tmp_path, synthetic_xy, spec, folds):
    features, target = synthetic_xy
    result = run_cv(
        make_pipeline(spec),
        features,
        target,
        folds,
        name="submit",
        fit_full=True,
    )
    row_ids = pd.Series(range(len(features)), name="RowId")
    path = tmp_path / "submission.csv"

    write_submission_from_result(result, features, row_ids, path)
    submission = validate_submission(path, row_ids)

    assert list(submission.columns) == ["RowId", "AGE_RANGE"]
    assert len(submission) == len(features)


def test_submission_requires_fit_full(tmp_path, synthetic_xy, spec, folds):
    features, target = synthetic_xy
    result = run_cv(make_pipeline(spec), features, target, folds, name="no_full")
    with pytest.raises(ValueError, match="fit_full"):
        write_submission_from_result(
            result,
            features,
            pd.Series(range(len(features))),
            tmp_path / "submission.csv",
        )


def test_write_submission_rejects_duplicate_ids(tmp_path, synthetic_xy, spec):
    features, target = synthetic_xy
    pipeline = make_pipeline(spec).fit(features, target)
    with pytest.raises(ValueError, match="unique"):
        write_submission(
            pipeline,
            features,
            np.zeros(len(features), dtype=int),
            tmp_path / "submission.csv",
        )

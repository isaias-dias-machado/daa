from __future__ import annotations

from daa.evaluation.cv import run_cv
from daa.evaluation.report import (
    feature_importance_table,
    permutation_importance_table,
    plot_confusion,
)
from daa.experiment import build_pipeline
from daa.features.preprocess import build_preprocessor


def test_report_helpers(tmp_path, synthetic_xy, spec, folds):
    features, target = synthetic_xy
    pipeline = build_pipeline(
        build_preprocessor(spec, variant="impute"),
        "decision_tree",
        max_depth=4,
        min_samples_leaf=5,
    )
    result = run_cv(pipeline, features, target, folds, name="report", fit_full=True)

    image = plot_confusion(result, tmp_path / "confusion.png")
    assert image.exists()

    importance = feature_importance_table(result, top_n=5)
    assert len(importance) == 5
    assert set(importance.columns) == {"feature", "importance"}

    permutation = permutation_importance_table(result, features, target, n_repeats=1)
    assert len(permutation) == len(features.columns)
    assert set(permutation.columns) == {
        "feature",
        "importance_mean",
        "importance_std",
    }

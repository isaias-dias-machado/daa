"""Integration tests for the ``run_experiment`` knobs.

These cover the named feature sets, the default preprocessing variant, the
search wrapper, the ``repeats`` shortcut, the ``estimator=`` escape hatch, and
reloading a saved run with ``load_run``.
"""

from __future__ import annotations

import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

from daa.data import explicit_spec, infer_column_spec
from daa.evaluation.cv import load_run, run_cv
from daa.evaluation.submit import validate_submission, write_submission_from_result
from daa.experiment import build_pipeline, run_experiment
from daa.features.preprocess import DEFAULT_VARIANT, build_preprocessor
from daa.features.selection import columns_fingerprint, save_selection
from daa.models.spaces import SEARCH_SPACES
from daa.splits import SplitConfig, make_folds


@pytest.fixture
def selected_path(tmp_path, synthetic_xy, monkeypatch):
    """A persisted shortlist that ``feature_set="selected"`` resolves to."""
    features, target = synthetic_xy
    spec = explicit_spec(infer_column_spec(features))
    path = save_selection(
        tmp_path / "selected.json",
        ["num_height", "cat_sex"],
        spec=spec,
        target=target,
    )
    monkeypatch.setattr("daa.features.selection.SELECTED_FEATURES_PATH", path)
    return path


def test_feature_set_selected_resolves_columns(synthetic_xy, folds, selected_path):
    features, target = synthetic_xy
    result = run_experiment(
        "dummy",
        name="selected",
        X=features,
        y=target,
        folds=folds,
        feature_set="selected",
        save=False,
    )

    assert result.params["feature_set"] == "selected"
    assert result.params["n_input_columns"] == 2
    assert result.params["feature_hash"] == columns_fingerprint(("num_height", "cat_sex"))[:12]


def test_columns_override_wins_over_feature_set(synthetic_xy, folds, selected_path):
    features, target = synthetic_xy
    result = run_experiment(
        "dummy",
        name="override",
        X=features,
        y=target,
        folds=folds,
        columns=["num_weight"],
        feature_set="selected",
        save=False,
    )

    assert result.params["n_input_columns"] == 1
    assert result.params["feature_hash"] == columns_fingerprint(("num_weight",))[:12]


def test_unknown_feature_set_raises(synthetic_xy, folds):
    features, target = synthetic_xy
    with pytest.raises(ValueError, match="unknown feature set"):
        run_experiment(
            "dummy",
            name="bad_set",
            X=features,
            y=target,
            folds=folds,
            feature_set="bogus",
            save=False,
        )


def test_preprocessor_none_resolves_default_variant(synthetic_xy, folds):
    features, target = synthetic_xy
    result = run_experiment(
        "dummy",
        name="default_variant",
        X=features,
        y=target,
        folds=folds,
        save=False,
    )

    assert result.params["preprocessor"] == DEFAULT_VARIANT


def test_feature_set_none_keeps_derived_label(synthetic_xy, folds):
    features, target = synthetic_xy
    subset = run_experiment(
        "dummy",
        name="subset",
        X=features,
        y=target,
        folds=folds,
        columns=["num_height"],
        feature_set=None,
        save=False,
    )
    full = run_experiment(
        "dummy",
        name="full",
        X=features,
        y=target,
        folds=folds,
        feature_set=None,
        save=False,
    )

    assert subset.params["feature_set"] == "subset"
    assert full.params["feature_set"] == "all"


def test_search_grid_with_explicit_space(synthetic_xy, folds):
    features, target = synthetic_xy
    result = run_experiment(
        "decision_tree",
        name="grid",
        X=features,
        y=target,
        folds=folds,
        search={"kind": "grid", "space": {"clf__max_depth": [1, 3]}, "n_jobs": 1},
        save=False,
    )

    assert len(result.fold_scores) == len(folds)
    assert result.n_features > 0
    search = result.params["search"]
    assert search["kind"] == "grid"
    assert search["space"] == ["clf__max_depth"]
    assert search["scoring"] == "macro_f1"


def test_search_string_uses_the_models_registered_space(synthetic_xy, folds):
    features, target = synthetic_xy
    result = run_experiment(
        "decision_tree",
        name="grid_default_space",
        X=features,
        y=target,
        folds=folds,
        search="grid",
        save=False,
    )

    assert result.params["search"]["space"] == sorted(SEARCH_SPACES["decision_tree"])


def test_search_random_records_n_iter(synthetic_xy, folds):
    features, target = synthetic_xy
    result = run_experiment(
        "decision_tree",
        name="random",
        X=features,
        y=target,
        folds=folds,
        search={"kind": "random", "n_iter": 2, "n_jobs": 1},
        save=False,
    )

    assert result.params["search"]["kind"] == "random"
    assert result.params["search"]["n_iter"] == 2


def test_search_rejects_unknown_options(synthetic_xy, folds):
    features, target = synthetic_xy
    with pytest.raises(ValueError, match="unknown search options"):
        run_experiment(
            "decision_tree",
            name="bad_search",
            X=features,
            y=target,
            folds=folds,
            search={"kind": "grid", "bogus_option": 1},
            save=False,
        )


def test_estimator_escape_hatch(synthetic_xy, folds):
    features, target = synthetic_xy
    result = run_experiment(
        name="wrapped",
        X=features,
        y=target,
        folds=folds,
        estimator=DecisionTreeClassifier(max_depth=3, random_state=0),
        save=False,
    )

    assert result.params["model"] == "DecisionTreeClassifier"
    assert len(result.fold_scores) == len(folds)


def test_model_name_required_without_estimator(synthetic_xy, folds):
    features, target = synthetic_xy
    with pytest.raises(ValueError, match="model_name is required"):
        run_experiment(name="no_model", X=features, y=target, folds=folds, save=False)


def test_search_string_needs_model_name_with_estimator(synthetic_xy, folds):
    features, target = synthetic_xy
    with pytest.raises(ValueError, match="explicit space"):
        run_experiment(
            name="bad_search",
            X=features,
            y=target,
            folds=folds,
            estimator=DecisionTreeClassifier(max_depth=3, random_state=0),
            search="grid",
            save=False,
        )


def test_repeats_limits_folds_and_is_recorded(synthetic_xy):
    features, target = synthetic_xy
    folds = make_folds(target, SplitConfig(n_splits=3, n_repeats=2, seed=0))

    partial = run_experiment(
        "dummy", name="partial", X=features, y=target, folds=folds, repeats=1, save=False
    )
    full = run_experiment(
        "dummy", name="full", X=features, y=target, folds=folds, save=False
    )

    assert set(partial.fold_scores["repeat"]) == {0}
    assert len(partial.fold_scores) == 3
    assert len(full.fold_scores) == 6
    assert partial.params["repeats"] == 1
    assert "repeats" not in full.params


def test_run_cv_rejects_bad_repeats(synthetic_xy, spec, folds):
    features, target = synthetic_xy
    pipeline = build_pipeline(build_preprocessor(spec), "decision_tree", max_depth=3)
    with pytest.raises(ValueError, match="repeats"):
        run_cv(pipeline, features, target, folds, repeats=0)


def test_load_run_roundtrip(tmp_path, synthetic_xy, folds):
    features, target = synthetic_xy
    result = run_experiment(
        "decision_tree",
        name="reload",
        X=features,
        y=target,
        folds=folds,
        search={"kind": "grid", "space": {"clf__max_depth": [1, 3]}, "n_jobs": 1},
        fit_full=True,
        output_dir=tmp_path / "runs",
    )
    run_dir = tmp_path / "runs" / result.run_id
    reloaded = load_run(run_dir)

    assert reloaded.run_id == result.run_id
    assert reloaded.name == result.name
    assert reloaded.timestamp == result.timestamp
    assert reloaded.metrics == result.metrics
    assert reloaded.params == result.params
    assert reloaded.labels == result.labels
    assert reloaded.n_features == result.n_features
    pd.testing.assert_frame_equal(reloaded.fold_scores, result.fold_scores, check_dtype=False)
    assert len(reloaded.oof) == len(result.oof)
    assert list(reloaded.per_class.columns) == list(result.per_class.columns)
    assert reloaded.confusion.shape == result.confusion.shape

    # fit_full=True stored the refit search object; it must still predict.
    assert reloaded.estimator is not None
    row_ids = pd.Series(range(len(features)), name="RowId")
    submission_path = tmp_path / "submission.csv"
    write_submission_from_result(reloaded, features, row_ids, submission_path)
    submission = validate_submission(submission_path, row_ids)
    assert len(submission) == len(features)


def test_load_run_without_estimator(tmp_path, synthetic_xy, folds):
    features, target = synthetic_xy
    result = run_experiment(
        "dummy",
        name="no_full",
        X=features,
        y=target,
        folds=folds,
        output_dir=tmp_path / "runs",
    )

    reloaded = load_run(tmp_path / "runs" / result.run_id)

    assert reloaded.estimator is None
    assert reloaded.params == result.params

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from daa.data import TARGET, ColumnSpec, explicit_spec, infer_column_spec
from daa.experiment import run_experiment
from daa.features.selection import (
    FEATURE_SET_NAMES,
    columns_fingerprint,
    load_selected_features,
    mutual_info_screen,
    ordinal_codes,
    rank_by_permutation,
    resolve_feature_set,
    save_selection,
    subset_spec,
)
from daa.models.registry import make_model


def make_signal_frame(n_samples: int = 240, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """One column that nearly determines the target, and one pure-noise column.

    The fold fixture is built for 240 rows, so the default matches it.
    """
    rng = np.random.default_rng(seed)
    target = pd.Series(rng.integers(0, 3, size=n_samples), name=TARGET)
    signal = np.where(rng.random(n_samples) < 0.9, target, rng.integers(0, 3, size=n_samples))
    frame = pd.DataFrame({"signal": signal, "noise": rng.integers(0, 3, size=n_samples)})
    return frame, target


# --------------------------------------------------------------------------
# subset_spec
# --------------------------------------------------------------------------


def test_subset_spec_keeps_roles_and_spec_order(synthetic_xy, spec):
    subset = subset_spec(spec, ["cat_sex", "num_height", "cat_region"])

    assert subset.numeric == ("num_height",)
    assert subset.ordinal == ()
    assert subset.nominal == ("cat_region", "cat_sex")
    assert subset.drop == spec.drop
    assert set(subset.input_columns) == {"cat_sex", "num_height", "cat_region"}


def test_subset_spec_ignores_input_order(spec):
    columns = ["cat_sex", "num_height", "cat_region"]
    assert subset_spec(spec, columns) == subset_spec(spec, list(reversed(columns)))


def test_subset_spec_rejects_unknown_column(spec):
    with pytest.raises(ValueError, match="not in the spec"):
        subset_spec(spec, ["num_height", "not_a_column"])


def test_subset_spec_rejects_dropped_column(spec):
    narrower = ColumnSpec(
        numeric=tuple(column for column in spec.numeric if column != "num_noise"),
        ordinal=spec.ordinal,
        nominal=spec.nominal,
        drop=spec.drop + ("num_noise",),
    )
    with pytest.raises(ValueError, match="dropped by the spec"):
        subset_spec(narrower, ["num_noise"])


def test_subset_spec_rejects_target_and_ids(spec):
    with pytest.raises(ValueError, match="target"):
        subset_spec(spec, [TARGET])
    with pytest.raises(ValueError, match="ID column"):
        subset_spec(spec, ["PID"])


def test_subset_spec_rejects_duplicates(spec):
    with pytest.raises(ValueError, match="duplicates"):
        subset_spec(spec, ["num_height", "num_height"])


def test_subset_spec_rejects_empty(spec):
    with pytest.raises(ValueError, match="empty"):
        subset_spec(spec, [])


# --------------------------------------------------------------------------
# ordinal_codes
# --------------------------------------------------------------------------


def test_ordinal_codes_are_integers_with_minus_one_for_nan(synthetic_xy, spec):
    features, _ = synthetic_xy
    frame = features[list(spec.input_columns)]

    codes = ordinal_codes(frame)

    assert list(codes.columns) == list(frame.columns)
    assert codes.notna().all().all()
    assert all(dtype == np.int64 for dtype in codes.dtypes)
    missing = frame["cat_region"].isna()
    assert (codes.loc[missing, "cat_region"] == -1).all()
    assert (codes.loc[~missing, "cat_region"] >= 0).all()


def test_ordinal_codes_rejects_non_dataframe():
    with pytest.raises(TypeError, match="DataFrame"):
        ordinal_codes(np.zeros((4, 2)))


# --------------------------------------------------------------------------
# mutual_info_screen
# --------------------------------------------------------------------------


def test_mutual_info_screen_shape_and_order(synthetic_xy, spec):
    features, target = synthetic_xy
    frame = features[list(spec.input_columns)]

    table = mutual_info_screen(frame, target, n_shuffles=2, seed=0)

    assert len(table) == len(frame.columns)
    assert list(table["rank"]) == list(range(1, len(frame.columns) + 1))
    assert table["mi_adjusted"].is_monotonic_decreasing
    assert table.equals(mutual_info_screen(frame, target, n_shuffles=2, seed=0))
    assert len(mutual_info_screen(frame, target, top_n=3, n_shuffles=2, seed=0)) == 3


def test_mutual_info_screen_ranks_signal_above_noise():
    frame, target = make_signal_frame()

    table = mutual_info_screen(frame, target, n_shuffles=3, seed=0).set_index("column")

    assert table.loc["signal", "mi_adjusted"] > table.loc["noise", "mi_adjusted"]
    assert table.loc["signal", "rank"] == 1


def test_mutual_info_screen_rejects_bad_arguments(synthetic_xy):
    features, target = synthetic_xy
    with pytest.raises(ValueError, match="different lengths"):
        mutual_info_screen(features, target.iloc[:-1])
    with pytest.raises(ValueError, match="n_shuffles"):
        mutual_info_screen(features, target, n_shuffles=0)


# --------------------------------------------------------------------------
# rank_by_permutation
# --------------------------------------------------------------------------


def test_rank_by_permutation_ranks_signal_above_noise(folds):
    frame, target = make_signal_frame()

    table = rank_by_permutation(
        make_model("decision_tree", max_depth=3),
        frame,
        target,
        folds,
        n_repeats=2,
        seed=0,
    )

    assert list(table["column"]) == ["signal", "noise"]
    assert list(table["rank"]) == [1, 2]
    assert (table["importance_std"] >= 0).all()
    assert table.loc[0, "importance_mean"] > table.loc[1, "importance_mean"]


def test_rank_by_permutation_rejects_unknown_score(folds):
    frame, target = make_signal_frame()
    with pytest.raises(ValueError, match="unknown score"):
        rank_by_permutation(make_model("decision_tree"), frame, target, folds, scoring="f1")


def test_rank_by_permutation_rejects_empty_folds():
    frame, target = make_signal_frame()
    with pytest.raises(ValueError, match="folds is empty"):
        rank_by_permutation(make_model("decision_tree"), frame, target, [])


# --------------------------------------------------------------------------
# save_selection / load_selected_features
# --------------------------------------------------------------------------


def test_save_and_load_roundtrip(tmp_path, synthetic_xy, spec):
    _, target = synthetic_xy
    columns = ["num_height", "cat_sex"]

    path = save_selection(
        tmp_path / "selected.json",
        columns,
        spec=spec,
        target=target,
        metadata={"method": "unit test"},
    )
    selection = load_selected_features(path, spec, target=target)

    assert selection.columns == tuple(columns)
    assert selection.k == 2
    assert selection.fingerprint == columns_fingerprint(columns)
    assert selection.metadata["method"] == "unit test"

    payload = json.loads(path.read_text())
    assert payload["seed"] == 42
    assert payload["git_sha"]
    assert payload["spec_counts"]["numeric"] == len(spec.numeric)
    assert payload["target_fingerprint"]


def test_save_selection_rejects_unknown_column(tmp_path, synthetic_xy, spec):
    _, target = synthetic_xy
    with pytest.raises(ValueError, match="not in the spec"):
        save_selection(tmp_path / "selected.json", ["not_a_column"], spec=spec, target=target)
    assert not (tmp_path / "selected.json").exists()


def test_load_rejects_tampered_columns(tmp_path, synthetic_xy, spec):
    _, target = synthetic_xy
    path = save_selection(tmp_path / "selected.json", ["num_height"], spec=spec, target=target)

    payload = json.loads(path.read_text())
    payload["columns"] = ["num_height", "num_weight"]
    payload["k"] = 2
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="fingerprint"):
        load_selected_features(path, spec)


def test_load_rejects_column_missing_from_spec(tmp_path, synthetic_xy):
    features, target = synthetic_xy
    spec = explicit_spec(infer_column_spec(features))
    path = save_selection(tmp_path / "selected.json", ["num_height"], spec=spec, target=target)

    narrower = ColumnSpec(
        numeric=tuple(column for column in spec.numeric if column != "num_height"),
        ordinal=spec.ordinal,
        nominal=spec.nominal,
        drop=spec.drop + ("num_height",),
    )
    with pytest.raises(ValueError, match="dropped by the spec"):
        load_selected_features(path, narrower)


def test_load_rejects_different_target(tmp_path, synthetic_xy, spec):
    _, target = synthetic_xy
    path = save_selection(tmp_path / "selected.json", ["num_height"], spec=spec, target=target)

    with pytest.raises(ValueError, match="different target"):
        load_selected_features(path, spec, target=target + 1)


def test_load_missing_file_raises(tmp_path, spec):
    with pytest.raises(FileNotFoundError):
        load_selected_features(tmp_path / "absent.json", spec)


# --------------------------------------------------------------------------
# The harness records which columns a run used
# --------------------------------------------------------------------------


def test_run_experiment_records_the_feature_set(synthetic_xy, spec, folds):
    features, target = synthetic_xy
    columns = ["num_height", "cat_sex"]

    result = run_experiment(
        "dummy",
        name="subset",
        X=features,
        y=target,
        folds=folds,
        columns=columns,
        feature_set="unit_test",
        save=False,
    )
    full = run_experiment(
        "dummy",
        name="full",
        X=features,
        y=target,
        folds=folds,
        save=False,
    )

    assert result.params["feature_set"] == "unit_test"
    assert result.params["n_input_columns"] == 2
    assert result.params["feature_hash"] == columns_fingerprint(("num_height", "cat_sex"))[:12]
    assert result.n_features < full.n_features
    assert full.params["feature_set"] == "all"
    assert full.params["n_input_columns"] == len(spec.input_columns)


def test_run_experiment_rejects_unknown_column(synthetic_xy, folds):
    features, target = synthetic_xy
    with pytest.raises(ValueError, match="not in the spec"):
        run_experiment(
            "dummy",
            name="bad",
            X=features,
            y=target,
            folds=folds,
            columns=["not_a_column"],
            save=False,
        )


# --------------------------------------------------------------------------
# resolve_feature_set
# --------------------------------------------------------------------------


def test_resolve_feature_set_all_returns_none(spec):
    assert resolve_feature_set("all", spec) is None
    assert set(FEATURE_SET_NAMES) == {"all", "selected"}


def test_resolve_feature_set_selected_reads_shortlist(tmp_path, synthetic_xy):
    features, target = synthetic_xy
    spec = explicit_spec(infer_column_spec(features))
    path = save_selection(
        tmp_path / "selected.json",
        ["num_height", "cat_sex"],
        spec=spec,
        target=target,
    )

    columns = resolve_feature_set("selected", spec, target=target, path=path)

    assert columns == ("num_height", "cat_sex")


def test_resolve_feature_set_uses_default_path(tmp_path, synthetic_xy, monkeypatch):
    features, target = synthetic_xy
    spec = explicit_spec(infer_column_spec(features))
    path = save_selection(tmp_path / "selected.json", ["num_height"], spec=spec, target=target)
    monkeypatch.setattr("daa.features.selection.SELECTED_FEATURES_PATH", path)

    assert resolve_feature_set("selected", spec, target=target) == ("num_height",)


def test_resolve_feature_set_unknown_name_lists_valid_names(spec):
    with pytest.raises(ValueError, match="unknown feature set 'bogus'.*'all', 'selected'"):
        resolve_feature_set("bogus", spec)


def test_resolve_feature_set_rejects_stale_target(tmp_path, synthetic_xy):
    features, target = synthetic_xy
    spec = explicit_spec(infer_column_spec(features))
    path = save_selection(tmp_path / "selected.json", ["num_height"], spec=spec, target=target)

    with pytest.raises(ValueError, match="different target"):
        resolve_feature_set("selected", spec, target=target + 1, path=path)

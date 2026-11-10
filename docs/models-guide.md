# Models guide — for the models track

You own `src/daa/models/`, `src/daa/evaluation/`, `src/daa/experiment.py`,
`tests/`, and `docs/`. Your job: register models, tune them, and get comparable
numbers on the leaderboard — **without ever naming columns or preprocessing
variants**. Those come from `feature_set=` names and
`features.preprocess.DEFAULT_VARIANT`; see the README section **"Global
variables and their effect on runs"** for the full list of globals and owners.

## Registering a model

1. Add a factory to `MODELS` in `src/daa/models/registry.py`. It must return a
   **fresh, unfitted** classifier (use `functools.partial` with
   `random_state=SEED` where the estimator takes one). `make_model` clones from
   the registry, never from a shared instance.
2. Add its default search space to `SEARCH_SPACES` in `src/daa/models/spaces.py`.
   Keys use the `clf__` prefix because the shared pipeline names its final step
   `clf`. Keep the grid small enough for a laptop.
3. Extend `tests/test_models.py`: `test_every_model_fits_a_pipeline` and
   `test_search_space_keys_are_valid` pick up new entries automatically; add a
   fast entry to `FAST_PARAMS` if fitting needs one.

## Running an experiment — one call

```python
from daa.experiment import run_experiment

result = run_experiment(
    "rf",                              # any name from daa.models.registry.MODELS
    {"n_estimators": 400},
    name="rf_selected",
    feature_set="selected",            # named shortlist; "all" is the default
    search="grid",                     # or "random", or {"kind", "space", "n_iter", "scoring", "n_jobs"}
    repeats=1,                         # fast iteration only — combine with save=False
    fit_full=True,                     # needed for submissions
)
result.metrics
```

What each knob does:

- `feature_set="selected"` resolves `outputs/selected_features.json` via
  `daa.features.selection.resolve_feature_set` (validated against the column
  and target fingerprints on every load). `columns=[...]` still exists as an
  explicit override for ablations and tests, and wins over the name.
- `preprocessor=None` (default) uses the encoding owner's
  `features.preprocess.DEFAULT_VARIANT`.
- `search=` wraps the pipeline with `daa.models.spaces.build_search`
  (`GridSearchCV` / `RandomizedSearchCV`, `refit=True`, scored with
  `evaluation.metrics.SCORERS`). Because `run_cv` clones the estimator per
  fold, the search runs as honest nested CV inside every fold. The string
  forms use `SEARCH_SPACES[model_name]`; pass a dict with an explicit `space`
  (and `n_iter` for random search) to override. Use `estimator=` instead of
  `model_name` if you need a pre-built/wrapped estimator as the `clf` step.
- `repeats=1` keeps only the first repeat of the frozen folds. For fast
  iteration only — combine with `save=False`, because a saved partial run is
  recorded with `repeats` in `params` and is not comparable to full 3×5 runs.

## What a run records

Every saved run lands in `outputs/runs/<run_id>/` and appends a row to
`outputs/leaderboard.csv`. `params` always carries `preprocessor`, `model`,
`model_params`, `feature_set`, `n_input_columns`, and a short `feature_hash`
(plus `search` and `repeats` when used) — so two rows are comparable only when
those match. The artifacts are `metrics.json`, `params.json`,
`fold_scores.csv`, `oof_predictions.csv`, `per_class.csv`,
`confusion_matrix.csv`, and `estimator.joblib` when `fit_full=True`.

## Submissions and reloading past runs

```python
from daa.data import load_test
from daa.evaluation import load_run, validate_submission, write_submission_from_result

X_test, row_ids = load_test()
write_submission_from_result(result, X_test, row_ids, "outputs/my_submission.csv")
validate_submission("outputs/my_submission.csv", row_ids)

past = load_run("outputs/runs/<run_id>")   # metrics, params, OOF, estimator
```

With `search=` + `fit_full=True` the stored estimator is the refit search
object; its `.predict` works, so `write_submission_from_result` is unaffected.

## Notebook rules

Notebooks stay thin drivers: one notebook per experiment, **Restart & Run
All**, strip outputs before committing (enforced by the `nbstripout`
pre-commit hook and CI). A notebook decides *what* to run (model, params,
`feature_set`, `search`); definitions live in `src/` with tests. Every result
worth keeping goes through `run_experiment` so it lands on the leaderboard.

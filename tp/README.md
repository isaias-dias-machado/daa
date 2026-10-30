# DAA group project — shared ML scaffolding

This repository holds the shared machinery for the DAA practical work. The task
is 16-class classification of `AGE_RANGE` from Portuguese survey data, scored
with **macro F1**.

The point of this scaffold is not the data or the first model. It is that three
people can work independently and still produce **comparable** numbers.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -e . --no-deps

python -m pytest
```

Open `notebooks/00_primer.ipynb` first (Pipeline, ColumnTransformer, leakage),
then `notebooks/01_dummy.ipynb` (end-to-end dummy baseline).

## The four contracts

| Layer | Function | Guarantee |
|---|---|---|
| Data | `daa.data.load_train`, `load_test`, `ColumnSpec` | Only this layer reads CSVs; IDs stay out of the model |
| Features | `daa.features.build_preprocessor(spec, variant)` | Raw DataFrame in, named DataFrame out; fitted inside CV |
| Models | `daa.models.make_model(name, **params)` | Fresh, unfitted, cloneable classifier |
| Evaluation | `daa.evaluation.run_cv`, `write_run` | Same folds, same metrics, same artifact schema |

The integration point is `daa.experiment.run_experiment`:

```python
from daa.experiment import run_experiment

result = run_experiment(
    "decision_tree",
    {"max_depth": 12, "min_samples_leaf": 20},
    name="decision_tree_baseline",
    fit_full=True,
)
result.metrics
```

A notebook experiment is usually just a `Pipeline` plus `run_cv`:

```python
from sklearn.pipeline import Pipeline
from daa.data import infer_column_spec, load_train
from daa.features.preprocess import build_preprocessor
from daa.evaluation.cv import run_cv
from daa.models.registry import make_model
from daa.splits import get_folds

X, y = load_train()
pipeline = Pipeline([
    ("prep", build_preprocessor(infer_column_spec(X), variant="impute")),
    ("clf", make_model("rf", n_estimators=400)),
])
result = run_cv(pipeline, X, y, get_folds(y), name="rf_v1")
```

## Layout and ownership

| Path | Owner | Responsibility |
|---|---|---|
| `src/daa/data.py` | A | CSV loaders and `ColumnSpec` (frozen) |
| `src/daa/features/` | A | Preprocessing variants and feature blocks |
| `src/daa/models/` | B | Model registry and search spaces |
| `src/daa/evaluation/` | C | Metrics, CV loop, reports, submission |
| `src/daa/splits.py` | C | Frozen fold indices |
| `src/daa/experiment.py` | C | Compose + run + persist |
| `tests/` | C | Contract, leakage, metric, and submission tests |
| `notebooks/` | all | Thin drivers; no reusable definitions |
| `splits/folds.npz` | frozen | 5-fold × 3 repeats, seed 42 |
| `outputs/leaderboard.csv` | shared | Append-only run log |

Ownership is decided after the joint Phase 0 sessions. Until then, the table is
the default and not a lock.

### Frozen core

Change these only by group agreement, because every historical result depends
on them:

1. `src/daa/splits.py` and `splits/folds.npz` — the fold assignment.
2. `src/daa/evaluation/metrics.py` — macro F1 primary, QWK and MAE secondary.
3. `src/daa/data.py::ColumnSpec` — column roles.
4. The run artifact schema and `outputs/leaderboard.csv` columns.

`get_folds` checks a fingerprint of the target vector. If the target changes,
it fails loudly instead of silently mixing experiments. Regenerate deliberately:

```bash
python -m daa.splits --force
```

## Run artifacts

Each saved run creates `outputs/runs/<run_id>/`:

```
metrics.json          aggregate + fold metrics, git sha
params.json           preprocessor, model, model params
fold_scores.csv       one row per fold
oof_predictions.csv   row_index, y_true, y_pred, proba_* per repeat
per_class.csv         precision / recall / F1 / support
confusion_matrix.csv
estimator.joblib      only when fit_full=True
```

`outputs/leaderboard.csv` gets one row per run. `outputs/runs/` and `*.joblib`
are ignored by git.

## Notebook rules

1. One notebook per person/experiment; never co-edit.
2. **Restart & Run All** before committing; strip outputs.
3. Notebooks contain *decisions* (which variant, which model, which params).
   *Definitions* live in `src/` with a test.
4. Every result worth keeping goes through `run_cv` or `run_experiment` so it
   lands on the leaderboard.

## Environment overrides

```bash
DAA_DATA_DIR=/path/to/data
DAA_OUTPUT_DIR=/tmp/daa_runs
DAA_SPLITS_PATH=/tmp/folds.npz
```

Useful for tests and for notebook smoke runs that should not pollute
`outputs/`.

## Data notes

- Training: 10,962 rows × 237 columns; test: 3,655 rows × 236 columns.
- Target: `AGE_RANGE`, 16 classes, 172–1,062 rows per class.
- `PID` and `HHID` are unique and do not overlap between train and test, so a
  plain `StratifiedKFold` is safe.
- `WGT` is survey metadata and is dropped when present.
- Accuracy is not the competition metric. Use macro F1; the harness reports it
  by default.

## Checks

```bash
python -m pytest          # 39 tests, synthetic data only
ruff check src tests
```

# DAA group project — shared ML scaffolding

This repository holds the shared machinery for the DAA practical work. The task
is 16-class classification of `AGE_RANGE` from Portuguese survey data, scored
with **macro F1**.

The point of this scaffold is not the data or the first model. It is that three
people can work independently and still produce **comparable** numbers.

## Quick start

From the **repo root** (one command sets up venv, dependencies and the
notebook-clean commit hook):

```bash
./setup.sh
source .venv/bin/activate
```

Manual equivalent:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -e . --no-deps
./.venv/bin/pre-commit install

python -m pytest
```

Open `notebooks/00_primer.ipynb` first (Pipeline, ColumnTransformer, leakage),
then `01_dummy.ipynb` (end-to-end dummy baseline), `02_raw_eda.ipynb` (the raw
files), `03_recoding.ipynb` (what the recoding does), and
`04_feature_selection.ipynb` (from 255 inputs to a shortlist).

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
    feature_set="selected",  # named shortlist, resolved from outputs/selected_features.json
    search="grid",           # or "random", or {"kind", "space", "n_iter", "scoring", "n_jobs"}
    fit_full=True,
)
result.metrics
```

Features are named, not listed: `feature_set="selected"` resolves the shortlist
through `daa.features.selection.resolve_feature_set`, which validates the
persisted file (column and target fingerprints) on every load. `feature_set`
defaults to `"all"`; `columns=[...]` remains as an explicit override for tests
and ablations and wins over name resolution. `preprocessor=None` (the default)
uses `daa.features.preprocess.DEFAULT_VARIANT`, pinned by the encoding owner.
`search=` wraps the pipeline with `daa.models.spaces.build_search` before the
CV loop — since `run_cv` clones per fold, this is honest nested CV, and the
refit search object still `predict`s for submissions. `repeats=1` keeps only
the first repeat for fast iteration (combine with `save=False`). Pass
`estimator=` instead of `model_name` to use a pre-built estimator as the `clf`
step. Every run records `feature_set`, `n_input_columns` and a short
`feature_hash` in `params` (plus `search` and `repeats` when used), so the
leaderboard says exactly which columns and configuration a row used.
`daa.features.selection` builds the ranking, writes the shortlist file, and
validates it on load; `daa.evaluation.load_run` reads a saved run back for
error analysis.

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
| `src/daa/config.py` | A | Paths, environment overrides, `SEED` |
| `src/daa/data.py` | A | CSV loaders and `ColumnSpec` (frozen) |
| `src/daa/splits.py` | A | Frozen fold indices |
| `src/daa/features/recode.py` | A | The fixed recoding rule-book |
| `src/daa/features/selection.py` | A | Ranking, the shortlist artifact, named feature sets |
| `src/daa/features/preprocess.py` | B | Preprocessor variants and `DEFAULT_VARIANT` |
| EDA / feature-engineering notebooks | B | Decisions about features and encoding |
| `src/daa/models/` | C | Model registry and search spaces |
| `src/daa/evaluation/` | C | Metrics, CV loop, reports, submission |
| `src/daa/experiment.py` | C | Compose + run + persist (resolves names, defines none) |
| `tests/` | C | Contract, leakage, metric, and submission tests |
| `docs/` | C | Role guides (encoding, models) |
| `notebooks/` | all | Thin drivers; no reusable definitions |
| `outputs/selected_features.json` | A | The persisted shortlist behind `feature_set="selected"` |
| `splits/folds.npz` | frozen | 5-fold × 3 repeats, seed 42 |
| `outputs/leaderboard.csv` | shared | Append-only run log |

This is the division of labour agreed by the group. `experiment.py` only
*composes*: it resolves names, it never defines them — columns come from the
feature-selection owner's artifact, the encoding default from the encoding
owner's module, and models/search spaces from `models/`.

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

## Global variables and their effect on runs

Every module-level constant below changes what a run does or records. "Frozen"
means change only by group agreement, because historical leaderboard rows depend
on it.

| Global | Module | Owner | Effect on runs | Status |
|---|---|---|---|---|
| `SEED` | `config.py` | A | Single seed for models, selection and folds; changing it makes all runs non-comparable | frozen |
| `DATA_DIR`, `TRAIN_PATH`, `TEST_PATH` | `config.py` | A | Which CSVs are read (`DAA_DATA_DIR` override) | |
| `OUTPUTS_DIR`, `RUNS_DIR` | `config.py` | A | Where run dirs and `leaderboard.csv` are written (`DAA_OUTPUT_DIR`) | |
| `SPLITS_PATH` | `config.py` | A | Which folds file is loaded (`DAA_SPLITS_PATH`) | |
| `SELECTED_FEATURES_PATH` | `config.py` | A | Which shortlist `feature_set="selected"` resolves to (`DAA_SELECTED_FEATURES_PATH`) | |
| `DEFAULT_SPLIT_CONFIG` (`n_splits=5`, `n_repeats=3`, `seed`) | `splits.py` | A | The CV scheme; persisted in `splits/folds.npz` with a target fingerprint | frozen |
| `VARIANTS`, `DEFAULT_VARIANT` | `features/preprocess.py` | B | Which preprocessing runs when `preprocessor=None`; recorded in `params["preprocessor"]` | |
| `FEATURE_SET_NAMES` | `features/selection.py` | A | Which `feature_set=` names are accepted; the resolved columns are recorded via `feature_hash` in `params` | |
| `MODELS`, `MODEL_NAMES` | `models/registry.py` | C | Available model names for `run_experiment`; recorded in `params["model"]` | |
| `SEARCH_SPACES` | `models/spaces.py` | C | Default grids for `search=` (`clf__` prefix); recorded in `params["search"]` | |
| `METRICS` | `evaluation/metrics.py` | C | Per-fold metrics aggregated into every run | frozen |
| `SCORERS` | `evaluation/metrics.py` | C | Valid `scoring=` names for search | |
| `SCORE_COLUMNS` | `evaluation/cv.py` | C | Which metrics get mean/std in `RunResult` | |
| `SUBMISSION_COLUMNS` | `evaluation/submit.py` | C | Submission CSV contract | |

`ColumnSpec` (`data.py`) and the run artifact schema / leaderboard columns are
frozen as well — see "Frozen core" above.

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

### Notebook hygiene is enforced twice

No tracked `.ipynb` may contain outputs or execution counts.

- **Locally** the `pre-commit` hook (installed by `./setup.sh`) runs
  `nbstripout` on committed notebooks and rewrites them on disk. To check
  everything by hand: `pre-commit run --all-files`.
- **Remotely** the `notebook-clean` GitHub Actions workflow reruns the same
  check on every PR and push to `master`, so bypassing the hook with
  `--no-verify` still fails CI. **After merge**, require the
  `notebook-clean / notebook-clean` status check on `master`
  (Settings → Branches → branch protection) to make it mandatory; until then
  CI results are advisory. Pushing a dirty branch is allowed — merging it
  into `master` is not.

## Environment overrides

```bash
DAA_DATA_DIR=/path/to/data
DAA_OUTPUT_DIR=/tmp/daa_runs
DAA_SPLITS_PATH=/tmp/folds.npz
DAA_SELECTED_FEATURES_PATH=/tmp/selected_features.json
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

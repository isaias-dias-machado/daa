# Encoding guide — for the EDA / feature-engineering / encoding track

You own `src/daa/features/preprocess.py` and the EDA / feature-engineering
notebooks. Everyone else (models included) reaches your work only through a
variant name — they never edit this file and never name your columns.

See the README section **"Global variables and their effect on runs"** for the
full list of globals and their owners; `VARIANTS` and `DEFAULT_VARIANT` are
yours.

## The contract

`build_preprocessor(spec, variant)` returns an **unfitted** `ColumnTransformer`:

- **Input**: a raw `DataFrame` containing the columns named in the `ColumnSpec`
  (numeric / ordinal / nominal roles come from `daa.data`; the shortlist from
  `daa.features.selection`).
- **Output**: a named `DataFrame` (`set_output(transform="pandas")`,
  `verbose_feature_names_out=False`) whose columns are `get_feature_names_out()`.
- **Fitted only inside the CV pipeline** — never on the full data. Fitting a
  preprocessor on the full dataset before cross-validation leaks validation
  information; notebook 00 demonstrates exactly that bug. `run_experiment` and
  `run_cv` clone and refit the whole pipeline per fold, so as long as the
  preprocessor stays a pipeline step, you are safe.

## Adding a variant

1. Add the name to `VARIANTS` in `src/daa/features/preprocess.py`.
2. Implement it in `build_preprocessor` next to the existing branches
   (`"impute"`, `"passthrough"`). Keep the contract above: no NaN in the output
   unless the variant explicitly supports NaN (like `passthrough` for
   `HistGradientBoostingClassifier`), and `handle_unknown` set on encoders.
3. Unknown variant names must raise `ValueError` listing `VARIANTS` (already the
   behaviour — keep it).

## Changing the default

Set `DEFAULT_VARIANT` at the top of `src/daa/features/preprocess.py`. That is
the variant used whenever a caller passes `preprocessor=None` — which is the
default in `run_experiment`. The resolved variant is recorded in
`params["preprocessor"]` of every run, so leaderboard rows always say which
encoding they used. Changing the default changes what every default run does;
that is why it lives here and not in the caller's code.

## Tests

Extend `tests/test_preprocess.py` (synthetic fixtures only, see
`tests/conftest.py`):

- one contract test per new variant: named `DataFrame` out, expected NaN
  behaviour, IDs/target never present;
- one unseen-category test if your variant encodes categoricals;
- `test_unknown_variant_raises` already covers the `ValueError`.

Run `python -m pytest` and `ruff check src tests` before committing.

## Notebook rules

Notebooks stay thin drivers: they *decide* which variant to run
(`preprocessor="my_variant"` or the default), definitions live in `src/` with a
test. Never fit a preprocessor outside a pipeline in a notebook.

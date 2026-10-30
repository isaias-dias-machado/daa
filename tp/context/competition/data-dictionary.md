<!--
Measured facts about the local data files. Method: pandas.read_csv(...).shape,
value_counts(), and ID-set intersections, run on 2026-10-30. Only facts that were
measured are recorded as measured; interpretations carry a status.
Authority: this file and data/ own measured dataset facts. Requirements are owned by
../assignment/enunciado.md; competition rules by kaggle.md.
-->

# Data dictionary — measured facts

**Measured:** 2026-10-30
**Method:** `pandas.read_csv` for shapes and class counts; Python set intersections for
ID overlap; direct inspection of the example submission file.
**Files:** `data/dataset_train.csv`, `data/dataset_test_X.csv`,
`data/example_submission_10.csv`.

## Shapes and columns

| File | Rows | Columns | Notes |
|---|---:|---:|---|
| `dataset_train.csv` | 10,962 | 237 | 236 features + target `AGE_RANGE` |
| `dataset_test_X.csv` | 3,655 | 236 | features only; no target |
| `example_submission_10.csv` | 3,655 | 2 | format template; not data |

- Train columns not present in test: `AGE_RANGE` only.
- Test columns are a subset of train columns; test has no target.
- `WGT` is **absent** from both current files. It was present in the older
  `data_v1/` (238 columns), whose outputs are stored in `reports/01`.
- The figure "475 columns" reported by the data explorer is **237 + 236 + 2**: the
  sum of the three files' column counts, not a union. The union of train and test
  columns is 238 (237 train columns + `RowId`).

## Identifiers

| Check | Result |
|---|---|
| `PID` unique within train | yes |
| `PID` unique within test | yes |
| `HHID` unique within train | yes |
| `HHID` unique within test | yes |
| Train/test `PID` overlap | 0 |
| Train/test `HHID` overlap | 0 |

`PID` and `HHID` are identifiers, not model features.

## Target distribution

`AGE_RANGE` takes 16 values, 0…15.

| Band | Count | Share | Band | Count | Share |
|---:|---:|---:|---:|---:|---:|
| 0 | 207 | 1.89% | 8 | 905 | 8.26% |
| 1 | 172 | 1.57% | 9 | 1,014 | 9.25% |
| 2 | 399 | 3.64% | 10 | 1,062 | 9.69% |
| 3 | 377 | 3.44% | 11 | 1,040 | 9.49% |
| 4 | 449 | 4.10% | 12 | 1,000 | 9.12% |
| 5 | 647 | 5.90% | 13 | 823 | 7.51% |
| 6 | 859 | 7.84% | 14 | 650 | 5.93% |
| 7 | 862 | 7.86% | 15 | 496 | 4.52% |

Sum: 10,962. Majority-class baseline: 1,062 / 10,962 = **9.69%**. Minimum class:
172 (band 1). Maximum class: 1,062 (band 10).

## Example submission

- Header: `RowId,AGE_RANGE`.
- 3,655 data rows; CRLF line endings.
- `RowId` runs 0…3654.
- Every `AGE_RANGE` value is `10`. This is a **placeholder**, not a prediction.

## `AGE_RANGE` coding — status **open**

No codebook is present in `data/`. Two hypotheses are compatible with a 16-class
partition:

1. **Official Kaggle table:** `0 = 15–17, 1 = 18–19, 2 = 20–24, …, 15 = 85+`.
2. **`reports/01` hypothesis:** `0 = 0–14, 1 = 15–19, 2 = 20–24, …, 15 = 85+`.

Band 0 evidence (207 rows): 204 are "Aluno, estudante…"; education is 163 "Básico 3º
ciclo", 39 "Básico 1º e 2º ciclo", 2 "Nenhum nível completado", 2 "Secundário",
1 "Pós secundário". That profile fits children aged 0–14, but it also fits
15–17-year-olds.

Band 1 counter-evidence: 172 rows, while band 2 (20–24) has 399. A five-year 15–19
band would normally be larger than 172 relative to band 2; this argues for band 1 =
18–19, i.e. the official table.

`0 = 0–14, 1 = 18–19` is not a valid partition because it skips 15–17. Do not state
any age label as fact until the professor answers. See `../../TODO.md`.

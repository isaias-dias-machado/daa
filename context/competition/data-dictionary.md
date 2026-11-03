<!--
Measured facts about the local data files. Method: pandas.read_csv(...).shape,
value_counts(), and ID-set intersections, run on 2026-10-30. Only facts that were
measured are recorded as measured; interpretations carry a status.
Authority: this file and data/ own measured dataset facts. Requirements are owned by
../assignment/enunciado.md; competition rules by kaggle.md.
-->

# Data dictionary — measured facts

**Measured:** 2026-10-30 for shapes, IDs, target, submission; 2026-11-03 for the
string-column, sentinel, `HHMM`, and recoding sections.
**Method:** `pandas.read_csv` for shapes and class counts; Python set intersections for
ID overlap; direct inspection of the example submission file; `value_counts()` and
regex scans of the string columns for the later sections.
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

## String columns that are quantities (measured 2026-11-03)

229 of the 237 train columns are strings. 23 of them are pure quantities once the
two sentinels are removed. `n_values` = non-sentinel rows; sentinel rows =
10,962 − `n_values`.

| column | n_values | n_unique | leading-zero values | example values |
|---|---:|---:|---:|---|
| `PE9` | 10,826 | 164 | 9,520 | `0200`, `0300`, `0400` |
| `IN57` | 4,216 | 142 | 0 | `40.0`, `20.0`, `50.0` |
| `IN55` | 1,928 | 133 | 0 | `248.0`, `50.0`, `7.0` |
| `IN56` | 949 | 118 | 0 | `5.0`, `12.0`, `100.0` |
| `BM2` | 10,664 | 101 | 0 | `69.0`, `54.0`, `72.0` (weight-like) |
| `PE7` | 2,873 | 100 | 2,729 | `0130`, `0100`, `1230` |
| `IN45` | 2,232 | 71 | 0 | `41.0`, `30.0`, `50.0` |
| `IN4` | 746 | 68 | 0 | `37.0`, `42.0`, `44.0` |
| `BM1` | 10,747 | 65 | 0 | `179.0`, `175.0`, `159.0` (height-like) |
| `IN3` | 1,273 | 63 | 0 | `27.0`, `38.0`, `16.0` |
| `IN59` | 844 | 60 | 0 | `40.0`, `150.0`, `30.0` |
| `IN2` | 1,433 | 54 | 0 | `4.0`, `13.0`, `34.0` |
| `IN47` | 3,273 | 50 | 0 | `20.0`, `15.0`, `16.0` |
| `SK2` | 1,514 | 35 | 0 | `10.0`, `20.0`, `8.0` |
| `AM5` | 5,255 | 15 | 0 | `1.0`, `0.0`, `3.0` |
| `DH2` | 7,272 | 15 | 0 | `2.0`, `1.0`, `3.0` |
| `DH4` | 3,956 | 14 | 0 | `2.0`, `1.0`, `4.0` |
| `AM3` | 8,168 | 13 | 0 | `0.0`, `1.0`, `2.0` |
| `PE2` | 10,820 | 8 | 0 | `7.0`, `5.0`, `0.0` |
| `PE4` | 10,753 | 8 | 0 | `0.0`, `4.0`, `3.0` |
| `PE8` | 10,748 | 8 | 0 | `0.0`, `2.0`, `3.0` |
| `PE6` | 10,787 | 8 | 0 | `0.0`, `2.0`, `3.0` |
| `IN29` | 10,906 | 4 | 0 | `2.0`, `3.0`, `1.0` |

Units and meaning are **not** confirmed: the table shows the shape, not the
interpretation.

## Mixed-format string columns (measured 2026-11-03)

`numeric_share` is the share of non-sentinel rows that are numbers.

| column | numeric_share | non-numeric part |
|---|---:|---|
| `HHNBPERS_COD` | 0.9964 | `'1.0'…'6.0'` plus a `'>=7'` bucket (40 rows) |
| `HHNBPERS_0_13_COD` | 0.9920 | `'0.0'…'2.0'` plus a `'>=3'` bucket (88 rows) |
| `HO12_COD` | 0.9915 | numeric codes plus a `'>= 25'` bucket (93 rows) |
| `HO34_COD` | 0.9990 | numeric codes plus a `'>= 60'` bucket (11 rows) |
| `PA1` | 0.2243 | `'Há mais de 1 ano'`, `'Nunca'`, `YYYYMM` values |
| `IN24` | 0.9723 | years 1997–2008 (`'2007.0'`, `'1999.0'`) |
| `IN25` | 0.9850 | numeric values plus `'53 ou mais'` (15 rows) |
| `IN26` | 0.9963 | numeric values plus `'33 ou mais'` (5 rows) |

`HHNBPERS_COD` levels: `'2.0'` 4,021; `'1.0'` 3,006; `'3.0'` 2,073; `'4.0'`
1,403; `'5.0'` 344; `'6.0'` 75; `'>=7'` 40.

## Sentinels (measured 2026-11-03)

- `'Sem resposta'` appears in 210 columns; `'Não aplicável'` in 109; at least one
  of the two in 215.
- Highest sentinel shares (% of rows): `IN13` 99.94, `IN62` 99.00, `IN12` 97.41,
  `AC_IN2` 97.15, `IN11` 96.46, `AC_IN1` 96.23, `AC2` 95.21, `PE5` 94.72,
  `IN49` 93.88, `IN4` 93.19.
- `IN13` levels: `'Não aplicável'` 10,955; `'Só com ajuda'` 3; `'Sozinho, sem
  dificuldade'` 2; `'Sozinho, mas com dificuldade'` 2.
- `IN16` levels: NaN 5,602; `'Vigilância / Rotina'` 1,352; `'Higienização'`
  1,255; `'Tratamento de um dente'` 864; `'Extração de um dente'` 578.
- Other sentinel spellings exist outside the parsed columns: `'NR'` in `IN19`,
  `PA5`, `SK5`; `'Não aplicável (nunca tentou ou não precisa de o fazer)'` in
  `HA1A`–`HA1G`.

## `HHMM` columns `PE7` / `PE9` (measured 2026-11-03)

Every non-sentinel value has exactly 4 digits, in train and test. Values above
23:59 exist, so the reading is a **duration** (hours and minutes), not a clock
time:

| column | values above 23:59 in train | in test |
|---|---|---|
| `PE7` | `'3000'` 10, `'2500'` 2, `'2400'` 1, `'2425'` 1 (13 rows) | 7 rows |
| `PE9` | `'2400'` 7 | none |

Minutes part: `'00'` 11,525; `'30'` 1,358; `'10'` 151; `'45'` 123; `'15'` 93.
Half-hour codes exist (`'0130'` 456, `'0230'` 236, `'0330'` 113); raw-minute
codes do not (`'0060'`, `'0090'`, `'0180'`: 0 rows). Maximum after parsing:
`PE7` 1,800 minutes (30h), `PE9` 1,440 minutes.

## Test-only levels (measured 2026-11-03)

20 columns have levels that appear only in test: `PE9` 15, `IN57` 13, `PE7` 13,
`IN55` 13, `IN56` 10, `IN59` 9, `IN3` 6, `IN47` 4, `IN2` 3, `IN4` 2, `BM1` 2,
`BM2` 2, `IN25` 2, `HHINACT` 1, `IN21` 1, `IN24` 1, `HO34_COD` 1 (`'33.0'`),
`IN62` 1, `IN45` 1, `DH2` 1.

## Train/test comparability (measured 2026-11-03)

Total variation distance is small in every column: maximum 0.0354 (`IN47`),
then `PA8_NAC` 0.027, `REGION` 0.026, `AL1` 0.026, `IN50` 0.026, `DH4` 0.026.
No evidence of a major distribution shift.

## Recoded frame (measured 2026-11-03)

`load_train()` / `load_test()` now apply `daa.features.recode.recode_frame`:

- 29 parsed columns (21 quantity + 2 `HHMM` + 6 bucket), all `float64`, plus 29
  `*_FLAG` columns. Train and test become 265 columns each.
- `infer_column_spec(load_train()[0])`: numeric 33, nominal 226, drop 6.
  `explicit_spec` adds `REFYEAR`, `REFMONTH`, `IN13`, `AC1A` to `drop`: numeric
  31, nominal 224, drop 10, so 255 input columns.
- `HHID` is unique in train, so household grouping effects cannot be modelled.


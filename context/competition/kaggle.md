<!--
Derived from the Kaggle competition page for ML@DAA 2026/2027. Written 2026-10-30.
Re-check the live competition page before submitting; rules can change.
Competition entry (this is not an invite token): https://www.kaggle.com/competitions/DAA
The private invite link is deliberately not recorded anywhere in this repository.
Requirements and deadlines are owned by context/assignment/enunciado.md; this file
owns competition rules, target coding and submission format.
-->

# Competition — ML@DAA 2026/2027

**Host:** VictorAlves · Community Prediction Competition (not a public competition)
**Course:** Data and Machine Learning (*Dados e Aprendizagem Automática*) — practical group work

## Overview

The goal is to develop and optimise Machine Learning models that predict an
individual's **age group** (`AGE_RANGE`) from a Portuguese nationwide health survey
collected in 2019. The survey covers health status, healthcare utilisation,
determinants and lifestyles, and general socioeconomic characteristics.

Prediction is one half of the work. The other half is using the model to detect
**profile inconsistencies and epidemiological anomalies** (for example, a
60-year-old biological profile in an 18-year-old respondent, or the reverse) and to
characterise each age group at health and socioeconomic level. The assignment has two
tasks; see `../assignment/enunciado.md`.

## Dataset

- Reference year: 2019 (one record dated 2020); country: Portugal.
- Train: `data/dataset_train.csv`, 10,962 rows × 237 columns.
- Test: `data/dataset_test_X.csv`, 3,655 rows × 236 columns.
- 236 explanatory features. The training table has 237 columns because it also
  contains the target `AGE_RANGE`.
- `PID` and `HHID` are identifiers and must not be used as model features.

### Feature groups

| Group | Description |
|---|---|
| `PID`, `HHID` | Individual and household identification numbers |
| `REGION`, `NUTS1`, `NUTS2_AG` | Geographical region of residence |
| `DEG_URB` | Degree of urbanisation (densely, intermediate, or thinly populated) |
| `SEX` | Sex of the individual |
| `HATLEVEL` | Highest level of education completed |
| `MAINSTAT` | Labour status |
| `JOBISCO_COD`, `LOCNACE_COD` | Main occupation and economic activity sector |
| `HHINCOME` | Quintile of net monthly income per equivalised adult |
| `HS1`, `HS2`, `HS3` | Self-perceived health, long-standing health problems, degree of limitation in daily activities |
| `CD1A`–`CD1P` | Binary indicators of chronic diseases/conditions in the past 12 months |
| `PL1`–`PL9` | Difficulty and use of aids for sensory and motor functions |
| `PC1A`–`PC1E` | Difficulty and need for help in personal care activities |
| `HA1A`–`HA1G` | Difficulty in household and administrative tasks |
| `BM1`, `BM2` | Anthropometric data |
| `PE1`–`PE8` | Physical effort at work, walking/cycling commuting, weekly exercise frequency |
| `DH1`–`IN71` | Food consumption frequency |
| `IN43`–`SK6` | Tobacco consumption status and history |
| `AL1`–`AL6` | Alcohol consumption frequency/volume and heavy episodic drinking |
| `SS1`–`SS3` | Social support, contact network, perceived help availability |
| `AGE_RANGE` | Target — age group of the individual |

### Regional distribution (train)

**Data-explorer-reported, not re-verified.** Do not publish these figures as measured
until they are recomputed on the current `data/`.

| Dimension | Distribution |
|---|---|
| Region | Continente 72% · RA Açores 14% · Other (487) 13% |
| NUTS1 | Centro (PT) 22% · Alentejo 16% · Other (2270) 62% |
| NUTS2_AG | Centro 18% · Lx V Tejo 17% · Other (2373) 65% |
| DEG_URB | Áreas mediamente povoadas 39% · Áreas pouco povoadas 32% · Other (1050) 29% |

## Target: 16 age groups

The **official** Kaggle table is:

| Code | Age group | Code | Age group |
|---|---|---|---|
| 0 | 15–17 | 8 | 50–54 |
| 1 | 18–19 | 9 | 55–59 |
| 2 | 20–24 | 10 | 60–64 |
| 3 | 25–29 | 11 | 65–69 |
| 4 | 30–34 | 12 | 70–74 |
| 5 | 35–39 | 13 | 75–79 |
| 6 | 40–44 | 14 | 80–84 |
| 7 | 45–49 | 15 | 85+ |

### Disputed: band 0

The data is ambiguous about band 0:

- Band 0 has 207 rows (1.9% of the sample): 204 are students, 163 completed
  "Básico 3º ciclo", 39 "Básico 1º e 2º ciclo", 2 "Nenhum nível completado",
  2 "Secundário", 1 "Pós secundário".
- That profile fits children aged 0–14, but it also fits 15–17-year-olds: most have
  completed basic 3rd cycle and have not completed secondary.
- Band 1 has only 172 rows, while band 2 (20–24) has 399. A five-year 15–19 band
  would normally be closer to band 2, which argues for the official table
  (`1 = 18–19`).
- The survey may cover only ages 15+, which would rule out a 0–14 band.

**Status: open.** Do not state either mapping as fact. The two valid partitions are
the official table above, or `0 = 0–14, 1 = 15–19, 2 = 20–24, …, 15 = 85+` (the
hypothesis in `reports/01`). `0 = 0–14, 1 = 18–19` is not valid because it skips
15–17. See `../../TODO.md` for the question to the professor.

## Evaluation

**Macro F1-score** across the 16 age-group classes. Every class carries the same
weight regardless of its frequency, so the minority bands 0 and 1 matter as much as
the large middle bands.

## Submission format

- Header `RowId,AGE_RANGE`.
- 3,655 data rows (one per test row) plus the header; `RowId` runs 0…3654.
- CRLF line endings in the provided example.
- `data/example_submission_10.csv` is a **format template**: every `AGE_RANGE` value
  is `10` as a placeholder. It is not a prediction and must not be submitted as one.
- Up to **3 valid submissions per day**; invalid submissions do not count towards
  the limit.
- Only **one file** (the one you choose) counts for the final ranking.
- Improper formatting or extra rows/columns cause errors.
- Do not build the file by hand.

## Dates

- Competition closes at the end of **2027-01-01** (23:59).
- Report and all elements as one zip: **2027-01-02** on e-learning.
- Checkpoint: **2026-11-30** and **2026-12-03** (during class).
- Presentations: **2027-01-11, 12, 13** (choose a slot; 10 minutes).

See `../assignment/deliverables.md` for the full checklist.

## Notes on stored results

`reports/01_feature_selection_review.md` contains model outputs produced from
`data_v1/` (238 columns, including `WGT`). Current `data/` has 237 columns and no
`WGT`; those outputs were not regenerated. Label them "stored outputs from
`data_v1/`" until they are rerun on the current data.

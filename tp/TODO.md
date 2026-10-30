# TODO — open questions

Only actionable questions. Resolved items are listed at the end.

## For the professor

### 1. Is `AGE_RANGE` band 0 `15–17` or `0–14`?

The Kaggle page labels the 16 classes as:

`0 = 15–17, 1 = 18–19, 2 = 20–24, …, 15 = 85+`.

The data is ambiguous:

- Band 0 has 207 rows: 204 students, 163 with “Básico 3º ciclo”, 39 with
  “Básico 1º e 2º ciclo”, 2 with “Nenhum nível completado”, 2 with “Secundário”,
  1 with “Pós secundário”. It is 1.9% of the sample.
- That profile fits children aged 0–14, but it also fits 15–17-year-olds: most
  15–17-year-olds have completed basic 3rd cycle and have not completed secondary.
- Band 1 has only 172 rows, while band 2 (20–24) has 399. A five-year 15–19 band
  would normally be closer to band 2; this argues for band 1 = 18–19, i.e. the
  official table.
- The survey may cover only ages 15+, which would rule out a 0–14 band.
- A valid 16-class partition is either the official one, or
  `0 = 0–14, 1 = 15–19, 2 = 20–24, …, 15 = 85+` (the `reports/01` hypothesis).
  `0 = 0–14, 1 = 18–19` is not valid because it skips 15–17.

**Blocks:** age-group characterisation, anomaly examples, any ordinal treatment of
the target, and the wording of the report.

**Until answered:** use the official labels for submission; label every age
interpretation “provisional”; do not put a band mapping in the report without a
caveat.

### 2. What is expected at the 2026-11-30 / 2026-12-03 checkpoint?

The enunciado only says “using the means it considers most appropriate”.

**Blocks:** what we prepare, and how much must be finished by then.

### 3. Are the two tasks weighted separately?

Analysis and Validation vs. Competition.

**Blocks:** where effort goes in December.

## For the group

- If the professor does not answer (1) before the report draft: keep the official
  label in every table and add a footnote that the data's band 0 is disputed.
- Re-measure the regional distribution before publishing it; if not re-measured,
  mark it “data-explorer-reported”.
- Regenerate `reports/01` outputs on current `data/` (237 columns, no `WGT`) before
  quoting them as current results; until then, label them “stored outputs from
  `data_v1/`”.

## Resolved

- `n_estimators` — number of trees / boosting rounds; see `context/notes/ml-reference.md` §10.
- Thresholds fragment — dropped as unreconstructable; see `context/notes/notas-audit.md`.
- Pair plot — kept as a plot; the “super power” claim was dropped.
- Multiclass logistic regression — multinomial (softmax) / one-vs-rest.

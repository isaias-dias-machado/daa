# Competition ML@DAA — 2026/2027 Edition

**Host:** VictorAlves · Community Prediction Competition (Private)
**Format:** Practical Group Work
**Deadline:** ~2 months remaining
**License:** Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)

---

## Objective

Build a Machine Learning model that predicts an individual's **age group (`AGE_RANGE`)** from their socioeconomic profile, self-perceived health status, chronic pathologies, autonomy in daily tasks, and lifestyle habits (alcohol, tobacco, diet, physical exercise).

Beyond prediction, the model must be used to identify **epidemiological anomalies**: individuals whose model-estimated age deviates significantly from their true age, signaling prematurely aged or atypically young biological profiles. Each age group must then be characterized in detail at both socioeconomic and health levels.

---

## Datasets

| File | Purpose |
|---|---|
| `dataset_train.csv` | Training data — includes the true label `AGE_RANGE`. Used to develop, train, and optimize the model. |
| `dataset_test.csv` | Test data — `AGE_RANGE` withheld. Predict this class for each entry. |

Evaluation on the test set measures generalization via the **Macro F1-Score**.

**Data Explorer summary:**

- 3 files, 47.3 MB total
- 475 columns
- `dataset_test_X.csv` (11.82 MB), `dataset_train.csv`, `example_submission_10.csv`
- Reference year: 2019 (one record dated 2020), country: PT

**Regional distribution (train):**

| Dimension | Distribution |
|---|---|
| Region | Continente 72% · RA Açores 14% · Other (487) 13% |
| NUTS1 | Centro (PT) 22% · Alentejo 16% · Other (2270) 62% |
| NUTS2_AG | Centro 18% · Lx V Tejo 17% · Other (2373) 65% |
| DEG_URB | Áreas mediamente povoadas 39% · Áreas pouco povoadas 32% · Other (1050) 29% |

Covered areas: Continente (Norte, Centro, Lisboa, Lx V Tejo, Alentejo, Algarve), RA Açores, RA Madeira — across densely, intermediate, and thinly populated areas.

---

## Feature Groups

| Group | Description |
|---|---|
| `PID`, `HHID` | Individual and household identification numbers |
| `REGION`, `NUTS1`, `NUTS2_AG` | Geographical region of residence |
| `DEG_URB` | Degree of urbanization (densely, intermediate, or thinly populated) |
| `SEX` | Sex of the individual |
| `HATLEVEL` | Highest level of education completed |
| `MAINSTAT` | Labor status |
| `JOBISCO_COD`, `LOCNACE_COD` | Main occupation and economic activity sector |
| `HHINCOME` | Quintile of net monthly income per equivalized adult |
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

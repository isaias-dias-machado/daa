# Review of notebook 01 — feature selection for `AGE_RANGE`

**Prepared:** 2026-10-26  
**Notebook reviewed:** `notebooks/01.ipynb`  
**Data checked:** `data/dataset_train.csv` (10,962 rows × 237 columns) and `data/dataset_test_X.csv` (3,655 × 236)  
**Stored notebook output:** produced from `data_v1/dataset_train.csv` (10,962 × 238, including `WGT`)  
**How the numbers were obtained:** the random-forest and permutation-importance numbers are read from the stored notebook outputs. The extra checks below (missingness, association, routing, train/test coverage) are cheap contingency-table and information-theoretic diagnostics; no model was refitted.

---

## Bottom line

1. **The model has real signal.** Five-fold CV accuracy is **0.442 ± 0.007**, holdout accuracy is **0.451**, and the majority-class baseline is **0.097**. A 16-class model at 44% is clearly learning something.
2. **The top-20 list is useful for plots, but it is not a validated feature set.** Cutting to the top 20 drops CV accuracy to **0.399** — a loss of **4.3 percentage points**, or **9.7% relative**. The notebook's own criterion says a large drop means the ranking discarded real signal. It did.
3. **The apparent 116-feature result is inflated by leakage.** `SelectFromModel` is fitted on all training rows before cross-validation, so each CV fold's validation rows helped choose the features. The reported **0.444** is optimistic; its +0.002 over the full model is well inside the 0.007 CV noise. It is not evidence that the smaller set is as good.
4. **Many top features are age proxies or survey-routing artifacts, not substantive causes of age.** Examples: at the plain “Não aplicável” level, `HA1D` is 100% in bands 0–8 and 0% in bands 9–15 (a separate, more specific not-applicable level appears within 55+); `PA7_NAC`/`PA8_NAC` are female-only and perfectly associated with `SEX`; household composition and employment status encode life stage.
5. **The correlation audit uses the wrong metric.** Pearson correlation on arbitrary ordinal codes is not valid for nominal variables. Cramér's V finds perfect associations (1.0) that Pearson reports as 0.74/0.68.
6. **The original notebook does not run on the current data.** `data/` has 237 columns and no `WGT`; the drop step raises `KeyError`. The stored outputs come from the 238-column `data_v1/`. The loader has been made tolerant of both versions, but the model outputs should be regenerated to confirm.

**Practical conclusion:** use the top-20 file for exploratory graphs. Do not present it as “the important features” or as a final model. The current evidence supports the opposite conclusion: 20 features are too few for this target, and the 116-feature result has not been validated.

---

## 1. What the notebook does

| Step | What happens | Comment |
|---|---|---|
| 0 | Drops `PID`, `HHID`, `WGT`, constants (`PROXY`, `COUNTRY`) and near-constant `IN13` | Correct for prediction; `WGT` is survey metadata |
| 1 | Ordinal-encodes every categorical; median-imputes numerics | Fast, but treats nominal categories as ordered numbers |
| 2 | One stratified 80/20 split; 400-tree random forest; permutation importance on the holdout | One split, one seed |
| 3 | Audits the top 20: Pearson correlation, one-hot re-encoding, cardinality | Correlation metric is invalid for nominals; one-hot audit is joint, not per-feature |
| 4 | Compares full vs top-20 vs `SelectFromModel` with 5-fold CV on the training split | `SelectFromModel` is leaky |

### Results at face value

| Feature set | Features | CV accuracy | What it means |
|---|---:|---:|---|
| Full | 231 | **0.4424** | Baseline for the cut |
| Top-20 | 20 | **0.3995** | −4.29 pp, −9.7% relative |
| `SelectFromModel` (median impurity) | 116 | **0.4443** | Leaky; +0.19 pp vs full, inside 0.007 CV noise |

- Target: 16 ordered bands, minimum class 172, maximum class 1,062, majority baseline **9.69%**.
- Full model: 5-fold CV **0.442 ± 0.007**, holdout **0.451**.
- Test set: 3,655 rows, 236 columns, no target, no `PID`/`HHID` overlap with train. It was correctly left untouched.

### A note on the age bands

The data does not include a codebook, but the observed patterns are consistent with the common coding **0 = 0–14, 1 = 15–19, 2 = 20–24, …, 15 = 85+**, not with 5-year bands starting at 0–4:

- 204 of the 207 respondents in band 0 are “Aluno, estudante” (students).
- Students are concentrated in bands 0–2: 204, 130, 155, then 24 in band 3 and 5 in band 4.
- Retirees become common from band 10 (60–64): 333; band 11 (65–69): 751; band 12 (70–74): 904.

So in this report, “band 9” means approximately **55–59 years old**, and “bands 9–15” means **55+**. Confirm this with the survey codebook before putting age labels in a final write-up.

---

## 2. What the model is actually learning

### 2.1 The signal is real, but accuracy is a coarse summary

With 16 classes, random guessing is 6.25% and the majority-class baseline is 9.69%. The model reaches 44.2% cross-validated and 45.1% holdout, so it is not a random model.

But accuracy treats every mistake the same. Predicting the neighbouring age band is penalised exactly like predicting ten bands away. For an ordered target, that is a poor summary. The notebook does not report:

- a confusion matrix,
- per-class recall,
- macro-F1 or balanced accuracy,
- mean absolute error in bands,
- quadratic weighted kappa (QWK).

Without those, we cannot tell whether 44% means “usually within one band” or “often wildly wrong.” The target is also imbalanced: the smallest classes have 172–449 rows, the largest about 1,000. A model can reach 44% overall while doing poorly on the youngest bands.

### 2.2 Age proxies: household and life-stage variables

The top of the permutation ranking is dominated by variables that describe life stage. Group means below are in band index; approximate ages assume the 0–14/5-year coding above.

| Feature | Level | Mean band | Approx. age | Why it is an age proxy |
|---|---|---:|---:|---|
| `MAINSTAT` | Student | 1.20 | 15–19 | Students are young |
| `MAINSTAT` | Retired | 12.38 | 70–74 | Retirees are old |
| `MARSTALEGAL` | Single | 5.22 | 35–39 | Never-married respondents are younger on average |
| `MARSTALEGAL` | Widowed | 12.62 | 70–74 | Widowhood is concentrated at older ages |
| `HATLEVEL` | Secondary | 6.02 | 40–44 | Educational attainment is cohort-related |
| `HATLEVEL` | No completed level | 12.71 | 70–74 | Older cohorts have less schooling |
| `HH_ACT` | 0 active members | 11.50 | 65–69 | Retired households have no active workers |
| `HH_ACT` | 2 active members | 6.10 | 40–44 | Younger households are more likely to have workers |
| `HHNBPERS_0_13_COD` | 0 children | 9.42 | 55–59 | Households with no young children are older |
| `HHNBPERS_0_13_COD` | ≥3 children | 5.42 | 35–39 | Households with young children are younger |

These are **correlations with age**, not causes of age. A model is allowed to use them for prediction. But “`HH_INACT` is the most important feature” does not mean “household inactivity causes age”; it means “given the other features, this forest uses household activity to separate life stages.”

The same point applies to `SEX`: its univariate association with age is almost zero (AMI 0.0013, rank 219 of 231), yet it ranks 20th in the forest. It is useful because it conditions the female-only questions, not because sex predicts age by itself. That is an interaction effect.

### 2.3 Survey routing: “Não aplicável” is often an eligibility flag

This is the most important interpretation issue. In a survey, a question is often asked only to eligible respondents. “Não aplicável” then means “this question was skipped,” not “the person has no health problem.” The data contains near-perfect age routing:

| Feature(s) | Bands 0–4 (≈0–34) | Bands 5–8 (≈35–54) | Bands 9–15 (≈55+) | Interpretation |
|---|---:|---:|---:|---|
| `HA1A`–`HA1G`, `PC1A`–`PC1E`, `IN15` | **100%** NA | **100%** NA | **0%** NA | Health/disability module asked only at 55+ |
| `FT_PT`, `JOBSTAT_COD`, `AW1`, `IN60`, `LOCNACE_COD` | 46.7% NA | 17.6% NA | 78.7% NA | Employment status; children/students and retirees are not in the labour force |
| `IN23`, `NT1`, `NT2`, `IN20`, `IN28` | ~46% NA | ~44% NA | ~98% NA | Labour-market routing |
| `IN16` | 34.7% missing | 39.9% missing | 61.4% missing | Dental-visit reason; missingness itself is age-related |

\*For `HA1A`–`HA1G`, the table counts only the plain `Não aplicável` level, which is the age-eligibility skip. Those columns also contain a separate `Não aplicável (nunca tentou ou não precisa de o fazer)` level, which is a substantive skip within the eligible 55+ group; it is not age routing. `PC1A`–`PC1E` and `IN15` have only the plain level.

The 13 health variables are especially revealing. Their plain **`Não aplicável` level is 100% for everyone under about 55 and 0% for everyone 55+**. A tree can split the sample at 55 with one of these features without learning anything about health. Their univariate association with age is therefore enormous (AMI ≈ 0.36–0.40), which is why they dominate the univariate top 20. They are not 13 independent discoveries; they are essentially one eligibility indicator plus a health response.

This is not direct target leakage: no feature is a copy of `AGE_RANGE`, and there is no train/test ID overlap. The problem is indirect — the survey asked different questions to different people.

This does **not** make them useless for prediction. It does mean:

- Do not interpret “health predicts age” from these features.
- Do not treat “Não aplicável” as a substantive answer category in an explanatory analysis.
- If the goal is explanation, analyse within eligible age groups or drop routing-only features.
- If the goal is prediction, routing is allowed, but say so explicitly.

`IN16` is a second example of informative missingness: 51.1% of all rows are missing, but the rate rises from 34.7% in bands 0–4 to 61.4% in bands 9–15. The notebook fills it with `__MISSING__` and never audits it. That is a missed age signal — and possibly another routing artifact.

### 2.4 Female-only screening questions

`PA7_NAC` and `PA8_NAC` are asked only to women:

- All **4,708 men** have `Não aplicável`.
- All women have a non-`Não aplicável` value (including `Sem resposta`).
- Cramér's V with `SEX` is **1.000** for both.
- Pearson on the ordinal codes gives only 0.736 and 0.682.

The top 20 contains `SEX`, `PA7_NAC` and `PA8_NAC`: three sex-related columns, not three independent findings. When the one-hot audit re-encodes them together, `PA7_NAC` rises from 0.0139 to 0.0189 while `PA8_NAC` collapses from 0.0041 to −0.0002. They compete for the same signal. Drop one (or both, keeping `SEX`) and the ranking changes.

### 2.5 Household identifiers do not group households

`PID` and `HHID` are both unique per row. `HHID` is just `PID` plus a two-digit suffix: the prefix equals `PID` for 100% of rows, and there are 10,962 unique prefixes for 10,962 rows — one row per “household.” There is no observable household grouping.

Consequences:

- Dropping both IDs is correct.
- There is no ID overlap between train and test, so no leakage through IDs.
- You cannot do grouped cross-validation by household, and you cannot check whether household features leak information from other members. The household features are still attributes of the respondent's household, but they cannot be validated as household-level effects.

### 2.6 The top 20 is not 20 independent features

The top 20 collapses into a handful of themes:

| Theme | Features in the top 20 |
|---|---|
| Household composition / activity | `HH_INACT`, `HH_ACT`, `HHNBPERS_0_13_COD`, `HHNBPERS_COD` |
| Marital / partnership | `MARSTALEGAL`, `PARTNERS` |
| Labour market | `MAINSTAT`, `FT_PT`, `LOCNACE_COD`, `IN23` |
| Sex / female screening | `SEX`, `PA7_NAC`, `PA8_NAC` |
| Health / disability | `HA1D` |
| Education | `HATLEVEL` |
| Income | `HHINCOME` |
| Other survey items | `IN63`, `IN70`, `PL1`, `IN54` |

Permutation importance splits credit within correlated groups. The effective number of independent signals is much smaller than 20. This is why the one-hot audit produces such large swings: the model is reallocating credit among substitutes, not discovering a new variable.

---

## 3. Problems, ranked

### Critical

#### 3.1 The original notebook does not run on the current data

As stored, the notebook uses the 238-column `data_v1/` file. The current `data/dataset_train.csv` has **237 columns and no `WGT`**, so the notebook as written does:

```python
X = df.drop(columns=[TARGET] + ID_COLS + [WEIGHT_COL])
```

That raises `KeyError: "['WGT'] not found in axis"`. The stored outputs are from `data_v1/dataset_train.csv`, which has 238 columns and includes `WGT`.

`data/` is `data_v1/` minus `WGT`. Since `WGT` was dropped before modelling anyway, the feature matrix should be unchanged after a conditional drop — but the notebook must be rerun to confirm and to update its stored outputs.

**Fix applied:** the loader now drops `WGT` only if it is present, and prints a note when it is absent. A reproducibility warning was added at the top of the notebook. The stored model outputs are still from the old run; they were not regenerated (that would require rerunning the forest).

#### 3.2 The top-20 cut loses too much accuracy to be a final model

The notebook's own Step 4 says:

> “a small drop (or none) means the cut is safe; a large drop means the ranking discarded real signal.”

The result is a **4.29 percentage-point drop** (0.4424 → 0.3995), or **9.7% relative**. By its own criterion, the cut is not safe.

The boundary is also arbitrary:

- rank 20: 0.002371
- rank 21: 0.002234
- rank 25: 0.001824

The gap between ranks 20 and 21 is 0.000137, about 5.8% of the rank-20 value. A different seed or split could easily swap them. Beyond the top 5, the permutation standard deviations are large relative to the means (see §3.9; full values in §6.1). The “top 20” is a convenient round number, not a natural boundary.

**What to do:** keep the top 20 for exploratory plots. For a final model, either keep more features or use a principled, leak-free selection procedure. Do not describe the top 20 as “the important features.”

#### 3.3 `SelectFromModel` cross-validation is leaky

The code is:

```python
sfm = SelectFromModel(RandomForestClassifier(...), threshold="median").fit(X_tr, y_tr)
kept = X_tr.columns[sfm.get_support()].tolist()
results.loc[...] = ["select-from-median", len(kept), cv_accuracy(kept)]
```

`sfm` is fitted on **all** of `X_tr`. Then `cv_accuracy(kept)` runs cross-validation on `X_tr` using the already-selected columns. Each fold's validation rows were used to choose the features, so the CV estimate is optimistic.

The reported 0.4443 is also only +0.0019 over the full model, while the full-model CV standard deviation is 0.007. Even without the leakage, that difference is noise. There is no evidence here that selection improves performance.

There is a second inconsistency: `SelectFromModel` thresholds **impurity** importance, while the notebook argues that **permutation** importance is preferable and less biased toward high-cardinality features. The 116-feature set is therefore not simply “the top half of the ranking”; it may include features the permutation ranking considers weak.

**Fix:** put selection inside a pipeline so it is refit within every CV fold:

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

pipe = Pipeline([
    ("select", SelectFromModel(
        RandomForestClassifier(n_estimators=400, random_state=0, n_jobs=-1),
        threshold="median",
    )),
    ("rf", RandomForestClassifier(n_estimators=400, random_state=0, n_jobs=-1)),
])

cv = StratifiedKFold(5, shuffle=True, random_state=0)
scores = cross_val_score(pipe, X_tr, y_tr, cv=cv)
```

This fixes the leakage. It does not fix the impurity-importance bias; for that, use `importance_getter` or a custom selector based on permutation importance. Nested CV is the more rigorous option if you also want to tune the threshold.

### High

#### 3.4 Nominal variables are ordinal-encoded

`OrdinalEncoder` assigns an arbitrary integer order to every categorical. For a tree, that is not fatal: a tree can split at any threshold and approximate an arbitrary ordering. But it is inefficient, it creates a false sense of distance between categories, and it makes importance comparisons unstable.

The notebook's own one-hot audit shows large changes when 18 low-cardinality features are re-encoded together:

| Feature | Ordinal importance | One-hot sum | Change |
|---|---:|---:|---:|
| `MAINSTAT` | 0.0064 | 0.0255 | **4.0×** |
| `HATLEVEL` | 0.0033 | 0.0119 | **3.6×** |
| `FT_PT` | 0.0025 | 0.0106 | **4.3×** |
| `IN63` | 0.0046 | 0.0085 | 1.8× |
| `HHNBPERS_0_13_COD` | 0.0180 | 0.0061 | **−66%** |
| `HHINCOME` | 0.0044 | −0.0034 | collapses |
| `HHNBPERS_COD` | 0.0046 | −0.0026 | collapses |
| `PA8_NAC` | 0.0041 | −0.0002 | collapses |
| `IN54` | 0.0024 | −0.0045 | collapses |

Caveat: this audit re-encodes all 18 features at once, so these are **joint** effects, not isolated encoding effects. A negative one-hot sum means shuffling that feature improved holdout accuracy — a sign of noise or competition, not a real effect. Still, the size of the swings is a red flag: the ranking is not encoding-robust.

**What to do:**

- Classify each variable as numeric, nominal, or genuinely ordinal.
- One-hot encode nominals, with `handle_unknown="ignore"` or an explicit `__UNKNOWN__` level.
- Use ordinal encoding only for true ordered categories (e.g., education levels, Likert scales), and document the order.
- Put the encoder inside a `ColumnTransformer` / `Pipeline` so it is fitted on training folds only.
- Do not one-hot encode 165-level features blindly; group rare levels or use cross-validated target encoding.

#### 3.5 Pearson correlation on ordinal codes is invalid for nominal variables

The notebook computes:

```python
corr = X_enc[top_k].corr()
```

`X_enc` contains arbitrary integer codes for nominal categories. Pearson measures linear association between those codes. Change the alphabetical order of the categories and the correlation changes. It is not a valid measure of association for nominal variables.

The practical consequence: the notebook's conclusion “no pairs above 0.9” is false under a valid measure. Cramér's V finds **perfect** associations:

| Pair | Pearson on codes | Cramér's V | What it shows |
|---|---:|---:|---|
| `PA7_NAC` – `SEX` | 0.736 | **1.000** | Female-only question; all men are `Não aplicável` |
| `PA8_NAC` – `SEX` | 0.682 | **1.000** | Same |
| `MAINSTAT` – `FT_PT` | 0.719 | 0.577 | Labour-status cluster |
| `LOCNACE_COD` – `FT_PT` | 0.697 | 0.598 | Labour-status cluster |
| `HH_ACT` – `FT_PT` | 0.677 | 0.427 | Household/labour overlap |
| `MARSTALEGAL` – `PARTNERS` | 0.666 | 0.522 | Marital-status cluster |
| `HH_ACT` – `HHNBPERS_COD` | 0.610 | 0.447 | Household composition |
| `PA7_NAC` – `PA8_NAC` | 0.604 | 0.460 | Female screening pair |
| `HHNBPERS_0_13_COD` – `HHNBPERS_COD` | 0.596 | 0.447 | Household size |
| `MAINSTAT` – `LOCNACE_COD` | 0.509 | 0.352 | Labour-status cluster |

Cramér's V is a 0–1 measure of association between two categorical variables: 0 means independent, 1 means perfect association. It does not assume the categories are ordered.

**What to do:**

- Use Cramér's V (or mutual information) for categorical–categorical pairs.
- Use Pearson only for numeric–numeric pairs.
- Use a correlation ratio or grouped summary for numeric–categorical pairs.
- Cluster correlated features, then use grouped permutation importance or keep one representative per cluster.
- Do not interpret a Pearson matrix of ordinal codes as a correlation matrix of the underlying variables.

#### 3.6 Correlated features hide each other; RF importance is conditional

Only **5** of the 20 random-forest features appear in the univariate top 20 by adjusted mutual information:

- Intersection: `HA1D`, `MAINSTAT`, `FT_PT`, `HH_ACT`, `IN23`.
- Jaccard similarity: 5 / 35 = **0.143**.

The univariate top 20 is dominated by the 13 health/routing features (`IN15`, `PC1A`–`PC1E`, `HA1A`–`HA1G`). The RF top 20 contains only one of them, `HA1D`. That does **not** mean the others are unimportant. It means they are highly correlated, and `HA1D` absorbs the shared signal. When one feature is shuffled, its correlated copies remain, so permutation importance understates the group's importance.

This is exactly the “correlated features” caveat the notebook cites, but the notebook then uses a Pearson matrix on arbitrary codes to audit it.

**What to do:**

- Group features by theme or by Cramér's V, then shuffle the whole group to measure group importance.
- Or drop one feature per correlated cluster and re-rank.
- Report cluster-level importance, not just individual ranks.
- Do not say “`HA1D` is the only health feature that matters.” Say “`HA1D` is the representative that survived in the presence of its correlates.”

#### 3.7 Survey routing is not separated from substantive responses

This is covered in §2.3, but it is also a methodological flaw: the notebook treats every `Não aplicável` as an ordinary category. That mixes two different things:

- **Eligibility routing:** the question was not asked because of age, sex, or employment status.
- **Substantive non-response:** the question was asked but the respondent did not answer.

A model can exploit either, but they have different interpretations. The health module's plain `Não aplicável` 100% / 0% split at 55+ is an eligibility flag. `IN16`'s 51% missingness may be a visit/no-visit pattern.

**What to do:**

- Get the survey codebook and document what each `Não aplicável` means.
- For each feature, record the eligible subgroup and the routing rule.
- Add explicit `asked` / `eligible` flags if you want to model both the routing and the response.
- For explanatory analysis, restrict to eligible respondents or drop routing-only features.
- For prediction, keep them but state that the model uses survey design.

### Medium

#### 3.8 The high-cardinality rule does nothing

The notebook flags features with more than 10% of rows as levels:

```python
high_card_threshold = 0.10 * n_rows  # 1,096 levels
```

No feature crosses 1,096. The maximum is `PE9` with 165 levels, followed by `IN57` 144, `IN55` 135, `IN56` 120, `PE7` 102, `BM2` 102. The check prints “none” and the notebook moves on. But 165 levels is still a lot for a 10,962-row dataset, and the test set contains unseen levels:

| Feature | Unseen test levels | Share of test rows |
|---|---:|---:|
| `PE9` | 15 | 0.438% |
| `IN55` | 13 | 0.410% |
| `IN57` | 13 | 0.383% |
| `PE7` | 13 | 0.356% |
| `IN56` | 10 | 0.328% |
| `IN59` | 9 | 0.246% |

`OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)` maps every unseen level to **−1**, which a tree treats as an ordered value below all seen codes. That is a silent assumption, not a neutral fallback.

`HH_INACT` is numeric, so the categorical coverage check misses it; the test set has one row with value **6**, which never occurs in training (training values are 0–5). Trees will extrapolate to the right of 5. For a count, that may be reasonable, but it should be a conscious choice.

None of the top-20 categorical features had unseen levels in this particular test split. The risk is in the full model and in future data.

**What to do:**

- Set a data-driven cardinality threshold (e.g., 20–30 levels) and group rare levels into `__OTHER__`.
- Or use cross-validated target encoding for high-cardinality features.
- Check train/test category coverage before modelling.
- Use `OneHotEncoder(handle_unknown="ignore")` or an explicit `__UNKNOWN__` category; do not let unseen levels become −1 in an ordinal code.
- For numeric counts, clip to the training range or treat as categorical if the values are few.

#### 3.9 One split, one seed, no stability analysis

The notebook uses one 80/20 split, one CV seed, and one forest seed. The permutation importance is repeated 10 times, so `importance_std` measures repeat-to-repeat noise, **not** uncertainty across splits or model fits.

For ranks 6–20, the repeat-to-repeat standard deviation is often large relative to the mean:

| Feature | Importance | Std / mean |
|---|---:|---:|
| `HATLEVEL` | 0.0033 | **1.04** |
| `FT_PT` | 0.0025 | **1.56** |
| `IN23` | 0.0024 | 0.90 |
| `PL1` | 0.0026 | 0.90 |
| `HHINCOME` | 0.0044 | 0.79 |
| `LOCNACE_COD` | 0.0041 | 0.82 |
| `PA8_NAC` | 0.0041 | 0.73 |
| `IN54` | 0.0024 | 0.69 |
| `SEX` | 0.0024 | 0.68 |

The top 5 are more stable (std/mean 0.05–0.20), but beyond that the ranking is noisy. A different split or seed could reshuffle ranks 6–25.

**What to do:**

- Repeat the split/CV with several seeds and report the distribution of ranks.
- Bootstrap the training data and count how often each feature appears in the top 20.
- Report confidence intervals for importance, not just the mean.
- Treat the top-20 boundary as provisional.

#### 3.10 Accuracy alone is the wrong metric for this target

The target has 16 ordered, imbalanced classes. Accuracy ignores:

- how far off the prediction is,
- whether small classes are predicted at all,
- whether the model is just predicting the large middle bands.

**What to do:**

- Report a confusion matrix.
- Report per-class recall and precision.
- Report macro-F1 and balanced accuracy.
- For the ordered target, report mean absolute error (in bands) and quadratic weighted kappa (QWK).
- Compare against a dummy classifier and a simple ordinal baseline.
- Consider ordinal regression or cost-sensitive learning; a random forest classifier treats the bands as unordered labels.

#### 3.11 Preprocessing is fitted before the split

`OrdinalEncoder` and median imputation are fitted on the full dataset before `train_test_split`. That means holdout information influences the encoding and imputation. The effect is small here — only `IN16` has missing values, and it is not in the top 20 — but it is still leakage.

**Fix:** split first, then fit all preprocessing on the training split only. The cleanest way is a `Pipeline` with a `ColumnTransformer`.

#### 3.12 The one-hot audit is not a controlled experiment

The audit re-encodes **all 18** low-cardinality categoricals at once, retrains the forest, and sums dummy importances. Problems:

- Differences are joint effects of 18 encoding changes, not the effect of one feature's encoding.
- `HH_INACT` and `HH_ACT` are numeric, so they are unchanged; their “one-hot sum” is just their importance in the new model.
- Summing dummy importances can produce negative values; a negative sum means shuffling the feature improved accuracy.
- The one-hot model has a different dimensionality, so the forest and the permutation importance are not directly comparable.

It is a useful **sensitivity check**, but it does not prove that `MAINSTAT`'s ordinal importance was “wrong” or that its one-hot importance is “right.” Use it to flag instability, not to establish a corrected ranking.

### Other issues

- **`REFYEAR` and `REFMONTH` are survey metadata, not person attributes.** Their adjusted mutual information with age is 0.0005 and 0.0003 (ranks 230 and 231 of 231). `IN13` has AMI 0.00009 and is dropped for near-zero variance; `REFYEAR`/`REFMONTH` survive because their values are common, not because they are informative. Drop them or justify them.
- **Geography is kept without justification.** `NUTS1`, `REGION`, `NUTS2_AG` and `DEG_URB` have AMI 0.003–0.005 (ranks 177–194). They may be legitimate contextual features, but the notebook does not audit or justify them.
- **`WGT` is dropped, which is correct for prediction but not for population inference.** The reported accuracy is unweighted. If the goal is a representative description of the Portuguese population, use the survey weights in descriptive statistics and evaluation.
- **`IN16` is the only column with missing values, and its missingness is age-related.** The notebook fills it with `__MISSING__` but never checks whether missingness is a routing flag. It is not in the top 20, but it remains in the full model.
- **`SelectFromModel` keeps 116 features, but the selected set is never inspected.** It could include survey metadata, high-cardinality noise, or redundant features. Report the selected columns and their cardinality.
- **No codebook / data dictionary.** Variable names are cryptic, `Não aplicável` is ambiguous, and the age-band mapping is not documented. This is a reproducibility and interpretation risk.
- **No simple baselines.** The only comparison is the majority class. A logistic regression, a decision tree, or an ordinal model would tell you whether the forest's complexity is earning its keep.

---

## 4. What is still useful

- **The signal is real.** 44% CV / 45% holdout vs 9.7% majority is a large gap. The survey contains substantial age-related structure.
- **The top-20 columns are a good exploratory set.** They cover the strongest forest features and the main life-stage themes. For plots, that is exactly what you want: a manageable file with the target and the key variables.
- **The univariate ranking is a useful complement.** It exposes the health/routing features that the forest's top 20 hides. Together, the two rankings tell a more honest story: the forest top 20 is about conditional usefulness; the univariate top 20 is about marginal association.
- **The data is relatively clean.** No duplicate feature rows, only one column with missing values, unique IDs, no train/test ID overlap, and a test set that was left untouched.
- **The test set is still clean.** Because it was not used for selection, it can be used once at the end to evaluate a final, leak-free pipeline. Do not use it to tune the top-20 cut.

---

## 5. Recommended next steps

### 5.1 Decide the goal first

| Goal | What to do |
|---|---|
| **Predict age as accurately as possible** | Keep routing/proxy features. Use a leak-free pipeline and more than 20 features. Optimise accuracy/QWK/MAE, not just accuracy. Consider an ordinal model. |
| **Explain what is associated with age** | Drop or stratify routing-only features. Analyse within eligible groups. Use Cramér's V/AMI and ordinal measures. Avoid causal language. |
| **Explore the data** | Use the top-20 CSV. Plot `"Não aplicável"` as its own category. Treat the top 20 as a lens, not a verdict. |

### 5.2 Fix the evaluation

1. Split train/holdout/test **before** any preprocessing.
2. Put imputation, encoding, and feature selection inside a `Pipeline`.
3. Use `cross_validate` with `StratifiedKFold`, repeated over several seeds.
4. For selection, use nested CV or a pipeline-wrapped selector; never fit the selector on all training rows before CV.
5. Report accuracy, balanced accuracy, macro-F1, per-class recall, confusion matrix, MAE and QWK.
6. Use the separate test set once, at the end, for the final pipeline.

### 5.3 Fix the encoding and correlation audit

1. Get the codebook and classify each feature as numeric, nominal, or ordinal.
2. One-hot encode nominals; use ordinal encoding only for true orders.
3. Use Cramér's V / mutual information for categorical pairs, not Pearson on codes.
4. Cluster correlated features and use grouped permutation importance.
5. Check train/test category coverage; handle unseen levels explicitly.

### 5.4 Audit the routing

1. For every `Não aplicável`, record the eligibility rule.
2. Separate “question not asked” from “answer missing.”
3. For explanatory work, restrict to eligible subgroups or drop routing-only features.
4. For prediction, keep them but document that the model uses survey design.

### 5.5 Handle cardinality and stability

1. Group rare levels or use cross-validated target encoding for high-cardinality features.
2. Lower the cardinality threshold from 10% of rows to something meaningful (e.g., 20–30 levels).
3. Bootstrap or repeat CV; report how often each feature is selected.
4. Treat the top-20 boundary as provisional.

### 5.6 Minimal pipeline sketch

This fixes the `SelectFromModel` leakage; it does not fix the encoding or the impurity-importance bias.

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

pipe = Pipeline([
    ("select", SelectFromModel(
        RandomForestClassifier(n_estimators=400, random_state=0, n_jobs=-1),
        threshold="median",
    )),
    ("rf", RandomForestClassifier(n_estimators=400, random_state=0, n_jobs=-1)),
])

cv = StratifiedKFold(5, shuffle=True, random_state=0)
scores = cross_val_score(pipe, X_tr, y_tr, cv=cv)
print(f"{scores.mean():.3f} ± {scores.std():.3f}")
```

For a complete solution, wrap the preprocessing in a `ColumnTransformer` and put it before the selector in the same pipeline.

---

## 6. Appendix

### 6.1 Top-20 features

`perm. imp.` and `std` are from the stored permutation-importance run on the holdout. `std/mean` is the repeat-to-repeat coefficient of variation, not model uncertainty. `one-hot sum` is from the joint one-hot audit; negative values mean shuffling improved accuracy (noise/competition). `AMI` is adjusted mutual information with `AGE_RANGE`; `AMI rank` is among the 231 features that enter the model. `Cramér's V` is bias-corrected. `levels` counts missing as a level; the top 20 have no missing values.

| Rank | Feature | Perm. imp. | Std | Std/mean | One-hot sum | AMI | AMI rank | Cramér's V | Levels |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `HH_INACT` | 0.0638 | 0.0031 | 0.05 | 0.0600* | 0.0702 | 49 | 0.2168 | 6 |
| 2 | `HH_ACT` | 0.0380 | 0.0030 | 0.08 | 0.0350* | 0.1683 | 17 | 0.3155 | 7 |
| 3 | `HHNBPERS_0_13_COD` | 0.0180 | 0.0022 | 0.12 | 0.0061 | 0.0891 | 35 | 0.3224 | 4 |
| 4 | `MARSTALEGAL` | 0.0144 | 0.0024 | 0.16 | 0.0158 | 0.1472 | 21 | 0.3919 | 5 |
| 5 | `PA7_NAC` | 0.0139 | 0.0027 | 0.20 | 0.0189 | 0.0933 | 34 | 0.2501 | 8 |
| 6 | `MAINSTAT` | 0.0064 | 0.0029 | 0.46 | 0.0255 | 0.2957 | 14 | 0.4207 | 9 |
| 7 | `HA1D` | 0.0049 | 0.0014 | 0.27 | 0.0016 | 0.3904 | 9 | 0.4288 | 7 |
| 8 | `IN63` | 0.0046 | 0.0021 | 0.46 | 0.0085 | 0.0949 | 33 | 0.3692 | 3 |
| 9 | `HHNBPERS_COD` | 0.0046 | 0.0024 | 0.52 | −0.0026 | 0.0779 | 42 | 0.2268 | 7 |
| 10 | `HHINCOME` | 0.0044 | 0.0035 | 0.79 | −0.0034 | 0.0382 | 66 | 0.2091 | 5 |
| 11 | `LOCNACE_COD` | 0.0041 | 0.0034 | 0.82 | 0.0040 | 0.1409 | 27 | 0.1945 | 15 |
| 12 | `PA8_NAC` | 0.0041 | 0.0030 | 0.73 | −0.0002 | 0.0550 | 57 | 0.1775 | 8 |
| 13 | `PARTNERS` | 0.0036 | 0.0023 | 0.63 | −0.0010 | 0.0808 | 40 | 0.3400 | 4 |
| 14 | `HATLEVEL` | 0.0033 | 0.0034 | 1.04 | 0.0119 | 0.1188 | 31 | 0.2656 | 9 |
| 15 | `IN70` | 0.0026 | 0.0015 | 0.59 | −0.0000 | 0.0067 | 167 | 0.0740 | 6 |
| 16 | `PL1` | 0.0026 | 0.0023 | 0.90 | 0.0007 | 0.0340 | 71 | 0.1943 | 4 |
| 17 | `FT_PT` | 0.0025 | 0.0038 | 1.56 | 0.0106 | 0.1748 | 15 | 0.4082 | 4 |
| 18 | `IN54` | 0.0024 | 0.0017 | 0.69 | −0.0045 | 0.0041 | 184 | 0.0522 | 8 |
| 19 | `IN23` | 0.0024 | 0.0021 | 0.90 | 0.0018 | 0.1532 | 20 | 0.4021 | 4 |
| 20 | `SEX` | 0.0024 | 0.0016 | 0.68 | 0.0040 | 0.0013 | 219 | 0.0661 | 2 |

\* `HH_INACT` and `HH_ACT` are numeric; they were not one-hot encoded, so this is their importance in the one-hot model, not a sum of dummies.

### 6.2 RF top 20 vs univariate top 20

- RF top 20: the 20 features in the table above.
- Univariate top 20 by AMI: `IN15`, `PC1A`, `PC1D`, `PC1E`, `HA1B`, `PC1C`, `HA1G`, `PC1B`, `HA1D`, `HA1C`, `HA1A`, `HA1E`, `HA1F`, `MAINSTAT`, `FT_PT`, `JOBSTAT_COD`, `HH_ACT`, `AW1`, `IN60`, `IN23`.
- Intersection: `HA1D`, `MAINSTAT`, `FT_PT`, `HH_ACT`, `IN23` — 5 features.
- Jaccard similarity: 5 / 35 = **0.143**.

The absence of the 13 health/routing features from the RF top 20 is consistent with their correlation with `HA1D`, not with them being weak. This is the clearest evidence that the two rankings answer different questions.

### 6.3 Data facts

| Item | Value |
|---|---|
| Train shape | 10,962 × 237 |
| Test shape | 3,655 × 236 |
| Target classes | 16 |
| Min / max class | 172 / 1,062 |
| Majority baseline | 9.69% |
| Columns with missing values | 1 (`IN16`, 51.1%) |
| Constant columns | `PROXY`, `COUNTRY` |
| Near-constant | `IN13` (10,955 NA + 7 responses) |
| `PID` unique | 10,962 / 10,962 |
| `HHID` unique | 10,962 / 10,962 |
| `HHID` prefix = `PID` | 100% |
| Unique `HHID` prefixes | 10,962 (one row each) |
| Train/test `PID` overlap | 0 |
| Train/test `HHID` overlap | 0 |
| Duplicate feature rows | 0 |
| Test columns = train minus target | Yes |
| Categorical columns with unseen test levels | 19 |
| Largest unseen coverage | `PE9`: 15 levels, 0.438% of test rows |

### 6.4 Target distribution

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

### 6.5 What not to conclude

- **Do not** conclude that `HH_INACT` is the most important cause of age. It is the feature the forest relies on most, given the others.
- **Do not** conclude that the 13 health/routing features are unimportant because they are absent from the RF top 20. They are correlated with `HA1D`.
- **Do not** conclude that 116 features are as good as 231 based on 0.444. That number is leaky.
- **Do not** conclude that the top 20 is a safe modelling set. The notebook's own CV says it costs about 10% relative accuracy.
- **Do not** conclude that `PA7_NAC` and `PA8_NAC` are independent health findings. They are female-only and perfectly associated with `SEX`.
- **Do not** compare feature sets by accuracy alone. Use uncertainty intervals and ordinal metrics.
- **Do not** use `"Não aplicável"` as evidence of a health condition. It often means “question skipped.”

---

## 7. Files generated

| File | Purpose |
|---|---|
| `reports/01_feature_selection_review.md` | This report |
| `outputs/dataset_train_top20.csv` | 23-column exploratory dataset: `PID`, `HHID`, `AGE_RANGE`, and the 20 RF features; all 10,962 rows |

The generation cell is at the end of `notebooks/01.ipynb`. It is self-contained and idempotent: it can run after a fresh kernel, and running it again overwrites the same file with identical contents. The CSV keeps the raw category strings, including `"Não aplicável"` and `"Sem resposta"`; those are survey answers, not missing values.

A good first plot is `AGE_RANGE` vs `HA1D` with the not-applicable level shown separately. It makes the 55+ routing visible immediately. The second useful plot is `AGE_RANGE` vs `PA7_NAC` split by `SEX`; it shows why the female-only questions are really a sex indicator plus a screening response.

# Feature selection for `AGE_RANGE` — stage report

**Prepared:** 2026-11-04
**Run:** `c60f8f8`, working tree clean (`git_dirty: false`)
**Notebook:** `notebooks/04_feature_selection.ipynb`
**Executed record:** `outputs/04_feature_selection_executed.ipynb`
**Artifacts:** `outputs/selected_features.json`, `outputs/feature_selection_k_sweep.csv`,
`outputs/feature_selection_ranking.csv`, `outputs/feature_selection_ranking_deployed.csv`

---

## 1. Purpose and audience

This is an **internal document**. It is written for the people steering this
project, not for the assessment. It is not a section of the 20-page report and it
is not written to a rubric. Its job is to let a reader who knows the course
material but not the extra material follow what this stage did, check that it was
done honestly, and decide what to do next.

The stage asked one question:

> Of the 255 input columns the model currently uses, how many can be removed
> before macro F1 drops?

The answer is **40 can be removed**, leaving 40 of the 136 screened columns
(255 raw inputs expand to 1,170 columns after one-hot encoding). That answer
comes with three qualifications that matter more than the number itself, and
they are the substance of this document:

1. The aggregate score hides a per-class reshuffle.
2. The best-scoring option was **rejected by a rule**, on a margin of 0.0002.
3. Once you account for the cost of *choosing* the columns, the honest estimate
   is **worse than using everything**.

Section 3 is a guided tour of the eleven concepts this stage uses that the course
does not teach. Each concept is introduced at the point in the method where it is
needed, in the order the notebook runs. A reader who follows section 3 can read
sections 4–6 without further background.

### A note on the age bands

Every age interpretation in this document uses the official Kaggle mapping
(`0 = 15–17, 1 = 18–19, …`). The data is ambiguous about band 0 and the question
is open with the professor; see `TODO.md`. Nothing in this stage depends on the
mapping, because the ranking treats labels as codes.

---

## 2. Executive summary

| # | Finding | Evidence | Confidence |
|---|---|---|---|
| 1 | 40 columns match the 255-column baseline **in aggregate**, but the per-class effect is a reshuffle: 6 classes improve, 10 get worse. | 0.4519 ± 0.0104 vs 0.4531 ± 0.0096; mean per-class delta −0.0098 | High |
| 2 | **K = 60 is the best mean and the rule rejected it** on a 0.0002 miss. The rule, not the data, chose 40. | 0.4587, worst repeat −0.0102 against a −0.010 guard | High |
| 3 | The Stage 3 check earned its cost: the ordinal ranking had `PC1A` 33rd, and it would have been dropped. | Spearman 0.492; `PC1A` rank 33 → 1 | High |
| 4 | Selection optimism is **+0.0137**, and the shortlist is **unstable**: 26–40 of the 40 columns change per fold. | Stage 4 0.4519 vs Stage 5 0.4382 ± 0.0133 | High |
| 5 | **48 of 136** screened columns have negative permutation importance — they actively hurt. | `feature_selection_ranking_deployed.csv` | High |
| 6 | Importance concentration does not predict sufficiency: the top 10 carry 84% of importance, yet K = 20 fails by 1.6 points. | K sweep | High |

**The one-line answer.** 40 columns preserve the baseline's cross-validated score
within noise, but selection optimism of +0.0137 puts the honest estimate at
**~0.438**, which is **below** the 255-column baseline of 0.4531. Whether that
gap is real is exactly what the two Kaggle scores will settle, because the test
set never saw the selection. **Those scores are not in yet; this document is
marked pending on that point.**

---

## 3. Guided tour: the material the course does not cover

The course covers feature selection rationale and wrapper/embedded methods
(Aula2b 20–22), label vs one-hot encoding (Aula2b 24), permutation importance and
MDI (PL_7 26–33), k-fold cross-validation (PL_2 39–48), and precision/recall/F1
(PL_2 14–27). Everything below is **not** in those decks. Each entry says what it
is, why this stage needed it, and the trap it sets.

### 3.1 Macro F1

**What it is.** Plain F1 (PL_2 slide 26) is defined for two classes. With 16
classes you must choose how to combine 16 per-class F1 scores. **Macro** F1 is
the unweighted mean of the 16 per-class scores. **Weighted** F1 weights each
class by its support. **Micro** F1 is just accuracy for single-label problems.

**Why it matters here.** The competition scores macro F1, so every number in this
document is macro F1. The difference is not cosmetic: band 0 has 207 rows (1.9%),
band 10 has 1,062. Under macro averaging the 207-row band counts as much as the
1,062-row band, so a model that is good on the big bands and bad on the rare ones
scores far worse under macro than under weighted or micro.

**The trap.** This is why the archived `data_v1` comparison is not comparable to
this one. That archive measured **accuracy** and reported that top-20 lost 4.3
percentage points. Macro F1 weights the rare bands more heavily, so the same
feature cut costs more under macro F1. Never put the 4.3-point figure next to a
macro-F1 number.

### 3.2 Repeated cross-validation

**What it is.** The course teaches one pass of k-fold: partition the data into k
folds, train k times, average the k scores. **Repeated** CV does that whole
procedure R times with different random partitions. Here R = 3 and k = 5, giving
**15 folds**.

**Why it matters here.** A single 5-fold pass gives you 5 numbers. Their standard
deviation is an estimate of how much the score moves with the *split*, and with 5
numbers that estimate is itself very noisy. Repeating gives 15 numbers and a
steadier estimate. Every "±" in this document is the standard deviation across
those 15 folds.

**The trap.** The 15 folds are **not independent**. The three repeats score the
same rows, just partitioned differently. So the standard deviation across folds
is a **descriptive** spread, not a confidence interval. Do not compute
"mean ± 2 SE" from it and do not treat it as a significance test. This is stated
in the notebook for exactly this reason.

### 3.3 Paired comparison on identical folds

**What it is.** When comparing two models, compare them **fold by fold on the
same rows**, not by comparing two means. For fold *i*, compute
`baseline_i − candidate_i`; then look at the distribution of those differences.

**Why it matters here.** Two models scored on the same fold share the fold's
difficulty. Fold 2 is hard for both; fold 4 is easy for both. Differencing
cancels that shared component, so the paired difference is much less noisy than
the two raw scores. The K sweep's `paired_difference_mean` column is exactly
this: baseline minus K, per fold, then averaged.

**The trap.** A paired difference near zero with a large spread is *not* evidence
of equality. It is evidence of **no detectable difference at this sample size**.
The distinction matters for finding 1: 40 columns were not shown to be as good as
255, only not shown to be worse.

### 3.4 Pre-registration of a decision rule

**What it is.** Write the decision rule down **before** seeing the result, then
apply it mechanically. It is standard practice in research and it is not a
course topic.

**Why it matters here.** The tempting failure mode is to run the sweep, see that
K = 60 scores best, and then define "within noise" so that 60 passes. That would
be choosing the rule to fit the answer. Two rules in this stage were fixed in
advance and are quoted verbatim in section 4.

**The trap.** Pre-registration does not make the rule *correct*; it makes it
**honest**. A badly chosen rule applied mechanically is still a badly chosen
rule. Finding 2 is exactly this case: the rule was followed, and the rule is
arguably too sharp. Both statements belong in the record.

### 3.5 Selection leakage and nested selection

**What it is.** This is the most important idea in the stage and the course does
not state it. Aula2b slide 20 says feature selection "reduces overfitting" and
stops there. It never says that **the act of selecting features using labels
that later appear in a validation fold inflates the score you measure**.

The mechanism: Stages 1 and 2 look at the labels of **every row** to decide which
columns are important. With 5-fold CV, every row is in some validation fold. So
every row's label influenced the choice of columns, and then those same rows were
used to score the model. The score is optimistic.

**Why it matters here.** The frozen folds protect the *preprocessor* from
leaking — the scaler and encoder are fitted inside each fold. They do **not**
protect you from looking at validation scores while choosing columns. That is a
different leak, and it is invisible unless you go looking for it.

**The fix — nested selection.** Run the *entire* selection procedure inside each
outer fold:

1. Take the outer fold's training rows only.
2. Carve a 20% inner holdout out of them.
3. Run the mutual-information screen on the remaining 80%, fit the ranker there,
   and rank by permutation **on the holdout** — so the permutation is
   out-of-sample, exactly like Stage 2.
4. Take the top K, fit the final model on the outer training rows, score on the
   outer validation fold.

**The trap.** If you fit the ranker on all outer training rows and then permute
on a subset of them, the permutation is **in-sample**. That inflates
high-cardinality columns and contaminates the gap you are trying to measure. The
notebook uses a genuine holdout for this reason.

### 3.6 Mutual information with a shuffle bias correction

**What it is.** Mutual information (MI) measures how much knowing a column
reduces uncertainty about the label. It detects non-linear dependence, unlike
correlation. The course offers the **chi-square** test for feature/target
dependence (Aula2b slide 21); MI is a different and more general tool.

**Why it matters here.** Stage 1 uses MI as a cheap first screen, cutting 255
inputs down to 136 candidate columns before the expensive permutation ranking.

**The trap — MI is biased upward by cardinality.** A column with many distinct
values can appear informative by chance alone. The fix used here is to **shuffle
the labels**, recompute MI, and subtract the shuffled value. What survives is
signal above what random association produces. Without this correction MI
systematically favours high-cardinality columns.

### 3.7 Permutation importance on macro F1

**What it is.** PL_7 slides 30–31 introduce permutation importance: shuffle one
column, measure how much the score drops. The course demonstrates it on
**accuracy**. Here the scorer is **macro F1**.

**Why it matters here.** The importance of a column is defined *relative to the
metric*. Shuffling a column that matters only for the rare bands barely moves
accuracy but moves macro F1 a lot. So this ranking answers "which columns matter
for macro F1", which is the competition metric.

**The trap — negative values are meaningful.** A column whose importance is
negative *improved* the score when shuffled, i.e. it was actively hurting on the
validation folds. This is not a bug. **48 of the 136 screened columns have
negative importance** (finding 5). They are real evidence that those columns
carry noise the model was fitting.

### 3.8 Spearman rank correlation

**What it is.** A correlation between two **rankings**, not two values. It asks:
when ranking A puts a column high, does ranking B also put it high? It ranges
from −1 to +1 and ignores the magnitudes, only the order.

**Why it matters here.** Stage 3 compares two rankings of the same 20 columns
produced by two different pipelines. Pearson correlation would be wrong — the
importance values have wildly different scales (0.005 vs 0.174). Spearman only
cares about order, which is what "do these two rankings agree" means.

**The trap.** Spearman 0.492 says the rankings **disagree**. It does not say
which one is right. There is no ground truth for "the correct importance"; the
only external check is the competition score.

### 3.9 Ordinal-encoding bias in tree importances

**What it is.** PL_7 says MDI is biased toward high-cardinality features. This is
a **different and sharper** effect: a **nominal** column (no natural order) that
is label-encoded gets an arbitrary numeric order. A tree must then make several
splits to isolate a category that one-hot encoding would isolate in one.

**Why it matters here.** This is what `PC1A` demonstrated. Under ordinal codes
the random forest ranked it **33rd** with importance 0.005. Under the deployed
one-hot pipeline it ranks **1st** with importance 0.174 — a factor of 35. A
3-level nominal needs several splits under ordinal codes and one under one-hot,
so the ordinal lens systematically underrates it.

**The trap.** Stage 3 changes **two things at once** — the model (RF →
`hist_gb`) and the representation (ordinal → one-hot) — so the 0.492 cannot be
blamed on the encoding alone. With n = 20 the error bars are wide. The
conclusion is qualitative: the choice of representation changed the answer.

### 3.10 `HistGradientBoostingClassifier` and `passthrough`

**What it is.** The course covers boosting and XGBoost (Aula5 slides 8–10).
This stage uses scikit-learn's `HistGradientBoostingClassifier`, which
histogram-bins continuous features for speed and handles missing values natively.
`passthrough` means the preprocessor passes raw columns through untouched,
letting the model do its own splitting rather than receiving pre-encoded columns.

**Why it matters here.** It is the model the group actually deploys, so the
ranking must be produced by *this* model on *this* representation. Ranking with
any other configuration measures that configuration, not the deployed one.

**The trap.** Native categorical handling means the model can see a nominal
column as a category, not a number — which is precisely why it disagrees with the
ordinal RF. The disagreement is a property of the pipeline, not an error in
either ranking.

### 3.11 Fingerprints and provenance

**What it is.** A fingerprint is a hash of a set of column names (and of the
target values). Provenance is the record of what produced an artifact: git SHA,
whether the tree was dirty, the seed, the fold specification.

**Why it matters here.** `selected_features.json` stores both. The loader
re-checks the fingerprint when the file is read: if the shortlist no longer
matches the data, it **fails loudly** rather than silently dropping a column.
A shortlist that has quietly stopped fitting the data is worse than no shortlist.

**The trap.** A fingerprint protects against *drift*, not against *error*. It
confirms the columns are the ones that were chosen; it says nothing about whether
choosing them was a good idea.

---

## 4. Method as executed

### 4.1 Stages

| Stage | What it does | Cost |
|---|---|---|
| 0 | Baseline: `hist_gb` on all 255 inputs, 15 folds | ~5 min |
| 1 | Mutual-information screen with shuffle correction: 255 → 136 columns | fast |
| 2 | Permutation ranking of the 136, ordinal-coded RF, 5 repeats | ~28 min |
| 3 | Representation consistency check: re-rank the top 20 with the deployed pipeline | ~5 min |
| 3b | **Fallback, ran here:** re-rank all 136 with `hist_gb` + `passthrough` | ~30 min |
| 4 | K sweep over {10, 15, 20, 30, 40, 60, 80}, one model, paired against baseline | ~20 min |
| 5 | Nested selection: the whole procedure redone inside each outer fold | ~22 min |
| 6 | Persistence with fingerprints | fast |
| 7 | Two submissions: selected K=40 and all features | ~15 min |

Total wall time was about 2 h 20 min, including the Stage 3 fallback.

### 4.2 The two pre-registered rules, verbatim

**Stage 3 branch** (fixed before the run; quoted from the notebook):

> If the correlation is high (> 0.7), the ordinal lens is fine and the shortlist
> is built from the Stage 2 ranking. If it is low, the shortlist is built from
> the representation the model actually uses: the whole screened set is re-ranked
> with `hist_gb` + `passthrough`.
>
> This is a pre-registered branch, not a choice made after seeing the score. The
> only thing the number decides is *which ranking was already declared to win*.

**Stage 4 pass rule** (fixed before the run; quoted from the notebook):

> K passes if the mean paired difference over the folds is **at most 0.005**
> and no single repeat shows a loss worse than 0.010.

**Pre-registered fallback:** if no K passed, not even K = 80, the shortlist would
be the full input set and the report would say selection did not survive
validation. This did not trigger.

### 4.3 Why Stage 5 mirrors the Stage 3 winner

Stage 5 uses the **same ranker that produced the shortlist**: the deployed
`hist_gb` pipeline, because Stage 3 failed. If it had used the ordinal RF
instead, the gap would measure *disagreement between two rankers* rather than
*selection optimism*. Mirroring keeps the measurement clean.

---

## 5. Results

### 5.1 Baseline and the K sweep

Baseline: `hist_gb` on all 255 inputs, **0.4531 ± 0.0096**
(run `20261103T225840834739_hist-gb-all-recoded`).

Paired difference is **baseline minus K**, so a positive number is a loss.

| K | macro F1 | std | paired diff | worst repeat | passes |
|---:|---:|---:|---:|---:|:--:|
| 10 | 0.4139 | 0.0096 | +0.0392 | +0.0347 | no |
| 15 | 0.4294 | 0.0093 | +0.0237 | +0.0199 | no |
| 20 | 0.4367 | 0.0094 | +0.0164 | +0.0100 | no |
| 30 | 0.4403 | 0.0104 | +0.0128 | +0.0098 | no |
| **40** | **0.4519** | **0.0104** | **+0.0012** | **−0.0049** | **yes** |
| 60 | **0.4587** | 0.0072 | −0.0056 | **−0.0102** | no |
| 80 | 0.4574 | 0.0090 | −0.0044 | −0.0070 | yes |

**K = 40 is the smallest set that passes the rule.**

### 5.2 The Stage 3 check

Spearman correlation between the ordinal-RF and deployed rankings: **0.492**.
Below 0.7, so the fallback ran and the shortlist comes from the deployed ranking
(`ranking_source: stage3_deployed_hist_gb`).

The single most consequential disagreement:

| column | ordinal rank | deployed rank | ordinal importance | deployed importance |
|---|---:|---:|---:|---:|
| `PC1A` | 33 | **1** | 0.005 | **0.174** |

### 5.3 The nested check

| fold | K | macro F1 | columns differing from Stage 4 |
|---:|---:|---:|---:|
| 0 | 40 | 0.4498 | 26 |
| 1 | 40 | 0.4278 | 40 |
| 2 | 40 | 0.4320 | 34 |
| 3 | 40 | 0.4551 | 38 |
| 4 | 40 | 0.4262 | 38 |

Stage 5 mean: **0.4382 ± 0.0133**. Stage 4 for the same K: 0.4519.
**Selection optimism: +0.0137.**

### 5.4 Per-class behaviour, K=40 against the baseline

| band | baseline F1 | K=40 F1 | delta | support |
|---:|---:|---:|---:|---:|
| 0 | 0.857 | 0.829 | −0.028 | 207 |
| 1 | 0.308 | **0.350** | **+0.041** | 172 |
| 2 | 0.601 | 0.585 | −0.016 | 399 |
| 3 | 0.373 | **0.388** | **+0.015** | 377 |
| 4 | 0.259 | **0.277** | **+0.018** | 449 |
| 5 | 0.347 | 0.344 | −0.003 | 647 |
| 6 | 0.358 | 0.340 | −0.018 | 859 |
| 7 | 0.389 | **0.401** | **+0.013** | 862 |
| 8 | 0.634 | 0.621 | −0.013 | 905 |
| 9 | 0.661 | 0.635 | −0.026 | 1014 |
| 10 | 0.645 | 0.619 | −0.026 | 1062 |
| 11 | 0.514 | **0.516** | **+0.002** | 1040 |
| 12 | 0.380 | 0.353 | −0.027 | 1000 |
| 13 | 0.280 | 0.252 | −0.028 | 823 |
| 14 | 0.313 | **0.314** | **+0.001** | 650 |
| 15 | 0.449 | 0.387 | **−0.063** | 496 |

**6 classes improve, 10 get worse, mean delta −0.0098.**

### 5.5 The shortlist

40 columns, in deployed rank order:

`PC1A`, `MAINSTAT`, `HH_INACT`, `MARSTALEGAL`, `HATLEVEL`, `PARTNERS`,
`HH_ACT`, `PA7_NAC`, `HHTYPE`, `HHNBPERS_0_13_COD`, `IN45`, `HHNBPERS_COD`,
`JOBSTAT_COD`, `NT1`, `IN47`, `BM2`, `PA1`, `IN63`, `HA2`, `HHINCOME`,
`FT_PT`, `BM1`, `IN65`, `PA5`, `PE9`, `PL9_tot`, `IN4`, `IN24`, `IN28`,
`IN15`, `MD1`, `SK2`, `IN16`, `PC2`, `DH6`, `HA1G`, `JOBISCO_COD`, `PL5`,
`AM3`, `IN22`

By theme:

- **Labour status and job** — `MAINSTAT`, `JOBSTAT_COD`, `JOBISCO_COD`,
  `FT_PT`, `HH_ACT`, `HH_INACT`
- **Education** — `HATLEVEL`, `PE9`, `NT1`
- **Household structure** — `HHTYPE`, `HHNBPERS_COD`, `HHNBPERS_0_13_COD`,
  `PARTNERS`, `MARSTALEGAL`
- **Income and wealth** — `HHINCOME`, `PL9_tot`, `PL5`, `MD1`
- **Health and disability** — `HA2`, `HA1G`, `DH6`
- **Migration and nationality** — `PA7_NAC`, `PA1`, `PA5`, `BM1`, `BM2`
- **Region** — `PC1A`, `PC2`
- **Attitudes and opinions** — 10 of the 40 are `IN*` columns

**Blocks never selected:** `CD*` (9 columns), `AL*` (4), `HS*` (3), `AW*` (2),
`HO*` (2), `PN*` (2), `UN*` (2), `LOCNACE` (1). These carry no permutation
signal for this target.

---

## 6. Interpretation

### 6.1 Forty columns match the baseline in aggregate, but reshuffle per class

The aggregate difference is +0.0012, about an eighth of a standard deviation. The
honest reading is **"no detectable loss"**, not "40 columns are as good as 255".

But the per-class table shows this is not a neutral cut. Six classes improve —
mostly the sparse middle bands (1, 3, 4, 7) — and ten get worse, including the
well-populated bands 9, 10 and 12 and, worst of all, **band 15 at −0.063**. The
40-column model is not the 255-column model with less noise; it is a **different**
model that trades some classes for others and happens to land at the same mean.

That matters for the project's stated goal. The assignment asks the model's
predictions to be used to detect profile anomalies, which needs the middle bands
to be well separated. A cut that improves bands 1 and 4 while losing band 15 is
not obviously the right trade for that purpose, even though macro F1 cannot tell
the difference.

### 6.2 K = 60 is the best mean and the rule rejected it

K = 60 scores **0.4587**, better than the baseline on the mean and better than
K = 40. It fails the guard by **0.0002** (−0.0102 against a −0.010 threshold).

Two things are true at once and both belong in the record:

- **The rule was fixed in advance and was followed.** That is the point of
  pre-registration, and the guard did exactly what it was designed to do.
- **The guard is a knife-edge at this sample size.** A 0.0002 miss is noise, not
  evidence of instability. K = 60 is *plausibly* the better operating point, and
  the rule — not the data — is what chose 40.

The sweep also shows **K = 80 passing** (0.4574, diff −0.0044) while K = 60
fails. That non-monotonicity is a warning: the guard is partly measuring fold
noise, not only model quality. A rule that rejects 60 but accepts 80 is not
tracking a real threshold.

**This is a decision for the humans, not a settled result.** Options are set out
in section 7.

### 6.3 The Stage 3 check earned its cost

The ordinal RF ranked `PC1A` 33rd; the deployed pipeline ranks it 1st, with 35×
the importance. Had the check passed, `PC1A` would have been **dropped from the
shortlist** — the single most important column under the model the group ships.

The methodological lesson is the reusable part: **rank features with the
representation the model uses, or your ranking measures your encoding, not your
data.** The ordinal lens was not wrong in a way that showed up as an error; it
was wrong in a way that silently reordered the answer.

Caveat: the check changes model and representation together, so 0.492 cannot be
attributed to encoding alone, and n = 20 is a wide error bar.

### 6.4 Selection optimism is +0.0137, and the shortlist is unstable

Stage 4 says 0.4519. Stage 5 — the same procedure with selection redone inside
each fold — says **0.4382**. Roughly **1.4 points of the apparent score were
bought by looking at the labels**.

The per-fold column counts are the sharper finding: **26 to 40 of the 40 columns
change in every fold, and in one fold all 40 change.** The exact shortlist is not
a stable object. What Stage 5 measures is the **procedure**, not the list.

**The uncomfortable consequence:** the honest estimate of ~0.438 is **below** the
255-column baseline of 0.4531. On this evidence the defensible claim is not
"40 columns match the baseline" but:

> 40 columns match the baseline's cross-validated score, but once the cost of
> choosing them is accounted for, the procedure may be slightly worse than using
> everything.

Whether that gap is real is unresolved. The cross-validated estimate is
optimistic for the selected set and not for the baseline, so the two numbers are
not directly comparable — but the test set never saw the selection, so the two
Kaggle scores are the clean comparison. **They are pending.**

### 6.5 Forty-eight columns actively hurt

48 of 136 screened columns have **negative** permutation importance: shuffling
them *improved* macro F1 on the validation folds. This is stronger than "the rest
are unimportant" — it says the model was fitting noise through those columns.

This is direct evidence for the course's own claim (Aula2b slide 20) that
removing unimportant features reduces overfitting. It is also a lead: a model
trained on the 88 non-negative columns might beat both the 40 and the 255.

### 6.6 Importance concentration does not predict sufficiency

The top 10 columns carry **84%** of total importance, and the top 20 carry
**95%** — yet K = 20 fails the rule by 1.6 points. Importance concentration and
model sufficiency are different questions. A column can matter greatly and still
be redundant with another: permutation importance splits credit among correlated
columns, so a column can look weak while its group is essential.

This is the same warning the notebook records about `HA1*`/`PC1*`: their plain
"not applicable" level is 100% for under-55 and 0% for 55+, so one split at 55
captures all of them. They are **one age-eligibility signal, not thirteen health
findings**. Report themes, not thirteen discoveries.

---

## 7. Decisions for the humans

These are open. This stage informs them; it does not settle them.

**D1. Which K to ship: 40 or 60?**
The rule chose 40. The mean favours 60. A defensible middle path is to submit
both to Kaggle and let the test set decide, which costs two of the three daily
submissions. **Recommendation:** submit K=40 and the all-features model first
(both already built), then K=60 if the daily budget allows.

**D2. Is the selection worth keeping at all?**
If the Kaggle score for K=40 is not at least as good as all-features, the
honest conclusion is that selection did not survive validation, and the group
should ship the full model. This was the pre-registered fallback for the *rule*;
it is now a live question for the *test set*.

**D3. Should the guard be changed?**
The −0.010 per-repeat guard rejected 60 and accepted 80. If the group wants to
keep using a guard, it should be re-derived — for example from the observed
repeat-to-repeat spread rather than a round number. Changing it now would be
post-hoc, so any change must be declared for the *next* stage, not retrofitted.

**D4. What to do with the 48 negative-importance columns?**
A run on the 88 non-negative columns is cheap and is a genuine hypothesis, not a
fishing expedition, because the sign was measured out-of-sample on the
validation folds.

**D5. The `IN*` attitude columns.**
10 of the 40 selected columns are attitudes and opinions. They rank high under
`hist_gb` and low under the ordinal RF. That is either real signal or the same
representation artifact. **Do not build a narrative on them** until someone
checks what those columns mean and whether the effect survives a different
representation.

---

## 8. Limitations

1. **The Kaggle scores are pending.** The central claim in 6.4 is unresolved
   without them. Everything else is cross-validated evidence only.
2. **The nested check uses 5 outer folds and one inner split per fold.** The
   optimism estimate is itself noisy; ±0.0133 across folds.
3. **The 15 folds are not independent**, so no significance test is computed and
   none should be.
4. **The Stage 3 check is confounded** by design (model and representation
   change together) and uses n = 20.
5. **The band-0 mapping is disputed** (`TODO.md`). No result here depends on it,
   because labels are treated as codes, but no age interpretation should be
   published without the caveat.
6. **`reports/01`'s stored outputs are from `data_v1`** and measured accuracy,
   not macro F1. They are not comparable and are not used here.
7. **Permutation importance splits credit among correlated columns.** A single
   member of a correlated group can look weak while the group is essential. This
   is why section 6.6 reports themes.
8. **Nothing here is causal.** `HATLEVEL` ranking high does not mean education
   causes age band; both reflect life stage.

---

## 9. Reproduction

```bash
# From the repo root, on c60f8f8 with a clean tree:
.venv/bin/jupyter nbconvert --to notebook --execute \
  notebooks/04_feature_selection.ipynb \
  --output-dir /tmp/opencode --output 04_executed.ipynb \
  --ExecutePreprocessor.timeout=14400
```

The tracked notebook runs in smoke mode by default (`SMOKE = True`, ~14 min,
numbers meaningless). For the real run, set `SMOKE = False` in a copy and execute
that; the artifacts land in `outputs/` regardless, because the paths come from
`daa.config`.

Artifacts written:

| file | contents |
|---|---|
| `outputs/selected_features.json` | the 40-column shortlist, fingerprints, provenance |
| `outputs/feature_selection_k_sweep.csv` | K, macro F1, paired diff, pass flag, run id |
| `outputs/feature_selection_ranking.csv` | Stage 2 ordinal-RF ranking, all 136 |
| `outputs/feature_selection_ranking_deployed.csv` | deployed `hist_gb` ranking, all 136 |
| `outputs/submission_selected_k40.csv` | submission, K=40 |
| `outputs/submission_all_features.csv` | submission, all 255 inputs |

Both submissions validated: 3,655 rows, `RowId` 0…3654, 16 classes.

**Pending:** upload both to Kaggle and record the two macro-F1 scores. That is
the only remaining evidence needed to close section 6.4.

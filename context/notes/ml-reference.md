<!--
Distilled and corrected from the DAA lectures. Standard-reference additions are
marked [ref]; heuristics are marked "rule of thumb". The full audit trail, including
every drop and correction, is in notas-audit.md.
Source: context.bak-2026-10-30/notas.md (Portuguese). This file is English and is
study material, not a requirement.
-->

# Machine-learning reference

## 1. Variable types

- Numeric: discrete vs. continuous. Categorical: nominal vs. ordinal.
- The cost of a wrong prediction follows the type: nominal is right/wrong (binary
  error); ordinal **can be** penalised by a rank distance (e.g. `|i−j|`), but the
  choice of loss is a modelling decision. `[ref]`
- A column's storage type is not its modelling type — the meaning you assign decides.

## 2. Descriptive statistics

- Mean is pulled by outliers; median is robust. Report the median when the
  distribution is skewed or outlier-heavy.
- Variance measures dispersion, in squared units. Standard deviation is its square
  root, back on the data's scale.
- Percentiles describe the shape of a distribution; the 50th percentile is the
  median. Q1–Q3 holds the middle 50% (the IQR).

## 3. Distribution shape

- Some algorithms assume (approximate) normality; check before choosing them.
- Normal reference: ±1σ ≈ 68.3% of mass (34.1% per side), ±2σ ≈ 95.4%,
  ±3σ ≈ 99.7%. `[ref]`
- Skewness and excess kurtosis are ~0 for a normal distribution.
  - Skewness > 0 → longer **right** tail; skewness < 0 → longer left tail.
  - Excess kurtosis > 0 → peaked, heavy tails (leptokurtic); < 0 → flat, light tails
    (platykurtic).

## 4. Association

- Covariance: `cov(X,Y) = mean((x−μx)(y−μy))` over the population; divide by `n−1`
  for the sample version. It is the mean of the co-deviations, not of the variances.
- Covariance carries units, so its magnitude is not comparable across variables;
  its sign is interpretable. Correlation normalises it:
  `r = cov(X,Y)/(σx·σy)`, bounded in [−1, 1].
- Correlation does not imply causation. A third variable can drive both: ice-cream
  sales and crime both rise with temperature.
- For categorical data use contingency tables with χ²/Cramér's V — not Pearson on
  category codes. `reports/01` found perfect associations (V = 1.000) that Pearson
  scored as 0.736 and 0.682.
- Dimensionality reduction is essential for visualising high-dimensional data.

## 5. Learning paradigms

- Symbolic vs. subsymbolic: whether a human can read the decision rule off the model.
- Supervised vs. unsupervised: without `y` there is no direct mapping to the inputs,
  so results are harder to interpret.
- Reinforcement learning is a third paradigm — neither supervised nor unsupervised;
  it learns from a reward signal through interaction.
- Concept drift: models degrade as the world moves away from the training
  distribution, so monitoring is part of the lifecycle.

## 6. Process

- Domain understanding → read the data → prepare → split → fit → tune → deploy →
  monitor.
- **Rule of thumb:** skipping domain understanding is a common failure. *(Heuristic,
  not a measured fact.)*
- Establish an end-to-end baseline **before** deep data work, so every later change
  has a reference point.
- Survivorship bias (Wald's returning aircraft): the missing data carries the
  information — look where the survivors have no damage.
- Evidence of generalisation comes from cross-validation spread (mean ± std), not
  from a single split.
- **Rule of thumb:** tree ensembles (random forest = bagged trees) are a strong
  default on tabular data. `[ref]`

## 7. Data quality and preparation

- Check for missing values, duplicates, noise, outliers. Detect via central tendency,
  dispersion, distribution, correlation/contingency tables and plots. **Rule of
  thumb:** a histogram is a common first plot. *(Heuristic.)*
- Pair plots show pairwise relationships and are useful for a first multivariate
  look. *(Plot kept; "super power" claim dropped.)*
- Scaling: irrelevant for scale-invariant models (decision trees); required for
  distance- and gradient-based ones.
  - Min-max → [0, 1]; sensitive to outliers.
  - Standardisation → mean 0, sd 1; **rule of thumb:** the safer default when the
    distribution is skewed or outlier-heavy. `[ref]`
- Outliers: detect with box/whisker plots, then drop or cap.
- Missing data: drop rows/columns, mean (shrinks variance), median (skewed data),
  interpolation, or an explicit "missing" mask. The choice depends on why it is
  missing.
- Encoding nominals: one-hot (adds columns), binary, label encoding (implies an
  order — unsafe for nominals in linear models), binning (groups continuous values
  that mean the same thing in context, and removes noise).
- Feature selection: drop invariant columns, near-zero-variance columns and highly
  correlated/redundant ones; then PCA (unsupervised), wrapper methods (search subsets
  by model score) or embedded methods (selection during training: L1, tree
  importance).
- Shuffle/stratify before splitting — file ordering can leak structure into train or
  test.

## 8. Linear models

- Verify linearity first: `f(x+y) = f(x)+f(y)` and `f(ax) = a·f(x)`.
- Classification: a line separates the classes. Regression: values lie near a line.
- Logistic regression is a classifier despite the name; it models `P(y=1|x)`. For
  more than two classes use multinomial (softmax) or one-vs-rest. `[ref]`
- Parameter estimation: least squares (closed form, expensive in the number of
  features) or gradient descent (needs a learning rate).
- Batch gradient descent is slow per step with many rows; stochastic and mini-batch
  are much faster per step but noisier.
- Regularisation: Lasso (L1) drives coefficients to exactly zero → automatic feature
  selection; Ridge (L2) shrinks but never to zero. `[ref]`

## 9. Decision trees

- Good first baseline; invariant to feature scale and monotone transforms, so little
  preprocessing is needed.
- Split quality comes from impurity (Gini/entropy). Feature importance is the
  weighted impurity decrease; important attributes **tend to** sit nearer the root,
  but a feature can accumulate importance from many deep splits. `[ref]`
- Unstable: adding data can change the structure. Infinite depth = overfitting.
- Stopping: `min_samples_split`, `min_samples_leaf`, `max_depth` (pre-pruning); stop
  when validation score stops improving; or post-pruning with cost-complexity and a
  tuned `α` (`ccp_alpha` in scikit-learn). `[ref]`

## 10. Ensembles

- Bagging: train the same model on bootstrap samples (n draws with replacement), in
  parallel, then aggregate. Random forest = bagging + feature subsampling per split.
  `[ref]`
- Boosting: sequential; each learner targets the errors of the previous ones.
  - AdaBoost reweights misclassified samples.
  - Gradient boosting fits each new learner to the negative gradient
    (pseudo-residuals). XGBoost is a regularised gradient-boosting implementation
    with second-order information; its hyperparameters are tuned by the user, not
    automatically. `[ref]`
- Stacking: a meta-learner trained on the base models' outputs. Blending is the
  holdout variant, avoiding repeated reuse of the same rows.
- Voting: hard (majority) or soft (averaged probabilities).
- Bagging/boosting are usually tree-based; stacking is the one that mixes
  heterogeneous model families.
- `n_estimators`: number of trees in a forest, or boosting rounds in gradient
  boosting. More trees reduce variance with diminishing returns, at higher compute
  cost; it is not a parameter to over-tune once the score has plateaued. `[ref]`

## 11. Neural networks

- Approximate arbitrary functions and generalise well, but they can fit noise — that
  is a tuning problem, not a property.
- A vanilla network has one hidden layer. `[ref]`
- Output activation follows the target: sigmoid for binary, softmax for multiclass,
  linear for regression (ReLU for non-negative regression outputs); ReLU in hidden
  layers. `[ref]`
- Feed-forward (no recurrence) + backpropagation is the standard workflow.
- Vanishing gradients: backpropagation can degenerate in deep networks.

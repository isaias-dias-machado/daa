<!--
Index of the 13 slide decks under context/slides/. Page counts measured with pdfinfo
on 2026-10-30; total 411. Topics are page ranges read from each deck's contents page
and slide titles. This is a map, not a substitute for the slides.
-->

# Slides index

## How to use

PDFs are binary. Text extraction loses diagrams, figures, and layout — for example,
`Aula5` slide 3 extracts to a bare title. Use this index to choose a deck and page
range, then extract only those pages:

```bash
pdftotext -layout -f N -l M "context/slides/<deck>.pdf" -
```

Read the actual slides for anything visual.

## Labels

| Label | Meaning |
|---|---|
| `direct` | Covers a method used in this project. |
| `background` | Concepts and vocabulary that support the project. |
| `optional` | May help, but is not required. |
| `ignore` | No project content. |

9 of the 13 decks are `direct`.

## Decks

| Deck | Real title | Pages | Topics (pages) | Use |
|---|---|---:|---|---|
| `Aula 0 - DAA 2627.pdf` | Course presentation / opening | 14 | Course overview and data (1–14) | ignore |
| `Aula2a - DAA 2627.pdf` | Conceitos e metodologias | 20 | Technology trends and ML vs Data Science (2–5); learning concepts and paradigms (5–14); SEMMA, CRISP-DM and pipeline (15–19) | background |
| `Aula2b - DAA 2627.pdf` | Data Exploration and Preparation | 28 | Data quality: survivorship, missing values, noise (3–6); exploration, contingency tables, correlation, visualisation (7–13); basic preparation (14–15); scaling (17–18), outliers (19), feature selection/PCA (20–22), missing values (23), nominal discretisation (24), binning (25), feature engineering (26–27) | direct |
| `Aula3 - DAA 2627_RL.pdf` | **Supervised Learning: Linear and Logistic Regression** (the filename says RL; the content is linear/logistic regression) | 23 | Linear models and linear regression (3–8); logistic regression, sigmoid, multiclass (9–12); least squares and gradient descent (13–19); overfitting, regularisation, standardisation (20–22) | direct |
| `Aula4 - DAA 2627_DT.pdf` | Decision Trees | 20 | Classification trees, Gini/entropy and continuous attributes (3–8); regression trees (9–14); pruning (15–17); ID3 and conclusions (18–19) | direct |
| `Aula5 - DAA 2627_Ensembles.pdf` | Ensemble Learning | 15 | Ensemble classes (3); voting and averaging (4–5); bagging and random forest (6–7); boosting, AdaBoost and XGBoost (8–10); stacking and blending (11–13); summary (14) | direct |
| `Aula6 -DAA 2627_ANN.pdf` | Artificial Neural Networks | 37 | Overfitting context (2–3); motivation and applications (4–7); neurons and activations (8–10); topologies and feedforward (11–17); training and backpropagation (18–22); topology/hyperparameters (23–25); overfitting and learning curves (26–35); strengths, weaknesses, frameworks (36–37) | optional |
| `DAA_PL_1(1).pdf` | Introduction to Data Science Concepts (Part I) | 45 | Data types (3–8); mean/median/mode (9–13); variance and standard deviation (14–17); probability density functions (18–20); percentiles and quartiles (21–23); covariance and correlation (24–29); virtual environments and setup (30–41); hands-on (42–45) | background |
| `DAA_PL_2(1).pdf` | Quality Metrics and Model Validation (Part II) | 49 | Decision-tree example (3–13); classification metrics: confusion matrix, accuracy, precision/recall, ROC/AUC, F1 (14–27); regression metrics: MAE, MSE, RMSE (28–38); hold-out, cross-validation, k-fold, leave-one-out (39–48); hands-on (49) | direct |
| `DAA_PL_3.pdf` | Data Exploration and Preparation (Part III) | 40 | Problem and ML pipeline (3–6); data exploration (7–13); data preparation (14–19); univariate analysis (20–24); multivariate analysis (25–38); critical analysis (39); hands-on (40) | direct |
| `DAA_PL_4(1).pdf` | Linear and Logistic Regression (Part IV) | 39 | Supervised-learning pipeline (3); linear regression with EDA, fit, metrics and LASSO (4–21); logistic regression with EDA, preprocessing, solvers and metrics (22–38); hands-on (39) | direct |
| `DAA_PL_5.pdf` | Feature Engineering / Decision Tree Pruning and Hyperparameter Tuning (Part V) | 55 | Feature engineering: missing values (9–13), unique values (14–15), categorical encoding (16–28), dates (30–35); decision-tree EDA (36–41); classifier and GridSearch (42–48); pruning and `ccp_alpha` (49–54); hands-on (55) | direct |
| `DAA_PL_6.pdf` | Ensemble Learning (Part VI) | 26 | Ensemble introduction (3–4); problem, EDA and preprocessing (5–9); comparison DT/logistic/RF (10–12); max voting (13–14); bagging and random forest (15–17); boosting and XGBoost (18–19); stacking (20–21); model comparison (22–25); hands-on (26) | direct |

## Version note

`Aula3 - DAA 2627_RL.pdf` is misnamed: "RL" here does not mean reinforcement
learning. Pages 2–5 show *Supervised Learning: Linear and Logistic Regression*.
Use the title column, not the filename, when choosing a deck.

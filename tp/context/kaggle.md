# 🏆 Competition ML@DAA — 2026/2027 Edition

**Host:** VictorAlves · Community Prediction Competition (Private)
**Course:** Data and Machine Learning (*Dados e Aprendizagem Automática*) — Practical Group Work

> Develop and optimize Machine Learning models to predict an individual's **age group** based on their profile — a nationwide health survey of the Portuguese resident population (2019).

---

## 📌 Competition Overview

This competition is the Group Practical Assignment of the course unit **Data and Machine Learning**, aimed at students of the following Master's programs:

- Informatics Engineering
- Mathematics and Computing
- Systems Engineering
- Data Science and Engineering
- Artificial Intelligence
- Cybersecurity
- Advanced Computing
- Computer Networks

### Dataset

The dataset originates from a nationwide survey conducted in Portugal during **2019**, covering:

- Health status
- Healthcare utilization
- Determinants and lifestyles
- General socioeconomic characteristics

**237 explanatory variables** across several domains:

| Domain | Examples |
|---|---|
| Demographics & work | Region (NUTS), education level (HATLEVEL), labor status (MAINSTAT) |
| Health status & chronic diseases | Self-perceived health (HS1), hypertension, diabetes, depression, chronic pain (CD1A–CD1P) |
| Limitations & autonomy | Vision/hearing/mobility difficulties (PL1–PL9), daily-task autonomy (PC1A–PC1E, HA1A–HA1G) |
| Lifestyle & habits | Physical exercise (PE1–PE8), tobacco (IN43–SK6), alcohol (AL1–AL6), dietary habits (DH1–IN71) |

Two datasets are provided: a **training set** (to train and tune the model) and a **test set** (to predict the age range, encoded on a 0–15 scale).

---

## 🎯 Objective

Apply Machine Learning models to predict each individual's **age group** from socioeconomic data, health status, healthcare utilization, determinants, and lifestyles.

The main focus goes beyond accurate prediction, centering on **detecting profile inconsistencies and anomalies** and characterizing each age group.

> ⚠️ **Example anomaly:** a model detecting a senior-profile (e.g., 60–64) in someone declaring age 18 — or vice versa — indicates an epidemiological anomaly, biological profile deviation, or recording error.

### Students must:

1. **Train and optimize** robust models to accurately predict the 16 age groups
2. **Identify profile anomalies** — cases with significant deviation between declared and predicted age
3. **Characterize the age groups** — extract socioeconomic, pathological, and behavioral patterns, plus the factors behind the anomalies

---

## 📅 Timeline

- **Start:** 15 days ago
- **Close:** 2 months to go

---

## 📊 Target: 16 Age Groups

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

---

## 📈 Evaluation Metric

**Macro F1-Score** across the 16 age group classes.

This ensures all age groups carry the **same weight** in evaluation, regardless of how many individuals fall into each bracket in the dataset.

---

## 📤 Submission Format

- Up to **3 valid submissions per day** (invalid ones don't count)
- Only **one file** (chosen by you) counts toward the final ranking
- **CSV format** with exactly **3655 entries** (same as the test set) plus the header `AGE_RANGE`
- Errors occur if the file is improperly formatted or has extra rows/columns
- An example submission file is available in the competition's Data tab

> 🚫 **Please do not build the file by hand!**

---

## 🏅 Prizes & Participation

| | |
|---|---|
| **Prizes** | Kudos — does not award Points or Medals |
| **Entrants** | 146 |
| **Participants** | 44 |
| **Teams** | 12 |
| **Submissions** | 56 |

**Tags:** F1 Score

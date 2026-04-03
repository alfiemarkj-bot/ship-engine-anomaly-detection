# Anomaly Detection in Ship Engine Data

Multi-method anomaly detection pipeline for 19,535 real-world ship engine 
sensor readings, combining statistical and machine learning approaches to 
enable predictive maintenance.

---

## Problem Statement
Unplanned engine failures at sea carry serious safety and financial 
consequences. This project identifies abnormal engine states before they 
escalate — shifting operations from reactive to predictive maintenance.

---

## Pipeline

| Stage 1 | Stage 2 | Stage 3 |
|---|---|---|
| Data Loading & EDA | IQR Statistical Detection | ML Detection (SVM & IF) |

---

## Exploratory Data Analysis

All six features show right-skewed distributions, with the mean exceeding 
the median in every case. No missing values or duplicates were found.

![Feature Distributions](images/histograms.png)
![Boxplots](images/boxplots.png)
![Correlation Heatmap](images/heatmap.png)

The near-zero correlations confirm that anomalies only emerge from 
combinations of features — motivating the use of multivariate ML models.

---

## Anomaly Detection Results

| Method | Anomalies Flagged | % of Dataset |
|---|---|---|
| IQR (T=2) | 422 | 2.16% |
| One-Class SVM (nu=0.01) | 198 | 1.01% |
| Isolation Forest (cont=0.01) | 196 | 1.00% |
| Two-Method Agreement | 176 | 0.90% |
| All-Method Agreement | 58 | 0.30% |

![One-Class SVM PCA](images/svm_pca.png)

---

## Key Finding

Anomalous states are driven by simultaneous elevation across multiple 
sensors — not a single runaway reading.

| Feature | Normal Mean | Anomaly Mean | % Change |
|---|---|---|---|
| Fuel Pressure | 6.617 | 10.513 | +58.9% |
| Coolant Pressure | 2.323 | 3.602 | +55.1% |
| Engine RPM | 788.5 | 1,058.1 | +34.2% |

---

## Tools & Libraries
Python · Pandas · NumPy · Scikit-learn · Matplotlib · Seaborn

## Files
- `Mini_Project_1_real.ipynb` — full annotated Colab notebook
- `Mini_Project_actualreport.pdf` — written report with recommendations

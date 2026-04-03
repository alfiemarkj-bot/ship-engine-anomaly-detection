# Anomaly Detection in Ship Engine Data

Detecting anomalous engine behaviour using statistical and machine learning methods, applied to 19,535 real-world sensor readings from a ship's engine.

## Project Overview
Unplanned engine failures at sea carry serious safety and financial consequences. This project builds a multi-method anomaly detection pipeline to identify abnormal engine states before they escalate, enabling a shift from reactive to predictive maintenance.

## Methods
- **Exploratory Data Analysis** — descriptive statistics, histograms, boxplots, and correlation heatmap
- **IQR (Interquartile Range)** — statistical baseline using a multi-feature threshold (T=2, flags 2.16% of data)
- **One-Class SVM** — machine learning boundary model (nu=0.01, flags 1.01%)
- **Isolation Forest** — ensemble anomaly detection (contamination=0.01, flags 1.00%)
- **Cross-method comparison** — identifies 58 high-confidence anomalies agreed upon by all three methods

## Key Finding
Anomalous engine states are characterised by simultaneous elevation of Fuel Pressure (+58.9%), Coolant Pressure (+55.1%), and Engine RPM (+34.2%) — consistent with multi-system fault patterns rather than single-sensor noise.

## Files
- `Mini_Project_1_real.ipynb` — full annotated Google Colab notebook
- `Mini_Project_actualreport.pdf` — written report with analysis and recommendations

## Tools & Libraries
Python · Pandas · NumPy · Scikit-learn · Matplotlib · Seaborn

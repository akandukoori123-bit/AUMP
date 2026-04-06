# AUMP — Adaptive Universal Monitoring Platform

AUMP is a machine learning–based early warning system for simulated bradycardia detection. The goal of the project is to move beyond simple threshold-based monitoring by using recent BPM (beats per minute) trends to estimate short-term crash risk before a dangerous threshold is crossed.

This project is designed as an early-stage research prototype that combines:
- physiological time-series simulation
- temporal feature engineering
- machine learning risk prediction
- real-time warning analysis
- visualization of crash risk over time

---

## Project Goal

Traditional threshold-based systems only react once a patient's BPM falls below a dangerous value. AUMP explores whether machine learning can identify **rising crash risk earlier** by looking at:
- current BPM
- recent BPM changes
- short-term trend
- short-term variability

The long-term goal is to develop a smarter early-warning layer that could eventually integrate with wearable health monitoring systems.

---

## Current Features

- Synthetic BPM sequence generation with gradual deterioration
- Probabilistic crash labeling
- Temporal feature engineering
- Logistic Regression baseline model
- Random Forest comparison model
- ROC-AUC evaluation
- Real-time risk prediction from recent BPM readings
- Early warning lead time calculation
- BPM-over-time and risk-over-time visualization

---

## Project Structure

```text
AUMP/
├── src/
│   ├── generate_data.py
│   ├── train_model.py
│   ├── predict.py
│   └── plot_risk.py
├── data/
│   └── simulated_data.csv
├── models/
│   └── model.pkl
├── results/
│   ├── roc_curve_comparison.png
│   ├── risk_predictions.csv
│   ├── bpm_over_time.png
│   └── risk_over_time.png
├── requirements.txt
├── README.md
└── .gitignore
"""
Train AUMP risk model with corrected methodology.

Fixes vs. previous version:
  1. Train/test split holds out whole sequences (no row-level leakage).
  2. Rolling/shift/diff features computed PER SEQUENCE so feature
     computation does not cross sequence boundaries.
  3. Compares Logistic Regression and Random Forest, saves the better one.
"""
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import joblib

FEATURES = ["bpm", "delta", "rolling_mean", "rolling_std", "acceleration", "trend"]


def compute_features(df):
    """Add temporal features per sequence (no cross-sequence leakage)."""
    parts = []
    for sid, g in df.groupby("sequence_id", sort=False):
        g = g.sort_values("time").copy()
        g["bpm_prev"]     = g["bpm"].shift(1)
        g["delta"]        = g["bpm"] - g["bpm_prev"]
        g["rolling_mean"] = g["bpm"].rolling(window=3).mean()
        g["rolling_std"]  = g["bpm"].rolling(window=3).std()
        g["acceleration"] = g["delta"].diff()
        g["trend"]        = g["rolling_mean"].diff()
        parts.append(g)
    out = pd.concat(parts, ignore_index=True)
    return out.dropna().reset_index(drop=True)


def split_by_sequence(df, test_frac=0.2, seed=42):
    """Hold out whole sequences for the test set."""
    rng = np.random.default_rng(seed)
    seq_ids = df["sequence_id"].unique().copy()
    rng.shuffle(seq_ids)
    n_test = int(len(seq_ids) * test_frac)
    test_ids = set(seq_ids[:n_test])
    test_mask = df["sequence_id"].isin(test_ids)
    return df[~test_mask].copy(), df[test_mask].copy()


def evaluate(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    print(f"\n=== {name} ===")
    print(classification_report(y_test, y_pred, digits=3))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    print(f"AUC: {roc_auc:.3f}")
    return fpr, tpr, roc_auc


# Load + features
df = pd.read_csv("data/simulated_data.csv")
df = compute_features(df)

# Group-aware split
train_df, test_df = split_by_sequence(df, test_frac=0.2, seed=42)
X_train, y_train = train_df[FEATURES], train_df["crash"]
X_test,  y_test  = test_df[FEATURES],  test_df["crash"]

print(f"Train rows: {len(X_train)} from {train_df['sequence_id'].nunique()} sequences")
print(f"Test rows:  {len(X_test)} from {test_df['sequence_id'].nunique()} sequences")
print(f"Train positive rate: {y_train.mean():.3f}")
print(f"Test positive rate:  {y_test.mean():.3f}")

# Models
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
fpr_lr, tpr_lr, auc_lr = evaluate(lr, X_test, y_test, "Logistic Regression")

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
fpr_rf, tpr_rf, auc_rf = evaluate(rf, X_test, y_test, "Random Forest")

# ROC comparison plot
os.makedirs("results", exist_ok=True)
plt.figure()
plt.plot(fpr_lr, tpr_lr, label=f"LogReg AUC = {auc_lr:.3f}")
plt.plot(fpr_rf, tpr_rf, label=f"RandomForest AUC = {auc_rf:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Comparison (held-out sequences)")
plt.legend()
plt.tight_layout()
plt.savefig("results/roc_curve_comparison.png")

# Save best model BEFORE showing plot, so a closed/killed plot
# can never prevent the model from being saved
best_model, best_name, best_auc = (
    (lr, "LogisticRegression", auc_lr) if auc_lr >= auc_rf
    else (rf, "RandomForest", auc_rf)
)
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/model.pkl")
print(f"\nSaved {best_name} (AUC {best_auc:.3f}) to models/model.pkl")

plt.show()
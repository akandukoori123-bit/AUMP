import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import joblib
import os

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv("data/simulated_data.csv")

# -----------------------------
# Feature Engineering (IMPORTANT)
# -----------------------------
df["bpm_prev"] = df["bpm"].shift(1)
df["delta"] = df["bpm"] - df["bpm_prev"]
df["rolling_mean"] = df["bpm"].rolling(window=3).mean()
df["rolling_std"] = df["bpm"].rolling(window=3).std()

# 🔥 NEW FEATURES (IMPORTANT UPGRADE)
df["acceleration"] = df["delta"].diff()
df["trend"] = df["rolling_mean"].diff()

df = df.dropna()

# -----------------------------
# Features / Labels
# -----------------------------
X = df[["bpm", "delta", "rolling_mean", "rolling_std", "acceleration", "trend"]]
y = df["crash"]

# -----------------------------
# Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Train model
# -----------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# -----------------------------
# Predictions
# -----------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# -----------------------------
# Evaluation
# -----------------------------
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# -----------------------------
# ROC Curve
# -----------------------------
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

os.makedirs("results", exist_ok=True)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.savefig("results/roc_curve.png")
plt.show()

# -----------------------------
# Save model
# -----------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.pkl")

print("Model saved.")
print("AUC:", roc_auc)
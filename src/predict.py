import pandas as pd
import joblib
import os

model = joblib.load("models/model.pkl")

recent_bpm = [75, 74, 73, 72, 70, 68, 65, 60, 58, 55, 52]

alert_threshold = 0.7
danger_threshold = 60

print("\n--- Real-Time Risk Prediction ---\n")

results = []
alert_time = None
danger_time = None

for i in range(4, len(recent_bpm) + 1):
    window = recent_bpm[:i]

    df = pd.DataFrame({"bpm": window})

    df["bpm_prev"] = df["bpm"].shift(1)
    df["delta"] = df["bpm"] - df["bpm_prev"]
    df["rolling_mean"] = df["bpm"].rolling(window=3).mean()
    df["rolling_std"] = df["bpm"].rolling(window=3).std()
    df["acceleration"] = df["delta"].diff()
    df["trend"] = df["rolling_mean"].diff()

    df = df.dropna()

    input_data = df[[
        "bpm",
        "delta",
        "rolling_mean",
        "rolling_std",
        "acceleration",
        "trend"
    ]].iloc[[-1]]

    prob = model.predict_proba(input_data)[0][1]
    bpm_now = window[-1]

    print(f"Time {i}: BPM={bpm_now} | Risk={prob:.3f}")

    results.append({
        "time": i,
        "bpm": bpm_now,
        "risk": prob
    })

    if alert_time is None and prob >= alert_threshold:
        alert_time = i

    if danger_time is None and bpm_now < danger_threshold:
        danger_time = i

os.makedirs("results", exist_ok=True)
pd.DataFrame(results).to_csv("results/risk_predictions.csv", index=False)

print("\nSaved predictions to results/risk_predictions.csv")

print("\n--- Early Warning Analysis ---")
print(f"Alert threshold: {alert_threshold}")
print(f"Danger threshold BPM: {danger_threshold}")

if alert_time is not None:
    print(f"Model alert time: {alert_time}")
else:
    print("Model never crossed alert threshold.")

if danger_time is not None:
    print(f"Danger threshold crossing time: {danger_time}")
else:
    print("BPM never crossed danger threshold.")

if alert_time is not None and danger_time is not None:
    lead_time = danger_time - alert_time
    print(f"Lead time: {lead_time} timesteps")

    if lead_time > 0:
        print("Model provided an early warning before danger threshold.")
    elif lead_time == 0:
        print("Model alerted at the same time as danger threshold.")
    else:
        print("Model alerted after danger threshold.")
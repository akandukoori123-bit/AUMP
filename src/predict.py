import pandas as pd
import joblib

model = joblib.load("models/model.pkl")

recent_bpm = [75, 74, 73, 72, 70, 68, 65, 60, 58]

print("\n--- Real-Time Risk Prediction ---\n")

for i in range(4, len(recent_bpm)):
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

    print(f"Time {i}: BPM={window[-1]} | Risk={prob:.3f}")
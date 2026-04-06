import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/risk_predictions.csv")

# Plot BPM over time
plt.figure()
plt.plot(df["time"], df["bpm"], marker="o")
plt.xlabel("Time")
plt.ylabel("BPM")
plt.title("Heart Rate Over Time")
plt.tight_layout()
plt.savefig("results/bpm_over_time.png")
plt.show(block=True)

# Plot risk over time
plt.figure()
plt.plot(df["time"], df["risk"], marker="o")
plt.axhline(0.7, linestyle="--", label="Alert Threshold")
plt.xlabel("Time")
plt.ylabel("Crash Risk")
plt.title("Predicted Crash Risk Over Time")
plt.legend()
plt.tight_layout()
plt.savefig("results/risk_over_time.png")
plt.show(block=True)
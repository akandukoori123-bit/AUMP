import numpy as np
import pandas as pd

def generate_sequence(length=30):
    bpm = []
    crash = []

    base = np.random.randint(70, 85)
    current = base

    # where deterioration begins
    crash_start = np.random.randint(12, 20)

    for t in range(length):

        # smooth noise (more realistic than randint)
        noise = np.random.normal(0, 1.5)
        current += noise

        # gradual deterioration after crash_start
        if t >= crash_start:
            drift = np.random.uniform(0.5, 2.5)
            current -= drift

        current = np.clip(current, 40, 100)

        bpm.append(current)

        # 🔥 PROBABILISTIC LABEL (THIS FIXES EVERYTHING)
        risk = 1 / (1 + np.exp((current - 60) / 4))
        crash.append(1 if np.random.rand() < risk else 0)

    return bpm, crash


def build_dataset(n_sequences=500):
    data = []

    for _ in range(n_sequences):
        bpm_seq, crash_seq = generate_sequence()

        for t in range(len(bpm_seq)):
            data.append([t, bpm_seq[t], crash_seq[t]])

    df = pd.DataFrame(data, columns=["time", "bpm", "crash"])
    return df


if __name__ == "__main__":
    df = build_dataset()
    df.to_csv("data/simulated_data.csv", index=False)
    print("Dataset saved.")
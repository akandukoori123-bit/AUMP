"""
Generate synthetic BPM sequences with FUTURE-looking crash labels.

Differences from the previous version:
  1. Adds sequence_id so train/test split can hold out whole sequences.
  2. Label is now future-looking:
        crash[t] = 1 if min(bpm[t+1 : t+1+HORIZON]) < DANGER_BPM
     The old label depended only on current BPM, which conflated the
     prediction target with the threshold baseline.
  3. Mix of crash and stable sequences (controlled by CRASH_PROBABILITY)
     so false-alarm rate can be measured on truly stable inputs.
"""
import os
import numpy as np
import pandas as pd

HORIZON = 10            # timesteps ahead the label looks
DANGER_BPM = 60        # threshold defining a "crash"
LENGTH = 30            # timesteps per sequence
N_SEQUENCES = 500
CRASH_PROBABILITY = 0.7   # fraction of sequences that deteriorate


def generate_sequence(length=LENGTH, has_crash=True, rng=None):
    """Generate one BPM sequence."""
    if rng is None:
        rng = np.random.default_rng()

    bpm = np.empty(length)
    current = float(rng.integers(70, 86))
    crash_start = int(rng.integers(12, 21)) if has_crash else None

    for t in range(length):
        current += rng.normal(0, 1.5)               # smooth noise
        if has_crash and t >= crash_start:
            current -= rng.uniform(0.5, 2.5)        # gradual drift down
        current = float(np.clip(current, 40, 100))
        bpm[t] = current

    return bpm, crash_start


def label_future_crash(bpm, horizon=HORIZON, danger=DANGER_BPM):
    """1 if any BPM in the next `horizon` steps falls below `danger`."""
    n = len(bpm)
    labels = np.zeros(n, dtype=int)
    for t in range(n):
        future = bpm[t + 1 : t + 1 + horizon]
        if len(future) > 0 and np.min(future) < danger:
            labels[t] = 1
    return labels


def build_dataset(n_sequences=N_SEQUENCES, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    for sid in range(n_sequences):
        has_crash = rng.random() < CRASH_PROBABILITY
        bpm, crash_start = generate_sequence(has_crash=has_crash, rng=rng)
        labels = label_future_crash(bpm)
        for t in range(len(bpm)):
            rows.append({
                "sequence_id": sid,
                "time": t,
                "bpm": float(bpm[t]),
                "crash": int(labels[t]),
                "has_crash_seq": int(has_crash),
                "crash_start": int(crash_start) if crash_start is not None else -1,
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = build_dataset()
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/simulated_data.csv", index=False)
    n_seq = df["sequence_id"].nunique()
    n_crash = df.groupby("sequence_id")["has_crash_seq"].first().sum()
    print(f"Saved {n_seq} sequences ({n_crash} with crash, "
          f"{n_seq - n_crash} stable) to data/simulated_data.csv")
    print(f"Positive label rate: {df['crash'].mean():.3f}")
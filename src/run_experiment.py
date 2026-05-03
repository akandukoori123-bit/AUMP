"""
src/run_experiment.py

Multi-scenario evaluation of AUMP vs threshold-only baseline.

For each of N synthetic scenarios (with varied baseline BPM, noise level,
drift speed, and crash onset), runs:
  1. Threshold-only monitoring (alerts when BPM < DANGER_BPM)
  2. AUMP at multiple alert probability thresholds (0.5–0.9)

Logs per-scenario alert times, lead times, and false-alarm flags to
results/experiment_log.csv. Saves a few full risk traces to
results/example_traces.csv for figure generation in evaluate.py.
"""
import os
import numpy as np
import pandas as pd
import joblib

# ------------------------- Config -------------------------

N_SCENARIOS       = 200
SEQUENCE_LENGTH   = 30
DANGER_BPM        = 60
WARMUP            = 4
ALERT_THRESHOLDS  = [0.5, 0.6, 0.7, 0.8, 0.9]
CRASH_PROBABILITY = 0.7
SEED              = 2026

FEATURES = ["bpm", "delta", "rolling_mean", "rolling_std",
            "acceleration", "trend"]

# Parameter ranges (sampled per scenario)
NOISE_SIGMAS      = [1.0, 1.5, 2.0, 2.5]
BASELINE_LOWS     = [65, 70, 75]
DRIFT_MIN_RANGE   = (0.3, 0.8)
DRIFT_MAX_RANGE   = (1.5, 3.0)
CRASH_START_RANGE = (10, 22)

EXAMPLE_INDICES = [0, 1, 2]  # full traces saved for these scenarios


# ------------------------- Simulator -------------------------

def sample_scenario_params(scenario_id, rng):
    has_crash = rng.random() < CRASH_PROBABILITY
    return {
        "scenario_id":  scenario_id,
        "length":       SEQUENCE_LENGTH,
        "has_crash":    has_crash,
        "noise_sigma":  float(rng.choice(NOISE_SIGMAS)),
        "baseline_low": int(rng.choice(BASELINE_LOWS)),
        "drift_min":    float(rng.uniform(*DRIFT_MIN_RANGE)),
        "drift_max":    float(rng.uniform(*DRIFT_MAX_RANGE)),
        "crash_start":  int(rng.integers(*CRASH_START_RANGE)) if has_crash else -1,
    }


def generate_scenario(params, rng):
    bpm = np.empty(params["length"])
    current = float(rng.integers(params["baseline_low"], params["baseline_low"] + 16))
    crash_start = params["crash_start"] if params["has_crash"] else None

    for t in range(params["length"]):
        current += rng.normal(0, params["noise_sigma"])
        if params["has_crash"] and t >= crash_start:
            current -= rng.uniform(params["drift_min"], params["drift_max"])
        current = float(np.clip(current, 40, 100))
        bpm[t] = current
    return bpm


# ------------------------- Feature computation -------------------------

def compute_features_at_t(bpm_window):
    """Last-timestep features given a growing BPM window. None if too short."""
    df = pd.DataFrame({"bpm": list(bpm_window)})
    df["bpm_prev"]     = df["bpm"].shift(1)
    df["delta"]        = df["bpm"] - df["bpm_prev"]
    df["rolling_mean"] = df["bpm"].rolling(window=3).mean()
    df["rolling_std"]  = df["bpm"].rolling(window=3).std()
    df["acceleration"] = df["delta"].diff()
    df["trend"]        = df["rolling_mean"].diff()
    df = df.dropna()
    if len(df) == 0:
        return None
    return df[FEATURES].iloc[[-1]]


# ------------------------- Detectors -------------------------

def threshold_only_alert_time(bpm_seq, danger_bpm=DANGER_BPM):
    for t, b in enumerate(bpm_seq):
        if b < danger_bpm:
            return t
    return None


def aump_risk_trace(bpm_seq, model):
    n = len(bpm_seq)
    risks = np.full(n, np.nan)
    for t in range(WARMUP - 1, n):
        feats = compute_features_at_t(bpm_seq[:t + 1])
        if feats is not None:
            risks[t] = float(model.predict_proba(feats)[0][1])
    return risks


def first_crossing(risks, threshold):
    for t, r in enumerate(risks):
        if not np.isnan(r) and r >= threshold:
            return t
    return None


# ------------------------- Main -------------------------

def main():
    model = joblib.load("models/model.pkl")
    print(f"Loaded model: {type(model).__name__}")

    rng = np.random.default_rng(SEED)
    log_rows = []
    example_traces = []

    for sid in range(N_SCENARIOS):
        params  = sample_scenario_params(sid, rng)
        bpm_seq = generate_scenario(params, rng)
        crashed = bool(np.any(bpm_seq < DANGER_BPM))

        thr_t = threshold_only_alert_time(bpm_seq)
        risks = aump_risk_trace(bpm_seq, model)

        row = {**params, "ground_truth_crashed": crashed,
               "threshold_alert_time": thr_t}

        for thr in ALERT_THRESHOLDS:
            aump_t = first_crossing(risks, thr)
            if aump_t is not None and thr_t is not None:
                lead = thr_t - aump_t
            else:
                lead = np.nan
            false_alarm = (aump_t is not None) and (not crashed)

            row[f"aump_alert_time@{thr}"] = aump_t if aump_t is not None else np.nan
            row[f"lead_time@{thr}"]       = lead
            row[f"false_alarm@{thr}"]     = bool(false_alarm)

        log_rows.append(row)

        if sid in EXAMPLE_INDICES:
            for t in range(SEQUENCE_LENGTH):
                example_traces.append({
                    "scenario_id":  sid,
                    "time":         t,
                    "bpm":          float(bpm_seq[t]),
                    "risk":         float(risks[t]) if not np.isnan(risks[t]) else np.nan,
                    "has_crash_seq": params["has_crash"],
                })

    os.makedirs("results", exist_ok=True)
    log_df = pd.DataFrame(log_rows)
    log_df.to_csv("results/experiment_log.csv", index=False)
    print(f"\nSaved {len(log_df)} scenarios to results/experiment_log.csv")

    traces_df = pd.DataFrame(example_traces)
    traces_df.to_csv("results/example_traces.csv", index=False)
    print(f"Saved {len(EXAMPLE_INDICES)} example traces to results/example_traces.csv")

    # Console summary
    print("\n--- Summary ---")
    print(f"Crashed sequences (ground truth): "
          f"{log_df['ground_truth_crashed'].sum()} / {len(log_df)}")
    print(f"Threshold-only fired on: "
          f"{log_df['threshold_alert_time'].notna().sum()} / {len(log_df)}")
    print()
    for thr in ALERT_THRESHOLDS:
        lead = log_df[f"lead_time@{thr}"].dropna()
        fa   = log_df[f"false_alarm@{thr}"].mean()
        fired = log_df[f"aump_alert_time@{thr}"].notna().sum()
        if len(lead) > 0:
            print(f"  thr={thr}: AUMP fired on {fired}/{len(log_df)}, "
                  f"median lead = {lead.median():+.1f} steps "
                  f"(IQR {lead.quantile(0.25):+.1f} to {lead.quantile(0.75):+.1f}), "
                  f"false-alarm rate = {fa:.1%}")
        else:
            print(f"  thr={thr}: AUMP fired on {fired}/{len(log_df)}, "
                  f"no lead-time data, false-alarm rate = {fa:.1%}")


if __name__ == "__main__":
    main()
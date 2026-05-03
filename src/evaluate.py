"""
src/evaluate.py

Reads results/experiment_log.csv from run_experiment.py and produces:
  - results/figure1_threshold_sweep.png       (headline figure)
  - results/figure2_lead_time_distribution.png
  - results/figure3_example_traces.png
  - results/summary_stats.csv
  - Console: paired Wilcoxon signed-rank tests at each threshold
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

ALERT_THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]
HEADLINE_THR = 0.6
SECONDARY_THR = 0.7
EXAMPLE_TRACES_PATH = "results/example_traces.csv"
LOG_PATH = "results/experiment_log.csv"


def summary_stats(log):
    rows = []
    for thr in ALERT_THRESHOLDS:
        lead = log[f"lead_time@{thr}"].dropna()
        aump_fired = log[f"aump_alert_time@{thr}"].notna().sum()
        crashed_seqs = log["ground_truth_crashed"].sum()
        stable_seqs = (~log["ground_truth_crashed"]).sum()
        false_alarms = log.loc[~log["ground_truth_crashed"],
                               f"aump_alert_time@{thr}"].notna().sum()
        fa_rate = false_alarms / stable_seqs if stable_seqs > 0 else np.nan

        rows.append({
            "alert_threshold": thr,
            "aump_fired_count": int(aump_fired),
            "crashed_sequences": int(crashed_seqs),
            "stable_sequences": int(stable_seqs),
            "false_alarm_rate": float(fa_rate),
            "median_lead_time": float(lead.median()) if len(lead) else np.nan,
            "iqr_lead_low": float(lead.quantile(0.25)) if len(lead) else np.nan,
            "iqr_lead_high": float(lead.quantile(0.75)) if len(lead) else np.nan,
            "n_with_lead_time": int(len(lead)),
        })
    return pd.DataFrame(rows)


def paired_wilcoxon(log):
    print("\n--- Paired Wilcoxon signed-rank tests ---")
    print("H0: AUMP alert time == threshold-only alert time")
    print("H1: AUMP alert time < threshold-only (one-sided)")
    print()
    rows = []
    for thr in ALERT_THRESHOLDS:
        paired = log[
            log["threshold_alert_time"].notna()
            & log[f"aump_alert_time@{thr}"].notna()
        ].copy()
        if len(paired) < 5:
            print(f"  thr={thr}: n={len(paired)}, too few for test")
            rows.append({"alert_threshold": thr, "n_paired": len(paired),
                         "median_diff": np.nan, "p_value": np.nan})
            continue
        diff = (paired[f"aump_alert_time@{thr}"]
                - paired["threshold_alert_time"]).values
        if np.all(diff == 0):
            print(f"  thr={thr}: n={len(paired)}, all diffs zero — skipped")
            rows.append({"alert_threshold": thr, "n_paired": len(paired),
                         "median_diff": 0.0, "p_value": np.nan})
            continue
        stat, p = wilcoxon(diff, alternative="less", zero_method="wilcox")
        med = float(np.median(diff))
        print(f"  thr={thr}: n={len(paired)}, "
              f"median(aump - threshold) = {med:+.1f} steps, "
              f"p = {p:.4g}")
        rows.append({"alert_threshold": thr, "n_paired": len(paired),
                     "median_diff": med, "p_value": float(p)})
    return pd.DataFrame(rows)


def plot_threshold_sweep(stats, out_path):
    fig, ax1 = plt.subplots(figsize=(7, 4.5))

    color1 = "#1f77b4"
    ax1.plot(stats["alert_threshold"], stats["median_lead_time"],
             "o-", color=color1, linewidth=2, label="Median lead time")
    ax1.fill_between(stats["alert_threshold"],
                     stats["iqr_lead_low"], stats["iqr_lead_high"],
                     alpha=0.2, color=color1, label="IQR")
    ax1.axhline(0, color="gray", linestyle=":", linewidth=1)
    ax1.set_xlabel("AUMP alert probability threshold")
    ax1.set_ylabel("Lead time vs. threshold-only (steps)", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = "#d62728"
    ax2.plot(stats["alert_threshold"], stats["false_alarm_rate"] * 100,
             "s--", color=color2, linewidth=2, label="False-alarm rate")
    ax2.set_ylabel("False-alarm rate on stable sequences (%)", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_ylim(bottom=0)

    plt.title("AUMP vs threshold-only baseline (N=200 scenarios)")
    fig.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_lead_time_distribution(log, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, thr in zip(axes, [HEADLINE_THR, SECONDARY_THR]):
        lead = log[f"lead_time@{thr}"].dropna()
        if len(lead) == 0:
            ax.set_title(f"Threshold {thr}: no data")
            continue
        ax.hist(lead, bins=range(int(lead.min()) - 1, int(lead.max()) + 2),
                edgecolor="black", color="#4a90e2", alpha=0.85)
        ax.axvline(lead.median(), color="red", linestyle="--", linewidth=2,
                   label=f"Median = {lead.median():+.1f}")
        ax.axvline(0, color="gray", linestyle=":", linewidth=1)
        ax.set_title(f"Alert threshold = {thr} (n = {len(lead)})")
        ax.set_xlabel("Lead time (steps)\n(positive = AUMP earlier than threshold)")
        ax.legend()
    axes[0].set_ylabel("Number of scenarios")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_example_traces(log, traces_df, out_path):
    sids = sorted(traces_df["scenario_id"].unique())
    fig, axes = plt.subplots(len(sids), 1, figsize=(8, 2.6 * len(sids)),
                             sharex=True)
    if len(sids) == 1:
        axes = [axes]
    for ax, sid in zip(axes, sids):
        trace = traces_df[traces_df["scenario_id"] == sid].sort_values("time")
        ax2 = ax.twinx()

        ax.plot(trace["time"], trace["bpm"], "-o", color="#1f77b4",
                markersize=3, label="BPM")
        ax.axhline(60, color="gray", linestyle=":", linewidth=1, label="Danger 60 BPM")
        ax.set_ylabel("BPM", color="#1f77b4")
        ax.set_ylim(40, 100)

        ax2.plot(trace["time"], trace["risk"], "-^", color="#d62728",
                 markersize=3, label="AUMP risk")
        ax2.axhline(HEADLINE_THR, color="#d62728", linestyle="--",
                    linewidth=1, alpha=0.7, label=f"Alert {HEADLINE_THR}")
        ax2.set_ylabel("Predicted risk", color="#d62728")
        ax2.set_ylim(0, 1)

        log_row = log[log["scenario_id"] == sid].iloc[0]
        thr_t = log_row["threshold_alert_time"]
        aump_t = log_row[f"aump_alert_time@{HEADLINE_THR}"]
        if pd.notna(aump_t):
            ax.axvline(aump_t, color="#d62728", linestyle="-",
                       alpha=0.5, linewidth=1.5)
        if pd.notna(thr_t):
            ax.axvline(thr_t, color="gray", linestyle="-",
                       alpha=0.5, linewidth=1.5)

        crashed = "crash" if log_row["ground_truth_crashed"] else "stable"
        ax.set_title(f"Scenario {sid} ({crashed}): "
                     f"AUMP@{HEADLINE_THR}={aump_t}, threshold={thr_t}")

    axes[-1].set_xlabel("Time step")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    log = pd.read_csv(LOG_PATH)
    print(f"Loaded {len(log)} scenarios from {LOG_PATH}")

    stats = summary_stats(log)
    os.makedirs("results", exist_ok=True)
    stats.to_csv("results/summary_stats.csv", index=False)
    print("\n--- Summary stats ---")
    print(stats.to_string(index=False))

    wilcox = paired_wilcoxon(log)
    wilcox.to_csv("results/wilcoxon_results.csv", index=False)

    plot_threshold_sweep(stats, "results/figure1_threshold_sweep.png")
    plot_lead_time_distribution(log, "results/figure2_lead_time_distribution.png")

    if os.path.exists(EXAMPLE_TRACES_PATH):
        traces = pd.read_csv(EXAMPLE_TRACES_PATH)
        plot_example_traces(log, traces, "results/figure3_example_traces.png")
    else:
        print(f"Skipped figure 3 — {EXAMPLE_TRACES_PATH} not found")

    print("\nDone. Files written to results/")


if __name__ == "__main__":
    main()
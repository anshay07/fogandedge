
import argparse
import random
import statistics
import csv
from datetime import datetime
import os

import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------
# Default parameter ranges
# ---------------------------
DEFAULTS = {
    "num_tasks": 100000,
    # fog ranges (ms)
    "fog_processing_min": 20,
    "fog_processing_max": 40,
    "fog_network_min": 10,
    "fog_network_max": 30,
    # cloud ranges (ms)
    "cloud_processing_min": 50,
    "cloud_processing_max": 100,
    "cloud_network_min": 80,
    "cloud_network_max": 120,
}

# ---------------------------
# Simulation function
# ---------------------------
def simulate_latency(
    num_tasks=DEFAULTS["num_tasks"],
    fog_proc_range=(DEFAULTS["fog_processing_min"], DEFAULTS["fog_processing_max"]),
    fog_net_range=(DEFAULTS["fog_network_min"], DEFAULTS["fog_network_max"]),
    cloud_proc_range=(DEFAULTS["cloud_processing_min"], DEFAULTS["cloud_processing_max"]),
    cloud_net_range=(DEFAULTS["cloud_network_min"], DEFAULTS["cloud_network_max"]),
    seed=None,
):
    """
    Simulate num_tasks IoT tasks and return a pandas DataFrame with:
      task_id, fog_processing, fog_network, fog_total,
      cloud_processing, cloud_network, cloud_total, timestamp
    All values are in milliseconds (ms).
    """
    if seed is not None:
        random.seed(seed)

    records = []
    for t in range(1, num_tasks + 1):
        fog_processing = random.uniform(*fog_proc_range)
        fog_network = random.uniform(*fog_net_range)
        fog_total = fog_processing + fog_network

        cloud_processing = random.uniform(*cloud_proc_range)
        cloud_network = random.uniform(*cloud_net_range)
        cloud_total = cloud_processing + cloud_network

        records.append(
            {
                "task_id": t,
                "fog_processing_ms": round(fog_processing, 3),
                "fog_network_ms": round(fog_network, 3),
                "fog_total_ms": round(fog_total, 3),
                "cloud_processing_ms": round(cloud_processing, 3),
                "cloud_network_ms": round(cloud_network, 3),
                "cloud_total_ms": round(cloud_total, 3),
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

    df = pd.DataFrame.from_records(records)
    return df

# ---------------------------
# Stats and comparison
# ---------------------------
def compute_summary_stats(df):
    """
    Compute summary statistics for fog and cloud totals.
    Returns a dictionary of metrics.
    """
    fog = df["fog_total_ms"].tolist()
    cloud = df["cloud_total_ms"].tolist()

    stats = {
        "num_tasks": len(fog),
        "fog_avg_ms": statistics.mean(fog),
        "fog_median_ms": statistics.median(fog),
        "fog_min_ms": min(fog),
        "fog_max_ms": max(fog),
        "fog_std_ms": statistics.pstdev(fog),
        "cloud_avg_ms": statistics.mean(cloud),
        "cloud_median_ms": statistics.median(cloud),
        "cloud_min_ms": min(cloud),
        "cloud_max_ms": max(cloud),
        "cloud_std_ms": statistics.pstdev(cloud),
        "improvement_pct_avg": (statistics.mean(cloud) - statistics.mean(fog))
        / statistics.mean(cloud)
        * 100.0,
    }
    return stats

# ---------------------------
# Plotting function
# ---------------------------
def plot_latencies(df, out_path="latency_comparison.png", show_plot=False):
    """
    Plot fog vs cloud total latency for each task and save the figure.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(df["task_id"], df["cloud_total_ms"], label="Cloud total latency (ms)", marker="o", linewidth=1)
    plt.plot(df["task_id"], df["fog_total_ms"], label="Fog total latency (ms)", marker="o", linewidth=1)
    plt.xlabel("Task ID")
    plt.ylabel("Latency (ms)")
    plt.title("Latency comparison: Fog vs Cloud")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    if show_plot:
        plt.show()
    plt.close()
    return out_path

# ---------------------------
# CSV export
# ---------------------------
def save_results_csv(df, out_csv="latency_results.csv"):
    df.to_csv(out_csv, index=False)
    return out_csv

# ---------------------------
# Main CLI
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="IoT Fog vs Cloud Latency Simulation")
    parser.add_argument("--tasks", type=int, default=DEFAULTS["num_tasks"], help="Number of IoT tasks to simulate")
    parser.add_argument("--seed", type=int, default=123, help="Random seed (for reproducibility)")
    parser.add_argument("--outdir", type=str, default=".", help="Output directory for plot and CSV")
    parser.add_argument("--show", action="store_true", help="Show plot interactively (useful on desktop)")
    args = parser.parse_args()

    # ensure outdir exists
    os.makedirs(args.outdir, exist_ok=True)

    # simulate
    df = simulate_latency(
        num_tasks=args.tasks,
        fog_proc_range=(DEFAULTS["fog_processing_min"], DEFAULTS["fog_processing_max"]),
        fog_net_range=(DEFAULTS["fog_network_min"], DEFAULTS["fog_network_max"]),
        cloud_proc_range=(DEFAULTS["cloud_processing_min"], DEFAULTS["cloud_processing_max"]),
        cloud_net_range=(DEFAULTS["cloud_network_min"], DEFAULTS["cloud_network_max"]),
        seed=args.seed,
    )

    # compute stats
    stats = compute_summary_stats(df)

    # print stats to console
    print("\n--- Simulation Summary ---")
    print(f"Simulated tasks: {stats['num_tasks']}")
    print(f"Average fog latency (ms): {stats['fog_avg_ms']:.3f}")
    print(f"Average cloud latency (ms): {stats['cloud_avg_ms']:.3f}")
    print(f"Fog latency range (min..max) ms: {stats['fog_min_ms']:.3f} .. {stats['fog_max_ms']:.3f}")
    print(f"Cloud latency range (min..max) ms: {stats['cloud_min_ms']:.3f} .. {stats['cloud_max_ms']:.3f}")
    print(f"Estimated average improvement with fog: {stats['improvement_pct_avg']:.2f}%")
    print("---------------------------\n")

    # save CSV
    csv_path = os.path.join(args.outdir, "latency_results.csv")
    save_results_csv(df, csv_path)
    print(f"Saved detailed results CSV -> {csv_path}")

    # plot and save
    plot_path = os.path.join(args.outdir, "latency_comparison.png")
    plot_latencies(df, plot_path, show_plot=args.show)
    print(f"Saved latency comparison plot -> {plot_path}")

    # also save a small summary txt
    summary_txt = os.path.join(args.outdir, "summary.txt")
    with open(summary_txt, "w") as f:
        for k, v in stats.items():
            f.write(f"{k}: {v}\n")
    print(f"Saved summary -> {summary_txt}\n")

    print("Done.")

if __name__ == "__main__":
    main()

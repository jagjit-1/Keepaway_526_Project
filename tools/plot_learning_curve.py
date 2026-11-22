#!/usr/bin/env python3
"""Plot learning curves for Keepaway experiments.

Based on Section 3.2 and Section 4 of:
  Stone et al. (2005), "Keepaway Soccer: From Machine Learning Testbed to Benchmark"

Computes a sliding-window moving average of episode durations and generates publication-style
learning curves comparing Tile Coding, Neural Networks, and baseline benchmarks.

Usage:
  python tools/plot_learning_curve.py results/benchmarks.csv --window 50 --save results/learning_curve.png
"""

import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Plot Keepaway learning curves")
    parser.add_argument("csv", help="Path to benchmarks.csv or episode log CSV")
    parser.add_argument("--window", type=int, default=50, help="Sliding window size (episodes)")
    parser.add_argument("--save", default="results/learning_curve.png", help="Output image file path")
    parser.add_argument("--title", default="Keepaway 3v2 Learning Curve", help="Plot title")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"File not found: {args.csv}")
        return

    df = pd.read_csv(args.csv)
    if "seconds" not in df.columns or "policy" not in df.columns:
        print("CSV must contain 'policy' and 'seconds' columns.")
        return

    plt.figure(figsize=(9, 5), dpi=150)
    policies = df["policy"].unique()

    # Benchmark baseline possession times from Table 1 in Stone et al. (2005)
    baselines = {
        "AlwaysHold (Paper)": 3.4,
        "Random (Paper)": 7.5,
        "Hand-coded (Paper)": 8.3,
    }

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(policies), 3)))

    for idx, pol in enumerate(policies):
        sub = df[df["policy"] == pol].copy().reset_index(drop=True)
        durations = sub["seconds"].values
        if len(durations) < 2:
            continue

        w = min(args.window, max(1, len(durations) // 5))
        rolling_mean = pd.Series(durations).rolling(window=w, min_periods=1).mean()
        episodes = np.arange(1, len(durations) + 1)

        plt.plot(episodes, rolling_mean, label=f"{pol} (window={w})", color=colors[idx], lw=2)

    # Reference baselines
    plt.axhline(8.3, color="green", linestyle="--", alpha=0.6, label="Hand-coded Benchmark (8.3s)")
    plt.axhline(7.5, color="orange", linestyle="--", alpha=0.6, label="Random Benchmark (7.5s)")

    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Possession Duration (seconds)", fontsize=12)
    plt.title(args.title, fontsize=14, fontweight="bold")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower right", framealpha=0.9)
    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
    plt.savefig(args.save)
    print(f"Saved learning curve plot to {args.save}")


if __name__ == "__main__":
    main()

"""
---------------------------------------------------------------------------
tools/plot_nn_results.py

Visualizes training progress, policy benchmarks, and comparisons for the
Keepaway Neural Network function approximators.

Generates:
  1. Learning curves (moving average episode duration over training)
  2. Policy evaluation comparison (Neural Net vs HandCoded vs Random vs AlwaysHold)
  3. Action distribution / End reason breakdown bar charts

Partially generated with assistance from AI (DeepMind Antigravity / Claude)
and adapted/verified by human authors.
---------------------------------------------------------------------------
"""

import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def moving_average(data, window_size=50):
    if len(data) < window_size:
        return data
    cumsum = np.cumsum(np.insert(data, 0, 0))
    return (cumsum[window_size:] - cumsum[:-window_size]) / float(window_size)

def read_csv(path):
    episodes = []
    seconds = []
    reasons = []
    with open(path, "r") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if not row or len(row) < 4:
                continue
            try:
                episodes.append(int(row[1]))
                seconds.append(float(row[2]))
                reasons.append(row[3])
            except ValueError:
                continue
    return np.array(episodes), np.array(seconds), reasons

def main():
    nn_dir = os.path.join("results", "nn")
    fig_dir = os.path.join("results", "figures")
    os.makedirs(fig_dir, exist_ok=True)

    print("Generating Neural Network figures...")

    # 1. Learning curves: Q-learning vs Sarsa
    qlearn_train_csv = os.path.join(nn_dir, "train_qlearn_sep.csv")
    sarsa_train_csv = os.path.join(nn_dir, "train_sarsa_sep.csv")
    shared_train_csv = os.path.join(nn_dir, "train_qlearn_shared.csv")

    plt.figure(figsize=(10, 5), dpi=300)
    plt.grid(True, linestyle="--", alpha=0.5)

    curves_plotted = 0
    if os.path.exists(qlearn_train_csv):
        _, sec_q, _ = read_csv(qlearn_train_csv)
        ma_q = moving_average(sec_q, 50)
        plt.plot(range(len(ma_q)), ma_q, label="Q-learning (3 separate 13-20-1 subnets)", color="#1f77b4", linewidth=2.0)
        curves_plotted += 1

    if os.path.exists(sarsa_train_csv):
        _, sec_s, _ = read_csv(sarsa_train_csv)
        ma_s = moving_average(sec_s, 50)
        plt.plot(range(len(ma_s)), ma_s, label="Sarsa (3 separate 13-20-1 subnets)", color="#ff7f0e", linewidth=2.0)
        curves_plotted += 1

    if os.path.exists(shared_train_csv):
        _, sec_sh, _ = read_csv(shared_train_csv)
        ma_sh = moving_average(sec_sh, 50)
        plt.plot(range(len(ma_sh)), ma_sh, label="Q-learning (13-20-3 shared hidden)", color="#2ca02c", linestyle="--", linewidth=1.8)
        curves_plotted += 1

    plt.title("Keepaway 3-v-2: Neural Network Learning Curves (Moving Avg Window = 50)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Training Episode", fontsize=11)
    plt.ylabel("Possession Time (seconds)", fontsize=11)
    plt.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=10)
    plt.tight_layout()
    curve_fig_path = os.path.join(fig_dir, "nn_learning_curves.png")
    plt.savefig(curve_fig_path)
    plt.close()
    print(f"Saved: {curve_fig_path}")

    # 2. Benchmark Comparison Bar Chart
    # Baselines from benchmark runs:
    # AlwaysHold: ~4.85 s
    # Random: ~10.90 s
    # HandCoded: ~22.89 s
    # NN_QLearn (Eval): ~13.8 s
    # NN_Sarsa (Eval): ~13.8 s
    # NN_Shared (Eval): ~12.7 s
    policies = ["Always Hold\n(Baseline)", "Random\n(Baseline)", "NN Shared\n(Q-learn)", "NN Separate\n(Q-learn)", "NN Separate\n(Sarsa)", "HandCoded\n(Expert Benchmark)"]
    means = [4.85, 10.90, 12.69, 13.77, 13.83, 22.89]
    errors = [0.80, 2.97, 4.87, 5.81, 5.96, 11.01]
    colors = ["#7f7f7f", "#aec7e8", "#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"]

    plt.figure(figsize=(9, 5.5), dpi=300)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    bars = plt.bar(policies, means, yerr=errors, capsize=5, color=colors, edgecolor="#333333", alpha=0.88, width=0.55)

    for bar, m in zip(bars, means):
        plt.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.5, f"{m:.2f}s", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.title("Keepaway 3-v-2 Policy Comparison: Mean Possession Time (Greedy Eval)", fontsize=13, fontweight="bold", pad=12)
    plt.ylabel("Mean Episode Duration (seconds)", fontsize=11)
    plt.ylim(0, 38)
    plt.tight_layout()
    bar_fig_path = os.path.join(fig_dir, "nn_policy_comparison.png")
    plt.savefig(bar_fig_path)
    plt.close()
    print(f"Saved: {bar_fig_path}")

    # 3. Termination Reason Analysis
    categories = ["NN Separate (Q-learn)", "NN Separate (Sarsa)", "Random", "HandCoded"]
    steals = [197, 197, 194, 175]
    out_bounds = [3, 3, 6, 25]

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=300)
    x = np.arange(len(categories))
    width = 0.4

    ax.grid(axis="y", linestyle="--", alpha=0.5)
    r1 = ax.bar(x - width/2, steals, width, label="Ball Stolen by Takers", color="#d95f02", alpha=0.85)
    r2 = ax.bar(x + width/2, out_bounds, width, label="Ball Out of Bounds", color="#7570b3", alpha=0.85)

    ax.set_ylabel("Count (out of 200 episodes)", fontsize=11)
    ax.set_title("Episode Termination Causes Across Policies", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9)
    plt.tight_layout()
    reasons_fig_path = os.path.join(fig_dir, "nn_termination_reasons.png")
    plt.savefig(reasons_fig_path)
    plt.close()
    print(f"Saved: {reasons_fig_path}")

    print("All figures successfully generated.")

if __name__ == "__main__":
    main()

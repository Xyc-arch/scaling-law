#!/usr/bin/env python3
# ablate_plot.py
# ----------------------------------------------------
# Final minimalist ablation plotter
# One baseline bar, no legend or title
# Y-axis labeled as "<metric> loss"
# ----------------------------------------------------

import os
import json
import matplotlib.pyplot as plt
import numpy as np

# ====================================================
# CONFIGURATION
# ====================================================

DATASETS = ["adult", "craft", "diabetes", "gender"]
ABLATION_TYPES = ["imb", "spurious"]
RESULTS_DIR = "/home/ubuntu/scaling-law/codebase/results_ablate"
SAVE_DIR = "/home/ubuntu/scaling-law/plots/ablate"

os.makedirs(SAVE_DIR, exist_ok=True)

BASELINES = ["synthetic_seed", "smote", "adasyn", "ros"]
LABELS = ["LLM", "SMOTE", "ADASYN", "ROS"]
COLORS = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"]  # green, blue, orange, red


# ====================================================
# HELPER FUNCTION TO PLOT
# ====================================================

def plot_ablate(dataset, ablation, data_dict):
    metrics = ["balanced_cross", "min_cross"]

    for metric in metrics:
        plt.figure(figsize=(6.5, 5))

        # --- collect base and balanced data ---
        imbalance_base_means = []
        imbalance_base_stds = []
        balance_means = []
        balance_stds = []

        for base in BASELINES:
            if base not in data_dict:
                continue
            sub = data_dict[base]["xgb"][metric]
            imbalance_base_means.append(sub["imbalance_base_mean"])
            imbalance_base_stds.append(sub["imbalance_base_std"])
            balance_means.append(sub["balance_base_mean"])
            balance_stds.append(sub["balance_base_std"])

        # --- positions ---
        x = np.arange(len(BASELINES) + 1)  # +1 for baseline
        width = 0.6

        # --- gray baseline bar ---
        base_mean = imbalance_base_means[0] if imbalance_base_means else 0
        base_std = imbalance_base_stds[0] if imbalance_base_stds else 0

        plt.bar(
            0,
            base_mean,
            width,
            yerr=base_std,
            capsize=5,
            color="gray",
            alpha=0.8,
        )

        # --- colored bars for each method ---
        for i, (mean, std, color) in enumerate(zip(balance_means, balance_stds, COLORS)):
            plt.bar(
                i + 1,
                mean,
                width,
                yerr=std,
                capsize=5,
                color=color,
            )

        # --- labels and style ---
        xtick_labels = ["Base"] + LABELS
        plt.xticks(np.arange(len(xtick_labels)), xtick_labels, fontsize=15)
        plt.ylabel(metric.replace("_", " ") + " loss", fontsize=16)
        plt.yticks(fontsize=14)
        plt.tight_layout()

        # --- save ---
        save_path = os.path.join(SAVE_DIR, f"{dataset}_{ablation}_{metric}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✅ Saved: {save_path}")


# ====================================================
# MAIN LOOP
# ====================================================

def main():
    if not os.path.exists(RESULTS_DIR):
        print(f"❌ Results directory not found: {RESULTS_DIR}")
        return

    for dataset in DATASETS:
        for ablation in ABLATION_TYPES:
            filename = f"{dataset}_ablate_{ablation}.json"
            result_path = os.path.join(RESULTS_DIR, filename)

            if not os.path.exists(result_path):
                print(f"⚠️ Missing file: {result_path}")
                continue

            with open(result_path, "r") as f:
                data = json.load(f)

            plot_ablate(dataset, ablation, data)


if __name__ == "__main__":
    main()

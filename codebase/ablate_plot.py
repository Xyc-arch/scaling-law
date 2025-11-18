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

# ---- add LLM-Ind as a new method (minimal change) ----
BASELINES = ["synthetic_seed", "synthetic_seed_ind", "smote", "adasyn", "ros"]
LABELS   = ["LLM",            "LLM-Ind",           "SMOTE", "ADASYN", "ROS"]
COLORS   = ["#2ca02c", "#9467bd", "#1f77b4", "#ff7f0e", "#d62728"]
# green, purple, blue, orange, red


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

        used_labels = []
        used_colors = []

        for base, label, color in zip(BASELINES, LABELS, COLORS):
            if base not in data_dict:
                continue
            sub = data_dict[base]["xgb"][metric]
            imbalance_base_means.append(sub["imbalance_base_mean"])
            imbalance_base_stds.append(sub["imbalance_base_std"])
            balance_means.append(sub["balance_base_mean"])
            balance_stds.append(sub["balance_base_std"])
            used_labels.append(label)
            used_colors.append(color)

        if len(balance_means) == 0:
            print(f"⚠️ No data for {dataset}, {ablation}, {metric}")
            plt.close()
            continue

        # --- positions ---
        x = np.arange(len(used_labels) + 1)  # +1 for baseline
        width = 0.6

        # --- gray baseline bar ---
        # use the first method's imbalance baseline as the "Base" value
        base_mean = imbalance_base_means[0]
        base_std = imbalance_base_stds[0]

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
        for i, (mean, std, color) in enumerate(zip(balance_means, balance_stds, used_colors)):
            plt.bar(
                i + 1,
                mean,
                width,
                yerr=std,
                capsize=5,
                color=color,
            )

        # --- labels and style ---
        xtick_labels = ["Base"] + used_labels
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
            base_filename = f"{dataset}_ablate_{ablation}.json"
            base_path = os.path.join(RESULTS_DIR, base_filename)

            if not os.path.exists(base_path):
                print(f"⚠️ Missing file: {base_path}")
                continue

            with open(base_path, "r") as f:
                data = json.load(f)

            # ---- minimal change: also load *_ind.json and merge ----
            ind_filename = f"{dataset}_ablate_{ablation}_ind.json"
            ind_path = os.path.join(RESULTS_DIR, ind_filename)
            if os.path.exists(ind_path):
                with open(ind_path, "r") as f_ind:
                    ind_data = json.load(f_ind)
                # this adds the "synthetic_seed_ind" entry
                data.update(ind_data)
            else:
                print(f"⚠️ IND file not found (skipping LLM-Ind for this setting): {ind_path}")

            plot_ablate(dataset, ablation, data)


if __name__ == "__main__":
    main()

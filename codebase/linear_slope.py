import os
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from utils import load_data_info

# ==========================
# Compute slope over log(sample size)
# using *raw* loss (no log transform on Y)
# ==========================

def compute_slope(data_name, json_dir, task, classifier, metric):
    """
    Compute slope of loss vs log(sample size)
    """
    file_name = f"{task}_{classifier}_synthetic_allseed_{metric}.json"
    file_path = os.path.join(json_dir, file_name)

    if not os.path.exists(file_path):
        print(f"⚠️ Missing file: {file_path}")
        return None

    # Load the JSON file
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Extract mean losses across augmentation levels
    means = []
    for i in range(10):  # 0 through 9 correspond to different synthetic sizes
        key = str(i)
        if "balance_augment_mean" not in data[key]:
            continue
        means.append(data[key]["balance_augment_mean"])

    # Convert to numpy
    means = np.array(means)
    # Synthetic sample sizes: 1, 400, 800, ..., 4000
    sample_sizes = np.arange(1, 4001, 400)

    # Use log(sample size) as X
    X = np.log(sample_sizes[1:]).reshape(-1, 1)  # skip 0 point to avoid bias
    y = means[1:]  # raw loss (no log)

    # Linear regression
    model = LinearRegression()
    model.fit(X, y)

    slope = model.coef_[0]

    print(f"✅ {data_name} | {task} | {metric} | slope={slope:.6f}")
    return slope


if __name__ == "__main__":
    data_names = {0: "craft", 1: "gender", 2: "diabetes", 3: "adult"}
    tasks = {
        0: "spurious_corr_vary_additional",
        1: "imbalanced_class_vary_additional"
    }
    metrics = {0: "balanced_log_cross", 1: "min_log_cross"}

    classifier = "xgb"  # only need XGBoost

    results = []

    for idx_data in [0, 1, 2, 3]:
        data_name = data_names[idx_data]
        for idx_task in [0, 1]:
            task = tasks[idx_task]
            # Skip known missing combo
            if data_name == "adult" and task == "spurious_corr_vary_additional":
                continue
            for idx_metric in [0, 1]:
                metric = metrics[idx_metric]

                info = load_data_info("data_info.json")
                data_info = info.get(data_name, {})
                json_dir = data_info.get("path")

                slope = compute_slope(data_name, json_dir, task, classifier, metric)
                if slope is not None:
                    results.append({
                        "dataset": data_name,
                        "task": task,
                        "metric": metric,
                        "slope": slope
                    })

    # Save results
    output_path = "/home/ubuntu/scaling-law/codebase/results_ablate/slope.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"\n✅ All slopes saved to {output_path}")

import pandas as pd
import numpy as np
import os
import json
from utils import *

RESULTS_DIR = "./results_ablate"
os.makedirs(RESULTS_DIR, exist_ok=True)


def divide_test_data(test_data, label_column="Target"):
    groups = {grp: df for grp, df in test_data.groupby(test_data[label_column])}
    for grp in [0, 1]:
        if grp not in groups:
            groups[grp] = pd.DataFrame(columns=test_data.columns)
    return groups


def run_experiment(parent_path, seed_size, classifier_type, target_column, add_data, metric):
    seed_path = os.path.join(parent_path, "seed.csv")
    train0_path = os.path.join(parent_path, "train0.csv")
    synthetic1_path = os.path.join(parent_path, f"{add_data}1.csv")
    test_path = os.path.join(parent_path, "test.csv")

    seed_data = pd.read_csv(seed_path)
    train0 = pd.read_csv(train0_path)
    synthetic1 = pd.read_csv(synthetic1_path)
    test_data = pd.read_csv(test_path)

    test_groups = divide_test_data(test_data.copy(), target_column)

    imbalance_ratio = 5
    imbalance_base_scores, balance_base_scores = [], []

    for seed in [1, 2, 6, 8, 42]:
        np.random.seed(seed)

        # Resample for this seed
        train0_use = train0.sample(n=imbalance_ratio * seed_size, random_state=seed)
        synthetic1_balance = synthetic1.sample(n=imbalance_ratio * seed_size, random_state=seed)

        # Imbalance baseline
        imbalance_base = pd.concat([seed_data, train0_use])
        X_train_imb = imbalance_base.drop(columns=[target_column])
        y_train_imb = imbalance_base[target_column]

        clf = get_classifier(classifier_type, seed)
        clf.fit(X_train_imb, y_train_imb)
        imbalance_base_scores.append(loss(metric, test_data, test_groups, target_column, clf, "imbalance"))

        # Balanced with synthetic
        balance_base = pd.concat([imbalance_base, synthetic1_balance])
        X_train_bal = balance_base.drop(columns=[target_column])
        y_train_bal = balance_base[target_column]

        clf = get_classifier(classifier_type, seed)
        clf.fit(X_train_bal, y_train_bal)
        balance_base_scores.append(loss(metric, test_data, test_groups, target_column, clf, "imbalance"))

    return {
        "imbalance_base_mean": np.mean(imbalance_base_scores),
        "imbalance_base_std": np.std(imbalance_base_scores),
        "balance_base_mean": np.mean(balance_base_scores),
        "balance_base_std": np.std(balance_base_scores),
    }


if __name__ == "__main__":
    data_names = {0: "craft", 1: "gender", 2: "diabetes", 3: "adult"}
    add_data_ls = {0: "synthetic_seed", 1: "smote", 2: "adasyn", 3: "ros"}
    classifier_types = {0: "rf", 1: "xgb"}
    metrics = {0: "balanced_cross", 1: "min_cross"}

    for idx_data_name in [3]:
        data_name = data_names[idx_data_name]
        info = load_data_info("data_info.json")
        data_info = info.get(data_name, {})
        seed_size = data_info.get("seed_size")
        label_col = data_info.get("label_col")
        parent_path = data_info.get("path")

        dataset_results = {}

        for idx_add_data in [0, 1, 2, 3]:
            add_data = add_data_ls[idx_add_data]
            dataset_results[add_data] = {}

            for idx_classifier in [1]:  # xgb only
                classifier_type = classifier_types[idx_classifier]
                dataset_results[add_data][classifier_type] = {}

                for idx_metric in [0, 1]:
                    metric = metrics[idx_metric]
                    res = run_experiment(parent_path, seed_size, classifier_type, label_col, add_data, metric)
                    dataset_results[add_data][classifier_type][metric] = res

        save_path = os.path.join(RESULTS_DIR, f"{data_name}_ablate_imb.json")
        with open(save_path, "w") as f:
            json.dump(dataset_results, f, indent=4)
        print(f"Saved {save_path}")

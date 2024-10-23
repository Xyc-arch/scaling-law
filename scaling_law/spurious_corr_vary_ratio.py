import pandas as pd
import numpy as np
import os
import json
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from utils import *



def divide_test_data(test_data, spurious_column, spurious_small_is_0, label_column='Target'):
    if spurious_small_is_0 == 1:
        small_value = 0
    else:
        small_value = test_data[spurious_column].min()
    test_data['spurious_value'] = test_data[spurious_column].apply(lambda x: 0 if x == small_value else 1)
    test_data['group'] = test_data['spurious_value'].astype(str) + test_data[label_column].astype(int).astype(str)
    
    # Create groups
    groups = {grp: df for grp, df in test_data.groupby('group')}

    # Ensure all groups are present and drop 'spurious_value' and 'group'
    for grp in ['00', '01', '10', '11']:
        if grp not in groups:
            groups[grp] = pd.DataFrame(columns=test_data.columns)
        else:
            groups[grp] = groups[grp].drop(columns=['spurious_value', 'group'])
    
    return groups


def perform_experiment(parent_path, seed_size, random_seed, classifier_type, spurious_column, spurious_small_is_0, target_column='Target', add_data="synthetic", metric="balanced", augment_w = 0.5):
    np.random.seed(random_seed)

    exp_results_path = os.path.join(parent_path, f"spurious_corr_vary_ratio_{classifier_type}_{add_data}_{metric}.json")
    seed_path = os.path.join(parent_path, "seed.csv")
    train_neg_path = os.path.join(parent_path, "train_neg.csv")
    synthetic_pos_path = os.path.join(parent_path, f"{add_data}_pos.csv")
    synthetic_neg_path = os.path.join(parent_path, f"{add_data}_neg.csv")
    test_path = os.path.join(parent_path, "test.csv")

    seed_data = pd.read_csv(seed_path)
    train_neg = pd.read_csv(train_neg_path)
    synthetic_pos = pd.read_csv(synthetic_pos_path)
    synthetic_neg = pd.read_csv(synthetic_neg_path)
    test_data = pd.read_csv(test_path)

    test_groups = divide_test_data(test_data.copy(), spurious_column, spurious_small_is_0, target_column)

    train_neg_use = train_neg.sample(n=5 * seed_size, random_state=random_seed)
    synthetic_pos_balance = synthetic_pos.sample(n=5 * seed_size, random_state=random_seed)
    synthetic_pos_remain = synthetic_pos.drop(synthetic_pos_balance.index)

    results = {}

    for i in range(10):
        print(i)
        imbalance_base_scores = []
        balance_base_scores = []
        balance_augment_scores = []

        for seed in [1, 2, 6, 8, 42]:
            np.random.seed(seed)

            num_samples_seed = int(i * seed_size / 2)
            num_samples_augment = 3 * seed_size

            if num_samples_seed > 0:
                train_neg_samples = train_neg_use.sample(n=num_samples_seed, random_state=seed)
                imbalance_base = pd.concat([seed_data, train_neg_samples])
            else:
                imbalance_base = seed_data.copy()

            X_train_imbalance = imbalance_base.drop(columns=[target_column])
            y_train_imbalance = imbalance_base[target_column]

            clf = get_classifier(classifier_type, seed)
            clf.fit(X_train_imbalance, y_train_imbalance)

            y_pred = clf.predict(test_data.drop(columns=[target_column]))

            if metric == "balanced":
                ba_score = balanced_accuracy_score(test_data[target_column], y_pred)
                imbalance_base_scores.append(ba_score)
            elif metric == "worst":
                group_accuracies = []
                for group_name, group_data in test_groups.items():
                    if len(group_data) > 0:
                        group_pred = clf.predict(group_data.drop(columns=[target_column]))
                        group_acc = accuracy_score(group_data[target_column], group_pred)
                        group_accuracies.append(group_acc)
                    else:
                        group_accuracies.append(0)
                imbalance_base_scores.append(min(group_accuracies))
            elif metric == "min":
                minority_groups = ['00', '11']
                combined_minority_data = pd.concat([test_groups[grp] for grp in minority_groups if grp in test_groups and len(test_groups[grp]) > 0])

                if len(combined_minority_data) > 0:
                    minority_pred = clf.predict(combined_minority_data.drop(columns=[target_column]))
                    imbalance_base_scores.append(accuracy_score(combined_minority_data[target_column], minority_pred))
                else:
                    imbalance_base_scores.append(0)

            if num_samples_seed > 0:
                synthetic_samples = synthetic_pos_balance.sample(n=num_samples_seed, random_state=seed)
                balance_base = pd.concat([imbalance_base, synthetic_samples])
            else:
                balance_base = imbalance_base.copy()

            X_train_balance = balance_base.drop(columns=[target_column])
            y_train_balance = balance_base[target_column]

            clf = get_classifier(classifier_type, seed)
            clf.fit(X_train_balance, y_train_balance)

            y_pred = clf.predict(test_data.drop(columns=[target_column]))

            if metric == "balanced":
                ba_score = balanced_accuracy_score(test_data[target_column], y_pred)
                balance_base_scores.append(ba_score)
            elif metric == "worst":
                group_accuracies = []
                for group_name, group_data in test_groups.items():
                    if len(group_data) > 0:
                        group_pred = clf.predict(group_data.drop(columns=[target_column]))
                        group_acc = accuracy_score(group_data[target_column], group_pred)
                        group_accuracies.append(group_acc)
                    else:
                        group_accuracies.append(0)
                balance_base_scores.append(min(group_accuracies))
            elif metric == "min":
                minority_groups = ['00', '11']
                combined_minority_data = pd.concat([test_groups[grp] for grp in minority_groups if grp in test_groups and len(test_groups[grp]) > 0])

                if len(combined_minority_data) > 0:
                    minority_pred = clf.predict(combined_minority_data.drop(columns=[target_column]))
                    balance_base_scores.append(accuracy_score(combined_minority_data[target_column], minority_pred))
                else:
                    balance_base_scores.append(0)

            if num_samples_augment > 0:
                half_augment = int(num_samples_augment / 2)
                augment_samples_pos = synthetic_pos_remain.sample(n=half_augment, random_state=seed)
                augment_samples_neg = synthetic_neg.sample(n=half_augment, random_state=seed)
                augment_samples = pd.concat([augment_samples_pos, augment_samples_neg], ignore_index=True)
                balance_augment = pd.concat([balance_base, augment_samples])
                sample_weights = np.concatenate([np.ones(len(balance_base)), np.full(len(augment_samples), augment_w)])
            else:
                balance_augment = balance_base.copy()
                sample_weights = np.ones(len(balance_base))

            X_train_augment = balance_augment.drop(columns=[target_column])
            y_train_augment = balance_augment[target_column]

            clf = get_classifier(classifier_type, seed)
            clf.fit(X_train_augment, y_train_augment, sample_weight=sample_weights)

            y_pred = clf.predict(test_data.drop(columns=[target_column]))

            if metric == "balanced":
                ba_score = balanced_accuracy_score(test_data[target_column], y_pred)
                balance_augment_scores.append(ba_score)
            elif metric == "worst":
                group_accuracies = []
                for group_name, group_data in test_groups.items():
                    if len(group_data) > 0:
                        group_pred = clf.predict(group_data.drop(columns=[target_column]))
                        group_acc = accuracy_score(group_data[target_column], group_pred)
                        group_accuracies.append(group_acc)
                    else:
                        group_accuracies.append(0)
                balance_augment_scores.append(min(group_accuracies))
            elif metric == "min":
                minority_groups = ['00', '11']
                combined_minority_data = pd.concat([test_groups[grp] for grp in minority_groups if grp in test_groups and len(test_groups[grp]) > 0])

                if len(combined_minority_data) > 0:
                    minority_pred = clf.predict(combined_minority_data.drop(columns=[target_column]))
                    balance_augment_scores.append(accuracy_score(combined_minority_data[target_column], minority_pred))
                else:
                    balance_augment_scores.append(0)

        results[i] = {
            'imbalance_base_mean': np.mean(imbalance_base_scores),
            'imbalance_base_std': np.std(imbalance_base_scores),
            'balance_base_mean': np.mean(balance_base_scores),
            'balance_base_std': np.std(balance_base_scores),
            'balance_augment_mean': np.mean(balance_augment_scores),
            'balance_augment_std': np.std(balance_augment_scores)
        }

    with open(exp_results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    
    
    
def get_classifier(classifier_type, random_state):
    if classifier_type == 'log':
        return LogisticRegression(random_state=random_state, max_iter=1000)
    elif classifier_type == 'rf':
        return RandomForestClassifier(random_state=random_state)
    elif classifier_type == 'cat':
        return CatBoostClassifier(random_state=random_state, verbose=0)
    elif classifier_type == 'gb':
        return GradientBoostingClassifier(random_state=random_state)
    elif classifier_type == 'xgb':
        return XGBClassifier(random_state=random_state)
    else:
        raise ValueError(f"Unsupported classifier type: {classifier_type}")



if __name__ == "__main__":
    
    data_names = {0: "craft", 1: "gender", 2: "diabetes", 3: "adult"}
    
    add_data_ls = {0: "synthetic_allseed", 1: "smote", 2: "adasyn", 3: "ros"}
    
    classifier_types = {0: "log", 1: "rf", 2: "xgb"}
    
    metrics = {0: "balanced", 1: "worst", 2: "min"}
    
    metric = metrics[1]
    
    for idx_metric in [0, 1, 2]:
        metric = metrics[idx_metric]
        for idx_data_name in [2, 3]:
            data_name = data_names[idx_data_name]
            for idx_add_data in [0, 1, 2, 3]:
                for idx_classifier in [0, 1, 2]:
                    add_data = add_data_ls[idx_add_data]
                    classifier_type = classifier_types[idx_classifier]
                    
                    info = load_data_info('data_info.json')
                    data_info = info.get(data_name, {})
                    seed_size = data_info.get('seed_size')
                    label_col = data_info.get('label_col')
                    spurious_col = data_info.get('spurious_col')
                    spurious_small_is_0 = data_info.get('spurious_small_is_0')
                    parent_path = data_info.get('path')

                    
                    
                    perform_experiment(
                        parent_path,
                        seed_size,
                        42,
                        classifier_type,
                        spurious_column=spurious_col,
                        spurious_small_is_0=spurious_small_is_0,
                        target_column=label_col,
                        add_data=add_data,
                        metric=metric
                    )
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



def divide_test_data(test_data, label_column='Target'):
    
    groups = {grp: df for grp, df in test_data.groupby(test_data[label_column])}

    for grp in [0, 1]:
        if grp not in groups:
            groups[grp] = pd.DataFrame(columns=test_data.columns)

    return groups


def perform_experiment(parent_path, seed_size, random_seed, classifier_type, target_column='Target', add_data="synthetic", metric="balanced", step = 1, augment_w = 0.5):
    np.random.seed(random_seed)

    exp_results_path = os.path.join(parent_path, f"imbalanced_class_vary_additional_{classifier_type}_{add_data}_{metric}.json")
    print(exp_results_path)
    seed_path = os.path.join(parent_path, "seed.csv")
    train0_path = os.path.join(parent_path, "train0.csv")
    train_all_path = os.path.join(parent_path, "train.csv")
    synthetic1_path = os.path.join(parent_path, f"{add_data}1.csv")
    synthetic0_path = os.path.join(parent_path, f"{add_data}0.csv")
    test_path = os.path.join(parent_path, "test.csv")


    seed_data = pd.read_csv(seed_path)
    train0 = pd.read_csv(train0_path)
    train_all = pd.read_csv(train_all_path)
    synthetic1 = pd.read_csv(synthetic1_path)
    synthetic0 = pd.read_csv(synthetic0_path)
    test_data = pd.read_csv(test_path)

    test_groups = divide_test_data(test_data.copy(), target_column)

    train0_use = train0.sample(n=5 * seed_size, random_state=random_seed)
    synthetic1_balance = synthetic1.sample(n=5 * seed_size, random_state=random_seed)
    synthetic1_remain = synthetic1.drop(synthetic1_balance.index)
    
    num_samples_seed = 4*seed_size
    train0_samples = train0_use.sample(n=num_samples_seed, random_state=random_seed)
    imbalance_base = pd.concat([seed_data, train0_samples])

    results = {}

    for i in range(10):
        print(i)
        
        balance_augment_scores = []
        

        for seed in [1, 2, 6, 8, 42]:
            np.random.seed(seed)

            num_samples_augment = i * seed_size * step

            
            if num_samples_seed > 0:
                synthetic_samples = synthetic1_balance.sample(n=num_samples_seed, random_state=seed)
                balance_base = pd.concat([imbalance_base, synthetic_samples])
            else:
                balance_base = imbalance_base.copy()


            if num_samples_augment > 0:
                half_augment = int(num_samples_augment / 2)
                augment_samples_pos = synthetic1_remain.sample(n=half_augment, random_state=seed)
                augment_samples_neg = synthetic0.sample(n=half_augment, random_state=seed)
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
                if len(test_groups[1]) > 0:
                    group_pred = clf.predict(test_groups[1].drop(columns=[target_column]))
                    group_acc = accuracy_score(test_groups[1][target_column], group_pred)
                else:
                    group_acc = 0  
                balance_augment_scores.append(group_acc)


        results[i] = {
            'balance_augment_mean': np.mean(balance_augment_scores),
            'balance_augment_std': np.std(balance_augment_scores)
        }
        
        
    X_train_all = train_all.drop(columns=[target_column])
    y_train_all = train_all[target_column]

    clf = get_classifier(classifier_type, seed)
    clf.fit(X_train_all, y_train_all)

    y_pred = clf.predict(test_data.drop(columns=[target_column]))

    if metric == "balanced":
        train_all_score = balanced_accuracy_score(test_data[target_column], y_pred)
    elif metric == "worst":
        group_accuracies = []
        for group_name, group_data in test_groups.items():
            if len(group_data) > 0:
                group_pred = clf.predict(group_data.drop(columns=[target_column]))
                group_acc = accuracy_score(group_data[target_column], group_pred)
                group_accuracies.append(group_acc)
            else:
                group_accuracies.append(0)
        train_all_score = min(group_accuracies)
    elif metric == "min":
        if len(test_groups[1]) > 0:
            group_pred = clf.predict(test_groups[1].drop(columns=[target_column]))
            group_acc = accuracy_score(test_groups[1][target_column], group_pred)
        else:
            group_acc = 0  
        train_all_score = group_acc
        
    results[i+1] = {
        'train_all_augment': train_all_score
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
    
    idx_add_data = 0
    
    
    for idx_metric in [0, 2]:
        metric = metrics[idx_metric]
        for idx_data_name in [2, 3]:
            data_name = data_names[idx_data_name]
            for idx_classifier in [0, 1, 2]:
                add_data = add_data_ls[idx_add_data]
                classifier_type = classifier_types[idx_classifier]
                
                info = load_data_info('data_info.json')
                data_info = info.get(data_name, {})
                seed_size = data_info.get('seed_size')
                label_col = data_info.get('label_col')
                parent_path = data_info.get('path')

                
                
                perform_experiment(
                    parent_path,
                    seed_size,
                    42,
                    classifier_type,
                    target_column=label_col,
                    add_data=add_data,
                    metric=metric,
                    step=2
                )

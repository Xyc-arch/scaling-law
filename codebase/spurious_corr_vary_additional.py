import pandas as pd
import numpy as np
import os
import json
from sklearn.metrics import balanced_accuracy_score, accuracy_score, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
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


def perform_experiment(parent_path, seed_size, random_seed, classifier_type, spurious_column, spurious_small_is_0, target_column='Target', add_data="synthetic", metric="balanced", step = 1, augment_w = 0.5):
    np.random.seed(random_seed)

    exp_results_path = os.path.join(parent_path, f"spurious_corr_vary_additional_{classifier_type}_{add_data}_{metric}.json")
    print(exp_results_path)
    seed_path = os.path.join(parent_path, "seed.csv")
    train_neg_path = os.path.join(parent_path, "train_neg.csv")
    train_all_path = os.path.join(parent_path, "train.csv")
    synthetic_pos_path = os.path.join(parent_path, f"{add_data}_pos.csv")
    synthetic_neg_path = os.path.join(parent_path, f"{add_data}_neg.csv")
    test_path = os.path.join(parent_path, "test.csv")


    seed_data = pd.read_csv(seed_path)
    train_neg = pd.read_csv(train_neg_path)
    train_all = pd.read_csv(train_all_path)
    synthetic_pos = pd.read_csv(synthetic_pos_path)
    synthetic_neg = pd.read_csv(synthetic_neg_path)
    test_data = pd.read_csv(test_path)

    test_groups = divide_test_data(test_data.copy(), spurious_column, spurious_small_is_0, target_column)

    train_neg_use = train_neg.sample(n=5 * seed_size, random_state=random_seed)
    synthetic_pos_balance = synthetic_pos.sample(n=5 * seed_size, random_state=random_seed)
    synthetic_pos_remain = synthetic_pos.drop(synthetic_pos_balance.index)
    
    num_samples_seed = 4*seed_size
    train_neg_samples = train_neg_use.sample(n=num_samples_seed, random_state=random_seed)
    imbalance_base = pd.concat([seed_data, train_neg_samples])

    results = {}

    for i in range(10):
        print(i)
        
        balance_augment_scores = []
        

        for seed in [1, 2, 6, 8, 42]:
            np.random.seed(seed)

            num_samples_augment =  i * seed_size * step

            
            if num_samples_seed > 0:
                synthetic_samples = synthetic_pos_balance.sample(n=num_samples_seed, random_state=seed)
                balance_base = pd.concat([imbalance_base, synthetic_samples])
            else:
                balance_base = imbalance_base.copy()


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

            balance_augment_scores.append(loss(metric, test_data, test_groups, target_column, clf, "spurious"))


        results[i] = {
            'balance_augment_mean': np.mean(balance_augment_scores),
            'balance_augment_std': np.std(balance_augment_scores)
        }
        
        
    X_train_all = train_all.drop(columns=[target_column])
    y_train_all = train_all[target_column]

    clf = get_classifier(classifier_type, seed)
    clf.fit(X_train_all, y_train_all)

    train_all_score = loss(metric, test_data, test_groups, target_column, clf, "spurious")
    
            
    results[i+1] = {
        'train_all_augment': train_all_score
    }

    with open(exp_results_path, 'w') as f:
        json.dump(results, f, indent=4)
    

if __name__ == "__main__":
    
    data_names = {0: "craft", 1: "gender", 2: "diabetes"}
    
    add_data_ls = {0: "synthetic_allseed", 1: "smote", 2: "adasyn", 3: "ros"}
    
    classifier_types = {0: "rf", 1: "xgb"}
    
    metrics = {0: "balanced_log_cross", 1: "min_log_cross"}
    
    idx_add_data = 0
    
    
    for idx_metric in [0, 1]:
        metric = metrics[idx_metric]
        for idx_data_name in [0, 1, 2]:
            data_name = data_names[idx_data_name]
            for idx_classifier in [0]:
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
                    metric=metric,
                    step=2
                )

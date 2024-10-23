import json
from sklearn.metrics import balanced_accuracy_score, accuracy_score, log_loss
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
import numpy as np
import pandas as pd


def load_data_info(json_file):
    with open(json_file, 'r') as file:
        data_info = json.load(file)
    return data_info


def loss(metric, test_data, test_groups, target_column, clf, task):
    y_pred = clf.predict(test_data.drop(columns=[target_column]))
    y_pred_prob = clf.predict_proba(test_data.drop(columns=[target_column]))
    if metric in ["balanced_log_cross", "balanced_cross"]:
        true_labels = test_data[target_column]
        class_weights = compute_class_weight(class_weight='balanced',
                                            classes=np.unique(true_labels),
                                            y=true_labels)
        class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}
        sample_weights = np.array([class_weights_dict[label] for label in true_labels])
        weighted_log_loss = log_loss(true_labels, y_pred_prob, sample_weight=sample_weights)
        if metric == "balanced_log_cross":
            return np.log(weighted_log_loss + 1e-15)
        elif metric == "balanced_cross":
            return weighted_log_loss

    
    elif metric in ["min_log_cross", "min_cross"]:
        if task == "imbalance":
            group_pred_prob = clf.predict_proba(test_groups[1].drop(columns=[target_column]))[:, 1] 
            true_labels = np.ones(len(test_groups[1])) 
            min_log_loss = log_loss(true_labels, group_pred_prob, labels=[0, 1])


            if metric == "min_log_cross":
                return np.log(min_log_loss + 1e-15)
            elif metric == "min_cross":
                return min_log_loss
        elif task == "spurious":
            minority_groups = ['00', '11']
            combined_minority_data = pd.concat([test_groups[grp] for grp in minority_groups if grp in test_groups and len(test_groups[grp]) > 0])
            minority_pred_prob = clf.predict_proba(combined_minority_data.drop(columns=[target_column]))
            min_log_loss = log_loss(combined_minority_data[target_column], minority_pred_prob)
            if metric == "min_log_cross":
                return np.log(min_log_loss + 1e-15)
            elif metric == "min_cross":
                return min_log_loss
            
        
    
def get_classifier(classifier_type, random_state):
    if classifier_type == 'log':
        return LogisticRegression(random_state=random_state, max_iter=1000)
    elif classifier_type == 'rf':
        return RandomForestClassifier(random_state=random_state, criterion='entropy')  
    elif classifier_type == 'cat':
        return CatBoostClassifier(random_state=random_state, verbose=0)
    elif classifier_type == 'gb':
        return GradientBoostingClassifier(random_state=random_state)
    elif classifier_type == 'xgb':
        return XGBClassifier(random_state=random_state)
    else:
        raise ValueError(f"Unsupported classifier type: {classifier_type}")


        

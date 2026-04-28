import os
import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from utils import *

def plot_query_data_multiple(data_name, json_dir, task, classifier, metric):
    
    metric_names = {
        "balanced": "balanced accuracy", 
        "worst": "worst accuracy", 
        "min": "minority accuracy", 
        "balanced_log_cross": "balanced log cross entropy loss", 
        "min_log_cross": "minority log cross entropy loss"
    }
    metric_name = metric_names[metric]
        
    task_names = {
        "spurious_corr_vary_additional": "Spurious Corr Augment Additional", 
        "imbalanced_class_vary_additional": "Imbalanced Class Augment Additional"
    }
    task_name = task_names[task]
    
    clf_names = {"xgb": "XGBoost", "rf": "Random Forest"}
    clf_name = clf_names[classifier]
    
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'pink']

    plt.figure(figsize=(10, 6))
    file_name = f"{task}_{classifier}_synthetic_allseed_{metric}.json"
    file_path = os.path.join(json_dir, file_name)
        
    if not os.path.exists(file_path):
        raise ValueError(f"File {file_name} not found in {json_dir}.")
    
    means = []
    stds = []

    with open(file_path, 'r') as f:
        data = json.load(f)

    for i in range(10):
        key = str(i)
        mean_key = "balance_augment_mean"
        std_key = "balance_augment_std"

        means.append(data[key][mean_key])
        stds.append(data[key][std_key])

    means = np.array(means)
    stds = np.array(stds)

    traces = np.arange(1, 4001, 400)
    
    if metric in ["balanced_log_cross", "min_log_cross"]:
        traces = np.log(traces)

    # Plotting the error bars without connecting lines
    plt.errorbar(traces, means, yerr=stds, fmt='o', color=colors[0], capsize=5, label="augment additional", linestyle='None')
    
    if metric in ["balanced_log_cross", "min_log_cross"]:
        x_margin = (traces[-1] - traces[1]) * 0.05
        plt.xlim(traces[1] - x_margin, traces[-1] + x_margin)

    # Fitting a linear regression from the second point to the last point
    x_fit = traces[1:].reshape(-1, 1)  # From the second point to the last point
    y_fit = means[1:]
    
    linear_model = LinearRegression()
    linear_model.fit(x_fit, y_fit)
    
    # Generate line values for the linear regression
    y_pred = linear_model.predict(x_fit)
    
    plt.plot(x_fit, y_pred, color='blue', label='fitted linear line')
    
    # Plotting the horizontal line for the 10th entry
    if "train_all_augment" in data["10"]:
        plt.axhline(y=data["10"]["train_all_augment"], color='r', linestyle='--', label=f'train all real {data["10"]["train_all_augment"]: .3f}')

    # Plotting the horizontal line through the i=0 "balance_augment_mean"
    plt.axhline(y=means[0], color='g', linestyle=':', label=f'balanced mean without additional, {means[0]:.3f}')

    plt.xlabel('# additional augmented synthetic samples', fontsize = 14)
    if metric in ["balanced_log_cross", "min_log_cross"]:
        plt.xlabel('log of # additional augmented synthetic samples', fontsize = 14)
    plt.ylabel(f'{metric_name}', fontsize = 14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title(f'Performance for Task: {task_name}, Classifier: {clf_name}')
    plt.legend(fontsize = 12)
    plt.grid(True)
    
    output_dir = f"/home/ubuntu/plots/{data_name}"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/{task}_{classifier}_{metric}.png")
    plt.show()


if __name__ == "__main__":
    
    data_names = {0: "craft", 1: "gender", 2: "diabetes", 3: "adult"}
    
    tasks = {0: "spurious_corr_vary_additional", 1: "imbalanced_class_vary_additional"}
    
    # metrics = {0: "balanced", 1: "worst", 2: "min"}
    
    metrics = {0: "balanced_log_cross", 1: "min_log_cross"}
    
    classifier_types = {0: "rf", 1: "xgb"}
    
    for idx_data_name in [0, 1, 2, 3]:
        data_name = data_names[idx_data_name]
        for idx_task in [0, 1]:
            task = tasks[idx_task]
            if data_name == "adult" and task == "spurious_corr_vary_additional":
                continue
            for idx_metric in [0, 1]:
                metric = metrics[idx_metric]
                for idx_classifier in [0, 1]:
                    classifier = classifier_types[idx_classifier]
        
                    info = load_data_info('data_info.json')
                    data_info = info.get(data_name, {})
                    path = data_info.get('path')
                    seed_size = data_info.get('seed_size')
                    

                    json_dir = path 
                    
                    plot_query_data_multiple(data_name, json_dir, task, classifier, metric)

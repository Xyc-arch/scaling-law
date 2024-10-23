import os
import json
import matplotlib.pyplot as plt
import numpy as np
from utils import *

def plot_query_data_multiple(data_name, json_dir, task, classifier, metric):
    
    metric_names = {"balanced": "Balanced Accuracy", "worst": "Worst Accuracy", "min": "Minority Accuracy"}
    metric_name = metric_names[metric]
        
    task_names = {"spurious_corr_vary_additional": "spruious corr augment additional", "imbalanced_class_vary_additional": "imbalanced class augment additional"}
    task_name = task_names[task]
    
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

    traces = np.arange(0, 4000, 400)

    plt.errorbar(traces, means, yerr=stds, fmt='-o', color=colors[0], capsize=5, label="augment additional")

    # Plotting the horizontal line for the 10th entry
    if "train_all_augment" in data["10"]:
        plt.axhline(y=data["10"]["train_all_augment"], color='r', linestyle='--', label=f'Train all {data["10"]["train_all_augment"]: .3f}')

    # Plotting the horizontal line through the i=0 "balance_augment_mean"
    plt.axhline(y=means[0], color='g', linestyle=':', label=f'Balanced augment mean at 0, {means[0]:.3f}')

    plt.xlabel('# additional augmented synthetic samples')
    plt.ylabel(f'{metric_name}')
    plt.title(f'Performance for Task: {task_name}, Classifier: {classifier}')
    plt.legend()
    plt.grid(True)
    
    output_dir = f"/home/ubuntu/plots/{data_name}"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/{task}_{classifier}_{metric}.png")
    plt.show()


if __name__ == "__main__":
    
    data_names = {0: "craft", 1: "gender", 2: "diabetes", 3: "adult"}
    
    tasks = {0: "spurious_corr_vary_additional", 1: "imbalanced_class_vary_additional"}
    
    metrics = {0: "balanced", 1: "worst", 2: "min"}
    
    classifier_types = {0: "log", 1: "rf", 2: "xgb"}
    
    for idx_data_name in [0, 1, 2, 3]:
        data_name = data_names[idx_data_name]
        for idx_task in [0, 1]:
            task = tasks[idx_task]
            for idx_metric in [0, 2]:
                metric = metrics[idx_metric]
                for idx_classifier in [0, 1, 2]:
                    classifier = classifier_types[idx_classifier]
        
                    info = load_data_info('data_info.json')
                    data_info = info.get(data_name, {})
                    path = data_info.get('path')
                    seed_size = data_info.get('seed_size')
                    

                    json_dir = path 
                    
                    plot_query_data_multiple(data_name, json_dir, task, classifier, metric)

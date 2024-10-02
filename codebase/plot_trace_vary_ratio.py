import os
import json
import matplotlib.pyplot as plt
import numpy as np
from utils import *

def plot_query_data_multiple(data_name, json_dir, task, classifier, queries, metric):
    
    metric_names = {"balanced": "Balanced Accuracy", "worst": "Worst Accuracy", "min": "Minority Accuracy"}
    metric_name = metric_names[metric]
        
    task_names = {"spurious_corr_vary_ratio": "spruious corr varying ratio", "imbalanced_class_vary_ratio": "imbalanced class varying ratio"}
    task_name = task_names[task]
    
    
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'pink']

    plt.figure(figsize=(10, 6))

    for idx, query in enumerate(queries):
        file_name = f"{task}_{classifier}_{query[0]}_{metric}.json"
        file_path = os.path.join(json_dir, file_name)
        
        if not os.path.exists(file_path):
            raise ValueError(f"File {file_name} not found in {json_dir}.")
        
        means = []
        stds = []

        with open(file_path, 'r') as f:
            data = json.load(f)

        for i in range(10):
            key = str(i)
            mean_key = f"{query[1]}_mean"
            std_key = f"{query[1]}_std"

            means.append(data[key][mean_key])
            stds.append(data[key][std_key])

        means = np.array(means)
        stds = np.array(stds)

        traces = np.arange(1, 11, 1)  # X-axis seed sizes
        if query[1] == "imbalance_base":
            legend_name = "raw"
        elif query[1] == "balance_augment" and query[0] == "synthetic_allseed":
            legend_name = "LLM balance + augment"
        elif query[0] == "synthetic_allseed":
            legend_name = "LLM balance"
        else:
            legend_name = query[0]
        

        plt.errorbar(traces, means, yerr=stds, fmt='-o', color=colors[idx % len(colors)], capsize=5, label=legend_name)

    plt.xlabel('Maj/Min')
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
    
    tasks = {0: "spurious_corr_vary_ratio", 1: "imbalanced_class_vary_ratio"}
    task = tasks[1]
    
    metrics = {0: "balanced", 1: "worst", 2: "min"}
    metric = metrics[1]
    
    add_data_ls = {0: "synthetic_allseed", 1: "smote", 2: "adasyn", 3: "ros"}
    
    classifier_types = {0: "log", 1: "rf", 2: "xgb"}
    
    classifier = classifier_types[2]
    
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
                    
                    q = [("smote", "balance_base"), ("smote", "imbalance_base"), ("adasyn", "balance_base"), ("synthetic_allseed", "balance_base"), ("ros", "balance_base"), ("synthetic_allseed", "balance_augment")]
                    plot_query_data_multiple(data_name, json_dir, task, classifier, q, metric)
                    
                    

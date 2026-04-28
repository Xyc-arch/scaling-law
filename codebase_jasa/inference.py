import time
import os
import pandas as pd
from be_great import GReaT
from utils import *

data_names = {0: "craft", 1: "gender", 2: "diabetes", 3: "spam"}
data_name = data_names[3]

for idx_data in [2]:
    
    data_name = data_names[idx_data]

    info = load_data_info('data_info.json')
    data_info = info.get(data_name, {})
    syn_samples = data_info.get('syn_samples')

    # Define paths for model save directory and synthetic data
    model_save_path = f"../model/{data_name}"
    synthetic_data_save_path = f"../data/{data_name}/synthetic.csv"
    os.makedirs(os.path.dirname(synthetic_data_save_path), exist_ok=True)

    # Timer to measure execution time
    start_time = time.time()

    # Step 1: Reinitialize a new model instance and load from the saved model
    print("Loading the model...")
    model = GReaT.load_from_dir(path=model_save_path)
    print(f"Model loaded from {model_save_path}")

    # Step 2: Sample synthetic data from the loaded model
    synthetic_data = model.sample(n_samples=syn_samples, max_length=2000)

    # # Step 3: Save the synthetic data to a CSV file
    # synthetic_data.to_csv(synthetic_data_save_path, index=False)
    # print(f"Synthetic data saved to {synthetic_data_save_path}")

    # Measure and print the elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken {data_name}: {elapsed_time} seconds")

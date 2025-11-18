# ablate_ind_great.py
import time
import os
from be_great import GReaT
import pandas as pd
from utils import *

import torch
print("CUDA available?", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())

data_names = {0: "craft", 1: "gender", 2: "diabetes", 3: "adult"}

for idx_data in [2]:   # SAME LOOP ORDER AS ORIGINAL
    data_name = data_names[idx_data]

    # Load dataset-specific config
    info = load_data_info('data_info.json')
    data_info = info.get(data_name, {})
    batch_size = data_info.get('batch_size')
    epochs = data_info.get('epochs')
    syn_samples = data_info.get('syn_samples')

    print(f"\n=== Training GReaT on IND seed for: {data_name} ===")

    # ---- IMPORTANT: Only this part is changed ----
    seed_data_path = f"../data/{data_name}/seed_ind.csv"
    model_save_path = f"../model/{data_name}_ind"
    synthetic_data_save_path = f"../data/{data_name}/synthetic_seed_ind.csv"
    # ------------------------------------------------

    os.makedirs(os.path.dirname(synthetic_data_save_path), exist_ok=True)

    # Timer
    start_time = time.time()

    # Load IND seed
    data = pd.read_csv(seed_data_path)
    data = data.sample(n=5000, random_state=0).reset_index(drop=True)
    print(data)

    # Train model
    print("Training the model...")
    model = GReaT(llm='gpt2', batch_size=batch_size, epochs=epochs, save_steps=400000)
    model.fit(data)

    # Save model
    model.save(path=model_save_path)
    print(f"Model saved to {model_save_path}")

    # Reload model
    print("Reinitializing the model and loading from saved model...")
    new_model = GReaT.load_from_dir(path=model_save_path)
    print(f"Model loaded from {model_save_path}")

    # Sample synthetic data
    synthetic_data = new_model.sample(n_samples=syn_samples, max_length=2000)

    # Save synthetic
    synthetic_data.to_csv(synthetic_data_save_path, index=False)
    print(f"Synthetic data saved to {synthetic_data_save_path}")

    # Time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken {data_name}: {elapsed_time} seconds")

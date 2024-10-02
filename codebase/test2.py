import time
import os
from be_great import GReaT
from sklearn.datasets import fetch_california_housing
import pandas as pd

# Define paths for data, model save directory, and synthetic data
data_save_path = "../data/test/data.csv"
model_save_path = "../model/california_llm"
synthetic_data_save_path = "../data/test/synthetic_data.csv"
os.makedirs(os.path.dirname(synthetic_data_save_path), exist_ok=True)

# Timer to measure execution time
start_time = time.time()

data = pd.read_csv(data_save_path)[:100]

# Step 2: Initialize and train the model
print("Training the model...")
model = GReaT(llm='gpt2', batch_size=16, epochs=200)
model.fit(data)

# Step 3: Save the trained model
model.save(path=model_save_path)
print(f"Model saved to {model_save_path}")

# Step 4: Reinitialize a new model instance and load from the saved model
print("Reinitializing the model and loading from saved model...")
new_model = GReaT.load_from_dir(path=model_save_path)
print(f"Model loaded from {model_save_path}")

# Step 5: Sample synthetic data from the loaded model
synthetic_data = new_model.sample(n_samples=4000)

# Step 6: Save the synthetic data to a CSV file
synthetic_data.to_csv(synthetic_data_save_path, index=False)
print(f"Synthetic data saved to {synthetic_data_save_path}")

# Measure and print the elapsed time
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Time taken to run the whole script: {elapsed_time} seconds")

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import os
from utils import *


# Function to calculate RMSE
def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


# Function to load data, train models, and calculate RMSE for 5 random seeds
def train_and_evaluate_random_seeds(data_name, method='linear', seed_list=[1, 2, 6, 8, 42], synthetic_sample_size=1000):
    
    # Load the paths and seed size from data_info.json
    info = load_data_info('data_info.json')
    data_info = info.get(data_name, {})
    
    parent_path = data_info.get('path')
    seed_size = data_info.get('seed_size', 100)  # Default seed size if not in the JSON

    # Load data
    train_data_path = os.path.join(parent_path, "train.csv")
    synthetic_data_path = os.path.join(parent_path, "synthetic.csv")
    test_data_path = os.path.join(parent_path, "test.csv")
    
    train_data = pd.read_csv(train_data_path)
    synthetic_data = pd.read_csv(synthetic_data_path)
    test_data = pd.read_csv(test_data_path)

    # Split the test data into features and target
    X_test = test_data.iloc[:, :-1]    # Features
    y_test = test_data.iloc[:, -1]     # Target

    # To store RMSE for each seed
    rmse_results_A_train = []
    rmse_results_B_train = []
    rmse_results_combined_train = []

    # Define model types
    model_map = {
        'linear': LinearRegression,
        'rf': RandomForestRegressor,
        'xgb': XGBRegressor
    }

    if method not in model_map:
        raise ValueError("Invalid method. Choose from 'linear', 'rf', or 'xgb'.")

    for seed in seed_list:
        # Set random seed
        np.random.seed(seed)

        # Sample seed data from the train data (A)
        train_sample = train_data.sample(n=seed_size, random_state=seed)
        X_train_seed = train_sample.iloc[:, :-1]  # Features from seed data
        y_train_seed = train_sample.iloc[:, -1]   # Target variable from seed data

        # Sample 1000 rows from synthetic data (B)
        synthetic_sample = synthetic_data.sample(n=synthetic_sample_size, random_state=seed)
        X_train_synthetic = synthetic_sample.iloc[:, :-1]  # Features from synthetic data
        y_train_synthetic = synthetic_sample.iloc[:, -1]   # Target variable from synthetic data

        # ---------------------------
        # 1. Train solely on seed data
        model_seed_train = model_map[method]()
        model_seed_train.fit(X_train_seed, y_train_seed)

        # Predict on the test set
        y_pred_seed_train = model_seed_train.predict(X_test)
        rmse_seed_train = calculate_rmse(y_test, y_pred_seed_train)
        rmse_results_A_train.append(rmse_seed_train)
        
        # ---------------------------
        # 2. Train solely on synthetic data
        model_synthetic_train = model_map[method]()
        model_synthetic_train.fit(X_train_synthetic, y_train_synthetic)

        # Predict on the test set
        y_pred_synthetic_train = model_synthetic_train.predict(X_test)
        rmse_synthetic_train = calculate_rmse(y_test, y_pred_synthetic_train)
        rmse_results_B_train.append(rmse_synthetic_train)

        # ---------------------------
        # 3. Train on seed data concatenated with synthetic data
        X_train_combined = pd.concat([X_train_seed, X_train_synthetic], axis=0)
        y_train_combined = pd.concat([y_train_seed, y_train_synthetic], axis=0)

        model_combined_train = model_map[method]()
        model_combined_train.fit(X_train_combined, y_train_combined)

        # Predict on the test set
        y_pred_combined_train = model_combined_train.predict(X_test)
        rmse_combined_train = calculate_rmse(y_test, y_pred_combined_train)
        rmse_results_combined_train.append(rmse_combined_train)
        
    # Calculate mean and std of RMSE for each case
    rmse_seed_train_mean = np.mean(rmse_results_A_train)
    rmse_seed_train_std = np.std(rmse_results_A_train)

    rmse_synthetic_train_mean = np.mean(rmse_results_B_train)
    rmse_synthetic_train_std = np.std(rmse_results_B_train)

    rmse_combined_train_mean = np.mean(rmse_results_combined_train)
    rmse_combined_train_std = np.std(rmse_results_combined_train)

    # Print the results
    print(f"RMSE for Seed Data (Mean ± Std): {rmse_seed_train_mean:.4f} ± {rmse_seed_train_std:.4f}")
    print(f"RMSE for Synthetic Data (Mean ± Std): {rmse_synthetic_train_mean:.4f} ± {rmse_synthetic_train_std:.4f}")
    print(f"RMSE for Combined Data (Mean ± Std): {rmse_combined_train_mean:.4f} ± {rmse_combined_train_std:.4f}")
if __name__ == "__main__":

    data_names = {0: "california", 1: "abalone", 2: "wine"}
    
    data_name = data_names[1]
    
    methods = {0: "linear", 1: "rf", 2: "xgb"}
    
    method = methods[1]
    train_and_evaluate_random_seeds(data_name, method, synthetic_sample_size=100)

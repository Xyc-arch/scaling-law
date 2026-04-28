import pandas as pd
import numpy as np
from utils import *

def train_seed_test_split(file_path, train_ratio, seed_size, output_seed, output_test, output_train, output_remain, random_seed=42):
    # Load the data
    df = pd.read_csv(file_path)
    
    # Shuffle the data
    df_shuffled = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    # Split into train and test sets based on train_ratio
    train_size = int(len(df_shuffled) * train_ratio)
    train_df = df_shuffled.iloc[:train_size]
    test_df = df_shuffled.iloc[train_size:]
    
    # Save test data
    test_df.to_csv(output_test, index=False)
    
    # Now split the train set into seed and remaining train
    seed_df = train_df.iloc[:seed_size]
    remain_train_df = train_df.iloc[seed_size:]
    
    # Save seed and remaining train data
    seed_df.to_csv(output_seed, index=False)
    remain_train_df.to_csv(output_remain, index=False)
    
    # Save full train data (including seed and remaining)
    train_df.to_csv(output_train, index=False)


if __name__ == "__main__":
    
    # Example configuration
    data_names = {0: "california", 1: "abalone", 2: "wine"}
    data_name = data_names[2]
    
    # Load data info
    info = load_data_info('data_info.json')
    data_info = info.get(data_name, {})
    
    # Extract necessary information from data_info
    seed_size = data_info.get('seed_size')
    train_ratio = data_info.get('train_ratio', 0.7)  # Default to 80% train, 20% test if not specified
    label_col = data_info.get('label_col')
    
    # Define paths
    parent_path = "/home/ubuntu/"
    file_path = parent_path + f"/data/{data_name}/{data_name}.csv"
    output_seed = parent_path + f"/data/{data_name}/seed.csv"
    output_test = parent_path + f"/data/{data_name}/test.csv"
    output_train = parent_path + f"/data/{data_name}/train.csv"
    output_remain = parent_path + f"/data/{data_name}/remain.csv"
    
    # Perform train-test-seed split
    train_seed_test_split(file_path, train_ratio, seed_size, output_seed, output_test, output_train, output_remain)

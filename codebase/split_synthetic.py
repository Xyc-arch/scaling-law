import pandas as pd
import numpy as np
import json
from utils import *


def split_and_save_syn(file_path, output_train1, output_train0, output_pos, output_neg, output_data_size, random_seed=42, label_column='Target', spurious_column='X6', spurious_small_is_0=1):
    
    # Read data
    data = pd.read_csv(file_path)

    # Determine the small value for the spurious column
    if spurious_small_is_0 == 1:
        small_value = 0
    else:
        small_value = data[spurious_column].min()

    # Create a binary indicator for the spurious column
    data['spurious_value'] = data[spurious_column].apply(lambda x: 0 if x == small_value else 1)

    # Create group labels based on spurious_value and label_column
    data['group'] = data['spurious_value'].astype(str) + data[label_column].astype(int).astype(str)

    # Split data into the 4 groups
    groups = {grp: df for grp, df in data.groupby('group')}

    # Ensure all groups are present
    for grp in ['00', '01', '10', '11']:
        if grp not in groups:
            groups[grp] = pd.DataFrame(columns=data.columns)
            print(f"Warning: Group {grp} is empty.")

    # Prepare subsets from the data
    train_groups = {grp: df for grp, df in data.groupby('group')}
    for grp in ['00', '01', '10', '11']:
        if grp not in train_groups:
            train_groups[grp] = pd.DataFrame(columns=data.columns)

    # Create datasets without the artificially created columns
    output_train1_data = pd.concat([train_groups['01'], train_groups['11']]).drop(['group', 'spurious_value'], axis=1).sample(frac=1, random_state=random_seed)  # Shuffle train1
    output_train0_data = pd.concat([train_groups['00'], train_groups['10']]).drop(['group', 'spurious_value'], axis=1).sample(frac=1, random_state=random_seed)  # Shuffle train0
    output_pos_data = pd.concat([train_groups['00'], train_groups['11']]).drop(['group', 'spurious_value'], axis=1).sample(frac=1, random_state=random_seed)  # Shuffle pos
    output_neg_data = pd.concat([train_groups['01'], train_groups['10']]).drop(['group', 'spurious_value'], axis=1).sample(frac=1, random_state=random_seed)  # Shuffle neg

    # Save the datasets
    output_train1_data.to_csv(output_train1, index=False)
    output_train0_data.to_csv(output_train0, index=False)
    output_pos_data.to_csv(output_pos, index=False)
    output_neg_data.to_csv(output_neg, index=False)

    # Count the sample sizes of each dataset
    data_size_info = {
        "train1_size": len(output_train1_data),
        "train0_size": len(output_train0_data),
        "pos_size": len(output_pos_data),
        "neg_size": len(output_neg_data)
    }

    # Save the data sizes in JSON format
    with open(output_data_size, "w") as json_file:
        json.dump(data_size_info, json_file, indent=4)






if __name__ == "__main__":
    
    data_names = {0: "craft", 1: "gender", 2: "diabetes", 3: "adult"}
    data_name = data_names[3]
    
    add_data_ls = {0: "synthetic_allseed", 1: "smote", 2: "adasyn", 3: "ros"}
    add_data = add_data_ls[0]
    
    info = load_data_info('data_info.json')
    data_info = info.get(data_name, {})
    path = data_info.get('path')
    seed_size = data_info.get('seed_size')
    label_col = data_info.get('label_col')
    spurious_col = data_info.get('spurious_col')
    spurious_small_is_0 = data_info.get('spurious_small_is_0')
    parent_path = "/home/ubuntu/"
    file_path = parent_path + f"/data/{data_name}/{add_data}.csv"
    output_train1 = parent_path + f"/data/{data_name}/{add_data}1.csv"
    output_train0 = parent_path + f"/data/{data_name}/{add_data}0.csv"
    output_neg = parent_path + f"/data/{data_name}/{add_data}_neg.csv"
    output_pos = parent_path + f"/data/{data_name}/{add_data}_pos.csv"
    output_data_size = parent_path + f"/data/{data_name}/{add_data}_data_size.json"
    
    
    split_and_save_syn(file_path, output_train1, output_train0, output_pos, output_neg, output_data_size, random_seed=42, label_column=label_col, spurious_column=spurious_col, spurious_small_is_0=spurious_small_is_0)

    



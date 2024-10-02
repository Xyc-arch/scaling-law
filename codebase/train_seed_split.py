import pandas as pd
import numpy as np
from utils import *

def train_seed_split(file_path, seed_size, output_seed, output_train, output_train1, output_train0, random_seed=42, label_column='label'):
    df = pd.read_csv(file_path)
    df_shuffled = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    # Split seed_size samples for the seed dataset
    seed_df = df_shuffled.iloc[:seed_size]
    seed_df.to_csv(output_seed, index=False)
    
    # The remaining samples are for training
    train_df = df_shuffled.iloc[seed_size:]
    train_df.to_csv(output_train, index=False)
    
    # Split the training data into label 1 and label 0 datasets
    train_df_label1 = train_df[train_df[label_column] == 1]
    train_df_label0 = train_df[train_df[label_column] == 0]
    
    train_df_label1.to_csv(output_train1, index=False)
    train_df_label0.to_csv(output_train0, index=False)
    


def train_seed_split_balanced(file_path, seed_size, output_seed, output_train, output_train1, output_train0, output_pos, output_neg, output_data_size, random_seed=42, label_column='Target', spurious_column='X6', spurious_small_is_0=1):
    
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
    data['group'] = data['spurious_value'].astype(str) + data[label_column].astype(str)

    # Split data into the 4 groups
    groups = {grp: df for grp, df in data.groupby('group')}

    # Ensure all groups are present
    for grp in ['00', '01', '10', '11']:
        if grp not in groups:
            groups[grp] = pd.DataFrame(columns=data.columns)
            print(f"Warning: Group {grp} is empty.")

    # Calculate seed sizes for each group
    seed_sizes = [seed_size // 4] * 4
    for i in range(seed_size % 4):
        seed_sizes[i] += 1

    # Sample seed data from each group
    np.random.seed(random_seed)
    seed_data_list = []
    for grp_label, n_samples in zip(['00', '01', '10', '11'], seed_sizes):
        group_df = groups[grp_label]
        actual_n_samples = min(n_samples, len(group_df))
        if actual_n_samples < n_samples:
            print(f"Warning: Not enough samples in group {grp_label}, sampling {actual_n_samples} instead of {n_samples}")
        if actual_n_samples > 0:
            seed_samples = group_df.sample(n=actual_n_samples, random_state=random_seed)
            seed_data_list.append(seed_samples)
    seed_data = pd.concat(seed_data_list)

    # Remove seed data from the main dataset to get train data
    train_data = data.drop(seed_data.index)

    # Prepare subsets from the train data
    train_groups = {grp: df for grp, df in train_data.groupby('group')}
    for grp in ['00', '01', '10', '11']:
        if grp not in train_groups:
            train_groups[grp] = pd.DataFrame(columns=train_data.columns)

    # Create datasets without the artificially created columns
    seed_data_clean = seed_data.drop(['group', 'spurious_value'], axis=1)
    train_data_clean = train_data.drop(['group', 'spurious_value'], axis=1)

    output_train1_data = pd.concat([train_groups['01'], train_groups['11']]).drop(['group', 'spurious_value'], axis=1)
    output_train0_data = pd.concat([train_groups['00'], train_groups['10']]).drop(['group', 'spurious_value'], axis=1)
    output_pos_data = pd.concat([train_groups['00'], train_groups['11']]).drop(['group', 'spurious_value'], axis=1)
    output_neg_data = pd.concat([train_groups['01'], train_groups['10']]).drop(['group', 'spurious_value'], axis=1)

    # Save seed and train data
    seed_data_clean = seed_data.drop(['group', 'spurious_value'], axis=1).sample(frac=1, random_state=random_seed)  
    train_data_clean = train_data.drop(['group', 'spurious_value'], axis=1).sample(frac=1, random_state=random_seed) 

    # Save the other datasets
    output_train1_data = pd.concat([train_groups['01'], train_groups['11']]).drop(['group', 'spurious_value'], axis=1).sample(frac=1, random_state=random_seed)  
    output_train0_data = pd.concat([train_groups['00'], train_groups['10']]).drop(['group', 'spurious_value'], axis=1).sample(frac=1, random_state=random_seed)  
    output_pos_data = pd.concat([train_groups['00'], train_groups['11']]).drop(['group', 'spurious_value'], axis=1).sample(frac=1, random_state=random_seed)
    output_neg_data = pd.concat([train_groups['01'], train_groups['10']]).drop(['group', 'spurious_value'], axis=1).sample(frac=1, random_state=random_seed)
    
    
    # save data
    seed_data_clean.to_csv(output_seed, index=False)
    train_data_clean.to_csv(output_train, index=False)
    
    output_train1_data.to_csv(output_train1, index=False)
    output_train0_data.to_csv(output_train0, index=False)
    output_pos_data.to_csv(output_pos, index=False)
    output_neg_data.to_csv(output_neg, index=False)
    


    # Count the sample sizes of each dataset
    data_size_info = {
        "seed_data_size": len(seed_data_clean),
        "train_data_size": len(train_data_clean),
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
    info = load_data_info('data_info.json')
    data_info = info.get(data_name, {})
    seed_size = data_info.get('seed_size')
    label_col = data_info.get('label_col')
    spurious_col = data_info.get('spurious_col')
    spurious_small_is_0 = data_info.get('spurious_small_is_0')
    parent_path = "/home/ubuntu/"
    file_path = parent_path + f"/data/{data_name}/use.csv"
    output_seed = parent_path + f"/data/{data_name}/seed.csv"
    output_train = parent_path + f"/data/{data_name}/train.csv"
    output_train1 = parent_path + f"/data/{data_name}/train1.csv"
    output_train0 = parent_path + f"/data/{data_name}/train0.csv"
    output_neg = parent_path + f"/data/{data_name}/train_neg.csv"
    output_pos = parent_path + f"/data/{data_name}/train_pos.csv"
    output_data_size = parent_path + f"/data/{data_name}/data_size.json"
    
    
    train_seed_split_balanced(file_path, seed_size, output_seed, output_train, output_train1, output_train0, output_pos, output_neg, output_data_size, random_seed=1, label_column=label_col, spurious_column=spurious_col, spurious_small_is_0=spurious_small_is_0)
    





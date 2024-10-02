
import os
import pandas as pd
from utils import *

def calculate_proportions(data_file, target_name, spurious_column, spurious_small_is_0=1):
    # Read data
    data = pd.read_csv(data_file)

    # Determine the small value for the spurious column
    if spurious_small_is_0 == 1:
        small_value = 0
        
    else:
        small_value = data[spurious_column].min()

    # Create a binary indicator for the spurious column
    data['spurious_value'] = data[spurious_column].apply(lambda x: 0 if x == small_value else 1)

    # Create group labels based on spurious_value and the target (label_column)
    data['group'] = data['spurious_value'].astype(str) + data[target_name].astype(str)

    # Split data into the 4 groups
    groups = {grp: df for grp, df in data.groupby('group')}

    # Calculate proportions for each group
    total_proportions = data.groupby(['group', target_name]).size() / len(data)
    total_proportions = total_proportions.reset_index(name='proportion')

    # Output the calculated proportions
    print(total_proportions)


def feature_target_correlation(data_file, target_name):
    # Read data
    data = pd.read_csv(data_file)

    # Calculate correlation of all features with the target
    correlations = data.corr()[target_name].drop(target_name)  

    # Sort by absolute correlation values, from smallest to largest
    sorted_correlations = correlations.reindex(correlations.abs().sort_values().index)

    # Return the sorted correlation values
    return sorted_correlations




if __name__ == "__main__":
    
    data_names = {0: "craft", 1: "gender", 3: "diabetes", 4: "adult"}
    data_name = data_names[3]
    info = load_data_info('data_info.json')
    data_info = info.get(data_name, {})
    seed_size = data_info.get('seed_size')
    label_col = data_info.get('label_col')
    path = data_info.get('path')
    
    data_file = os.path.join(path, f"{data_name}.csv")
    
    calculate_proportions(data_file, target_name = label_col, spurious_column = 'gender', spurious_small_is_0=1)
    
    order_corr = feature_target_correlation(data_file, target_name = label_col)
    print(order_corr)
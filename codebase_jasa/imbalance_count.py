import pandas as pd
from utils import *

def count_labels(file_path, label_column):
    df = pd.read_csv(file_path)
    label_counts = df[label_column].value_counts()
    label_1_count = label_counts.get(1, 0)
    label_0_count = label_counts.get(0, 0)
    print(f"Total number of label 1 samples: {label_1_count}")
    print(f"Total number of label 0 samples: {label_0_count}")
    
    return label_1_count, label_0_count
    
    
if __name__ == "__main__":
    
    
    data_names = {0: "craft", 1: "gender", 2: "diabetes", 3: "heart"}
    data_name = data_names[2]
    
    info = load_data_info('data_info.json')
    
    data_info = info.get(data_name, {})
    path = data_info.get('path')
    label_col = data_info.get('label_col')
    spurious_col = data_info.get('spurious_col')
    
    count_labels(path + f"/use.csv", label_col)


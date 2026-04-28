import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
from utils import *

def encode_categorical_columns(data_file, output_file):
    # Load the dataset
    data = pd.read_csv(data_file)

    # Iterate through each categorical column
    for col in data.select_dtypes(include=['object']).columns:
        unique_values = data[col].nunique()

        if unique_values == 2:
            # Binary column, map to 0 and 1
            data[col] = data[col].map({data[col].unique()[0]: 0, data[col].unique()[1]: 1})
        else:
            # Multiclass column, use Label Encoding (integers starting from 0)
            label_encoder = LabelEncoder()
            data[col] = label_encoder.fit_transform(data[col])

    # Save the encoded dataset to the specified output file
    data.to_csv(output_file, index=False)
    print(f"Encoded dataset saved to {output_file}")



if __name__ == "__main__":
    
    
    data_name = "adult"
    info = load_data_info('data_info.json')
    data_info = info.get(data_name, {})
    seed_size = data_info.get('seed_size')
    label_col = data_info.get('label_col')
    path = data_info.get('path')
    
    input_file = os.path.join(path, f"{data_name}_raw.csv")
    output_file = os.path.join(path, f"{data_name}.csv")
    encode_categorical_columns(input_file, output_file)

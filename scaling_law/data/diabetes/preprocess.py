import pandas as pd
import os

# Set the current working directory to the script's location
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load the dataset
diabetic_data_path = 'diabetes_raw.csv'
diabetic_data = pd.read_csv(diabetic_data_path)

# Replace '>30' and '<30' with 'YES', and keep 'NO' unchanged
diabetic_data['readmitted'] = diabetic_data['readmitted'].replace({'>30': 'YES', '<30': 'YES'})

# Filter rows where 'readmitted' is 'NO' or 'YES'
filtered_data = diabetic_data[diabetic_data['readmitted'].isin(['NO', 'YES'])]

# Identify and drop irrelevant columns
irrelevant_columns = ['encounter_id', 'patient_nbr']  # Exclude IDs that shouldn't be part of the model
data_for_model = filtered_data.drop(columns=irrelevant_columns)

# Print the top several samples (first 5 rows)
print(data_for_model.head())

# Save the prepared dataset for logistic regression as CSV
output_model_data_path = 'diabetes_filter.csv'
data_for_model.to_csv(output_model_data_path, index=False)

# Print the shape of the dataset
print(data_for_model.shape)

# Print the count of each 'readmitted' label category
label_counts = data_for_model.groupby('readmitted').size()
print("\nLabel counts for each 'readmitted' category:")
print(label_counts)

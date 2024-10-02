import pandas as pd
from sklearn.datasets import fetch_openml

# Fetch the dataset
adult_data = fetch_openml(name='adult', version=2, as_frame=True)

# Combine features and target into a single DataFrame
adult_df = pd.concat([adult_data.data, adult_data.target], axis=1)

# Save the DataFrame to a CSV file
adult_df.to_csv('adult.csv', index=False)

print("Dataset saved as 'adult.csv'")

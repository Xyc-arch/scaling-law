import pandas as pd
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from utils import load_data_info

# Function for SMOTE
def generate_smote_data(X_seed, y_seed, desired_size, random_state):
    """
    Generate an oversampled dataset using SMOTE.

    Parameters
    ----------
    X_seed : pandas.DataFrame
        Feature matrix for the seed dataset.
    y_seed : pandas.Series
        Binary labels for the seed dataset.
    desired_size : int
        Desired total number of samples after oversampling, split evenly across the two classes.
    random_state : int
        Random seed used by the SMOTE oversampler.

    Returns
    -------
    tuple
        A tuple containing the resampled feature matrix and resampled label vector.
    """
    oversampler = SMOTE(sampling_strategy={0: desired_size // 2, 1: desired_size // 2}, random_state=random_state)
    X_resampled, y_resampled = oversampler.fit_resample(X_seed, y_seed)
    return X_resampled, y_resampled

# Function for ADASYN
def generate_adasyn_data(X_seed, y_seed, desired_size, random_state):
    """
    Generate an oversampled dataset using ADASYN.

    Parameters
    ----------
    X_seed : pandas.DataFrame
        Feature matrix for the seed dataset.
    y_seed : pandas.Series
        Binary labels for the seed dataset.
    desired_size : int
        Desired total number of samples after oversampling, split evenly across the two classes.
    random_state : int
        Random seed used by the ADASYN oversampler.

    Returns
    -------
    tuple
        A tuple containing the resampled feature matrix and resampled label vector.
    """
    oversampler = ADASYN(sampling_strategy={0: desired_size // 2, 1: desired_size // 2}, random_state=random_state)
    X_resampled, y_resampled = oversampler.fit_resample(X_seed, y_seed)
    return X_resampled, y_resampled

# Function for RandomOverSampler (ROS)
def generate_ros_data(X_seed, y_seed, desired_size, random_state):
    """
    Generate an oversampled dataset using random over-sampling.

    Parameters
    ----------
    X_seed : pandas.DataFrame
        Feature matrix for the seed dataset.
    y_seed : pandas.Series
        Binary labels for the seed dataset.
    desired_size : int
        Desired total number of samples after oversampling, split evenly across the two classes.
    random_state : int
        Random seed used by the random over-sampler.

    Returns
    -------
    tuple
        A tuple containing the resampled feature matrix and resampled label vector.
    """
    oversampler = RandomOverSampler(sampling_strategy={0: desired_size // 2, 1: desired_size // 2}, random_state=random_state)
    X_resampled, y_resampled = oversampler.fit_resample(X_seed, y_seed)
    return X_resampled, y_resampled

# Main function to generate synthetic data based on selected method
def generate_synthetic_data(seed_data_path, oversample_method, output_file, desired_size, label_column='label', random_state=42):
    """
    Generate synthetic data using a selected baseline oversampling method and save it to disk.

    Parameters
    ----------
    seed_data_path : str
        Path to the seed dataset CSV file.
    oversample_method : str
        Oversampling method to use. Supported values are 'smote', 'adasyn', and 'ros'.
    output_file : str
        Path where the generated synthetic dataset CSV file is saved.
    desired_size : int
        Desired total number of samples after oversampling, split evenly across the two classes.
    label_column : str, optional
        Name of the binary label column. Default is 'label'.
    random_state : int, optional
        Random seed used for oversampling and shuffling. Default is 42.

    Returns
    -------
    None
        Writes the generated synthetic dataset to disk.
    """
    data = pd.read_csv(seed_data_path)
    X_seed = data.drop(columns=[label_column])
    y_seed = data[label_column]

    if oversample_method == 'smote':
        X_resampled, y_resampled = generate_smote_data(X_seed, y_seed, desired_size, random_state)
    elif oversample_method == 'adasyn':
        X_resampled, y_resampled = generate_adasyn_data(X_seed, y_seed, desired_size, random_state)
    elif oversample_method == 'ros':
        X_resampled, y_resampled = generate_ros_data(X_seed, y_seed, desired_size, random_state)
    else:
        raise ValueError(f"Unsupported oversample method: {oversample_method}")

    # Keep only synthetic data (excluding original seed data)
    X_synthetic = X_resampled[len(X_seed):]
    y_synthetic = y_resampled[len(y_seed):]

    # Combine synthetic data into a DataFrame
    synthetic_data = pd.DataFrame(X_synthetic, columns=X_seed.columns)
    synthetic_data[label_column] = y_synthetic

    # Shuffle and save the synthetic data
    synthetic_data = synthetic_data.sample(frac=1, random_state=random_state).reset_index(drop=True)
    synthetic_data.to_csv(output_file, index=False)
    print(f"Synthetic dataset saved to '{output_file}'")

# Main script to run the process
if __name__ == "__main__":
    data_names = {0: "craft", 1: "gender", 2: "diabetes", 3: "adult"}
    data_name = data_names[3]
    
    add_data_ls = {0: "smote", 1: "adasyn", 2: "ros"}
    add_data = add_data_ls[2]
    
    info = load_data_info('data_info.json')
    data_info = info.get(data_name, {})
    path = data_info.get('path')
    label_col = data_info.get('label_col')
    
    seed_data_path = path + "/seed.csv"
    output_path = path + f"/{add_data}.csv"
    
    # Call the generate_synthetic_data function
    generate_synthetic_data(seed_data_path, add_data, output_path, 50000, label_col)
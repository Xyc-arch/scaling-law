import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def generate_causal_dataset_rounded(n_samples, noise_level=1):
    # Set seed for reproducibility
    np.random.seed(42)

    # Generate independent features, rounded to one decimal place
    X1 = np.round(np.random.randn(n_samples), 1)
    X2 = np.round(np.random.randn(n_samples), 1)
    
    # X3 is causally dependent on X1 and X2
    X3 = np.round(0.5 * X1 + 0.3 * X2 + np.random.normal(0, 0.5, n_samples), 1)

    # Interaction and other features
    X4 = np.round(np.random.randn(n_samples) * X1, 1)  # Interaction between X1 and X4
    X5 = np.round(np.random.randn(n_samples) + 0.5 * X3, 1)  # X5 is influenced by X3
    X6 = np.random.choice([-1, 1], size=n_samples)  # Binary feature, 1 or -1
    X7 = np.round(np.random.randn(n_samples), 1)
    X8 = np.round(X2 * X3, 1)  # Direct interaction between X2 and X3
    X9 = np.round(X1 * X2, 1)  # Additional interaction between X1 and X2

    # Generate binary target with controllable noise level
    noise = np.random.normal(0, noise_level, n_samples)
    linear_combination = 1.5 + 0.7 * X1 - 0.6 * X2 + 0.8 * X3 + 0.4 * X9 + noise
    threshold = np.median(linear_combination)
    Y = (linear_combination > threshold).astype(int)

    # Create DataFrame
    data = pd.DataFrame({
        'X1': X1, 'X2': X2, 'X3': X3, 'X4': X4, 'X5': X5, 'X6': X6,
        'X7': X7, 'X8': X8, 'X9': X9, 'Target': Y
    })

    return data



# 100 training points and 400 test points
# Noise Level: 1.0, Accuracy: 0.7450
# Noise Level: 1.5, Accuracy: 0.6850
# Noise Level: 2.0, Accuracy: 0.6600
# Noise Level: 2.5, Accuracy: 0.6250
# Noise Level: 3.0, Accuracy: 0.6100



if __name__ == "__main__":
    craft = generate_causal_dataset_rounded(8000)
    craft.to_csv("/home/ubuntu/data/craft/craft.csv", index=False)
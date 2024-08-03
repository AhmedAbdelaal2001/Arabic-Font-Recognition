import numpy as np
from typing import List, Tuple, Dict

def load_features_from_file(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads features and labels from a CSV file.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the feature matrix X and the label vector y.
    
    Steps:
        1. Reads the data from the CSV file.
        2. Splits the data into features (X) and labels (y).
    """
    data: np.ndarray = np.genfromtxt(filepath, delimiter=",")
    X: np.ndarray = data[:, :-1]
    y: np.ndarray = data[:, -1]

    return X, y
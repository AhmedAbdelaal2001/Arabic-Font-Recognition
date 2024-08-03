import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from typing import List, Tuple, Dict

def read_processed_data(base_path: str, max_num_of_entries_per_folder: int = float('inf')) -> Tuple[List[np.ndarray], List[int]]:
    """
    Reads and processes image data from a given directory, converting images to grayscale and binarizing them.

    Args:
        base_path (str): Path to the base directory containing subfolders with images.
        max_num_of_entries_per_folder (int): Maximum number of entries to read per folder. Default is infinity.

    Returns:
        Tuple[List[np.ndarray], List[int]]: A tuple containing the list of processed images and their corresponding labels.
    
    Steps:
        1. List the predefined folders in the directory.
        2. Create a dictionary to hold the labels for each folder.
        3. Prepare lists to store image data and labels.
        4. Loop through each folder and read images, converting them to grayscale and binarizing.
        5. Append the processed image data and corresponding label to the lists.
        6. Return the lists of images and labels.
    """
    # List the folders in the directory
    folders: List[str] = ['IBM Plex Sans Arabic', 'Lemonada', 'Marhey', 'Scheherazade New']

    # Create a dictionary to hold the labels for each folder
    labels: Dict[str, int] = {folder: i for i, folder in enumerate(folders)}

    # Prepare lists to store the image data and labels
    images: List[np.ndarray] = []
    image_labels: List[int] = []

    # Loop through each folder and each image within the folder
    for folder in folders:
        count: int = 0
        folder_path: str = os.path.join(base_path, folder)
        for filename in os.listdir(folder_path):
            if filename.endswith('.jpeg'):  # Assuming the images are in JPEG format
                image_path: str = os.path.join(folder_path, filename)
                # Load the image
                image: Image.Image = Image.open(image_path)
                # Optionally, convert the image to 'L' to ensure it's in grayscale
                if image.mode != 'L':
                    image = image.convert('L')
                # Convert image data to array
                image_data: np.ndarray = np.array(image)
                # Binarize the image data (0 or 1) directly instead of normalization to [0, 1] range
                image_data = (image_data > 127).astype(np.uint8)  # Assuming binary threshold at the middle (127)
                # Append the image data and label to the list
                images.append(image_data)
                image_labels.append(labels[folder])
                count += 1
                if count == max_num_of_entries_per_folder:
                    break

    return images, image_labels


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


def remove_highly_correlated_features(X: np.ndarray, threshold: float = 0.95) -> Tuple[np.ndarray, List[int]]:
    """
    Removes highly correlated features from the feature matrix.

    Args:
        X (np.ndarray): Feature matrix.
        threshold (float): Correlation threshold for removing features. Default is 0.95.

    Returns:
        Tuple[np.ndarray, List[int]]: A tuple containing the reduced feature matrix and the list of dropped feature indices.
    
    Steps:
        1. Computes the correlation matrix of the features.
        2. Identifies features with correlations above the threshold.
        3. Removes the identified features from the feature matrix.
    """
    corr_matrix: pd.DataFrame = pd.DataFrame(X).corr().abs()
    upper: pd.DataFrame = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop: List[int] = [column for column in upper.columns if any(upper[column] > threshold)]
    X_reduced: np.ndarray = np.delete(X, to_drop, axis=1)
    return X_reduced, to_drop

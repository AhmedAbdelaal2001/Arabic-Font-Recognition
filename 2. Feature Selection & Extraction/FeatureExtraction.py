import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import pandas as pd
from typing import Tuple, Optional, List

from BoVW import BoVW
from GaborExtractor import GaborExtractor
from LawsExtractor import LawsExtractor

class FeatureExtraction:
    """
    FeatureExtraction is a class that handles the loading, preprocessing, and extraction of features from image data.
    It includes methods for loading features from a file, removing highly correlated features, and extracting features
    using various extraction techniques such as Bag of Visual Words, Gabor filters, and Laws' texture energy measures.
    """

    def __init__(self, kmeans: Optional[KMeans] = None, scaler: Optional[MinMaxScaler] = None, dropped_features: Optional[List[int]] = None) -> None:
        """
        Initializes the FeatureExtraction class with optional k-means, scaler, and dropped features.

        Args:
            kmeans (Optional[KMeans]): Pre-trained k-means model for BoVW. Default is None.
            scaler (Optional[MinMaxScaler]): Pre-fitted scaler for normalizing features. Default is None.
            dropped_features (Optional[List[int]]): List of features to be dropped after extraction. Default is None.
        """
        self.kmeans = kmeans
        self.scaler = scaler
        self.dropped_features = dropped_features

    def load_features_from_file(self, filename: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Loads features and labels from a CSV file.

        Args:
            filename (str): Path to the CSV file.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing feature matrix X and label vector y.
        
        Steps:
            1. Reads the data from the CSV file.
            2. Splits the data into features (X) and labels (y).
        """
        data: np.ndarray = np.genfromtxt(filename, delimiter=",")
        X: np.ndarray = data[:, :-1]
        y: np.ndarray = data[:, -1]
        return X, y

    def remove_highly_correlated_features(self, X: np.ndarray, threshold: float = 0.95) -> Tuple[np.ndarray, List[int]]:
        """
        Removes highly correlated features from the feature matrix.

        Args:
            X (np.ndarray): Feature matrix.
            threshold (float): Correlation threshold for removing features. Default is 0.95.

        Returns:
            Tuple[np.ndarray, List[int]]: A tuple containing the reduced feature matrix and the list of dropped features.
        
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

    def extract_features(self, X: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Extracts features from the input data using various feature extraction techniques.

        Args:
            X (Optional[np.ndarray]): Input data for feature extraction. Default is None.

        Returns:
            np.ndarray: Extracted feature matrix.
        
        Steps:
            1. Initializes the Bag of Visual Words (BoVW) extractor.
            2. Extracts BoVW features from the input data.
            3. Optionally, extracts Gabor and Laws' texture energy features and concatenates them.
            4. Optionally, normalizes the features using the scaler.
            5. Optionally, removes specified features from the feature matrix.
        """
        # Define Gabor filter parameters
        # orientations = [k * np.pi / 8 for k in range(1, 9)]
        # frequencies = np.linspace(0.2, 0.5, 3)
        # sigmas = np.linspace(3, 1, 3)
        
        BoVW_extractor = BoVW()
        # gabor_extractor = GaborExtractor()
        # laws_extractor = LawsExtractor()
        
        # Extract BoVW features using the provided k-means model
        X, _ = BoVW_extractor.extract_BoVW(data=X, kmeans=self.kmeans)
        
        # Extract Gabor features and concatenate (commented out)
        # X_Gabor = gabor_extractor.extract_gabor_features(X, orientations, frequencies, sigmas)
        
        # Extract Laws' texture energy measures and concatenate (commented out)
        # X_Laws = laws_extractor.extract_laws_texture_energy_measures(X)
        
        # Concatenate all feature sets (commented out)
        # X = np.concatenate((X_BoVW, X_Gabor), axis=1)
        # X = np.concatenate((X, X_Laws), axis=1)
        
        # Normalize the features using the scaler (commented out)
        # if self.scaler:
        #     X = self.scaler.transform(X)
        
        # Remove specified features from the feature matrix (commented out)
        # if self.dropped_features:
        #     X = np.delete(X, self.dropped_features, axis=1)
        
        return X

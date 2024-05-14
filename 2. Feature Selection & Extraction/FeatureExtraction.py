import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

from BoVW import BoVW
from GaborExtractor import GaborExtractor
from LawsExtractor import LawsExtractor

class FeatureExtraction:
    def __init__(self, features_saved=False, features_concatenated=False, dataset_type = 'training', kmeans=None, scaler=None, dropped_features=None):
        self.features_saved = features_saved
        self.features_concatenated = features_concatenated
        self.dataset_type = dataset_type
        self.kmeans = kmeans
        self.scaler = scaler
        self.dropped_features = dropped_features
    
    def load_features_from_file(self, filename):
        data = np.genfromtxt(filename, delimiter=",")
        X = data[:, :-1]
        y = data[:, -1]

        return (X, y)
    
    # Remove highly correlated features
    def remove_highly_correlated_features(self, X, threshold=0.95):
        corr_matrix = pd.DataFrame(X).corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        X_reduced = np.delete(X, to_drop, axis=1)
        return X_reduced, to_drop
    
    def extract_features(self, X=None):
        if self.features_concatenated: 
            return self.load_features_from_file(f"Final_features_{self.dataset_type}.csv")
        elif self.features_saved: 
            X_BoVW, y = self.load_features_from_file(f"BoVW_features_{self.dataset_type}.csv")
            X_Gabor, _ = self.load_features_from_file(f"Gabor_features_{self.dataset_type}.csv")
            X_Laws, _ = self.load_features_from_file(f"Laws_features_{self.dataset_type}.csv")

            # Concatenate the feature vectors horizontally
            X = np.concatenate((X_BoVW, X_Gabor), axis=1)
            X = np.concatenate((X, X_Laws), axis=1)
            if not self.scaler: 
                self.scaler = MinMaxScaler()
                X = self.scaler.fit_transform(X)
                X, self.dropped_features = self.remove_highly_correlated_features(X)
            else:
                X = self.scaler.transform(X)
                X = np.delete(X, self.dropped_features, axis=1)
            return (X, y)
        else:
            orientations = [k * np.pi / 8 for k in range(1, 9)]
            frequencies = np.linspace(0.2, 0.5, 3)
            sigmas = np.linspace(4, 1, 3)
            BoVW_extractor = BoVW()
            gabor_extractor = GaborExtractor()
            laws_extractor = LawsExtractor()
            X_BoVW, self.kmeans = BoVW_extractor.extract_BoVW(data=X, kmeans=self.kmeans)
            X_Gabor = gabor_extractor.extract_gabor_features(X, orientations, frequencies, sigmas)
            X_Laws = laws_extractor.extract_laws_texture_energy_measures(X)

            X = np.concatenate((X_BoVW, X_Gabor), axis=1)
            X = np.concatenate((X, X_Laws), axis=1)
            if not self.scaler: 
                self.scaler = MinMaxScaler()
                X = self.scaler.fit_transform(X)
            else:
                X = self.scaler.transform(X)
            X = np.delete(X, self.dropped_features, axis=1)

            return X
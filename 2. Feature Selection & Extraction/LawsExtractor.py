import numpy as np
import cv2
from typing import List

class LawsExtractor:
    """
    LawsExtractor is a class that creates Laws' texture energy filters and extracts texture energy measures from images.
    It uses these filters to compute feature vectors based on the texture properties of the images.
    """

    def __init__(self) -> None:
        """
        Initializes the LawsExtractor class, setting up the masks for Laws' filters and empty lists for filters and feature vectors.
        """
        self.filters: List[np.ndarray] = []
        self.feature_vectors: List[np.ndarray] = []
        L5: np.ndarray = np.array([1, 4, 6, 4, 1])
        E5: np.ndarray = np.array([-1, -2, 0, 2, 1])
        S5: np.ndarray = np.array([-1, 0, 2, 0, -1])
        W5: np.ndarray = np.array([-1, 2, 0, -2, 1])
        R5: np.ndarray = np.array([1, -4, 6, -4, 1])
        self.masks: List[np.ndarray] = [L5, E5, S5, W5, R5]

    def create_laws_kernels(self) -> None:
        """
        Creates Laws' texture energy kernels by computing the outer product of the masks.
        
        Steps:
            1. Iterates over each pair of masks.
            2. Computes the outer product of the masks to create 2D kernels.
            3. Appends the created kernels to the filters list.
        """
        for i in range(5):
            for j in range(5):
                self.filters.append(np.outer(self.masks[i], self.masks[j]))

    def extract_laws_texture_energy_measures(self, data: List[np.ndarray]) -> np.ndarray:
        """
        Extracts Laws' texture energy measures from a list of images using the created Laws' kernels.
        
        Args:
            data (List[np.ndarray]): List of images in numpy array format.

        Returns:
            np.ndarray: A 2D array where each row corresponds to the feature vector of an image.
        
        Steps:
            1. Creates Laws' texture energy kernels.
            2. Iterates over each image and applies the Laws' kernels.
            3. For each kernel, computes the filtered image, its energy, and standard deviation.
            4. Appends these measures to the feature vector.
            5. Converts the list of feature vectors to a 2D numpy array and returns it.
        """
        self.create_laws_kernels()  # Create the Laws' texture energy kernels
        i: int = 0

        for image in data:
            print(i)
            feature_vector: List[float] = []

            for kernel in self.filters:
                filtered_image: np.ndarray = np.abs(cv2.filter2D(image, -1, kernel))  # Apply the kernel to the image
                energy: float = np.mean(filtered_image)  # Compute the mean energy of the filtered image
                std: float = np.std(filtered_image)  # Compute the standard deviation of the filtered image
                feature_vector.extend([energy, std])  # Append energy and standard deviation to the feature vector

            self.feature_vectors.append(feature_vector)
            i += 1

        # Convert the list of feature vectors to a 2D numpy array
        self.feature_vectors = np.array(self.feature_vectors)
        return self.feature_vectors

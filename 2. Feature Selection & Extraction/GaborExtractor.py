import numpy as np
import cv2
from typing import List, Tuple

class GaborExtractor:
    """
    GaborExtractor is a class that builds Gabor filters and extracts Gabor features from images.
    It uses these filters to compute feature vectors based on the responses of the filters.
    """

    def __init__(self) -> None:
        """
        Initializes the GaborExtractor class, setting up empty lists for filters and feature vectors.
        """
        self.filters: List[Tuple[np.ndarray, np.ndarray]] = []
        self.feature_vectors: List[np.ndarray] = []

    def build_gabor_filters(self, orientations: List[float], frequencies: List[float], sigmas: List[float]) -> None:
        """
        Builds Gabor filters for given orientations, frequencies, and sigmas.
        
        Args:
            orientations (List[float]): List of orientations (in radians) for the Gabor filters.
            frequencies (List[float]): List of frequencies for the Gabor filters.
            sigmas (List[float]): List of standard deviations for the Gaussian envelope of the Gabor filters.

        Steps:
            1. Iterates over each orientation.
            2. For each combination of frequency and sigma, calculates the wavelength.
            3. Creates Gabor kernels with specified parameters.
            4. Ensures the kernel size is odd.
            5. Appends the created Gabor kernels to the filters list.
        """
        for θ in orientations:
            for frequency, σ in zip(frequencies, sigmas):
                λ = 1 / frequency  # Calculate the wavelength
                γ = 0.5  # Spatial aspect ratio
                kernel_size = int(3 * σ) if int(3 * σ) % 2 == 1 else int(3 * σ) + 1  # Ensure kernel size is odd
                zero_kernel = cv2.getGaborKernel((kernel_size, kernel_size), σ, θ, λ, γ, 0, ktype=cv2.CV_32F)
                neg_kernel = cv2.getGaborKernel((kernel_size, kernel_size), σ, θ, λ, γ, np.pi/2, ktype=cv2.CV_32F)
                self.filters.append((zero_kernel, neg_kernel))  # Append the kernels as a tuple

    def extract_gabor_features(
        self,
        data: List[np.ndarray],
        orientations: List[float],
        frequencies: List[float],
        sigmas: List[float]
    ) -> np.ndarray:
        """
        Extracts Gabor features from a list of images using the built Gabor filters.
        
        Args:
            data (List[np.ndarray]): List of images in numpy array format.
            orientations (List[float]): List of orientations (in radians) for the Gabor filters.
            frequencies (List[float]): List of frequencies for the Gabor filters.
            sigmas (List[float]): List of standard deviations for the Gaussian envelope of the Gabor filters.

        Returns:
            np.ndarray: A 2D array where each row corresponds to the feature vector of an image.
        
        Steps:
            1. Builds Gabor filters using the specified parameters.
            2. Iterates over each image and applies the Gabor filters.
            3. For each filter, computes the magnitude response and clamps its values.
            4. Extracts mean and standard deviation from the response and appends to the feature vector.
            5. Computes and appends the maximum response's mean and standard deviation to the feature vector.
            6. Converts the list of feature vectors to a 2D numpy array and returns it.
        """
        self.build_gabor_filters(orientations, frequencies, sigmas)
        print("Prepared Filters")
        num_images: int = len(data)

        for i in range(num_images):
            print(i)
            feature_vector: np.ndarray = np.zeros(2 * (1 + len(self.filters)), dtype=np.float32)
            feature_vector_index: int = 0
            image_responses: List[np.ndarray] = []

            for zero_kernel, neg_kernel in self.filters:
                zero_filtered: np.ndarray = cv2.filter2D(data[i], cv2.CV_8U, zero_kernel)
                neg_filtered: np.ndarray = cv2.filter2D(data[i], cv2.CV_8U, neg_kernel)
                E: np.ndarray = np.sqrt(zero_filtered ** 2 + neg_filtered ** 2)
                E = np.clip(E, -1e10, 1e10)  # Clamp the values to a reasonable range
                image_responses.append(E)
                E_mean: float = np.mean(E)
                E_std: float = np.std(E)
                feature_vector[feature_vector_index] = E_mean
                feature_vector[feature_vector_index + 1] = E_std
                feature_vector_index += 2

            max_response: np.ndarray = np.max(image_responses, axis=0)
            max_response = np.clip(max_response, -1e10, 1e10)  # Clamp the values to a reasonable range
            max_response_mean: float = np.mean(max_response)
            max_response_std: float = np.std(max_response)
            feature_vector[feature_vector_index] = max_response_mean
            feature_vector[feature_vector_index + 1] = max_response_std
            self.feature_vectors.append(feature_vector)

        # Convert the list of feature vectors to a 2D numpy array
        self.feature_vectors = np.array(self.feature_vectors)
        return self.feature_vectors

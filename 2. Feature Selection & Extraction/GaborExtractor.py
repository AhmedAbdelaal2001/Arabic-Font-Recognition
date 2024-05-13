import numpy as np
import cv2

class GaborExtractor:
    def __init__(self, data):
        self.data = data
        self.filters = []
        self.responses = None
        self.feature_vectors = None

    def build_gabor_filters(self, orientations, frequencies, sigmas):
        for θ in orientations:
            for frequency, σ in zip(frequencies, sigmas):
                λ = 1 / frequency  # Wavelength
                γ = 0.5  # Spatial aspect ratio
                kernel_size = int(8 * σ) if int(8 * σ) % 2 == 1 else int(8 * σ) + 1  # Ensure kernel size is odd
                zero_kernel = cv2.getGaborKernel((kernel_size, kernel_size), σ, θ, λ, γ, 0, ktype=cv2.CV_32F)
                neg_kernel = cv2.getGaborKernel((kernel_size, kernel_size), σ, θ, λ, γ, np.pi/2, ktype=cv2.CV_32F)
                self.filters.append([zero_kernel, neg_kernel])

    def apply_gabor_filters(self):
        num_images = self.data.shape[0]
        num_filters = len(self.filters)
        test_filter = self.filters[0][0]
        filtered_example = cv2.filter2D(self.data[0], -1, test_filter)
        filtered_shape = filtered_example.shape
        self.responses = np.zeros((num_images, num_filters + 1, filtered_shape[0], filtered_shape[1]))

        for i in range(num_images):
            for j, (zero_kernel, neg_kernel) in enumerate(self.filters):
                zero_filtered = cv2.filter2D(self.data[i], -1, zero_kernel)
                neg_filtered = cv2.filter2D(self.data[i], -1, neg_kernel)
                E = np.sqrt(zero_filtered ** 2 + neg_filtered ** 2)
                self.responses[i, j] = E
            self.responses[i, -1] = np.max(self.responses[i, :-1], axis=0)

    def prepare_features(self):
        num_images = self.responses.shape[0]
        num_responses = self.responses.shape[1]
        # Initialize feature_vectors with 2 * num_responses elements per image
        self.feature_vectors = np.zeros((num_images, 2 * num_responses))

        for i in range(num_images):
            feature_index = 0
            for j in range(num_responses):
                response_mean = np.mean(self.responses[i, j])
                response_std = np.std(self.responses[i, j])
                self.feature_vectors[i, feature_index] = response_mean
                self.feature_vectors[i, feature_index + 1] = response_std
                feature_index += 2

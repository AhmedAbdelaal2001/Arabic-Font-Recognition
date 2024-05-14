import numpy as np
import cv2

class GaborExtractor:
    def __init__(self):
        self.filters = []
        self.feature_vectors = []

    def build_gabor_filters(self, orientations, frequencies, σ=1.0):
        for θ in orientations:
            for frequency in frequencies:
                λ = 1 / frequency  # Wavelength
                γ = 0.5  # Spatial aspect ratio
                kernel_size = int(8 * σ) if int(8 * σ) % 2 == 1 else int(8 * σ) + 1  # Ensure kernel size is odd
                zero_kernel = cv2.getGaborKernel((kernel_size, kernel_size), σ, θ, λ, γ, 0, ktype=cv2.CV_32F)
                neg_kernel = cv2.getGaborKernel((kernel_size, kernel_size), σ, θ, λ, γ, np.pi/2, ktype=cv2.CV_32F)
                self.filters.append([zero_kernel, neg_kernel])

    def extract_gabor_features(self, data, orientations, frequencies):
        self.build_gabor_filters(orientations, frequencies)
        num_images = data.shape[0]
        for i in range(num_images):
            feature_vector = np.zeros(2 * (1 + len(self.filters)), dtype=np.float32)
            feature_vector_index = 0
            image_responses = []
            for zero_kernel, neg_kernel in self.filters:
                zero_filtered = cv2.filter2D(data[i], -1, zero_kernel)
                neg_filtered = cv2.filter2D(data[i], -1, neg_kernel)
                E = np.sqrt(zero_filtered ** 2 + neg_filtered ** 2)
                #E = np.clip(E, -1e10, 1e10)  # Clamp the values to a reasonable range
                image_responses.append(E)
                E_mean = np.mean(E)
                E_std = np.std(E)
                feature_vector[feature_vector_index] = E_mean
                feature_vector[feature_vector_index + 1] = E_std
                feature_vector_index += 2

            max_response = np.max(image_responses, axis=0)
            #max_response = np.clip(max_response, -1e10, 1e10)  # Clamp the values to a reasonable range
            max_response_mean = np.mean(max_response)
            max_response_std = np.std(max_response)
            feature_vector[feature_vector_index] = max_response_mean
            feature_vector[feature_vector_index + 1] = max_response_std
            self.feature_vectors.append(feature_vector)

        # Convert the list of feature vectors to a 2D numpy array
        self.feature_vectors = np.array(self.feature_vectors)
        return self.feature_vectors
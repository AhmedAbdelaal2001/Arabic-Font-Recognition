import numpy as np
import cv2

class LawsExtractor:
    def __init__(self):
        self.filters = []
        self.feature_vectors = []
        L5 = np.array([1, 4, 6, 4, 1])
        E5 = np.array([-1, -2, 0, 2, 1])
        S5 = np.array([-1, 0, 2, 0, -1])
        W5 = np.array([-1, 2, 0, -2, 1])
        R5 = np.array([1, -4, 6, -4, 1])
        self.masks = [L5, E5, S5, W5, R5]

    def create_laws_kernels(self):
        for i in range(5):
            for j in range(5):
                self.filters.append(np.outer(self.masks[i], self.masks[j]))
    
    def extract_laws_texture_energy_measures(self, data):
        self.create_laws_kernels()
        i = 0
        for image in data:
            print(i)
            feature_vector = []
            for kernel in self.filters:
                filtered_image = np.abs(cv2.filter2D(image, -1, kernel))
                energy = np.mean(filtered_image)
                std = np.std(filtered_image)
                feature_vector.extend([energy, std])
            self.feature_vectors.append(feature_vector)
            i += 1
        self.feature_vectors = np.array(self.feature_vectors)
        return self.feature_vectors

    

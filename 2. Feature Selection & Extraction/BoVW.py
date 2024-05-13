import cv2
import numpy as np
from sklearn.cluster import KMeans

class BoVW:
    def __init__(self):
        self.feature_vectors = []

    def extract_SIFT_descriptors(self, data):
        descriptors = []
        sift = cv2.SIFT_create()
        for img in data:
            img = (255 * img).astype(np.uint8)
            if len(img.shape) == 3: 
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            kp, desc = sift.detectAndCompute(img, None)
            if desc is not None:
                descriptors.append(desc)
            else:
                print("No descriptors found for an image.")
        return descriptors
    
    def extract_BoVW(self, data, k=112, kmeans=None, sift_features=None):

            if not sift_features:
                sift_features = self.extract_SIFT_descriptors(data)
            
            if not kmeans:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=1)
                kmeans.fit(np.vstack(sift_features))
            
            # Create histograms for each image
            for desc in sift_features:
                if desc is not None and len(desc) > 0:

                    # Predict cluster assignments for the reduced descriptors using the trained k-means model
                    cluster_predictions = kmeans.predict(desc)

                    # Create a histogram of cluster assignments (visual words)
                    hist, _ = np.histogram(cluster_predictions, bins=np.arange(kmeans.n_clusters + 1), density=True)

                    self.feature_vectors.append(hist)
                else:
                    # If no descriptors were found for this image, use an empty histogram
                    self.feature_vectors.append(np.zeros(self.kmeans.n_clusters))
            
            return np.array(self.feature_vectors), kmeans

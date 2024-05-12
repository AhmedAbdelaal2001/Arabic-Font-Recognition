from SIFT import *

class FeatureExtraction:
    def __init__(self, data, kmeans, sift_features=None):
        self.data = data
        self.kmeans = kmeans
        self.feature_vectors = []
        self.sift_features = sift_features
    
    def bag_of_visual_words_SIFT(self):

        if not self.sift_features:
            sift_extractor = SIFT(self.data)
            sift_extractor.extract_descriptors()
            self.sift_features = sift_extractor.descriptors
        
        # Create histograms for each image
        for desc in self.sift_features:
            if desc is not None and len(desc) > 0:

                # Predict cluster assignments for the reduced descriptors using the trained k-means model
                cluster_predictions = self.kmeans.predict(desc)

                # Create a histogram of cluster assignments (visual words)
                hist, _ = np.histogram(cluster_predictions, bins=np.arange(self.kmeans.n_clusters + 1), density=True)

                self.feature_vectors.append(hist)
            else:
                # If no descriptors were found for this image, use an empty histogram
                self.feature_vectors.append(np.zeros(self.kmeans.n_clusters))
        
        return np.array(self.feature_vectors)

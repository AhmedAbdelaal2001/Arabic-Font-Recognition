import cv2
import numpy as np
from sklearn.cluster import KMeans
from typing import List, Tuple, Optional

class BoVW:
    """
    BoVW (Bag of Visual Words) is a class that implements the extraction of SIFT descriptors from images
    and the creation of histograms of visual words using k-means clustering. It can be used to generate
    feature vectors for images based on their visual content.
    """

    def __init__(self) -> None:
        """
        Initializes the BoVW class, setting up an empty list to store feature vectors.
        """
        self.feature_vectors: List[np.ndarray] = []

    def extract_SIFT_descriptors(self, data: List[np.ndarray]) -> List[np.ndarray]:
        """
        Extracts SIFT (Scale-Invariant Feature Transform) descriptors from a list of images.
        
        Args:
            data (List[np.ndarray]): List of images in numpy array format.

        Returns:
            List[np.ndarray]: List of SIFT descriptors for each image.
        
        Steps:
            1. Initializes a SIFT detector.
            2. Iterates over each image, converting it to the correct format.
            3. Detects keypoints and computes descriptors.
            4. Collects and returns the descriptors.
        """
        descriptors: List[np.ndarray] = []
        sift = cv2.SIFT_create()  # Initialize the SIFT detector

        for img in data:
            img = (255 * img).astype(np.uint8)  # Convert image to uint8 format
            if len(img.shape) == 3:  # Convert color image to grayscale
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            kp, desc = sift.detectAndCompute(img, None)  # Detect keypoints and compute descriptors

            if desc is not None:  # Check if descriptors were found
                descriptors.append(desc)
            else:
                print("No descriptors found for an image.")  # Print a message if no descriptors were found

        return descriptors

    def extract_BoVW(
        self,
        data: List[np.ndarray],
        k: int = 112,
        kmeans: Optional[KMeans] = None,
        sift_features: Optional[List[np.ndarray]] = None
    ) -> Tuple[np.ndarray, KMeans]:
        """
        Extracts Bag of Visual Words (BoVW) features for a list of images using k-means clustering on SIFT descriptors.
        
        Args:
            data (List[np.ndarray]): List of images in numpy array format.
            k (int): Number of clusters for k-means. Default is 112.
            kmeans (Optional[KMeans]): Pre-trained k-means model. If not provided, a new model will be trained.
            sift_features (Optional[List[np.ndarray]]): Pre-extracted SIFT descriptors. If not provided, they will be extracted.

        Returns:
            Tuple[np.ndarray, KMeans]: A tuple containing the feature vectors and the k-means model.
        
        Steps:
            1. Extract SIFT descriptors if not provided.
            2. Train k-means on the SIFT descriptors if a model is not provided.
            3. Create histograms of visual words for each image based on cluster assignments.
            4. Append the histograms to the feature_vectors list.
            5. Return the feature vectors and the k-means model.
        """
        
        # Extract SIFT descriptors if not already provided
        if not sift_features:
            sift_features = self.extract_SIFT_descriptors(data)
        
        # Train k-means if a model is not provided
        if not kmeans:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=1)
            kmeans.fit(np.vstack(sift_features))  # Flatten the list of descriptors for k-means training
        
        # Create histograms for each image
        for desc in sift_features:
            if desc is not None and len(desc) > 0:
                # Predict cluster assignments for the descriptors using the trained k-means model
                cluster_predictions = kmeans.predict(desc)
                # Create a histogram of cluster assignments (visual words)
                hist, _ = np.histogram(cluster_predictions, bins=np.arange(kmeans.n_clusters + 1), density=True)
                self.feature_vectors.append(hist)
            else:
                # If no descriptors were found for this image, use an empty histogram
                self.feature_vectors.append(np.zeros(kmeans.n_clusters))
        
        return np.array(self.feature_vectors), kmeans

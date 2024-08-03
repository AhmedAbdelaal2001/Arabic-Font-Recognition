import os
import cv2
import numpy as np
from typing import List
from ImagePreprocessor import ImagePreprocessor

class DataPreprocessor:
    """
    DataPreprocessor is responsible for preprocessing images. It contains methods to preprocess images from a source 
    directory and save the processed images to a destination directory, as well as to preprocess test images and 
    return them as a list of binarized numpy arrays.
    """

    def preprocess_and_write(self, src_dir: str, dst_dir: str) -> None:
        """
        Preprocesses images found in the source directory and writes the processed images to the destination directory.
        
        Args:
            src_dir (str): The source directory containing folders of images to be processed.
            dst_dir (str): The destination directory where processed images will be saved.

        Steps:
            1. Ensures the destination directory exists.
            2. Iterates over each folder in the source directory.
            3. For each image in a folder, reads the image in grayscale mode.
            4. Preprocesses the image using ImagePreprocessor.
            5. Saves the processed image to the corresponding folder in the destination directory.
            6. Prints the save path of each processed image for debugging.
            7. Handles errors if an image cannot be read.
        """

        # Ensure the destination directory exists; create it if it does not
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)

        # Iterate over each folder in the source directory
        for folder in os.listdir(src_dir):
            folder_path: str = os.path.join(src_dir, folder)  # Construct the full path to the source folder
            save_path: str = os.path.join(dst_dir, folder)  # Construct the corresponding path in the destination directory

            # Ensure the save path exists; create it if it does not
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            # Iterate over each file in the current folder
            for file in os.listdir(folder_path):
                img_path: str = os.path.join(folder_path, file)  # Construct the full path to the current image file
                img: np.ndarray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read the image in grayscale mode

                if img is not None:  # Check if the image was read successfully
                    processor: ImagePreprocessor = ImagePreprocessor(img)  # Create an instance of ImagePreprocessor with the image
                    processed_img: np.ndarray = processor.preprocess_image()  # Preprocess the image using the custom method
                    save_img_path: str = os.path.join(save_path, file)  # Construct the path where the processed image will be saved
                    print(save_img_path)  # Print the save path for debugging purposes
                    cv2.imwrite(save_img_path, processed_img)  # Write the processed image to the save path
                else:
                    print(f"Failed to read image: {img_path}")  # Print an error message if the image could not be read

    def preprocess_test_data(self, src_dir: str) -> List[np.ndarray]:
        """
        Preprocesses test images found in the source directory and returns them as a list of binarized numpy arrays.
        
        Args:
            src_dir (str): The source directory containing images to be processed.

        Returns:
            List[np.ndarray]: A list of binarized numpy arrays representing the processed test images.

        Steps:
            1. Iterates over each file in the source directory.
            2. Reads each image in grayscale mode.
            3. Preprocesses the image using ImagePreprocessor.
            4. Binarizes the processed image by thresholding at 127.
            5. Appends the binarized image to the list of preprocessed images.
            6. Handles errors if an image cannot be read.
            7. Returns the list of preprocessed images.
        """

        preprocessed_images: List[np.ndarray] = []  # Initialize an empty list to hold the preprocessed images

        # Iterate over each file in the source directory
        for file in os.listdir(src_dir):
            img_path: str = os.path.join(src_dir, file)  # Construct the full path to the current image file
            img: np.ndarray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read the image in grayscale mode

            if img is not None:  # Check if the image was read successfully
                processor: ImagePreprocessor = ImagePreprocessor(img)  # Create an instance of ImagePreprocessor with the image
                processed_img: np.ndarray = processor.preprocess_image()  # Preprocess the image using the custom method
                # Binarize the processed image by thresholding at 127; convert to 8-bit unsigned integers
                binarized_img: np.ndarray = (processed_img > 127).astype(np.uint8)
                preprocessed_images.append(binarized_img)  # Append the binarized image to the list
            else:
                print(f"Failed to read image: {img_path}")  # Print an error message if the image could not be read

        return preprocessed_images  # Return the list of preprocessed images

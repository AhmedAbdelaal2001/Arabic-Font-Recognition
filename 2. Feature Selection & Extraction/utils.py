import os
from PIL import Image
import numpy as np

def read_processed_data():
    # Define the path to the directory containing the folders
    base_path = '../Preprocessed Dataset'

    # List the folders in the directory
    folders = ['IBM Plex Sans Arabic', 'Lemonada', 'Marhey', 'Scheherazade New']

    # Create a dictionary to hold the labels for each folder
    labels = {folder: i for i, folder in enumerate(folders)}

    # Prepare a list to store the image data and labels
    data = []

    # Loop through each folder and each image within the folder
    for folder in folders:
        folder_path = os.path.join(base_path, folder)
        for filename in os.listdir(folder_path):
            if filename.endswith('.jpeg'):  # assuming the images are in PNG format
                image_path = os.path.join(folder_path, filename)
                # Load the image
                image = Image.open(image_path)
                # Optionally, convert the image to 'L' to ensure it's in grayscale
                if image.mode != 'L':
                    image = image.convert('L')
                # Convert image data to array
                image_data = np.array(image)
                # Binarize the image data (0 or 1) directly instead of normalization to [0, 1] range
                image_data = (image_data > 127).astype(np.uint8)  # Assuming binary threshold at the middle (127)
                # Append the image data and label to the list
                data.append((image_data, labels[folder]))

    data, labels =  zip(*data)
    return np.array(data), np.array(labels)

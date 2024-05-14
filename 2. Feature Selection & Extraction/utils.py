import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

def read_processed_data(base_path, max_num_of_entries_per_folder = float('inf')):
    # List the folders in the directory
    folders = ['IBM Plex Sans Arabic', 'Lemonada', 'Marhey', 'Scheherazade New']

    # Create a dictionary to hold the labels for each folder
    labels = {folder: i for i, folder in enumerate(folders)}

    # Prepare a list to store the image data and labels
    images = []
    image_labels = []
    # Loop through each folder and each image within the folder
    for folder in folders:
        count = 0
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
                images.append(image_data)
                image_labels.append(labels[folder])
                count += 1
                if count == max_num_of_entries_per_folder: break

    return images, image_labels

def shuffle_and_partition_data(images, labels, random_state=42):
    # Combine images and labels into a list of tuples
    data = list(zip(images, labels))

    # Shuffle the data
    np.random.seed(random_state)
    np.random.shuffle(data)

    # Unzip the shuffled data back into images and labels
    images, labels = zip(*data)

    # Convert images and labels to lists
    images, labels = list(images), list(labels)

    # Split the data into training (60%), validation (20%), and test (20%) sets
    X_temp, X_test, y_temp, y_test = train_test_split(images, labels, test_size=0.2, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=random_state)

    return X_train, X_val, X_test, y_train, y_val, y_test

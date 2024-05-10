import numpy as np
import cv2
import matplotlib.pyplot as plt

class Preprocessing:
    def __init__(self, img):
        """
        Initialize the Preprocessing class with an image.
        Args:
            img (numpy.ndarray): The image to be processed, assumed to be in grayscale.
        """
        self.img = img  # the image to be processed
        self.angles = []  # list to store the angles of text orientations
    
    def fix_color(self):
        """
        Inverts the image colors if the average color of the border pixels is light.
        This is to ensure that the text is darker than the background for better processing.
        """
        # Extract the border pixels
        top = self.img[0]
        bottom = self.img[-1]
        left = self.img[:,0]
        right = self.img[:,-1]
        
        # Calculate the average color of the border pixels
        avg = np.mean([np.mean(top), np.mean(bottom), np.mean(left), np.mean(right)])
        
        # Invert the image colors if the average is light
        if avg > 128:
            self.img = 255 - self.img

    def binarize_image(self):
        """
        Applies median blurring and Otsu's thresholding to binarize the image.
        After thresholding, it calls fix_color to ensure proper contrast.
        """
        # Apply median blurring twice to reduce noise
        self.img = cv2.medianBlur(self.img, 3)
        self.img = cv2.medianBlur(self.img, 3)
        
        # Apply Otsu's thresholding
        _, self.img = cv2.threshold(self.img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Ensure the text is darker than the background
        self.fix_color()

    def get_rectangle_angles(self, structuring_element_size=20, display_rectangles=False):
        """
        Detects contours in the image and computes the orientation of each contour's minimum area rectangle.
        Optionally displays these rectangles overlaid on the image.
        Args:
            structuring_element_size (int): Size of the structuring element for morphological operations.
            display_rectangles (bool): If True, display the image with rectangles drawn around detected contours.
        """
        # Apply morphological closing to make the contours more detectable
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (structuring_element_size, structuring_element_size))
        processed = cv2.morphologyEx(self.img, cv2.MORPH_CLOSE, kernel)

        # Find contours in the processed image
        contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Optional: prepare image for rectangle visualization
        if display_rectangles: 
            color_img = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)

        # Process each detected contour
        for contour in contours:
            rect = cv2.minAreaRect(contour)  # get the minimum area rectangle
            angle = rect[2]  # extract the angle
            
            # Adjust angle for a consistent representation
            if rect[1][0] < rect[1][1]:
                angle -= 90
            
            self.angles.append(angle)

            # If displaying rectangles, draw them on the image
            if display_rectangles:
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(color_img, [box], 0, (0, 255, 0), 2)

        # Show the image with drawn rectangles
        if display_rectangles:
            plt.imshow(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))
            plt.show()

    def plot_angles_histogram(self):
        """
        Plots a histogram of the angles of text orientations.
        This only works if there are angles collected in the self.angles list.
        """
        if self.angles:
            plt.figure(figsize=(10, 6))
            plt.hist(self.angles, bins=30, color='green', alpha=0.7)
            plt.title('Distribution of Text Orientations')
            plt.xlabel('Angle (degrees)')
            plt.ylabel('Frequency')
            plt.grid(True)
            plt.show()

    def find_best_angle(self):
        """
        Finds the most common text orientation angle from the histogram of angles.
        Returns:
            float: The average angle from the most populated bin in the histogram.
        """
        hist, bin_edges = np.histogram(self.angles, bins=30)
        max_bin_index = np.argmax(hist)
        return np.mean([bin_edges[max_bin_index], bin_edges[max_bin_index + 1]])

    def rotate_image(self, angle):
        """
        Rotates the image by a specified angle.
        Args:
            angle (float): The angle to rotate the image by, in degrees.
        """
        (h, w) = self.img.shape[:2]  # image dimensions
        center = (w / 2, h / 2)  # image center

        # Compute the rotation matrix for the rotation and the scale
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Apply the rotation to the image
        rotated_img = cv2.warpAffine(self.img, M, (w, h))
        
        # Update the image
        self.img = rotated_img

    def find_text_region(self, block_size=(300, 300)):
        """
        Find the 300x300 block containing the maximum textual information.

        Args:
            block_size (tuple): Desired output block size.

        Returns:
            numpy.ndarray: The cropped 300x300 region with the most textual information.
        """
        # Extract the image dimensions and the block size
        img_h, img_w = self.img.shape
        block_h, block_w = block_size

        # Ensure the block size is smaller than or equal to the image size
        if block_h > img_h or block_w > img_w:
            raise ValueError("Block size exceeds image dimensions.")

        max_text_block = None
        max_text_density = 0

        # Iterate over all possible blocks
        for y in range(0, img_h - block_h + 1, 60):  # Step of 60 to avoid overly granular sliding
            for x in range(0, img_w - block_w + 1, 60):
                # Extract the current block
                block = self.img[y:y + block_h, x:x + block_w]
                
                # Calculate text density by counting the number of dark pixels
                text_density = np.sum(block == 255)  # Assuming text pixels are black (0) after binarization

                # Update the block with the maximum text density found
                if text_density > max_text_density:
                    max_text_density = text_density
                    max_text_block = block

        return max_text_block

    def preprocess_image(self, structuring_element_size=20, display_rectangles=False):
        """
        Executes the full preprocessing pipeline on the image, which includes binarization,
        detecting text orientations, finding the best rotation angle, and rotating the image.
        Args:
            structuring_element_size (int): Size of the kernel for morphological operations.
            display_rectangles (bool): Whether to display the rectangles around detected contours.
        Returns:
            The rotated image.
        """
        self.binarize_image()
        self.get_rectangle_angles(structuring_element_size, display_rectangles)
        best_angle = self.find_best_angle()
        self.rotate_image(best_angle)
        return self.img

import cv2
import numpy as np

class SIFT:
    def __init__(self, data):
        self.data = (data * 255).astype('uint8')
        self.descriptors = []
    
    def _process_image(self, img):
        sift = cv2.SIFT_create()
        _ , desc = sift.detectAndCompute(img, None)
        return desc

    def extract_descriptors(self):
        sift = cv2.SIFT_create()
        for img in self.data:
            kp, desc = sift.detectAndCompute(img, None)
            if desc is not None:
                self.descriptors.append(desc)

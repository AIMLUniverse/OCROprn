import numpy as np
import os
import cv2
import glob
import shutil
import pytesseract
from utils import Utils

class OCR_Enhancer(object):

    def __init__(self, source_folder, filename):
        self.filename = filename
        self.source_folder = source_folder
        self.util = Utils()
        self.config = self.util.read_config()

    def apply_threshold(self, img):
        return cv2.threshold(cv2.medianBlur(img, 3), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        #return cv2.threshold(cv2.GaussianBlur(img, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    def invert_black_header(self, img_path):
        im = cv2.imread(img_path)
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40,1))
        remove_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        cnts = cv2.findContours(remove_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            x,y,w,h = cv2.boundingRect(c)
            if h > 40 and w > 200:
                im[y:y+h, x:x+w] = 255 - im[y:y+h, x:x+w]
        return im

    def remove_lines(self, im):
        try:
            result = im.copy()
            gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

            # Remove horizontal lines
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40,1))
            remove_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
            cnts = cv2.findContours(remove_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            for c in cnts:
                x,y,w,h = cv2.boundingRect(c)
                if h <= 10:
                    cv2.drawContours(result, [c], -1, (255,255,255), 5)

            # Remove vertical lines
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,40))
            remove_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
            cnts = cv2.findContours(remove_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            for c in cnts:
                cv2.drawContours(result, [c], -1, (255,255,255), 5)

            return result
        except Exception as e:
            logging.error('Error in remove lines - {0}'.format(e))
            return im


    def crop_image(self, img, start_x, start_y, end_x, end_y):
        cropped = img[start_y:end_y, start_x:end_x]
        return cropped


    def preprocess_image(self, clean_document, train=False):
        # Read image using opencv
        img_path = self.source_folder + self.filename
        img = self.invert_black_header(img_path)
        if clean_document:
            img = self.remove_lines(img)
        # Convert to gray
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply dilation and erosion to remove some noise
        #kernel = np.ones((1, 1), np.uint8)
        #img = cv2.dilate(img, kernel, iterations=1)
        #img = cv2.erode(img, kernel, iterations=1)

        #  Apply threshold to get image with only black and white
        #img = self.apply_threshold(img)
        if train:
            out_path = self.config["train_prep_folder"]
        else:
            if clean_document:
                out_path = self.config["no_contour_image_folder"]
            else:
                out_path = self.config["preprocessed_folder"]
        save_path = os.path.join(out_path, self.filename)
        cv2.imwrite(save_path, img)
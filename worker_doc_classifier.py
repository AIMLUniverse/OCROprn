import cv2
import os
from PIL import Image
import pytesseract
from pytesseract import Output
from PIL import Image
from elasticsearch import Elasticsearch


from utils import Utils
import re
import numpy as np

#Prediction Imports

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms

class WorkerDocClassifier(object):
    def __init__(self):
        self.name='Worker_Doc_Classifier'
        self.util = Utils()
        self.class_config = self.util.read_doc_classifier_config()
        config = self.util.read_config()
        self.SOURCE_FOLDER = config["source_folder"]
        self.PREPROCESSED_FOLDER = config["classifier_prep_folder"]

    def get_ocr_text(self, document_name):
        img = Image.open(os.path.join(self.SOURCE_FOLDER, document_name))
        width, height = img.size
        results = pytesseract.image_to_data(img, output_type=Output.DICT, config="--psm 3 --oem 1 -c  classify_enable_learning=0, tessedit_char_whitelist=    \/$%@#-.abcdefghijklmnopqrstuvxywzABCDEFGHIJKLMNOPQRSTUVXYWZ0123456789")
        return width, height, results

    def get_image_data(self, document_name):
        im = cv2.imread(os.path.join(self.SOURCE_FOLDER, document_name), cv2.IMREAD_GRAYSCALE)
        #im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        image_data = im.copy()
        return im, image_data

    def format_text_with_coords(self, ocr_text):
        text_with_coords = []
        text = []
        for i in range(0, len(ocr_text["text"])):
            d = {}
            if ocr_text["text"][i] == '':
                continue
            # extract the bounding box coordinates of the text region from
            # the current result
            x = ocr_text["left"][i]
            y = ocr_text["top"][i]
            w = ocr_text["width"][i]
            h = ocr_text["height"][i]
            # extract the OCR text itself along with the confidence of the
            # text localization
            t = ocr_text["text"][i]
            if re.match(r'\b\S+\b', t.strip()) is None:
                continue
            text.append(t.upper())
            d['text'] = t.upper()
            d['box'] = {'X': x, 'Y': y, 'W': w, 'H': h}
            text_with_coords.append(d)
        return text_with_coords, text

    def get_matching_word_index(self, text, alias):
        word_index = []
        r = re.compile(r'\b\S+\b')
        data = ' '.join(text)
        dic = { i :(m.start(0), m.end(0), m.group(0)) for i, m in enumerate(r.finditer(data))}
        matches = []
        for alias_item in alias:
            match = (re.search(alias_item, data))
            if match is not None:
                for match in re.finditer(alias_item, data, re.IGNORECASE):
                    matches.append(match.span())
        for loc in matches:
            (start, end) = loc
            for k, v in dic.items():
                w_start, w_end, word = v
                if w_start >= start and w_end <= end:
                    word_index.append(k)
        word_index = list(set(word_index))
        return word_index

    def select_terms_by_word_index(self, text_with_coords, word_index):
        arr = np.array(text_with_coords)
        selected_terms = list(arr[word_index])
        return selected_terms

    def overlay_image(self, image_data, selected_terms, overlay_color):
        if len(selected_terms) == 0:
            return image_data
        for term in selected_terms:
            x1 = term['box']['X']
            x2 = term['box']['X'] + term['box']['W']
            y1 = term['box']['Y']
            y2 = term['box']['Y'] + term['box']['H']
            overlay = cv2.rectangle(image_data, (x1, y1), (x2, y2), overlay_color, -1)
        return overlay

    def save_image(self, im, overlay, document_name):
        alpha = 0.4  # Transparency factor.
        im = cv2.cvtColor(im,cv2.COLOR_GRAY2RGB)
        img_new = cv2.addWeighted(overlay, alpha, im, 1 - alpha, 0)
        filePath = os.path.join(self.PREPROCESSED_FOLDER, document_name)
        cv2.imwrite(filePath, img_new)

    def preprocess_document(self, document_name):
        width, height, results = self.get_ocr_text(document_name)
        im, image_data = self.get_image_data(document_name)
        image_data = cv2.cvtColor(image_data,cv2.COLOR_GRAY2RGB)
        text_with_coords, text = self.format_text_with_coords(results) 
        alias = self.getAlias()
        for item in alias:
            class_label = item['class']
            class_highlight_color = tuple(map(int, item['color'].split(', ')))
            class_terms = item['terms']
            word_index = self.get_matching_word_index(text, class_terms)
            selected_terms = self.select_terms_by_word_index(text_with_coords, word_index)
            image_data = self.overlay_image(image_data, selected_terms, class_highlight_color)
        self.save_image(im, image_data, document_name)
        return True

    def predict_document(self, document_name):
        model_name_or_path = 'classifier_models/doc_classifier_resnet152_100epchs.pth'
        input_path = os.path.join(self.PREPROCESSED_FOLDER, document_name)
        class_names = self.getClasses() 
        model = torch.load(model_name_or_path, map_location=torch.device('cpu'))
        data_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        img = Image.open(input_path)
        img_tensor = data_transforms(img)
        input_batch = img_tensor.unsqueeze(0) # create a mini-batch as expected by the model
        input_batch = input_batch.to('cpu')
        outputs = model(input_batch)
        _, preds = torch.max(outputs,1)
        predicted_class = class_names[preds]
        print('Predicted Class: ', predicted_class)
        return predicted_class

    def getClasses(self):
        classes = []
        for item in self.class_config['alias']:
            classes.append(item['class'])
        '''TODO: Retrain the model with right model order. Reversing order'''
        classes = ['PO', 'Invoice'] 
        return classes
    
    def getAlias(self):
        return self.class_config['alias']

if __name__ == '__main__':
    w = WorkerDocClassifier()
    doc_name = 'wexco.jpg'
    w.preprocess_document(doc_name)
    print(w.predict_document(doc_name))
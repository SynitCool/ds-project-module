import cv2
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from utils import *
from config import *
from dataset import *


class Preprocessing:
    def __init__(self, 
                 dataset: Dataset, 
                 extractions: list[str] = None,
                 feature_selected: bool = False,
                 n_components: int = 10,
                 alias: str = '',
                 include_cnn: bool = False):
        
        # private
        self.__dataset = dataset
        self.__extractions = extractions
        self.__feature_selected = feature_selected
        self.__n_components = n_components
        self.__alias = alias
        self.__X = []
        self.__y = []
        self.__X_CNN = []
        self.__y_CNN = []

        # public
        self.images_path = self.__dataset.get_images_path()

        self.__process_images()
        if include_cnn:
            for i in range(len(self.images_path)):
                self.__y_CNN.append(self.images_path[i].split("/")[-2])

                img = self.images_path[i]
                img = cv2.imread(img)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (256, 256))

                self.__X_CNN.append(img)

            self.__X_CNN = np.array(self.__X_CNN)
            self.__y_CNN = np.array(self.__y_CNN)

            le = LabelEncoder()
            self.__y_CNN = le.fit_transform(self.__y_CNN)

    def __process_images(self):
        for i in range(len(self.images_path)):
            self.__y.append(self.images_path[i].split("/")[-2])
            self.__X.append(self.__single_image(self.images_path[i]))

        self.__X = np.concatenate(tuple(self.__X), axis=0)
        self.__y = np.array(self.__y)

        le = LabelEncoder()
        self.__y = le.fit_transform(self.__y)

        # Selection feature PCA and RFE
        if self.__feature_selected:
            self.__X, self.__y = feature_selection(self.__X, self.__y, pca_component=self.__n_components)

        print(f"Preprocessing is Done: X has shape: {self.__X.shape}, y has shape: {self.__y.shape}")

    def __single_image(self, image:str):
        # change to grayscale
        img = cv2.imread(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (256, 256))
        
        # noise removing
        gaussian_blur = cv2.GaussianBlur(img, (5, 5), 0)  # Adjust kernel size (5, 5) as needed
        # median_blur = cv2.medianBlur(img, 5)  # Adjust kernel size (5) as needed

        # concat
        if self.__extractions:
            if len(self.__extractions) > 0:
                extracted = features_extraction(gray_image=img, gaussian_blur=gaussian_blur, extractions=self.__extractions)
            
            # normalize
            gaussian_blur = gaussian_blur.astype(np.float32)
            gaussian_blur /= 255.0
            
            # reshaping
            gaussian_blur = np.ravel(gaussian_blur)

            gaussian_blur = gaussian_blur.reshape((1, gaussian_blur.shape[0]))
            return np.concatenate((gaussian_blur, extracted), axis=1)

        # normalize
        gaussian_blur = gaussian_blur.astype(np.float32)
        gaussian_blur /= 255.0
        
        # reshaping
        gaussian_blur = np.ravel(gaussian_blur)

        gaussian_blur = gaussian_blur.reshape((1, gaussian_blur.shape[0]))
        
        return gaussian_blur
    
    def plot_histogram(self):
      data =  self.__X.ravel()

      plt.figure(figsize=(8, 6))
      sns.histplot(data, kde=True)
        
      plt.title('Histogram of Data')
      plt.xlabel('Data')
      plt.ylabel('Frequency')

      plt.savefig(f"save_{self.__alias}_histogram.png")
      plt.show()

    def get_X(self):
        return self.__X
    
    def get_y(self):
        return self.__y

    def get_X_CNN(self):
        return self.__X_CNN
    
    def get_y_CNN(self):
        return self.__y_CNN

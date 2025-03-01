import cv2
import os
import torch
import torchvision

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from torch import tensor
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF # Import the functional module

from PIL import Image

from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from skimage.feature import graycomatrix, graycoprops
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize

from config import *


def save_roc_multiclass(y_true, y_pred_prob, name: str):
    n_classes = len(np.unique(y_true))

    y_true_bin = label_binarize(y_true, classes=[0, 1, 2, 3, 4])

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_prob[:, i])
        roc_auc[i] = [roc_auc_score(y_true_bin[:, i], y_pred_prob[:, i])]

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_pred_prob.ravel())
    roc_auc["micro"] = roc_auc_score(y_true_bin, y_pred_prob, average="micro")
    
    for key, _ in fpr.items():
      if key == "micro":
        continue
      fpr_tpr = pd.DataFrame({"fpr": fpr[key], "tpr": tpr[key]})

      fpr_tpr.to_excel(f"fpr_tpr_{name}_class_{LABEL_CONVERTER[str(key)]}.xlsx", index=False)

    auc_df = pd.DataFrame(roc_auc)
    auc_df.to_excel(f"auc_{name}_all_class.xlsx", index=False)

    fpr_tpr = pd.DataFrame({"fpr": fpr["micro"], "tpr": tpr["micro"]})
    fpr_tpr.to_excel(f"fpr_tpr_{name}_micro.xlsx", index=False)

    fpr_tpr_auc = pd.DataFrame({"fpr": fpr["micro"], "tpr": tpr["micro"], "auc":roc_auc["micro"]})
    fpr_tpr_auc.to_excel(f"fpr_tpr_auc_{name}_micro.xlsx", index=False)

    return fpr, tpr, roc_auc

def features_extraction(gray_image, gaussian_blur, extractions: list[str]):
  extract_func = {"fourier": fourier_extraction, "glcm": glcm_extraction}
  all_extract = []
  
  for ext in extractions:
    if "fourier" == ext:
      all_extract.append(extract_func[ext](gray_image))
    elif "glcm" == ext:
      all_extract.append(extract_func[ext](gaussian_blur))

  return np.concatenate(tuple(all_extract), axis=1)

def fourier_extraction(gray_image):
  # fourier
  DFT = cv2.dft(np.float32(gray_image), flags=cv2.DFT_COMPLEX_OUTPUT)
  
  shift = np.fft.fftshift(DFT)
  row, col = gray_image.shape
  center_row, center_col = row // 2, col // 2
  
  mask = np.zeros((row, col, 2), np.uint8)
  mask[center_row - 30:center_row + 30, center_col - 30:center_col + 30] = 1
  
  fft_shift = shift * mask
  fft_ifft_shift = np.fft.ifftshift(fft_shift)
  imageThen = cv2.idft(fft_ifft_shift)
  
  imageThen = cv2.magnitude(imageThen[:,:,0], imageThen[:,:,1])

  imageThen /= 255.0
  imageThen = np.ravel(imageThen)
  imageThen = imageThen.reshape((1, imageThen.shape[0]))

  return imageThen

def glcm_extraction(gaussian_blur):
  # GLCM
  distances = [1]  
  angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  
  glcm = graycomatrix(gaussian_blur, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)

  # normalize
  glcm /= 255.0
  

  # reshaping
  glcm = np.ravel(glcm)
  

  glcm = glcm.reshape((1, glcm.shape[0]))

  return glcm




def feature_selection(features, label, pca_component=50):
  pca = PCA(n_components=pca_component)

  features = pca.fit_transform(features)
  
  estimator = RandomForestClassifier()
  
  selector = RFE(estimator)

  features = selector.fit_transform(features, label)
  
  return features, label


def unet_segmentation(img: str):
  model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
    in_channels=3, out_channels=1, init_features=32, pretrained=True)

  # Load image
  image_np = cv2.imread(img)
  image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
  image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)

  # segemented
  output = model(image_tensor.float())

  output_np = output.squeeze().detach().cpu().numpy()

  del model
  del image_tensor
  del image_tensor
  del output

  return output_np

def rcnn_segmentation(img: str):
  # Load model
  model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
    pretrained=True, progress=True, num_classes=91, pretrained_backbone=True)

  model.eval()

  # Load image
  image_np = Image.open(img)

  np_img_np = np.array(image_np.convert("RGB"))

  # Segmented
  transformed_img = torchvision.transforms.transforms.ToTensor()(
          torchvision.transforms.ToPILImage()(np_img_np))

  result = model([transformed_img])

  box = result[0]["boxes"][0].detach().numpy().tolist()

  im1 = np.array(image_np.crop((box[0], box[1], box[2], box[3])))

  del model
  del image_np
  del np_img_np
  del transformed_img
  del result
  del box

  return im1

def canny_image(gray_image):
  return cv2.Canny(gray_image, 50, 150)

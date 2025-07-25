import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch

from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize

from torch.utils.data import DataLoader, TensorDataset
from torch import nn

from model import SimpleCNN

from preprocessing import Preprocessing

from utils import save_roc_multiclass

from config import *

import lime
import lime.lime_tabular
import shap
from IPython.display import display

shap.initjs()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TrainingNoSplit:
    def __init__(
            self, 
            train_preprocessing: Preprocessing, 
            val_preprocessing: Preprocessing,
            test_preprocessing: Preprocessing, 
            model_name: dict[str, dict],
            alias: str = ''):
        # private
        self.__alias = alias
        self.__train_preprocessing = train_preprocessing
        self.__val_preprocessing = val_preprocessing
        self.__test_preprocessing = test_preprocessing
        self.__model_name = model_name

        raise Exception("Training No Split is not supported yet!")

    def __calc_roc_auc(self, y_true, y_pred_prob):
      n_classes = len(np.unique(y_true))

      y_true_bin = label_binarize(y_true, classes=[i for i in range(n_classes)])

      fpr = dict()
      tpr = dict()
      roc_auc = dict()

      for i in range(n_classes):
          fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_prob[:, i])
          roc_auc[i] = [roc_auc_score(y_true_bin[:, i], y_pred_prob[:, i])]

      # Compute micro-average ROC curve and ROC area
      fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_pred_prob.ravel())
      roc_auc["micro"] = roc_auc_score(y_true_bin, y_pred_prob, average="micro")

      return fpr, tpr, roc_auc

    def __plot_roc_auc(self, train, val, test, name: str):
      fpr_train = train["fpr"]
      tpr_train = train["tpr"]
      roc_auc_train = train["roc_auc"]

      fpr_val = val["fpr"]
      tpr_val = val["tpr"]
      roc_auc_val = val["roc_auc"]

      fpr_test = test["fpr"]
      tpr_test = test["tpr"]
      roc_auc_test = test["roc_auc"]
      
      plt.figure(figsize=(8, 6))
      plt.plot([0, 1], [0, 1], 'k--', lw=2)  # Diagonal line

      #Plot for each set
      for fpr, tpr, roc_auc, set_name in zip(
          [fpr_train, fpr_val, fpr_test], 
           [tpr_train, tpr_val, tpr_test], 
            [roc_auc_train, roc_auc_val, roc_auc_test], 
             ['train','val','test']):
          
          plt.plot(fpr["micro"], tpr["micro"],
              label=f'micro-average {set_name} ROC curve (area = {roc_auc["micro"]:0.2f})')
      
      plt.xlim([0.0, 1.0])
      plt.ylim([0.0, 1.05])
      plt.xlabel('False Positive Rate')
      plt.ylabel('True Positive Rate')
      plt.title(f'Receiver operating characteristic {name} {self.__alias}')
      plt.savefig(f"{name}_{self.__alias}_roc_auc.png")
      plt.legend(loc="lower right")
      plt.show()

    def train_test_method_grid_search(self, cv):
        X_train = self.__train_preprocessing.get_X()
        y_train = self.__train_preprocessing.get_y()

        X_val = self.__val_preprocessing.get_X()
        y_val = self.__val_preprocessing.get_y()

        X_test = self.__test_preprocessing.get_X()
        y_test = self.__test_preprocessing.get_y()

        model = {}
        for name, spec in self.__model_name.items():
            model[name] = GridSearchCV(spec["model"], spec["param"], cv=cv)

        for name, mdl in model.items():
            mdl.fit(X_train, y_train)

            # TRAIN
            y_pred_train = mdl.predict(X_train)
            train_metrics = {
                "Accuracy": [accuracy_score(y_train, y_pred_train)]
            }

            func_metric = [f1_score, recall_score, precision_score]
            func_name = ["F1-Score", "Recall", "Precision"]
            for func, fname in zip(func_metric, func_name):
                metrics = func(y_train, y_pred_train, average=None)
                for i, met in enumerate(metrics):
                    train_metrics[f"{fname}_{LABEL_CONVERTER[str(i)]}"] = met
            
            confusion_matrix_train = confusion_matrix(y_train, y_pred_train)

            train_metrics = pd.DataFrame(train_metrics)
            confusion_matrix_train = pd.DataFrame(confusion_matrix_train)

            # VAL
            y_pred_val = mdl.predict(X_val)
            val_metrics = {
                "Accuracy": [accuracy_score(y_val, y_pred_val)]
            }

            for func, fname in zip(func_metric, func_name):
                metrics = func(y_val, y_pred_val, average=None)
                for i, met in enumerate(metrics):
                    val_metrics[f"{fname}_{LABEL_CONVERTER[str(i)]}"] = met
            
            confusion_matrix_val = confusion_matrix(y_val, y_pred_val)

            val_metrics = pd.DataFrame(val_metrics)
            confusion_matrix_val = pd.DataFrame(confusion_matrix_val)
            
            # TEST
            y_pred_test = mdl.predict(X_test)
            test_metrics = {
                "Accuracy": [accuracy_score(y_test, y_pred_test)],
            }

            for func, fname in zip(func_metric, func_name):
                metrics = func(y_test, y_pred_test, average=None)
                for i, met in enumerate(metrics):
                    test_metrics[f"{fname}_{LABEL_CONVERTER[str(i)]}"] = met
            
            confusion_matrix_test = confusion_matrix(y_test, y_pred_test)

            test_metrics = pd.DataFrame(test_metrics)
            confusion_matrix_test = pd.DataFrame(confusion_matrix_test)

            # save to excel
            train_metrics.to_excel(f"{name}_train_metric.xlsx", index=False)
            confusion_matrix_train.to_excel(f"{name}_confusion_matrix_train.xlsx", index=False)

            val_metrics.to_excel(f"{name}_val_metric.xlsx", index=False)
            confusion_matrix_val.to_excel(f"{name}_confusion_matrix_val.xlsx", index=False)
            
            test_metrics.to_excel(f"{name}_test_metric.xlsx", index=False)
            confusion_matrix_test.to_excel(f"{name}_confusion_matrix_test.xlsx", index=False)

            # save roc
            save_roc_multiclass(y_train, mdl.predict_proba(X_train), f"{name}_train")
            save_roc_multiclass(y_val, mdl.predict_proba(X_val), f"{name}_val")
            save_roc_multiclass(y_test, mdl.predict_proba(X_test), f"{name}_test") 

            # plot roc auc
            fpr_train, tpr_train, roc_auc_train = self.__calc_roc_auc(y_train, mdl.predict_proba(X_train))
            fpr_val, tpr_val, roc_auc_val = self.__calc_roc_auc(y_val, mdl.predict_proba(X_val))
            fpr_test, tpr_test, roc_auc_test = self.__calc_roc_auc(y_test, mdl.predict_proba(X_test))

            self.__plot_roc_auc(
                train={"fpr": fpr_train, "tpr": tpr_train, "roc_auc": roc_auc_train}, 
                val={"fpr": fpr_val, "tpr": tpr_val, "roc_auc": roc_auc_val},
                test={"fpr": fpr_test, "tpr": tpr_test, "roc_auc": roc_auc_test},
                name=name
            )
        
    def train_test_method(self):
        X_train = self.__train_preprocessing.get_X()
        y_train = self.__train_preprocessing.get_y()

        X_val = self.__val_preprocessing.get_X()
        y_val = self.__val_preprocessing.get_y()

        X_test = self.__test_preprocessing.get_X()
        y_test = self.__test_preprocessing.get_y()

        model = {}
        for name in self.__model_name:
            if name == "LogisticRegression":
                model[name] = LogisticRegression(penalty='l2', multi_class="multinomial")
            elif name == "SVM":
                model[name] = SVC(probability=True)
            elif name == "RF":
                model[name] = RandomForestClassifier()
            elif name == "CNN":
                print("CNN is not supported for train test method for no split")

        for name, mdl in model.items():
            mdl.fit(X_train, y_train)

            # Coef & Feature Important
            if name == "LogisticRegression":
                imp = mdl.coef_
            elif name == "SVM":
                imp = mdl.dual_coef_
            elif name == "RF":
                imp = mdl.feature_importances_
                imp = imp.reshape((1, imp.shape[0]))

            imp = pd.DataFrame(imp)
            imp.to_excel(f"{name}_{self.__alias}_coef_feature_importance.xlsx", index=False)

            # TRAIN
            y_pred_train = mdl.predict(X_train)
            train_metrics = {
                "Accuracy": [accuracy_score(y_train, y_pred_train)]
            }

            func_metric = [f1_score, recall_score, precision_score]
            func_name = ["F1-Score", "Recall", "Precision"]
            for func, fname in zip(func_metric, func_name):
                metrics = func(y_train, y_pred_train, average=None)
                for i, met in enumerate(metrics):
                    train_metrics[f"{fname}_{LABEL_CONVERTER[str(i)]}"] = met
            
            confusion_matrix_train = confusion_matrix(y_train, y_pred_train)

            train_metrics = pd.DataFrame(train_metrics)
            confusion_matrix_train = pd.DataFrame(confusion_matrix_train)

            # VAL
            y_pred_val = mdl.predict(X_val)
            val_metrics = {
                "Accuracy": [accuracy_score(y_val, y_pred_val)]
            }

            for func, fname in zip(func_metric, func_name):
                metrics = func(y_val, y_pred_val, average=None)
                for i, met in enumerate(metrics):
                    val_metrics[f"{fname}_{LABEL_CONVERTER[str(i)]}"] = met
            
            confusion_matrix_val = confusion_matrix(y_val, y_pred_val)

            val_metrics = pd.DataFrame(val_metrics)
            confusion_matrix_val = pd.DataFrame(confusion_matrix_val)
            
            # TEST
            y_pred_test = mdl.predict(X_test)
            test_metrics = {
                "Accuracy": [accuracy_score(y_test, y_pred_test)],
            }

            for func, fname in zip(func_metric, func_name):
                metrics = func(y_test, y_pred_test, average=None)
                for i, met in enumerate(metrics):
                    test_metrics[f"{fname}_{LABEL_CONVERTER[str(i)]}"] = met
            
            confusion_matrix_test = confusion_matrix(y_test, y_pred_test)

            test_metrics = pd.DataFrame(test_metrics)
            confusion_matrix_test = pd.DataFrame(confusion_matrix_test)

            # save to excel
            train_metrics.to_excel(f"{name}_{self.__alias}_train_metric.xlsx", index=False)
            confusion_matrix_train.to_excel(f"{name}_{self.__alias}_confusion_matrix_train.xlsx", index=False)

            val_metrics.to_excel(f"{name}_{self.__alias}_val_metric.xlsx", index=False)
            confusion_matrix_val.to_excel(f"{name}_{self.__alias}_confusion_matrix_val.xlsx", index=False)
            
            test_metrics.to_excel(f"{name}_{self.__alias}_test_metric.xlsx", index=False)
            confusion_matrix_test.to_excel(f"{name}_{self.__alias}_confusion_matrix_test.xlsx", index=False)

            # save roc
            save_roc_multiclass(y_train, mdl.predict_proba(X_train), f"{name}_{self.__alias}_train")
            save_roc_multiclass(y_val, mdl.predict_proba(X_val), f"{name}_{self.__alias}_val")
            save_roc_multiclass(y_test, mdl.predict_proba(X_test), f"{name}_{self.__alias}_test")

            # plot roc auc
            fpr_train, tpr_train, roc_auc_train = self.__calc_roc_auc(y_train, mdl.predict_proba(X_train))
            fpr_val, tpr_val, roc_auc_val = self.__calc_roc_auc(y_val, mdl.predict_proba(X_val))
            fpr_test, tpr_test, roc_auc_test = self.__calc_roc_auc(y_test, mdl.predict_proba(X_test))

            self.__plot_roc_auc(
                train={"fpr": fpr_train, "tpr": tpr_train, "roc_auc": roc_auc_train}, 
                val={"fpr": fpr_val, "tpr": tpr_val, "roc_auc": roc_auc_val},
                test={"fpr": fpr_test, "tpr": tpr_test, "roc_auc": roc_auc_test},
                name=name
            )




class Training:
    def __init__(self, preprocessing: Preprocessing, model_name: dict[str, dict], alias:str =''):
        # private
        self.__preprocessing = preprocessing
        self.__model_name = model_name
        self.__alias = alias

        # public
        self.X = self.__preprocessing.get_X()
        self.y = self.__preprocessing.get_y()

        self.X_CNN = self.__preprocessing.get_X_CNN()
        self.y_CNN = self.__preprocessing.get_y_CNN()

    def __minimum_dataset(self, label_data):
      if (len(np.unique(label_data)) < NUM_CLASS):
        raise Exception("You have insufficient data to training the data! Please add more data!")


    def train_test_method_grid_search(self, cv):
        raise Exception("Train test using grid search is not supported yet!")
        X_train, X_temp, y_train, y_temp = train_test_split(self.X, self.y, test_size=TRAIN_70_30_PORTION, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=VAL_TEST_70_30_PORTION, random_state=42)

        self.__minimum_dataset(y_train)
        self.__minimum_dataset(y_val)
        self.__minimum_dataset(y_test)


        model = {}

        for name, spec in self.__model_name.items():
            model[name] = GridSearchCV(spec["model"], spec["param"], cv=cv)

        for name, mdl in model.items():
            mdl.fit(X_train, y_train)

            # TRAIN
            y_pred_train = mdl.predict(X_train)
            train_metrics = {
                "Accuracy": [accuracy_score(y_train, y_pred_train)]
            }

            func_metric = [f1_score, recall_score, precision_score]
            func_name = ["F1-Score", "Recall", "Precision"]
            for func, fname in zip(func_metric, func_name):
                metrics = func(y_train, y_pred_train, average=None)
                for i, met in enumerate(metrics):
                    train_metrics[f"{fname}_{LABEL_CONVERTER[str(i)]}"] = met

            confusion_matrix_train = confusion_matrix(y_train, y_pred_train)

            train_metrics = pd.DataFrame(train_metrics)
            confusion_matrix_train = pd.DataFrame(confusion_matrix_train)

            # VAL
            y_pred_val = mdl.predict(X_val)
            val_metrics = {
                "Accuracy": [accuracy_score(y_val, y_pred_val)]
            }

            for func, fname in zip(func_metric, func_name):
                metrics = func(y_val, y_pred_val, average=None)
                for i, met in enumerate(metrics):
                    val_metrics[f"{fname}_{LABEL_CONVERTER[str(i)]}"] = met

            confusion_matrix_val = confusion_matrix(y_val, y_pred_val)

            val_metrics = pd.DataFrame(val_metrics)
            confusion_matrix_val = pd.DataFrame(confusion_matrix_val)

            # TEST
            y_pred_test = mdl.predict(X_test)
            test_metrics = {
                "Accuracy": [accuracy_score(y_test, y_pred_test)],
            }

            for func, fname in zip(func_metric, func_name):
                metrics = func(y_test, y_pred_test, average=None)
                for i, met in enumerate(metrics):
                    test_metrics[f"{fname}_{LABEL_CONVERTER[str(i)]}"] = met

            confusion_matrix_test = confusion_matrix(y_test, y_pred_test)

            test_metrics = pd.DataFrame(test_metrics)
            confusion_matrix_test = pd.DataFrame(confusion_matrix_test)

            # save to excel
            train_metrics.to_excel(f"{name}_{self.__alias}_train_metric.xlsx", index=False)
            confusion_matrix_train.to_excel(f"{name}_{self.__alias}_confusion_matrix_train.xlsx", index=False)

            val_metrics.to_excel(f"{name}_{self.__alias}_val_metric.xlsx", index=False)
            confusion_matrix_val.to_excel(f"{name}_{self.__alias}_confusion_matrix_val.xlsx", index=False)

            test_metrics.to_excel(f"{name}_{self.__alias}_test_metric.xlsx", index=False)
            confusion_matrix_test.to_excel(f"{name}_{self.__alias}_confusion_matrix_test.xlsx", index=False)

            # save roc
            save_roc_multiclass(y_train, mdl.predict_proba(X_train), f"{name}_{self.__alias}_train")
            save_roc_multiclass(y_val, mdl.predict_proba(X_val), f"{name}_{self.__alias}_val")
            save_roc_multiclass(y_test, mdl.predict_proba(X_test), f"{name}_{self.__alias}_test")

            # plot roc auc
            fpr_train, tpr_train, roc_auc_train = self.__calc_roc_auc(y_train, mdl.predict_proba(X_train))
            fpr_val, tpr_val, roc_auc_val = self.__calc_roc_auc(y_val, mdl.predict_proba(X_val))
            fpr_test, tpr_test, roc_auc_test = self.__calc_roc_auc(y_test, mdl.predict_proba(X_test))

            self.__plot_roc_auc(
                train={"fpr": fpr_train, "tpr": tpr_train, "roc_auc": roc_auc_train},
                val={"fpr": fpr_val, "tpr": tpr_val, "roc_auc": roc_auc_val},
                test={"fpr": fpr_test, "tpr": tpr_test, "roc_auc": roc_auc_test},
                name=name
            )

    def validate_kfold(self, n_splits: int):
        raise Exception("Train test using KFold is not supported yet!")
        model = {}
        for name in self.__model_name:
            if name == "LogisticRegression":
                model[name] = LogisticRegression(penalty='l2', multi_class="multinomial")
            elif name == "SVM":
                model[name] = SVC(probability=True)
            elif name == "RF":
                model[name] = RandomForestClassifier()
            elif name == "CNN":
                print("CNN is not supported for K-Fold validation")

        kf = KFold(n_splits=n_splits)
        for name, mdl in model.items():
            label_test = []
            label_pred = []
            for fold, (train_index, test_index) in enumerate(kf.split(self.X)):
                X_train, y_train = self.X[train_index], self.y[train_index]
                X_test, y_test = self.X[test_index], self.y[test_index]

                mdl.fit(X_train, y_train)

                # Coef & Feature Important
                if name == "LogisticRegression":
                    imp = mdl.coef_
                elif name == "SVM":
                    imp = mdl.dual_coef_
                elif name == "RF":
                    imp = mdl.feature_importances_
                    imp = imp.reshape((1, imp.shape[0]))

                imp = pd.DataFrame(imp)
                imp.to_excel(f"{name}_split_{fold+1}_coef_feature_importance.xlsx", index=False)

                func_metric = [f1_score, recall_score, precision_score]
                func_name = ["F1-Score", "Recall", "Precision"]

                # TRAIN
                y_pred_train = mdl.predict(X_train)
                train_metrics = {
                    "Accuracy": [accuracy_score(y_train, y_pred_train)]
                }

                for func, fname in zip(func_metric, func_name):
                    metrics = func(y_train, y_pred_train, average=None)
                    for i, met in enumerate(metrics):
                        train_metrics[f"{fname}_{LABEL_CONVERTER[str(i)]}"] = met

                confusion_matrix_train = confusion_matrix(y_train, y_pred_train)

                train_metrics = pd.DataFrame(train_metrics)
                confusion_matrix_train = pd.DataFrame(confusion_matrix_train)

                # TEST
                y_pred_test = mdl.predict(X_test)
                test_metrics = {
                    "Accuracy": [accuracy_score(y_test, y_pred_test)],
                }

                for func, fname in zip(func_metric, func_name):
                    metrics = func(y_test, y_pred_test, average=None)
                    for i, met in enumerate(metrics):
                        test_metrics[f"{fname}_{LABEL_CONVERTER[str(i)]}"] = met

                confusion_matrix_test = confusion_matrix(y_test, y_pred_test)

                test_metrics = pd.DataFrame(test_metrics)
                confusion_matrix_test = pd.DataFrame(confusion_matrix_test)

                # save to excel
                train_metrics.to_excel(f"{name}_{self.__alias}_train_split_{fold+1}_metric.xlsx", index=False)
                confusion_matrix_train.to_excel(f"{name}_{self.__alias}_train_split_{fold+1}_confusion_matrix.xlsx", index=False)

                test_metrics.to_excel(f"{name}_{self.__alias}_test_split_{fold+1}_metric.xlsx", index=False)
                confusion_matrix_test.to_excel(f"{name}_{self.__alias}_test_split_{fold+1}_confusion_matrix.xlsx", index=False)

    def __calc_roc_auc(self, y_true, y_pred_prob):
      n_classes = NUM_CLASS

      y_true_bin = label_binarize(y_true, classes=[i for i in range(n_classes)])

      fpr = dict()
      tpr = dict()
      roc_auc = dict()

      for i in range(n_classes):
          fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_prob[:, i])
          roc_auc[i] = [roc_auc_score(y_true_bin[:, i], y_pred_prob[:, i])]

      # Compute micro-average ROC curve and ROC area
      fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_pred_prob.ravel())
      roc_auc["micro"] = roc_auc_score(y_true_bin, y_pred_prob, average="micro")

      return fpr, tpr, roc_auc

    def __plot_roc_auc(self, train, val, test, name: str):
      fpr_train = train["fpr"]
      tpr_train = train["tpr"]
      roc_auc_train = train["roc_auc"]

      fpr_val = val["fpr"]
      tpr_val = val["tpr"]
      roc_auc_val = val["roc_auc"]

      fpr_test = test["fpr"]
      tpr_test = test["tpr"]
      roc_auc_test = test["roc_auc"]

      plt.figure(figsize=(8, 6))
      plt.plot([0, 1], [0, 1], 'k--', lw=2)  # Diagonal line

      #Plot for each set
      for fpr, tpr, roc_auc, set_name in zip(
          [fpr_train, fpr_val, fpr_test],
           [tpr_train, tpr_val, tpr_test],
            [roc_auc_train, roc_auc_val, roc_auc_test],
             ['train','val','test']):

          plt.plot(fpr["micro"], tpr["micro"],
              label=f'micro-average {set_name} ROC curve (area = {roc_auc["micro"]:0.2f})')

      plt.xlim([0.0, 1.0])
      plt.ylim([0.0, 1.05])
      plt.xlabel('False Positive Rate')
      plt.ylabel('True Positive Rate')
      plt.title(f'Receiver operating characteristic {name} {self.__alias}')
      plt.savefig(f"{name}_{self.__alias}_roc_auc.png")
      plt.legend(loc="lower right")
      plt.show()

    def train_test_method(self, portion="70_20_10"):
        portion_ratio_1 = 0
        portion_ratio_2 = 0
        if portion == "70_20_10":
          portion_ratio_1 = 0.33
          portion_ratio_2 = 0.33
        elif portion == "60_20_20":
          portion_ratio_1 = 0.4
          portion_ratio_2 = 0.5
        else:
          raise Exception(f"Invalid portion ratio for {portion}")

        X_train, X_temp, y_train, y_temp = train_test_split(self.X, self.y, test_size=portion_ratio_1, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=portion_ratio_2, random_state=42)


        X_train_CNN, X_temp_CNN, y_train_CNN, y_temp_CNN = train_test_split(self.X_CNN, self.y_CNN, test_size=portion_ratio_1, random_state=42)
        X_val_CNN, X_test_CNN, y_val_CNN, y_test_CNN = train_test_split(X_temp_CNN, y_temp_CNN, test_size=portion_ratio_2, random_state=42)

        self.__minimum_dataset(y_train)
        self.__minimum_dataset(y_val)
        self.__minimum_dataset(y_test)
        self.__minimum_dataset(y_train_CNN)
        self.__minimum_dataset(y_val_CNN)
        self.__minimum_dataset(y_test_CNN)


        model = {}
        for name in self.__model_name:
            if name == "LogisticRegression":
                model[name] = LogisticRegression(penalty='l2', multi_class="multinomial")
            elif name == "SVM":
                model[name] = SVC(probability=True)
            elif name == "RF":
                model[name] = RandomForestClassifier()
            elif name == "CNN":
                input_shape = self.X_CNN.shape[1:][::-1]
                model[name] = SimpleCNN(input_size=input_shape, num_classes=len(np.unique(self.y)))

        for name, mdl in model.items():
            if name == "CNN":
                X_train_CNN = X_train_CNN.reshape((X_train_CNN.shape[0], *X_train_CNN.shape[1:][::-1]))
                X_val_CNN = X_val_CNN.reshape((X_val_CNN.shape[0], *X_val_CNN.shape[1:][::-1]))
                X_test_CNN = X_test_CNN.reshape((X_test_CNN.shape[0], *X_test_CNN.shape[1:][::-1]))

                X_train_CNN = torch.from_numpy(X_train_CNN).to(torch.float32)
                X_val_CNN = torch.from_numpy(X_val_CNN).to(torch.float32)
                X_test_CNN = torch.from_numpy(X_test_CNN).to(torch.float32)

                y_train_CNN = torch.from_numpy(y_train)
                y_val_CNN = torch.from_numpy(y_val)
                y_test_CNN = torch.from_numpy(y_test)

            # Fit Model
            if name == "CNN":
                dataset = TensorDataset(X_train_CNN, y_train_CNN)
                train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

                criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(mdl.parameters(), lr=LR)

                mdl.train_model(
                    train_loader=train_loader,
                    criterion=criterion,
                    optimizer=optimizer,
                    epochs=EPOCHS,
                    device=device
                )
            else:
                mdl.fit(X_train, y_train)


            # Coef & Feature Important
            if name == "LogisticRegression":
                imp = mdl.coef_
            elif name == "SVM":
                imp = mdl.dual_coef_
            elif name == "RF":
                imp = mdl.feature_importances_
                imp = imp.reshape((1, imp.shape[0]))

            if name != "CNN":
              imp = pd.DataFrame(imp)
              imp.to_excel(f"{name}_{self.__alias}_{portion}_coef_feature_importance.xlsx", index=False)

            # LIME and SHAP for non-CNN models
            if name != "CNN":
                # LIME
                print(f"--- LIME Analysis for {name} ---")
                try:
                    explainer = lime.lime_tabular.LimeTabularExplainer(
                        X_train,
                        mode='classification',
                        class_names=[LABEL_CONVERTER[str(i)] for i in range(NUM_CLASS)],
                        feature_names=[f'feature_{i}' for i in range(X_train.shape[1])]
                    )
                    exp = explainer.explain_instance(X_test[0], mdl.predict_proba, num_features=5)
                    exp.show_in_notebook(show_table=True)
                except:
                    print("Error has occured to create lime analysis, so we're skipping that. It could be that you need more data!")

                # SHAP
                print(f"\n--- SHAP Analysis for {name} ---")
                try:
                    # Summarize the background data for KernelExplainer
                    X_train_summary = shap.kmeans(X_train, 50)
                    explainer = shap.KernelExplainer(mdl.predict_proba, X_train_summary)
                    shap_values = explainer.shap_values(X_test)
    
                    # Summary plot - now this should work correctly
                    print(f"SHAP Summary Plot for {name}")
                    shap.summary_plot(shap_values, X_test, feature_names=[f'feature_{i}' for i in range(X_train.shape[1])], class_names=[LABEL_CONVERTER[str(i)] for i in range(NUM_CLASS)])
                except:
                    print("Error has occured to create shap analysis, so we're skipping that. It could be that you need more data!")


            # TRAIN
            if name == "CNN":
                y_pred_train, y_pred_train_output = mdl.predict(X_train_CNN, device=device)
                y_pred_train = y_pred_train.cpu().numpy()
                y_pred_train_output = y_pred_train_output.cpu().numpy()
            else:
                y_pred_train = mdl.predict(X_train)

            train_metrics = {
                "Accuracy": [accuracy_score(y_train, y_pred_train)]
            }

            func_metric = [f1_score, recall_score, precision_score]
            func_name = ["F1-Score", "Recall", "Precision"]
            for func, fname in zip(func_metric, func_name):
                metrics = func(y_train, y_pred_train, average=None)
                for i, met in enumerate(metrics):
                    train_metrics[f"{fname}_{LABEL_CONVERTER[str(i)]}"] = met

            confusion_matrix_train = confusion_matrix(y_train, y_pred_train)

            train_metrics = pd.DataFrame(train_metrics)
            confusion_matrix_train = pd.DataFrame(confusion_matrix_train)

            # VAL
            if name == "CNN":
                y_pred_val, y_pred_val_output = mdl.predict(X_val_CNN, device=device)
                y_pred_val = y_pred_val.cpu().numpy()
                y_pred_val_output = y_pred_val_output.cpu().numpy()
            else:
                y_pred_val = mdl.predict(X_val)

            val_metrics = {
                "Accuracy": [accuracy_score(y_val, y_pred_val)]
            }

            for func, fname in zip(func_metric, func_name):
                metrics = func(y_val, y_pred_val, average=None)
                for i, met in enumerate(metrics):
                    val_metrics[f"{fname}_{LABEL_CONVERTER[str(i)]}"] = met

            confusion_matrix_val = confusion_matrix(y_val, y_pred_val)

            val_metrics = pd.DataFrame(val_metrics)
            confusion_matrix_val = pd.DataFrame(confusion_matrix_val)

            # TEST
            # VAL
            if name == "CNN":
                y_pred_test, y_pred_test_output = mdl.predict(X_test_CNN, device=device)
                y_pred_test = y_pred_test.cpu().numpy()
                y_pred_test_output = y_pred_test_output.cpu().numpy()
            else:
                y_pred_test = mdl.predict(X_test)

            test_metrics = {
                "Accuracy": [accuracy_score(y_test, y_pred_test)],
            }

            func_metric = [f1_score, recall_score, precision_score]
            func_name = ["F1-Score", "Recall", "Precision"]
            for func, fname in zip(func_metric, func_name):
                metrics = func(y_test, y_pred_test, average=None)
                for i, met in enumerate(metrics):
                    test_metrics[f"{fname}_{LABEL_CONVERTER[str(i)]}"] = met

            confusion_matrix_test = confusion_matrix(y_test, y_pred_test)

            test_metrics = pd.DataFrame(test_metrics)
            confusion_matrix_test = pd.DataFrame(confusion_matrix_test)

            # save to excel
            train_metrics.to_excel(f"{name}_{self.__alias}_{portion}_train_metric.xlsx", index=False)
            confusion_matrix_train.to_excel(f"{name}_{self.__alias}_confusion_matrix_{portion}_train.xlsx", index=False)

            val_metrics.to_excel(f"{name}_{self.__alias}_{portion}_val_metric.xlsx", index=False)
            confusion_matrix_val.to_excel(f"{name}_{self.__alias}_confusion_matrix_{portion}_val.xlsx", index=False)

            test_metrics.to_excel(f"{name}_{self.__alias}_{portion}_test_metric.xlsx", index=False)
            confusion_matrix_test.to_excel(f"{name}_{self.__alias}_confusion_matrix_{portion}_test.xlsx", index=False)

            # save roc
            if name == "CNN":
                save_roc_multiclass(y_train, y_pred_train_output, f"{name}_{self.__alias}_{portion}_train")
                save_roc_multiclass(y_val, y_pred_val_output, f"{name}_{self.__alias}_{portion}_val")
                save_roc_multiclass(y_test, y_pred_test_output, f"{name}_{self.__alias}_{portion}_test")
            else:
                save_roc_multiclass(y_train, mdl.predict_proba(X_train), f"{name}_{self.__alias}_{portion}_train")
                save_roc_multiclass(y_val, mdl.predict_proba(X_val), f"{name}_{self.__alias}_{portion}_val")
                save_roc_multiclass(y_test, mdl.predict_proba(X_test), f"{name}_{self.__alias}_{portion}_test")

            # plot roc auc
            if name == "CNN":
                fpr_train, tpr_train, roc_auc_train = self.__calc_roc_auc(y_train, y_pred_train_output)
                fpr_val, tpr_val, roc_auc_val = self.__calc_roc_auc(y_val, y_pred_val_output)
                fpr_test, tpr_test, roc_auc_test = self.__calc_roc_auc(y_test, y_pred_test_output)
            else:
                fpr_train, tpr_train, roc_auc_train = self.__calc_roc_auc(y_train, mdl.predict_proba(X_train))
                fpr_val, tpr_val, roc_auc_val = self.__calc_roc_auc(y_val, mdl.predict_proba(X_val))
                fpr_test, tpr_test, roc_auc_test = self.__calc_roc_auc(y_test, mdl.predict_proba(X_test))

            self.__plot_roc_auc(
                train={"fpr": fpr_train, "tpr": tpr_train, "roc_auc": roc_auc_train},
                val={"fpr": fpr_val, "tpr": tpr_val, "roc_auc": roc_auc_val},
                test={"fpr": fpr_test, "tpr": tpr_test, "roc_auc": roc_auc_test},
                name=name+portion
            )

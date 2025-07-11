import os
import cv2

import matplotlib.pyplot as plt

import numpy as np

import torchvision.transforms as transforms
import torchvision.transforms.functional as TF # Import the functional module

from PIL import Image

from utils import canny_image, rcnn_segmentation, unet_segmentation, mask_rcnn_segmentation


import warnings

warnings.filterwarnings("ignore")

class Dataset:
    def __init__(self, src_folder: str, alias: str=''):
        # public
        self.src_folder = src_folder
        
        # private
        self.__images = []
        self.__alias = alias

        self.__get_all_images()    

    def __get_all_images(self):
        for label in os.listdir(self.src_folder):
            fol_label = os.path.join(self.src_folder, label)
            for img in os.listdir(fol_label):
                img_path = os.path.join(fol_label, img)

                self.__images.append(img_path)

    def get_images_path(self):
        return self.__images
    
    def save_segmentation(self, dest_folder_path: str, segmentation: list[str]):
        segment_func = {"canny": canny_image, "rcnn": rcnn_segmentation, "unet": unet_segmentation, "mask_rcnn": mask_rcnn_segmentation}

        if not os.path.exists(dest_folder_path):
            os.mkdir(dest_folder_path)

        for image in self.__images:
            fname = image.split("/")[-1].split(".")[0]
            format = image.split("/")[-1].split(".")[1]

            label = image.split("/")[-2]
            if not os.path.exists(os.path.join(dest_folder_path, label)):
                os.mkdir(os.path.join(dest_folder_path, label))
            
            img = cv2.imread(image)
            ori_filename = f"{fname}_original.{format}"
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ori_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            for segment in segmentation:
                filename = f"{fname}_{segment}.{format}"

                try:
                    if segment == "canny":
                        segment_img = segment_func[segment](gray_image)
                    elif segment == "rcnn":
                        segment_img = segment_func[segment](image)
                        segment_img = cv2.cvtColor(segment_img, cv2.COLOR_BGR2RGB)
                    elif segment == "mask_rcnn":
                        segment_img = segment_func[segment](image)
                        segment_img = cv2.cvtColor(segment_img, cv2.COLOR_BGR2RGB)
                    elif segment == "unet":
                        segment_img = segment_func[segment](image)
                        segment_img = (segment_img * 255).astype('uint8')

                    cv2.imwrite(os.path.join(dest_folder_path, label, filename), segment_img)
                    del segment_img
                except:
                    print(f"{segment} for {image} failed")
                
            cv2.imwrite(os.path.join(dest_folder_path, label, ori_filename), ori_img)
    
    def plot_segmentation(self):
        random_image = np.random.choice(self.__images)
        
        img = cv2.imread(random_image)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        canny_result = canny_image(gray_img)
        try:
            # unet_result = unet_segmentation(random_image)
            unet_result = None
            rcnn_result = rcnn_segmentation(random_image)
        except Exception as e:
            print(f"Error during segmentation: {e}, [Please try again!]")
            unet_result = None
            rcnn_result = None
        
        try:
            mask_rcnn_result = mask_rcnn_segmentation(random_image)
        except Exception as e:
            print(f"Error during segmentation: {e}, [Please try again!]")
            mask_rcnn_result = None


        plt.figure(figsize=(15, 10))

        plt.subplot(2, 3, 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')

        plt.subplot(2, 3, 2)
        plt.imshow(canny_result, cmap='gray')
        plt.title('Canny Edge Detection')

        if mask_rcnn_result is not None:
            plt.subplot(2, 3, 3)
            plt.imshow(mask_rcnn_result)
            plt.title('Mask R-CNN Segmentation')

        if unet_result is not None:
            plt.subplot(2, 3, 4)
            plt.imshow(unet_result, cmap='gray')
            plt.title('U-Net Segmentation')

        if rcnn_result is not None:
            plt.subplot(2, 3, 5)
            plt.imshow(rcnn_result)
            plt.title('Faster R-CNN Segmentation')

        plt.tight_layout()
        plt.savefig(f"plot_{self.__alias}_segmentation.png")
        plt.show()

    def plot_augmentation(self, augmentation: dict):
      random_image = np.random.choice(self.__images)
      
      for name, augmen in augmentation.items():
        trans = transforms.Compose([
            augmen,
            transforms.ToTensor()
        ])

        img = Image.open(random_image)
        img = trans(img)

        img = img.squeeze() 
        img = img.cpu()  
        img = TF.to_pil_image(img)

        plt.figure(figsize=(15, 10))

        plt.subplot(1, 1, 1)
        plt.imshow(img)
        plt.title(f'{name} Augmentation')

        plt.tight_layout()
        plt.savefig(f"plot_{name}_{self.__alias}_augmentation.png")
        plt.show()



    def save_augmentation(self, dest_folder_path: str, augmentation: dict):
        if not os.path.exists(dest_folder_path):
            os.mkdir(dest_folder_path)

        for name, augmen in augmentation.items():
            trans = transforms.Compose([
                augmen,
                transforms.ToTensor()
            ])

            for image in self.__images:
                filename = image.split("/")[-1]
                label = image.split("/")[-2]

                if not os.path.exists(os.path.join(dest_folder_path, label)):
                    os.mkdir(os.path.join(dest_folder_path, label))

                img = Image.open(image)
                img = trans(img)

                img = img.squeeze() 
                img = img.cpu()  
                img = TF.to_pil_image(img)
                img.save(f"{dest_folder_path}/{label}/{name}_{filename}")

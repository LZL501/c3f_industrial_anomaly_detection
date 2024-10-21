import os, yaml, pickle, shutil, tarfile, glob
import cv2
from PIL import Image
import numpy as np
import torch
import glob
import torchvision as tv
from functools import partial
from PIL import Image, ImageFilter
from tqdm import tqdm
from torch.utils.data import Dataset, Subset
import pandas as pd
import math


classes = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']


class VisA(Dataset):
    def __init__(self, data_path, object_label):
        df = pd.read_csv(os.path.join(data_path, 'split_csv/1cls.csv'))

        train_list = df[df['split'].str.contains('train')]
        
        train_list = train_list[train_list["object"] == object_label]

        train_list = train_list.to_dict(orient='list')

        train_img_list = train_list["image"]
        
        self.images = [Image.open(os.path.join(data_path, img_path)).convert("RGB") for img_path in train_img_list]
        self.tf_1 = tv.transforms.Resize((256, 256))
        self.tf_2 = tv.transforms.Compose([tv.transforms.ToTensor(), 
                                           tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        # self.tf_1 = tv.transforms.Compose([tv.transforms.Resize((256, 256)), 
        #                                    tv.transforms.ToTensor()])
        self.images = [self.tf_1(image) for image in self.images]
        # self.images = [cv2.medianBlur(np.array(image), 7) for image in self.images]
        self.images = [self.tf_2(image) for image in self.images]

        
    def __getitem__(self, idx):
        return self.images[idx]
    
    
    def __len__(self):
        return len(self.images)


class VisA_validate(Dataset):
    def __init__(self, data_path, object_label):
        # image_paths = glob.glob(data_path + '/Data/Images/Anomaly/*.jpg')
        df = pd.read_csv(os.path.join(data_path, 'split_csv/1cls.csv'))
        test_list = df[df['split'].str.contains('test')]
        test_list = test_list[test_list["object"] == object_label]

        test_list = test_list.to_dict(orient='list')

        test_img_list = test_list["image"]
        test_mask_list = test_list["mask"]
        
        for i in range(len(test_mask_list)):
            if isinstance(test_mask_list[i], str):
                pass
            else:
                if math.isnan(test_mask_list[i]):
                    test_mask_list[i] = "good"
        self.images = [Image.open(os.path.join(data_path, img_path)).convert("RGB") for img_path in test_img_list]
        self.masks = []
        self.labels = []

        for i, mask in enumerate(test_mask_list):
            if mask == "good":
                self.masks.append(Image.fromarray(np.zeros((self.images[i].size[1], self.images[i].size[0]))))
                self.labels.append(0)
            else:
                self.masks.append(Image.open(os.path.join(data_path, mask)))
                self.labels.append(1)
                     
        # self.tf_1 = self.tf_1 = tv.transforms.Compose([tv.transforms.Resize((256, 256)), 
        #                                                 tv.transforms.ToTensor()])
        self.tf_1 = tv.transforms.Resize((256, 256))
        self.tf_2 = tv.transforms.ToTensor()
        self.tf_3 = tv.transforms.Compose([tv.transforms.ToTensor(), 
                                           tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.images = [self.tf_1(image) for image in self.images]
        # self.images = [cv2.medianBlur(np.array(image), 7) for image in self.images]
        self.images = [self.tf_3(image) for image in self.images]
        self.masks = [self.tf_2(self.tf_1(mask)) for mask in self.masks]


    def __getitem__(self, idx):
        return self.images[idx], self.masks[idx], self.labels[idx]
    
    
    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    dataset = VisA('/data/arima/VisA', 'fryum')
    print(len(dataset))
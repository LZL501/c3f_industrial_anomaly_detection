import os, yaml, pickle, shutil, tarfile, glob
import cv2
import albumentations
from PIL import Image
import numpy as np
import torch
import glob
import torchvision as tv
from omegaconf import OmegaConf
from functools import partial
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, Subset

import taming.data.utils as tdu
from taming.data.imagenet import str_to_indices, give_synsets_from_indices, download, retrieve
from taming.data.imagenet import ImagePaths

import imgaug.augmenters as iaa

from ldm.data.perlin import rand_perlin_2d_np

class MVTec(Dataset):
    def __init__(self, data_path, texture_source_dir = None, structure_grid_size = 8, transparency_range = [0.15, 1.], \
                    perlin_scale = 6, min_perlin_scale = 0, perlin_noise_threshold = 0.5):
        image_paths = glob.glob(data_path + '/train/good/*.png')
        # image_paths = glob.glob(os.path.join(data_path, '/*/train/good/*.png'))
        self.images = [Image.open(path).convert("RGB") for path in image_paths]
        self.masks = []
        self.labels = []
        self.resize = (256, 256)
        
        
        self.tf_1 = tv.transforms.Resize(self.resize)
        # self.tf_2 = tv.transforms.Compose([tv.transforms.CenterCrop(224), 
        #                                     tv.transforms.ToTensor()])
        self.tf_2 = tv.transforms.ToTensor()
        self.tf_3 = tv.transforms.Compose([
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(
                mean = (0.485, 0.456, 0.406),
                std  = (0.229, 0.224, 0.225)
            )
        ])
        # self.tf_1 = tv.transforms.Compose([tv.transforms.Resize((256, 256)), 
        #                                    tv.transforms.ToTensor()])
        self.images = [self.tf_1(image) for image in self.images]
        # self.images = [cv2.medianBlur(np.array(image), 3) for image in self.images]
        self.images = [self.tf_3(image) for image in self.images]

    def __getitem__(self, idx):
        # example = dict()
        # example["image"] = (self.tf_2(self.images[idx]) * 2.0 - 1.0)
        # return example
        # return (self.images[idx] * 2.0 - 1.0)\
        
        img = self.images[idx]
        # img[img > 150 / 255] = 150 / 255
        # img = 1 - img
        # img_gray = cv2.cvtColor((img.permute(1, 2, 0).numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        # target_background_masks = torch.from_numpy(cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]).unsqueeze(0).repeat(3, 1, 1)
        # img[target_background_masks > 0] = 0.5
        # return img * 2 - 1
        return img
    
    def __len__(self):
        return len(self.images)


class MVTec_validate(Dataset):
    def __init__(self, data_path):
        classes = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
        image_paths = glob.glob(data_path + '/test/**/*.png')
        self.images = [Image.open(path).convert("RGB") for path in image_paths]
        
        self.masks = []
        self.labels = []
        for i, path in enumerate(image_paths):
            mask_path = path.replace("test", "ground_truth").replace(".png", "_mask.png")
            if os.path.exists(mask_path):
                self.masks.append(Image.open(mask_path))
                self.labels.append(1)
            else:
                self.masks.append(Image.fromarray(np.zeros((self.images[i].size[1], self.images[i].size[0]))))
                self.labels.append(0)
        self.tf_1 = tv.transforms.Resize((256, 256))
        self.tf_2 = tv.transforms.ToTensor()
        self.tf_3 = tv.transforms.Compose([
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(
                mean = (0.485, 0.456, 0.406),
                std  = (0.229, 0.224, 0.225)
            )
        ])
        # self.tf_2 = tv.transforms.Compose([tv.transforms.CenterCrop(224), 
        #                                     tv.transforms.ToTensor()])
        self.images = [self.tf_1(image) for image in self.images]
        # self.images = [cv2.medianBlur(np.array(image), 3) for image in self.images]
        self.images = [self.tf_3(image) for image in self.images]
        self.masks = [self.tf_2(self.tf_1(mask)) for mask in self.masks]

    def __getitem__(self, idx):
        img = self.images[idx]
        # img[img > 150 / 255] = 150 / 255
        # img = 1 - img
        # img_gray = cv2.cvtColor((img.permute(1, 2, 0).numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        # target_background_masks = torch.from_numpy(cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]).unsqueeze(0).repeat(3, 1, 1)
        # img[target_background_masks > 0] = 0.5
        # return (self.images[idx] * 2.0 - 1.0), self.masks[idx], self.labels[idx]
        # return (img * 2.0 - 1.0), self.masks[idx], self.labels[idx]
        return img, self.masks[idx], self.labels[idx]
    
    def __len__(self):
        return len(self.images)


class MVTec_validate2(Dataset):
    def __init__(self, data_path):
        classes = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
        image_paths = glob.glob(data_path + '/test/**/*.png')
        self.images = [Image.open(path).convert("RGB") for path in image_paths]
        
        self.masks = []
        self.labels = []
        for i, path in enumerate(image_paths):
            mask_path = path.replace("test", "ground_truth").replace(".png", "_mask.png")
            if os.path.exists(mask_path):
                self.masks.append(Image.open(mask_path))
                self.labels.append(1)
            else:
                self.masks.append(Image.fromarray(np.zeros((self.images[i].size[1], self.images[i].size[0]))))
                self.labels.append(0)
        self.tf_1 = tv.transforms.Resize((256, 256))
        # self.tf_2 = tv.transforms.ToTensor()
        
        # self.tf_1 = tv.transforms.Compose([tv.transforms.Resize((256, 256)), 
        #                                    tv.transforms.ToTensor()])
        self.images = [self.tf_1(image) for image in self.images]
        self.images = [cv2.medianBlur(np.array(image), 3) for image in self.images]
        self.images = [self.tf_2(image) for image in self.images]
        self.masks = [self.tf_2(self.tf_1(mask)) for mask in self.masks]

        img_grays = [cv2.cvtColor((img.permute(1, 2, 0).numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) for img in self.images]
        self.target_background_masks = [cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)[1] / 255 for img_gray in img_grays]

    def __getitem__(self, idx):
        img = self.images[idx]
        # img_gray = cv2.cvtColor((img.permute(1, 2, 0).numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        # target_background_masks = torch.from_numpy(cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]).unsqueeze(0).repeat(3, 1, 1)
        # img[target_background_masks > 0] = 0.5
        return (self.images[idx] * 2.0 - 1.0), self.masks[idx], self.labels[idx]
        # return (img * 2.0 - 1.0), self.masks[idx], self.labels[idx], self.target_background_masks[idx]
    
    def __len__(self):
        return len(self.images)



from enum import Enum
import PIL
from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"

class MVTecDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for MVTec.
    """

    def __init__(
        self,
        source,
        classname,
        resize=256,
        imagesize=224,
        split=DatasetSplit.TRAIN,
        train_val_split=1.0,
        **kwargs,
    ):
        """
        Args:
            source: [str]. Path to the MVTec data folder.
            classname: [str or None]. Name of MVTec class that should be
                       provided in this dataset. If None, the datasets
                       iterates over all available images.
            resize: [int]. (Square) Size the loaded image initially gets
                    resized to.
            imagesize: [int]. (Square) Size the resized loaded image gets
                       (center-)cropped to.
            split: [enum-option]. Indicates if training or test split of the
                   data should be used. Has to be an option taken from
                   DatasetSplit, e.g. mvtec.DatasetSplit.TRAIN. Note that
                   mvtec.DatasetSplit.TEST will also load mask data.
        """
        super().__init__()
        self.source = source
        self.split = split
        self.classnames_to_use = [classname] if classname is not None else _CLASSNAMES
        self.train_val_split = train_val_split
        self.transform_mean = IMAGENET_MEAN
        self.transform_std = IMAGENET_STD
        self.imgpaths_per_class, self.data_to_iterate = self.get_image_data()

        self.transform_img = [
            transforms.Resize(resize),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        self.transform_img = transforms.Compose(self.transform_img)

        self.transform_mask = [
            transforms.Resize(resize),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
        ]
        self.transform_mask = transforms.Compose(self.transform_mask)

        self.imagesize = (3, imagesize, imagesize)

    def __getitem__(self, idx):
        classname, anomaly, image_path, mask_path = self.data_to_iterate[idx]
        image = PIL.Image.open(image_path).convert("RGB")
        image = self.transform_img(image)

        if self.split == DatasetSplit.TEST and mask_path is not None:
            mask = PIL.Image.open(mask_path)
            mask = self.transform_mask(mask)
        else:
            mask = torch.zeros([1, *image.size()[1:]])

        return {
            "image": image,
            "mask": mask,
            "classname": classname,
            "anomaly": anomaly,
            "is_anomaly": int(anomaly != "good"),
            "image_name": "/".join(image_path.split("/")[-4:]),
            "image_path": image_path,
        }

    def __len__(self):
        return len(self.data_to_iterate)

    def get_image_data(self):
        imgpaths_per_class = {}
        maskpaths_per_class = {}

        for classname in self.classnames_to_use:
            classpath = os.path.join(self.source, classname, self.split.value)
            maskpath = os.path.join(self.source, classname, "ground_truth")
            anomaly_types = os.listdir(classpath)

            imgpaths_per_class[classname] = {}
            maskpaths_per_class[classname] = {}

            for anomaly in anomaly_types:
                anomaly_path = os.path.join(classpath, anomaly)
                anomaly_files = sorted(os.listdir(anomaly_path))
                imgpaths_per_class[classname][anomaly] = [
                    os.path.join(anomaly_path, x) for x in anomaly_files
                ]

                if self.train_val_split < 1.0:
                    n_images = len(imgpaths_per_class[classname][anomaly])
                    train_val_split_idx = int(n_images * self.train_val_split)
                    if self.split == DatasetSplit.TRAIN:
                        imgpaths_per_class[classname][anomaly] = imgpaths_per_class[
                            classname
                        ][anomaly][:train_val_split_idx]
                    elif self.split == DatasetSplit.VAL:
                        imgpaths_per_class[classname][anomaly] = imgpaths_per_class[
                            classname
                        ][anomaly][train_val_split_idx:]

                if self.split == DatasetSplit.TEST and anomaly != "good":
                    anomaly_mask_path = os.path.join(maskpath, anomaly)
                    anomaly_mask_files = sorted(os.listdir(anomaly_mask_path))
                    maskpaths_per_class[classname][anomaly] = [
                        os.path.join(anomaly_mask_path, x) for x in anomaly_mask_files
                    ]
                else:
                    maskpaths_per_class[classname]["good"] = None

        # Unrolls the data dictionary to an easy-to-iterate list.
        data_to_iterate = []
        for classname in sorted(imgpaths_per_class.keys()):
            for anomaly in sorted(imgpaths_per_class[classname].keys()):
                for i, image_path in enumerate(imgpaths_per_class[classname][anomaly]):
                    data_tuple = [classname, anomaly, image_path]
                    if self.split == DatasetSplit.TEST and anomaly != "good":
                        data_tuple.append(maskpaths_per_class[classname][anomaly][i])
                    else:
                        data_tuple.append(None)
                    data_to_iterate.append(data_tuple)

        return imgpaths_per_class, data_to_iterate
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 22:08:23 2020

@author: hluo_
"""

import os, sys, glob, json, shutil
from PIL import Image
import numpy as np
from tqdm import tqdm

import torch
torch.manual_seed(0)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from  albumentations  import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomContrast, RandomBrightness, Flip, OneOf, Compose,
    Resize, RandomCrop)

class SVHNDataSet(Dataset):
    """
        Tranning data and Validation data
    """
    def __init__(self, root, dtype = 'train', transform = None, aug = True):
        super(SVHNDataSet, self).__init__()
        
        self.dtype = dtype
        self.transform = transform
        self.aug = aug
        if self.dtype == 'train':
            self.img_path = os.path.join(root, 'train/')
            self.label_json = os.path.join(root, 'train.json')
            self.label = json.load(open(self.label_json))
        elif self.dtype == 'val':
            self.img_path = os.path.join(root, 'val/')
            self.label_json = os.path.join(root, 'val.json')
            self.label = json.load(open(self.label_json))
        self.Data_index()
        self.resize = self.__resize()
        self.augment = self.__augment()


    def Data_index(self):
        """
            Assign the image and label
        """
        self.files = []
        for fil in sorted(os.listdir(self.img_path)):
            self.files.append({
                                'img' : os.path.join(self.img_path, fil),
                                'label' : self.label[fil]
                                    })
                
    def __resize(self):
        if self.dtype == 'train':
            return Compose([Resize(64,128),
                            RandomCrop(60, 120)])
        else:
            return Compose([Resize(60, 120)])
        
    def __augment(self, p=0.5):
        return Compose([OneOf([IAAAdditiveGaussianNoise(),
                               GaussNoise(),], 
                              p=0.2),
                        OneOf([MotionBlur(p=.4),
                               MedianBlur(blur_limit=3, p=.4),
                               Blur(blur_limit=3, p=.1),], 
                              p=0.2),
                        ShiftScaleRotate(shift_limit=0.3, scale_limit=0.2, rotate_limit=10, p=.6),
                        OneOf([
                            OpticalDistortion(p=0.3),
                            GridDistortion(p=.1),
                            IAAPiecewiseAffine(p=0.3),
                            ], p=0.2),
                        OneOf([
                            CLAHE(clip_limit=2),
                            IAASharpen(),
                            IAAEmboss(),
                            RandomContrast(),
                            RandomBrightness(),
                            ], p=0.3),
                        HueSaturationValue(p=1),
                        ], p=p)
    
    
    def __getitem__(self, index):
    
        img = Image.open(self.files[index]['img']).convert('RGB')
        img_array = np.array(img)
        if self.aug:
            img_array = self.augment(image = img_array)['image']
        label = self.files[index]['label']
        img_array = self.resize(image = img_array)['image']
        if self.transform is not None:
            img_array = self.transform(img_array)
        
        lbl = label['label']+[10]*(5-len(label['label']))
        
        return img_array, torch.from_numpy(np.array(lbl[:4]))
    
    def __len__(self):
        return len(self.files)


class SVHNTest(Dataset):
    """
        Tranning data and Validation data
    """
    def __init__(self, root, transform = None):
        super(SVHNTest, self).__init__()
        
        self.transform = transform
        self.img_path = os.path.join(root, 'test/')
        # self.val_img_path = os.path.join(root, 'val/')
        # self.val_label_json = os.path.join(root, 'val.json')
        self.Data_index()
        
    def Data_index(self):
        """
            Assign the image and label
        """
        self.files = []
        for fil in sorted(os.listdir(self.img_path)):
            self.files.append({
                                'img' : os.path.join(self.img_path, fil),
                                })
        
   
    def __getitem__(self, index):
        img = Image.open(self.files[index]['img']).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img
    
    def __len__(self):
        return len(self.files)




def DataLoader(root, dtype = 'train', num_workers = 2):
    if dtype == 'train':
        data_loader = torch.utils.data.DataLoader(
        SVHNDataSet(root, dtype,
                    transforms.Compose([
                    # transforms.Resize((64, 128)),
                    # transforms.CenterCrop((60, 120)),
                    # transforms.ColorJitter(0.3, 0.3, 0.2),
                    # transforms.RandomRotation(10),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]), aug = True), 
        batch_size=40, 
        shuffle=True, 
        num_workers=num_workers,
        )
    elif dtype == 'val':
        data_loader = torch.utils.data.DataLoader(
        SVHNDataSet(root,dtype,
                    transforms.Compose([
                    # transforms.Resize((60, 120)),
                   #transforms.RandomCrop((60, 120)),
                   #transforms.ColorJitter(0.3, 0.3, 0.2),
                   #transforms.RandomRotation(10),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]), aug = False),
        batch_size=40,
        shuffle=False,
        num_workers = num_workers,
        )
    elif dtype == 'test':
        data_loader = torch.utils.data.DataLoader(
        SVHNTest(root,
                 transforms.Compose([
                 transforms.Resize((60, 120)),
                #transforms.RandomCrop((60, 120)),
                #transforms.ColorJitter(0.3, 0.3, 0.2),
                #transforms.RandomRotation(10),
                 transforms.ToTensor(),
                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])),
        batch_size=40,
        shuffle = False,
        num_workers = num_workers,
        )
    
    return data_loader
    
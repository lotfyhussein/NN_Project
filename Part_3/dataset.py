import os
from pathlib import Path
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms as tfs
import torch
import cv2
import random
import matplotlib.pyplot as plt
import subprocess

def load_images(split="train"):
    all_data_path = Path("../leftImg8bit")
    all_labels_path = Path("../gtFine/")
    image_paths = []
    label_paths = []
    all_data_path = all_data_path / split
    all_labels_path = all_labels_path / split
    
    folders = sorted(os.listdir(all_data_path))
    labels_folders = sorted(os.listdir(all_labels_path))
    
    for folder in folders:
        folder_path = all_data_path / folder
        files_in_folder = sorted(os.listdir(folder_path))
        for f in files_in_folder:
            image_paths.append(str(folder_path / f))
            
    for folder in labels_folders:
        folder_path = all_labels_path / folder
        files_in_folder = sorted(os.listdir(folder_path))
        for f in files_in_folder:
            if f[-12:-4] == "labelIds":
                label_paths.append(str(folder_path / f))
    
    return image_paths,label_paths

def transforms(im,label,crop_size):
    resize = tfs.Resize((crop_size[1],crop_size[0]),interpolation=Image.NEAREST )
    im = resize(im)
    label = resize(label)
    image_transform=tfs.Compose([
    tfs.ToTensor(),
    tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    im = image_transform(im)

    #Transform to tensor
    im = np.array(im)
    label = np.array(label)
    im = torch.from_numpy(im).float()
    label = torch.from_numpy(label).long()
    return im,label

class cityscape_dataset(Dataset):
    
    def __init__(self, split, transforms, crop_size):
        self.crop_size=crop_size
        self.transforms=transforms
        self.data_list,self.label_list=load_images(split)
        print('Read'+str(len(self.data_list))+'images')

    def __getitem__(self,idx):
        img = self.data_list[idx]
        label = self.label_list[idx]
        img = Image.open(img).convert("RGB")
        label = Image.open(label)
        # Mapping of ignore categories and valid ones (numbered from 1-19)
        mapping_20 = { 
            0: 0,
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 0,
            7: 1,
            8: 2,
            9: 0,
            10: 0,
            11: 3,
            12: 4,
            13: 5,
            14: 0,
            15: 0,
            16: 0,
            17: 6,
            18: 0,
            19: 7,
            20: 8,
            21: 9,
            22: 10,
            23: 11,
            24: 12,
            25: 13,
            26: 14,
            27: 15,
            28: 16,
            29: 0,
            30: 0,
            31: 17,
            32: 18,
            33: 19,
            -1: 0
        }
        img, label=self.transforms(img,label,self.crop_size)
        label_mask = np.zeros_like(label)
        for k in mapping_20:
            label_mask[label == k] = mapping_20[k]
        return img, label_mask 
    
    def __len__(self):
        return len(self.data_list)

def get_loader(split='train'):
    input_shape=(512,256)
    bs = 1
    if split == 'train':
        train_dataset = cityscape_dataset(split="train",transforms=transforms, crop_size=input_shape)
        return DataLoader(train_dataset,batch_size=bs,shuffle=True), train_dataset
    else:
        valid_dataset = cityscape_dataset(split="val",transforms=transforms,crop_size=input_shape)
        return DataLoader(valid_dataset,batch_size=bs,shuffle=True), valid_dataset

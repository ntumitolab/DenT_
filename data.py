import os
from pathlib import Path # 
import torch
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from torch.utils.data import Dataset
import numpy as np
from scipy import ndimage, misc
from skimage import io
import random

def normalize(img):
    """Subtract mean, set STD to 1.0"""
    result = img.astype(np.float64)
    result -= np.mean(result)
    result /= np.std(result)
    return result

def resize_3D(img, factor):
    factors = (1, factor, factor)
    result = ndimage.zoom(img, factors, mode='nearest')
    return result

def random_rotation(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k, (1,2)).copy()
    label = np.rot90(label, k, (1,2)).copy()
    return image, label

def random_flip(image, label):
    axis = np.random.randint(1, 3)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label

class SegDataset(Dataset):
    def __init__(self, args, mode="train"):
        self.model = args.model
        self.patch = args.patch
        self.mode = mode
        self.root_dir = os.path.join(args.data_path, mode)
        
        #for tiff        
        self.img_dir = os.path.join(self.root_dir, args.source_image); print(f"img : {Path(self.img_dir).resolve()}") # : data\{DataSet}_DenT\train\source
        self.seg_dir = os.path.join(self.root_dir, args.target_image); print(f"seg : {Path(self.seg_dir).resolve()}") # 
        self.path_lists = [(os.path.join(self.img_dir, img), os.path.join(self.seg_dir, img)) for img in os.listdir(self.img_dir)]
        
    def __len__(self):
        return len(self.path_lists)

    def __getitem__(self, index):
        #read 3D image and target mask (tiff)
        img_name = self.path_lists[index][0]
        seg_name = self.path_lists[index][1]

        img = io.imread(img_name) #TL (32,917,917)
        img = normalize(img)
        img = resize_3D(img, 0.2792) #917->256(2792)  917->512(5583)
        
        seg = io.imread(seg_name)
        seg = resize_3D(seg, 0.2792)
        seg = seg/255
        
        #data augmentation
        if self.mode != "test":
            if random.random() > 0.5:
                img, seg = random_rotation(img, seg)
            if random.random() > 0.5:
                img, seg = random_flip(img, seg)

        img = torch.from_numpy(img.astype(float)).float()
        img = torch.unsqueeze(img, 0) #(32,512,512) -> (1,32,512,512)
        seg = torch.from_numpy(seg).float()
        seg = torch.unsqueeze(seg, 0)

        #patches
        if self.patch == True:
            kc, kh, kw = (32, 256, 256) #kernel size
            dc, dh, dw = (32, 256, 256) #stride 
            img = img.unfold(1, kc, dc).unfold(2, kh, dh).unfold(3, kw, dw)
            img = img.permute(1,2,3,0,4,5,6) # C,pc,ph,pw,C,H,W->pc,....C,C,H,W
            img = img.contiguous().view(-1, 1, kc, kh, kw)
                
            seg = seg.unfold(1, kc, dc).unfold(2, kh, dh).unfold(3, kw, dw)
            seg = seg.permute(1,2,3,0,4,5,6)
            seg = seg.contiguous().view(-1, 1, kc, kh, kw)

            if self.mode == "val" or self.mode == "train":
                rand = torch.randint(len(img), (16,)) #choose the number of patches to use
                img = img[rand]
                seg = seg[rand]
     
        return img_name.split(os.sep)[-1], img, seg # 
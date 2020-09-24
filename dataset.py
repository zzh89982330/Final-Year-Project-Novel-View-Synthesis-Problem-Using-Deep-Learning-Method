from torch.utils.data import Dataset
from PIL.Image import Image
import os.path as path
import numpy as np
import cv2
import torch.nn as nn
from graphics import *

def read_raw_tensorimg(folder, x_idx):
    img_name = "Frame_" + str(x_idx).zfill(3) + '.png'
    img_path = path.join(folder, img_name)
    img = cv2.imread(img_path) / 255
    return img

class TrainingDataset(Dataset):#需要继承data.Dataset
    def __init__(self, training_df, img_folder):
        # TODO
        # 1. Initialize file path or list of file names.
        self.training_df = training_df
        self.img_folder = img_folder
        self.depth_levels = list(range(1, 100, 5))
        self.f = 10
        
    def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        #这里需要注意的是，第一步：read one data，是一个data
        reference_x, other_xs, target_x = list(self.training_df.iloc[index, :])
        other_xs = np.array(other_xs)
        ref_img = read_raw_tensorimg(self.img_folder, reference_x)
        other_imgs = []
        for other_x in other_xs:
            other_img = read_raw_tensorimg(self.img_folder, other_x)
            other_imgs.append(other_img)
        #other_imgs = tuple(other_imgs)
        #other_imgs = np.concatenate(other_imgs, axis=0)
        other_imgs = np.array(other_imgs)
        input_feat = plane_sweep(reference_x, other_xs, ref_img, other_imgs, self.depth_levels, self.f)
        input_feat = np.transpose(input_feat, (2, 0, 1))
        #input_feat = torch.from_numpy(input_feat)
        #target_x = torch.from_numpy(np.array(target_x))
        return input_feat, reference_x, target_x
        
    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.training_df)
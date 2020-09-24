# data input processing:
# image input processing
# classes:
# input and processing:
"""
TSampleSelector: select training sample groups, able to be written to the disk, to store the training samples used
######params######: 
 1. num_pose_images: number of images for the pose images to do the plane sweep
 2. training_samples_df: dataframe storing the training samples, <reference_I, otherImg_1, otherImg_2, ..., t_image, 
 pose_ref, pose_other1, pose_other2,..., pose_t>, define the middle image as the 0 in x axis
 3. img_folder: folder path storing the images
 4. num_mag: number of single side magnification x
 5. num_total: total number of images in the training dataset (per dataset)
 
 采样方法：以较小的采样间隔，较小的最大最小间距差进行训练（初步定为4个image进行训练，一个reference image，3个other images），
 并且测试一倍的magnification（左右各多2个，两边拓展4个）

######methods#####:
 1. sampling: pre-make all sample pairs, store them into the training_samples_df
"""
import pandas as pd
import torch
from torch.utils.data import DataLoader

class TSampleSelector:
    def __init__(self, img_folder):
        self.img_folder = img_folder
        self.training_samples_df = pd.DataFrame(columns=['reference_x', 'other_xs', 'target_x'])
        self.num_mag = 2
        self.num_pose_images = 1
        self.num_total = 100
        
    def sampling(self):
        num_pergroup = 2 * self.num_mag + self.num_pose_images + 1
        for i in range(0, self.num_total - num_pergroup):
            mag_indxes_1 = list(range(i, i + self.num_mag))
            mag_indees_2 = list(range(i + self.num_mag + self.num_pose_images + 1, i + self.num_mag * 2 + self.num_pose_images + 1))
            ref_idx = i + self.num_mag
            pose_idxes = list(range(i + self.num_mag + 1, i + self.num_mag + self.num_pose_images + 1))
            mag_idxes = mag_indxes_1 + mag_indees_2
            for t in mag_idxes:
                new_df = pd.DataFrame([[ref_idx, pose_idxes, t]], columns=['reference_x', 'other_xs', 'target_x'])
                self.training_samples_df = self.training_samples_df.append(new_df)
        return self.training_samples_df
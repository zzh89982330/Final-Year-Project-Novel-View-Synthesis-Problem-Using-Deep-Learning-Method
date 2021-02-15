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
import os
import cv2
from dataset import *

class TSampleSelector:
    def __init__(self, img_folder, crop_size):
        self.img_folder = img_folder
        self.training_samples_df = pd.DataFrame(columns=['reference_x', 'other_xs', 'target_x', 'img_cod'])
        self.num_mag = 10
        self.num_pose_images = 4
        self.num_total = 100
        img_list = os.listdir(img_folder)
        sample = cv2.imread(os.path.join(img_folder, img_list[0]))
        self.h, self.w, _ = sample.shape
        self.crop_size = crop_size

    def sampling(self):
        self.training_samples_df = pd.DataFrame(columns=['reference_x', 'other_xs', 'target_x', 'img_cod'])
        num_pergroup = 2 * self.num_mag + self.num_pose_images + 1
        for i in range(0, self.num_total - num_pergroup):
            mag_indxes_1 = list(range(i, i + self.num_mag))
            mag_indees_2 = list(range(i + self.num_mag + self.num_pose_images + 1, i + self.num_mag * 2 + self.num_pose_images + 1))
            ref_idx = i + self.num_mag
            pose_idxes = list(range(i + self.num_mag + 1, i + self.num_mag + self.num_pose_images + 1))
            mag_idxes = mag_indxes_1 + mag_indees_2
            num_w = self.w // self.crop_size - 1
            num_h = self.h // self.crop_size - 1

            data_frame_data = []
            for t in mag_idxes:
                for i_w in range(num_w):
                    for i_h in range(num_h):
                        data_frame_data.append([ref_idx, pose_idxes, t,
                                                [i_h * self.crop_size, (i_h + 1) * self.crop_size, i_w * self.crop_size, (i_w + 1) * self.crop_size]])
            append_dataframe = pd.DataFrame(data_frame_data, columns=['reference_x', 'other_xs', 'target_x', 'img_cod'])
            self.training_samples_df = self.training_samples_df.append(append_dataframe)
        return self.training_samples_df

    def sampling_whole(self):
        self.training_samples_df = pd.DataFrame(columns=['reference_x', 'other_xs', 'target_x'])
        num_pergroup = 2 * self.num_mag + self.num_pose_images + 1
        for i in range(0, self.num_total - num_pergroup):
            mag_indxes_1 = list(range(i, i + self.num_mag))
            mag_indees_2 = list(
                range(i + self.num_mag + self.num_pose_images + 1, i + self.num_mag * 2 + self.num_pose_images + 1))
            ref_idx = i + self.num_mag
            pose_idxes = list(range(i + self.num_mag + 1, i + self.num_mag + self.num_pose_images + 1))
            mag_idxes = mag_indxes_1 + mag_indees_2
            self.training_samples_df = pd.DataFrame(columns=['reference_x', 'other_xs', 'target_x'])
            for t in mag_idxes:
                new_df = pd.DataFrame([[ref_idx, pose_idxes, t]], columns=['reference_x', 'other_xs', 'target_x'])
                self.training_samples_df = self.training_samples_df.append(new_df)
        return self.training_samples_df

def prepareDataset(graphics, train=True, crop_size=None, getDmap=False):
    if train == False:
        sselector = TSampleSelector('../dataset/test/WorkShop', crop_size=crop_size)
        training_df = sselector.sampling_whole()
        img_folder = '../dataset/test/WorkShop'
        tdataset = TestingDataset(training_df, img_folder, graphics)
        return tdataset
    else:
        dataset_list = []
        folder_list = os.listdir("../dataset/train")
        # folder_list.remove('Server Project Folder')
        for dsname in folder_list:
            sselector = TSampleSelector('../dataset/train/' + dsname, crop_size=crop_size)
            training_df = sselector.sampling()
            img_folder = '../dataset/train/' + dsname
            tdataset = TrainingDataset(training_df, img_folder, graphics, getDmap)
            dataset_list.append(tdataset)
        tdataset = torch.utils.data.ConcatDataset(dataset_list)
        return tdataset
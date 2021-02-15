from torch.utils.data import Dataset
import os.path as path
import cv2
from graphics import *

def read_raw_tensorimg(folder, x_idx):
    img_name = "Frame_" + str(int(x_idx)).zfill(3) + '.png'
    img_path = path.join(folder, img_name)
    img = cv2.imread(img_path) / 255
    return img

class TrainingDataset(Dataset):#需要继承data.Dataset
    def __init__(self, training_df, img_folder, graphics, get_depthmap=False):
        # TODO
        # 1. Initialize file path or list of file names.
        self.training_df = training_df
        self.img_folder = img_folder
        self.graphcis = graphics
        self.getdmap = get_depthmap
        
    def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        #这里需要注意的是，第一步：read one data，是一个data
        reference_x, other_xs, target_x, crop_cor = list(self.training_df.iloc[index, :])
        other_xs = np.array(other_xs)
        ref_img = read_raw_tensorimg(self.img_folder, reference_x)[crop_cor[0] : crop_cor[1], crop_cor[2] : crop_cor[3]]
        other_imgs = []
        for other_x in other_xs:
            other_img = read_raw_tensorimg(self.img_folder, other_x)[crop_cor[0] : crop_cor[1], crop_cor[2] : crop_cor[3]]
            other_imgs.append(other_img)
        #other_imgs = tuple(other_imgs)
        #other_imgs = np.concatenate(other_imgs, axis=0)
        other_imgs = np.array(other_imgs)
        input_feat = self.graphcis.plane_sweep(reference_x, other_xs, ref_img, other_imgs)
        input_feat = np.transpose(input_feat, (2, 0, 1))
        target_img = np.transpose(read_raw_tensorimg(self.img_folder, target_x)[crop_cor[0] : crop_cor[1], crop_cor[2] : crop_cor[3]], (2, 0, 1))
        ref_img = np.transpose(ref_img, (2, 0, 1))
        #input_feat = torch.from_numpy(input_feat)
        #target_x = torch.from_numpy(np.array(target_x))
        if not self.getdmap:
            return input_feat, reference_x, target_x, target_img, ref_img
        else:
            return input_feat, reference_x, target_x, target_img, ref_img, self.get_depth_map(reference_x, other_xs[-1])[:, crop_cor[0] : crop_cor[1], crop_cor[2] : crop_cor[3]]
        
    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.training_df)

    def get_depth_map(self, reference_x, other_x):
        # laod the reference image and the other_x image:
        reference_img = np.mean(read_raw_tensorimg(self.img_folder, reference_x) * 255, 2).astype(np.uint8) # output shape: h, w
        other_img = np.mean(read_raw_tensorimg(self.img_folder, other_x) * 255, 2).astype(np.uint8) # output shape: h, w
        # calculate the stereo depth image:
        stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
        disparity_map = stereo.compute(reference_img, other_img)
        disparity_map = (disparity_map - min(disparity_map.reshape(-1))) / (
                    max(disparity_map.reshape(-1)) - min(disparity_map.reshape(-1)))
        return np.expand_dims(disparity_map, 0) # shape: 1, h, w

class TestingDataset(Dataset):
    def __init__(self, training_df, img_folder, graphics):
        # TODO
        # 1. Initialize file path or list of file names.
        self.training_df = training_df
        self.img_folder = img_folder
        self.graphics = graphics

    def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        # 这里需要注意的是，第一步：read one data，是一个data
        reference_x, other_xs, target_x = list(self.training_df.iloc[index, :])
        other_xs = np.array(other_xs)
        ref_img = read_raw_tensorimg(self.img_folder, reference_x)
        other_imgs = []
        for other_x in other_xs:
            other_img = read_raw_tensorimg(self.img_folder, other_x)
            other_imgs.append(other_img)
        # other_imgs = tuple(other_imgs)
        # other_imgs = np.concatenate(other_imgs, axis=0)
        other_imgs = np.array(other_imgs)
        input_feat = self.graphics.plane_sweep(reference_x, other_xs, ref_img, other_imgs)
        input_feat = np.transpose(input_feat, (2, 0, 1))
        ref_img = np.transpose(ref_img, (2, 0, 1))
        return input_feat, reference_x, target_x, ref_img

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.training_df)
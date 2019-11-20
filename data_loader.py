#coding=utf-8

import pandas as pd
import numpy as np
from torch.utils.data import Dataset,DataLoader
import copy
import glob
import json
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# from train import config


def plot_sample(x, y, axis):
    """

    :param x: (9216,)
    :param y: (4,2)
    :param axis:
    :return:
    """
    img = x.reshape(128, 128)
    axis.imshow(img, cmap='gray')
    axis.scatter(y[:,0], y[:,1], marker='x', s=10)

def plot_demo(X,y):
    fig = plt.figure(figsize=(6, 6))
    fig.subplots_adjust(
        left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    for i in range(16):
        ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
        plot_sample(X[i], y[i], ax)

    plt.show()


class KFDataset(Dataset):
    def __init__(self,config,X=None,gts=None):
        """

        :param X: (N,128*128)
        :param gts: (N,4,2)
        """
        self.__X = X
        self.__gts = gts
        self.__sigma = config['sigma']
        self.__debug_vis = config['debug_vis']
        self.__fname = config['fname']
        self.__is_test = config['is_test']        
        fnames = glob.glob(config['fname']+"*.jpg")
        self.__X = fnames
        gtnames = [name.replace("jpg","json") for name in fnames]
        self.__gts = gtnames

 

    def __len__(self):
        return len(self.__X)

    def __getitem__(self, item):
        H,W = 128,128
        x_name = self.__X[item]
        x = Image.open(x_name)
        x = x.resize((128,128),Image.ANTIALIAS)

        gt_name = self.__gts[item]
        gt = []
        with open(gt_name,'r') as f:
            gt_json = json.load(f) 
            gt.append(gt_json["shapes"][0]["points"][0])
            gt.append(gt_json["shapes"][1]["points"][0])
            gt.append(gt_json["shapes"][2]["points"][0])
            gt.append(gt_json["shapes"][3]["points"][0])
        gt = np.array(gt)
        gt = gt/4.0
        
        # gt 4x2, H 256, W 256
        # stride = 1
        heatmaps = self._putGaussianMaps(gt,H,W,1,self.__sigma)
        
        x = np.array(x)
        
        if self.__debug_vis == True:
            for i in range(heatmaps.shape[0]):
                x = cv2.cvtColor(x,cv2.COLOR_RGB2GRAY)
                img = copy.deepcopy(x).astype(np.uint8).reshape((H,W))
                self.visualize_heatmap_target(img,copy.deepcopy(heatmaps),1)
        
        x = x.reshape((3,128,128)).astype(np.float32)
        x = x / 255.
        heatmaps = heatmaps.astype(np.float32)
        return x,heatmaps,gt

    def _putGaussianMap(self, center, visible_flag, crop_size_y, crop_size_x, stride, sigma):
        """
        根据一个中心点,生成一个heatmap
        :param center:
        :return:
        """
        grid_y = int(crop_size_y / stride) # 256/1
        grid_x = int(crop_size_x / stride) # 256/1
        if visible_flag == False:
            return np.zeros((grid_y,grid_x))
        start = stride / 2.0 - 0.5
        y_range = [i for i in range(grid_y)]
        x_range = [i for i in range(grid_x)]
        xx, yy = np.meshgrid(x_range, y_range)
        xx = xx * stride + start
        yy = yy * stride + start
        d2 = (xx - center[0]) ** 2 + (yy - center[1]) ** 2
        exponent = d2 / 2.0 / sigma / sigma
        heatmap = np.exp(-exponent)
        return heatmap

    def _putGaussianMaps(self,keypoints,crop_size_y, crop_size_x, stride, sigma):
        """

        :param keypoints: (4,2)
        :param crop_size_y: int  128
        :param crop_size_x: int  128
        :param stride: int  1
        :param sigma: float   1e-4
        :return:
        """
        all_keypoints = keypoints #4,2
        point_num = all_keypoints.shape[0]  # 4
        heatmaps_this_img = []
        for k in range(point_num):  # 0,1,2,3
            flag = ~np.isnan(all_keypoints[k,0])
            heatmap = self._putGaussianMap(all_keypoints[k],flag,crop_size_y,crop_size_x,stride,sigma)
            heatmap = heatmap[np.newaxis,...]
            heatmaps_this_img.append(heatmap)
        heatmaps_this_img = np.concatenate(heatmaps_this_img,axis=0) # (num_joint,crop_size_y/stride,crop_size_x/stride)
        return heatmaps_this_img

    def visualize_heatmap_target(self,oriImg,heatmap,stride):
        plt.subplot(2,2,1)
        plt.imshow(oriImg)
        plt.imshow(heatmap[0], alpha=.5)
        plt.subplot(2,2,2)
        plt.imshow(oriImg)
        plt.imshow(heatmap[1], alpha=.5)
        plt.subplot(2,2,3)
        plt.imshow(oriImg)
        plt.imshow(heatmap[2], alpha=.5)
        plt.subplot(2,2,4)
        plt.imshow(oriImg)
        plt.imshow(heatmap[3], alpha=.5)
        plt.show()

if __name__ == '__main__':
    from train import config
    dataset = KFDataset(config)
    
    dataLoader = DataLoader(dataset=dataset,batch_size=8,shuffle=True)
    for i, (x, y ,gt) in enumerate(dataLoader):
        print(x.size())
        print(y.size())
        print(gt.size())
